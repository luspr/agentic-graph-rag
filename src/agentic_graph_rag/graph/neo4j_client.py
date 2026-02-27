from collections import defaultdict
from types import TracebackType
from typing import Any

from neo4j import AsyncDriver, NotificationDisabledClassification
from neo4j import AsyncGraphDatabase as Neo4jAsyncGraphDatabase
from neo4j.exceptions import Neo4jError

from agentic_graph_rag.config import Settings
from agentic_graph_rag.graph.base import (
    GraphDatabase,
    GraphSchema,
    NodeType,
    QueryResult,
    RelationshipType,
)

_NODE_PROPERTIES_QUERY = (
    "CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName, propertyTypes"
)

_NODE_COUNTS_QUERY = "MATCH (n) RETURN labels(n) AS labels, count(*) AS count"

_REL_STRUCTURE_QUERY = (
    "MATCH (a)-[r]->(b) "
    "RETURN DISTINCT type(r) AS type, "
    "labels(a) AS start_labels, labels(b) AS end_labels"
)

_REL_PROPERTIES_QUERY = (
    "CALL db.schema.relationshipTypeProperties() "
    "YIELD relationshipType, propertyName, propertyTypes"
)

_REL_PROPERTIES_SAMPLE_LIMIT = 10_000

_REL_PROPERTIES_FALLBACK_QUERY = (
    "MATCH ()-[r]->() "
    "WITH r LIMIT $rel_limit "
    "WITH type(r) AS relationshipType, r "
    "UNWIND keys(r) AS propertyName "
    "WITH relationshipType, propertyName, valueType(r[propertyName]) AS propertyType "
    "RETURN relationshipType, propertyName, collect(DISTINCT propertyType) AS propertyTypes"
)


def _canonical_labels(raw_labels: Any) -> tuple[str, ...]:
    """Normalize Neo4j label arrays into a deterministic label-set key."""
    if not isinstance(raw_labels, list):
        return ()
    labels = [label for label in raw_labels if isinstance(label, str) and label]
    return tuple(sorted(set(labels)))


def _label_expression(labels: tuple[str, ...]) -> str:
    """Return a display-friendly label expression for prompt rendering."""
    return ":".join(labels) if labels else "UNLABELED"


def _normalize_relationship_type(raw_type: Any) -> str:
    """Normalize relationship type strings from Neo4j schema procedures."""
    if not isinstance(raw_type, str):
        return ""
    return raw_type.lstrip(":")


class Neo4jClient(GraphDatabase):
    """Async Neo4j client with connection pool management."""

    def __init__(self, settings: Settings) -> None:
        """Initialize with application settings."""
        self._settings = settings
        self._driver: AsyncDriver | None = None
        self._gds_available: bool | None = None

    async def __aenter__(self) -> "Neo4jClient":
        """Open connection pool on context entry."""
        self._driver = Neo4jAsyncGraphDatabase.driver(
            self._settings.neo4j_uri,
            auth=(self._settings.neo4j_user, self._settings.neo4j_password),
            notifications_disabled_classifications=[
                NotificationDisabledClassification.DEPRECATION,
                NotificationDisabledClassification.GENERIC,
            ],
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close connection pool on context exit."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    @property
    def _active_driver(self) -> AsyncDriver:
        """Return the driver, raising if not connected."""
        if self._driver is None:
            raise RuntimeError(
                "Client is not connected. Use 'async with Neo4jClient(...)' to connect."
            )
        return self._driver

    async def execute(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> QueryResult:
        """Execute a Cypher query and return records with summary."""
        try:
            async with self._active_driver.session() as session:
                result = await session.run(cypher, params or {})
                records = await result.data()
                summary = await result.consume()
                return QueryResult(
                    records=records,
                    summary={"query_type": summary.query_type},
                )
        except Neo4jError as e:
            return QueryResult(records=[], summary={}, error=str(e))

    async def get_schema(self) -> GraphSchema:
        """Retrieve graph schema with node labels, relationships, and properties."""
        node_types = await self._fetch_node_types()
        relationship_types = await self._fetch_relationship_types()
        return GraphSchema(
            node_types=node_types,
            relationship_types=relationship_types,
        )

    async def validate_query(self, cypher: str) -> tuple[bool, str | None]:
        """Validate Cypher syntax using EXPLAIN without executing."""
        try:
            async with self._active_driver.session() as session:
                result = await session.run(f"EXPLAIN {cypher}")
                await result.consume()
            return (True, None)
        except Neo4jError as e:
            return (False, str(e))

    async def has_gds(self) -> bool:
        """Check whether the Neo4j Graph Data Science library is available.

        Caches the result after the first successful probe.
        """
        if self._gds_available is not None:
            return self._gds_available
        result = await self.execute("RETURN gds.version() AS version")
        self._gds_available = result.error is None
        return self._gds_available

    async def _fetch_node_types(self) -> list[NodeType]:
        """Query node classes (exact label sets) with their properties and counts."""
        props_result = await self.execute(_NODE_PROPERTIES_QUERY)
        counts_result = await self.execute(_NODE_COUNTS_QUERY)

        if props_result.error or counts_result.error:
            return []

        counts: dict[tuple[str, ...], int] = {}
        for row in counts_result.records:
            labels = _canonical_labels(row.get("labels", []))
            count = row.get("count")
            if isinstance(count, int):
                counts[labels] = count

        properties: dict[tuple[str, ...], dict[str, str]] = defaultdict(dict)
        for row in props_result.records:
            labels = _canonical_labels(row.get("nodeLabels", []))
            property_name = row.get("propertyName")
            if not isinstance(property_name, str) or not property_name:
                continue
            prop_types = row.get("propertyTypes", [])
            property_type = prop_types[0] if prop_types else "Unknown"
            properties[labels][property_name] = str(property_type)

        all_label_sets = set(counts.keys()) | set(properties.keys())
        sorted_label_sets = sorted(
            all_label_sets,
            key=lambda labels: (-counts.get(labels, 0), _label_expression(labels)),
        )
        return [
            NodeType(
                labels=labels,
                label_expression=_label_expression(labels),
                properties=properties.get(labels, {}),
                count=counts.get(labels, 0),
            )
            for labels in sorted_label_sets
        ]

    async def _fetch_relationship_types(self) -> list[RelationshipType]:
        """Query relationship types with full source/target label sets and properties."""
        structure_result = await self.execute(_REL_STRUCTURE_QUERY)
        props_result = await self.execute(_REL_PROPERTIES_QUERY)

        if structure_result.error:
            return []

        properties: dict[str, dict[str, str]] = defaultdict(dict)
        props_records: list[dict[str, Any]] = []
        if not props_result.error and props_result.records:
            props_records = props_result.records
        else:
            fallback_result = await self.execute(
                _REL_PROPERTIES_FALLBACK_QUERY,
                {"rel_limit": _REL_PROPERTIES_SAMPLE_LIMIT},
            )
            if not fallback_result.error:
                props_records = fallback_result.records

        for row in props_records:
            prop_types = row.get("propertyTypes", [])
            relationship_type = _normalize_relationship_type(
                row.get("relationshipType")
            )
            property_name = row.get("propertyName")
            if (
                not relationship_type
                or not isinstance(property_name, str)
                or not property_name
            ):
                continue
            properties[relationship_type][property_name] = (
                str(prop_types[0]) if prop_types else "Unknown"
            )

        relationships: list[RelationshipType] = []
        seen_keys: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
        for row in structure_result.records:
            relationship_type = _normalize_relationship_type(row.get("type"))
            if not relationship_type:
                continue

            start_labels = _canonical_labels(row.get("start_labels", []))
            end_labels = _canonical_labels(row.get("end_labels", []))
            key = (relationship_type, start_labels, end_labels)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            relationships.append(
                RelationshipType(
                    type=relationship_type,
                    start_labels=start_labels,
                    end_labels=end_labels,
                    start_label_expression=_label_expression(start_labels),
                    end_label_expression=_label_expression(end_labels),
                    properties=properties.get(relationship_type, {}),
                )
            )

        return sorted(
            relationships,
            key=lambda rel: (
                rel.type,
                rel.start_label_expression,
                rel.end_label_expression,
            ),
        )
