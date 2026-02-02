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

_NODE_COUNTS_QUERY = "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count"

_REL_STRUCTURE_QUERY = (
    "MATCH (a)-[r]->(b) "
    "RETURN DISTINCT type(r) AS type, "
    "labels(a)[0] AS start_label, labels(b)[0] AS end_label"
)

_REL_PROPERTIES_QUERY = (
    "CALL db.schema.relationshipTypeProperties() "
    "YIELD relationshipType, propertyName, propertyTypes"
)


class Neo4jClient(GraphDatabase):
    """Async Neo4j client with connection pool management."""

    def __init__(self, settings: Settings) -> None:
        """Initialize with application settings."""
        self._settings = settings
        self._driver: AsyncDriver | None = None

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

    async def _fetch_node_types(self) -> list[NodeType]:
        """Query node labels with their properties and counts."""
        props_result = await self.execute(_NODE_PROPERTIES_QUERY)
        counts_result = await self.execute(_NODE_COUNTS_QUERY)

        if props_result.error or counts_result.error:
            return []

        counts: dict[str, int] = {
            row["label"]: row["count"]
            for row in counts_result.records
            if row.get("label")
        }

        properties: dict[str, dict[str, str]] = defaultdict(dict)
        for row in props_result.records:
            labels = row.get("nodeLabels", [])
            prop_types = row.get("propertyTypes", [])
            if labels:
                label = labels[0]
                properties[label][row["propertyName"]] = (
                    prop_types[0] if prop_types else "Unknown"
                )

        all_labels = set(counts.keys()) | set(properties.keys())
        return [
            NodeType(
                label=label,
                properties=properties.get(label, {}),
                count=counts.get(label, 0),
            )
            for label in sorted(all_labels)
        ]

    async def _fetch_relationship_types(self) -> list[RelationshipType]:
        """Query relationship types with source/target labels and properties."""
        structure_result = await self.execute(_REL_STRUCTURE_QUERY)
        props_result = await self.execute(_REL_PROPERTIES_QUERY)

        if structure_result.error:
            return []

        properties: dict[str, dict[str, str]] = defaultdict(dict)
        if not props_result.error:
            for row in props_result.records:
                prop_types = row.get("propertyTypes", [])
                properties[row["relationshipType"]][row["propertyName"]] = (
                    prop_types[0] if prop_types else "Unknown"
                )

        return [
            RelationshipType(
                type=row["type"],
                start_label=row.get("start_label") or "",
                end_label=row.get("end_label") or "",
                properties=properties.get(row["type"], {}),
            )
            for row in structure_result.records
            if row.get("type")
        ]
