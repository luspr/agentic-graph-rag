"""Hybrid retriever combining vector search with graph expansion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agentic_graph_rag.graph.base import GraphDatabase
from agentic_graph_rag.llm.base import LLMClient
from agentic_graph_rag.retriever.base import (
    RetrievalResult,
    RetrievalStep,
    RetrievalStrategy,
    Retriever,
)
from agentic_graph_rag.vector.base import VectorStore

_PROPERTY_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class _QueryVariant:
    name: str
    text: str


class HybridRetriever(Retriever):
    """Retriever that combines vector search and graph traversal."""

    def __init__(
        self,
        graph_db: GraphDatabase,
        vector_store: VectorStore,
        llm_client: LLMClient,
        uuid_property: str = "uuid",
    ) -> None:
        """Initialize the HybridRetriever.

        Args:
            graph_db: Graph database client for Cypher execution.
            vector_store: Vector store for semantic search.
            llm_client: LLM client for embedding generation.
            uuid_property: Node property name used as the stable UUID identifier.
        """
        if not _PROPERTY_NAME_PATTERN.match(uuid_property):
            raise ValueError(f"Invalid UUID property name: {uuid_property}")
        self._graph_db = graph_db
        self._vector_store = vector_store
        self._llm_client = llm_client
        self._uuid_property = uuid_property

    async def retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Execute a hybrid retrieval action.

        Args:
            query: Natural language query or node UUID depending on action.
            context: Context dict containing action and options.

        Returns:
            RetrievalResult containing data and retrieval steps.
        """
        action = (context or {}).get("action", "vector_search")
        if action == "vector_search":
            return await self._vector_search(query, context or {})
        if action == "expand_node":
            return await self._expand_node(query, context or {})

        step = RetrievalStep(
            action="unknown_action",
            input={"query": query, "context": context or {}},
            output={},
            error=f"Unknown hybrid action: {action}",
        )
        return RetrievalResult(
            data=[],
            steps=[step],
            success=False,
            message=f"Unknown hybrid action: {action}",
        )

    @property
    def strategy(self) -> RetrievalStrategy:
        """Return the retrieval strategy type."""
        return RetrievalStrategy.HYBRID

    async def _vector_search(
        self, query: str, context: dict[str, Any]
    ) -> RetrievalResult:
        """Run a vector search and return matching node UUIDs."""
        limit = _coerce_positive_int(context.get("limit"), default=5)
        filters = context.get("filters")
        rrf_k = _coerce_positive_int(context.get("rrf_k"), default=_DEFAULT_RRF_K)
        variants = _build_query_variants(query)

        step = RetrievalStep(
            action="vector_search",
            input={
                "query": query,
                "limit": limit,
                "filters": filters,
                "rrf_k": rrf_k,
                "query_variants": [
                    {"variant": variant.name, "query": variant.text}
                    for variant in variants
                ],
            },
            output={},
            error=None,
        )

        variant_results: list[tuple[_QueryVariant, list[dict[str, Any]]]] = []
        try:
            for variant in variants:
                embedding = await self._llm_client.embed(variant.text)
                results = await self._vector_store.search(
                    embedding,
                    limit=limit,
                    filter=filters,
                )
                variant_results.append(
                    (
                        variant,
                        [
                            {
                                "uuid": result.id,
                                "vector_score": result.score,
                                "payload": result.payload,
                            }
                            for result in results
                        ],
                    )
                )
        except Exception as exc:
            step.error = str(exc)
            return RetrievalResult(
                data=[],
                steps=[step],
                success=False,
                message=f"Vector search failed: {exc}",
            )

        data = _fuse_with_rrf(variant_results, limit=limit, rrf_k=rrf_k)
        step.output = {
            "variant_results": [
                {
                    "variant": variant.name,
                    "query": variant.text,
                    "hits": len(results),
                }
                for variant, results in variant_results
            ],
            "data": data,
        }
        return RetrievalResult(
            data=data,
            steps=[step],
            success=True,
            message=(
                f"Retrieved {len(data)} fused vector matches from "
                f"{len(variants)} query variants"
            ),
        )

    async def _expand_node(
        self, query: str, context: dict[str, Any]
    ) -> RetrievalResult:
        """Expand from a UUID node into the graph."""
        node_uuid = context.get("node_id") or query
        relationship_types = context.get("relationship_types")
        depth = _coerce_positive_int(context.get("depth"), default=1)
        direction = context.get("direction", "both")
        if direction not in _VALID_DIRECTIONS:
            direction = "both"
        max_paths = _coerce_positive_int(context.get("max_paths"), default=20)
        raw_branching = context.get("max_branching")
        max_branching: int | None = (
            _coerce_positive_int(raw_branching, default=0) or None
            if raw_branching is not None
            else None
        )

        step = RetrievalStep(
            action="expand_node",
            input={
                "node_id": node_uuid,
                "relationship_types": relationship_types,
                "depth": depth,
                "direction": direction,
                "max_paths": max_paths,
                "max_branching": max_branching,
            },
            output={},
            error=None,
        )

        if not node_uuid:
            step.error = "Missing node UUID."
            return RetrievalResult(
                data=[],
                steps=[step],
                success=False,
                message="Missing node UUID for expansion.",
            )

        cypher = _build_expand_query(
            self._uuid_property,
            depth,
            relationship_types,
            direction,
            max_paths,
        )
        params: dict[str, Any] = {
            "uuid": node_uuid,
            "relationship_types": relationship_types,
            "max_paths": max_paths,
        }
        result = await self._graph_db.execute(cypher, params)

        if result.error:
            step.error = result.error
            step.output = {"records": [], "summary": result.summary}
            return RetrievalResult(
                data=[],
                steps=[step],
                success=False,
                message=f"Graph expansion failed: {result.error}",
            )

        records = _apply_max_branching(result.records, max_branching)

        step.output = {"records": records, "summary": result.summary}
        return RetrievalResult(
            data=records,
            steps=[step],
            success=True,
            message=f"Expanded {len(records)} paths from node",
        )


_VALID_DIRECTIONS = {"out", "in", "both"}

_DIRECTION_PATTERNS: dict[str, str] = {
    "out": "-[*1..{depth}]->",
    "in": "<-[*1..{depth}]-",
    "both": "-[*1..{depth}]-",
}


def _build_expand_query(
    uuid_property: str,
    depth: int,
    relationship_types: list[str] | None,
    direction: str = "both",
    max_paths: int = 20,
) -> str:
    """Build a Cypher query that returns one record per path.

    Returns path-level records with start_uuid, end_uuid, node_labels,
    path_length, ordered path_nodes, and ordered path_rels with direction.
    """
    rel_pattern = _DIRECTION_PATTERNS.get(
        direction, _DIRECTION_PATTERNS["both"]
    ).format(
        depth=depth,
    )
    match_clause = (
        f"MATCH p = (start {{{uuid_property}: $uuid}}){rel_pattern}(end_node)"
    )
    where_clause = ""
    if relationship_types:
        where_clause = (
            " WHERE ALL(rel IN relationships(p) WHERE type(rel) IN $relationship_types)"
        )
    return (
        f"{match_clause}{where_clause}"
        f" WITH p, start, end_node, length(p) AS path_length"
        f" ORDER BY path_length"
        f" LIMIT $max_paths"
        f" RETURN start.{uuid_property} AS start_uuid,"
        f" end_node.{uuid_property} AS end_uuid,"
        f" labels(end_node) AS node_labels,"
        f" path_length,"
        f" [n IN nodes(p) | {{uuid: n.{uuid_property},"
        f" labels: labels(n),"
        f" name: coalesce(n.name, n.title, null)}}] AS path_nodes,"
        f" [rel IN relationships(p) | {{type: type(rel),"
        f" from_uuid: startNode(rel).{uuid_property},"
        f" to_uuid: endNode(rel).{uuid_property}}}] AS path_rels"
    )


def _apply_max_branching(
    records: list[dict[str, Any]],
    max_branching: int | None,
) -> list[dict[str, Any]]:
    """Filter paths so no parent node exceeds *max_branching* distinct children.

    Processes paths in order (assumed shortest-first from Cypher ORDER BY).
    Tracks ``parent_uuid → {child_uuid}`` across all kept paths and drops any
    path where a parent would exceed the limit.

    Returns *records* unchanged when *max_branching* is ``None``.
    """
    if max_branching is None:
        return records

    parent_children: dict[str, set[str]] = {}
    kept: list[dict[str, Any]] = []

    for record in records:
        path_nodes: list[dict[str, Any]] = record.get("path_nodes", [])
        if len(path_nodes) < 2:
            kept.append(record)
            continue

        # Check every consecutive (parent, child) edge in the path
        exceeds = False
        edges: list[tuple[str, str]] = []
        for i in range(len(path_nodes) - 1):
            parent_uuid = path_nodes[i].get("uuid", "")
            child_uuid = path_nodes[i + 1].get("uuid", "")
            existing = parent_children.get(parent_uuid, set())
            if child_uuid not in existing and len(existing) >= max_branching:
                exceeds = True
                break
            edges.append((parent_uuid, child_uuid))

        if exceeds:
            continue

        # Commit edges
        for parent_uuid, child_uuid in edges:
            parent_children.setdefault(parent_uuid, set()).add(child_uuid)
        kept.append(record)

    return kept


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _build_query_variants(query: str) -> list[_QueryVariant]:
    normalized_query = " ".join(query.split())
    return [
        _QueryVariant(name="original", text=query),
        _QueryVariant(
            name="entity_focused",
            text=f"Entity-focused rewrite: {normalized_query}",
        ),
        _QueryVariant(
            name="hypothesis_style",
            text=f"Hypothesis-style rewrite: {normalized_query}",
        ),
    ]


def _fuse_with_rrf(
    variant_results: list[tuple[_QueryVariant, list[dict[str, Any]]]],
    *,
    limit: int,
    rrf_k: int,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}

    for variant, results in variant_results:
        for rank, result in enumerate(results, start=1):
            uuid = result["uuid"]
            candidate = fused.setdefault(
                uuid,
                {
                    "rrf_score": 0.0,
                    "best_rank": rank,
                    "best_vector_score": result["vector_score"],
                    "payload": result["payload"],
                    "provenance": [],
                },
            )
            candidate["rrf_score"] += 1.0 / (rrf_k + rank)
            candidate["provenance"].append(
                {
                    "variant": variant.name,
                    "query": variant.text,
                    "rank": rank,
                    "vector_score": result["vector_score"],
                }
            )
            if rank < candidate["best_rank"] or (
                rank == candidate["best_rank"]
                and result["vector_score"] > candidate["best_vector_score"]
            ):
                candidate["best_rank"] = rank
                candidate["best_vector_score"] = result["vector_score"]
                candidate["payload"] = result["payload"]

    sorted_items = sorted(
        fused.items(),
        key=lambda item: (
            -item[1]["rrf_score"],
            item[1]["best_rank"],
            item[0],
        ),
    )
    return [
        {
            "uuid": uuid,
            "score": data["rrf_score"],
            "payload": data["payload"],
            "provenance": data["provenance"],
        }
        for uuid, data in sorted_items[:limit]
    ]
