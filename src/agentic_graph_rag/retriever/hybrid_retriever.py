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
        if action == "shortest_path":
            return await self._shortest_path(query, context or {})
        if action == "pagerank":
            return await self._pagerank(query, context or {})

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

    async def _shortest_path(
        self, query: str, context: dict[str, Any]
    ) -> RetrievalResult:
        """Find shortest path(s) between two nodes."""
        source_id = context.get("source_id", "")
        target_id = context.get("target_id", "")
        relationship_types = context.get("relationship_types")
        max_length = _coerce_positive_int(context.get("max_length"), default=10)
        all_shortest = bool(context.get("all_shortest", False))

        step = RetrievalStep(
            action="shortest_path",
            input={
                "source_id": source_id,
                "target_id": target_id,
                "relationship_types": relationship_types,
                "max_length": max_length,
                "all_shortest": all_shortest,
            },
            output={},
            error=None,
        )

        if not source_id or not target_id:
            step.error = "Both source_id and target_id are required."
            return RetrievalResult(
                data=[],
                steps=[step],
                success=False,
                message="Missing source_id or target_id for shortest path.",
            )

        cypher = _build_shortest_path_query(
            self._uuid_property,
            relationship_types,
            max_length,
            all_shortest,
        )
        params: dict[str, Any] = {
            "source_uuid": source_id,
            "target_uuid": target_id,
        }
        result = await self._graph_db.execute(cypher, params)

        if result.error:
            step.error = result.error
            step.output = {"records": [], "summary": result.summary}
            return RetrievalResult(
                data=[],
                steps=[step],
                success=False,
                message=f"Shortest path query failed: {result.error}",
            )

        step.output = {"records": result.records, "summary": result.summary}
        return RetrievalResult(
            data=result.records,
            steps=[step],
            success=True,
            message=f"Found {len(result.records)} shortest path(s)",
        )

    async def _pagerank(self, query: str, context: dict[str, Any]) -> RetrievalResult:
        """Run Personalized PageRank from seed nodes."""
        source_ids: list[str] = context.get("source_ids", [])
        damping = context.get("damping", 0.85)
        limit = _coerce_positive_int(context.get("limit"), default=20)
        max_depth = _coerce_positive_int(context.get("max_depth"), default=3)
        relationship_types = context.get("relationship_types")

        step = RetrievalStep(
            action="pagerank",
            input={
                "source_ids": source_ids,
                "damping": damping,
                "limit": limit,
                "max_depth": max_depth,
                "relationship_types": relationship_types,
            },
            output={},
            error=None,
        )

        if not source_ids:
            step.error = "source_ids list is required and must not be empty."
            return RetrievalResult(
                data=[],
                steps=[step],
                success=False,
                message="Missing source_ids for PageRank.",
            )

        backend = "cypher_fallback"
        records: list[dict[str, Any]] = []

        if await self._graph_db.has_gds():
            cypher = _build_ppr_gds_query(self._uuid_property, relationship_types)
            params: dict[str, Any] = {
                "source_uuids": source_ids,
                "damping": damping,
                "limit": limit,
            }
            result = await self._graph_db.execute(cypher, params)
            if not result.error:
                backend = "gds"
                records = result.records

        if not records and backend != "gds":
            cypher = _build_ppr_cypher_fallback(
                self._uuid_property, max_depth, relationship_types
            )
            params = {
                "source_uuids": source_ids,
                "damping": damping,
                "limit": limit,
            }
            result = await self._graph_db.execute(cypher, params)
            if result.error:
                step.error = result.error
                step.output = {
                    "records": [],
                    "summary": result.summary,
                    "backend": backend,
                }
                return RetrievalResult(
                    data=[],
                    steps=[step],
                    success=False,
                    message=f"PageRank query failed: {result.error}",
                )
            records = result.records

        step.output = {
            "records": records,
            "summary": {},
            "backend": backend,
        }
        return RetrievalResult(
            data=records,
            steps=[step],
            success=True,
            message=(f"PageRank returned {len(records)} nodes (backend: {backend})"),
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


def _build_shortest_path_query(
    uuid_property: str,
    relationship_types: list[str] | None,
    max_length: int,
    all_shortest: bool,
) -> str:
    """Build a Cypher shortest-path query between two nodes.

    Uses built-in shortestPath() or allShortestPaths() — no GDS required.
    """
    func = "allShortestPaths" if all_shortest else "shortestPath"
    rel_filter = ""
    if relationship_types:
        rel_filter = ":" + "|".join(relationship_types)
    return (
        f"MATCH (src {{{uuid_property}: $source_uuid}}),"
        f" (tgt {{{uuid_property}: $target_uuid}})"
        f" MATCH p = {func}((src)-[{rel_filter}*..{max_length}]-(tgt))"
        f" RETURN src.{uuid_property} AS start_uuid,"
        f" tgt.{uuid_property} AS end_uuid,"
        f" length(p) AS path_length,"
        f" [n IN nodes(p) | {{uuid: n.{uuid_property},"
        f" labels: labels(n),"
        f" name: coalesce(n.name, n.title, null)}}] AS path_nodes,"
        f" [rel IN relationships(p) | {{type: type(rel),"
        f" from_uuid: startNode(rel).{uuid_property},"
        f" to_uuid: endNode(rel).{uuid_property}}}] AS path_rels"
    )


def _build_ppr_gds_query(
    uuid_property: str,
    relationship_types: list[str] | None,
) -> str:
    """Build a GDS Personalized PageRank query using anonymous graph projection."""
    node_proj = "'*'"
    if relationship_types:
        rel_proj = (
            "{"
            + ", ".join(f"{rt}: {{type: '{rt}'}}" for rt in relationship_types)
            + "}"
        )
    else:
        rel_proj = "'*'"
    return (
        f"MATCH (seed) WHERE seed.{uuid_property} IN $source_uuids"
        f" WITH collect(seed) AS seeds"
        f" CALL gds.pageRank.stream({node_proj}, {rel_proj},"
        f" {{sourceNodes: seeds, dampingFactor: $damping}})"
        f" YIELD nodeId, score"
        f" WITH gds.util.asNode(nodeId) AS n, score"
        f" RETURN n.{uuid_property} AS uuid,"
        f" labels(n) AS labels,"
        f" coalesce(n.name, n.title, null) AS name,"
        f" score AS ppr_score"
        f" ORDER BY ppr_score DESC LIMIT $limit"
    )


def _build_ppr_cypher_fallback(
    uuid_property: str,
    max_depth: int,
    relationship_types: list[str] | None,
) -> str:
    """Build a Cypher heuristic for Personalized PageRank using path counting.

    Scores each reachable node as SUM(damping^distance) normalized by seed count.
    """
    rel_filter = ""
    if relationship_types:
        rel_filter = ":" + "|".join(relationship_types)
    return (
        f"MATCH (seed) WHERE seed.{uuid_property} IN $source_uuids"
        f" WITH collect(seed) AS seeds, count(seed) AS seed_count"
        f" UNWIND seeds AS s"
        f" MATCH p = (s)-[{rel_filter}*1..{max_depth}]-(target)"
        f" WHERE target <> s"
        f" WITH target, seed_count,"
        f" length(p) AS dist,"
        f" count(p) AS path_count,"
        f" $damping AS damping"
        f" WITH target, seed_count,"
        f" sum(reduce(acc = 1.0, _ IN range(1, dist) |"
        f" acc * damping) * path_count) AS raw_score,"
        f" min(dist) AS min_distance,"
        f" sum(path_count) AS total_paths"
        f" RETURN target.{uuid_property} AS uuid,"
        f" labels(target) AS labels,"
        f" coalesce(target.name, target.title, null) AS name,"
        f" raw_score / seed_count AS ppr_score,"
        f" min_distance,"
        f" total_paths AS path_count"
        f" ORDER BY ppr_score DESC LIMIT $limit"
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


@dataclass(frozen=True)
class HybridScoreWeights:
    """Configurable weights for hybrid score blending."""

    vector_weight: float = 0.4
    graph_weight: float = 0.3
    novelty_weight: float = 0.15
    relation_prior_weight: float = 0.15
    hub_penalty_factor: float = 0.1
    path_length_decay: float = 0.2


_DEFAULT_WEIGHTS = HybridScoreWeights()


def blend_scores(
    candidates: list[dict[str, Any]],
    graph_paths: list[dict[str, Any]] | None = None,
    weights: HybridScoreWeights | None = None,
) -> list[dict[str, Any]]:
    """Blend vector RRF scores with graph-aware signals for candidate ranking.

    For each candidate, computes:
    - **vector_component**: normalized RRF score * vector_weight
    - **graph_component**: path quality based on shortest path length
    - **relation_prior_component**: bonus for diverse relationship types
    - **novelty_component**: bonus for unique node labels
    - **hub_penalty**: penalty for nodes appearing in many paths

    Returns candidates sorted by blended_score descending, with tie-break
    by UUID for determinism. Each candidate gets ``blended_score`` and
    ``score_components`` added to its dict.

    Args:
        candidates: RRF-fused candidates (uuid, score, payload, provenance).
        graph_paths: Expansion paths (end_uuid, path_length, path_nodes, path_rels).
        weights: Scoring weights; uses defaults when None.
    """
    if not candidates:
        return []

    w = weights or _DEFAULT_WEIGHTS
    paths = graph_paths or []

    # Pre-compute max RRF score for normalization
    max_rrf = max(c["score"] for c in candidates) if candidates else 1.0
    if max_rrf == 0.0:
        max_rrf = 1.0

    # Index graph paths by end_uuid for fast lookup
    paths_by_uuid: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        end_uuid = path.get("end_uuid", "")
        paths_by_uuid.setdefault(end_uuid, []).append(path)

    # Count total paths and per-uuid appearances for hub penalty
    total_paths = len(paths) if paths else 1

    # Collect all labels seen across candidates for novelty calculation
    seen_labels: set[str] = set()

    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        uuid = candidate["uuid"]
        rrf_score = candidate["score"]

        # --- vector component ---
        vector_component = (rrf_score / max_rrf) * w.vector_weight

        # --- graph component ---
        candidate_paths = paths_by_uuid.get(uuid, [])
        if candidate_paths:
            min_path_length = min(p.get("path_length", 1) for p in candidate_paths)
            graph_quality = 1.0 / (1.0 + w.path_length_decay * min_path_length)
        else:
            graph_quality = 0.0
        graph_component = graph_quality * w.graph_weight

        # --- relation prior component ---
        rel_types: set[str] = set()
        for p in candidate_paths:
            for rel in p.get("path_rels", []):
                rel_type = rel.get("type", "")
                if rel_type:
                    rel_types.add(rel_type)
        # More unique rel types = higher score (capped at 1.0)
        rel_diversity = min(len(rel_types) / 3.0, 1.0) if rel_types else 0.0
        relation_prior_component = rel_diversity * w.relation_prior_weight

        # --- novelty component ---
        candidate_labels: set[str] = set()
        payload = candidate.get("payload", {})
        if isinstance(payload, dict):
            for label in payload.get("labels", []):
                candidate_labels.add(label)
        # Also extract from graph path nodes
        for p in candidate_paths:
            for node in p.get("path_nodes", []):
                for label in node.get("labels", []):
                    candidate_labels.add(label)

        new_labels = candidate_labels - seen_labels
        novelty = (
            len(new_labels) / max(len(candidate_labels), 1) if candidate_labels else 0.0
        )
        novelty_component = novelty * w.novelty_weight
        seen_labels.update(candidate_labels)

        # --- hub penalty ---
        appearance_count = len(candidate_paths)
        if appearance_count > 1:
            hub_penalty = w.hub_penalty_factor * ((appearance_count - 1) / total_paths)
            hub_penalty = min(hub_penalty, w.hub_penalty_factor)
        else:
            hub_penalty = 0.0

        # --- blended score ---
        blended_score = (
            vector_component
            + graph_component
            + relation_prior_component
            + novelty_component
            - hub_penalty
        )

        scored.append(
            {
                **candidate,
                "blended_score": blended_score,
                "score_components": {
                    "vector": vector_component,
                    "graph": graph_component,
                    "relation_prior": relation_prior_component,
                    "novelty": novelty_component,
                    "hub_penalty": hub_penalty,
                },
            }
        )

    scored.sort(key=lambda c: (-c["blended_score"], c["uuid"]))
    return scored
