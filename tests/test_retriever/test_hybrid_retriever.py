"""Unit tests for HybridRetriever with mocked dependencies."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_graph_rag.graph.base import GraphDatabase, QueryResult
from agentic_graph_rag.llm.base import LLMClient
from agentic_graph_rag.retriever.base import RetrievalStrategy
from agentic_graph_rag.retriever.hybrid_retriever import (
    HybridRetriever,
    HybridScoreWeights,
    _apply_max_branching,
    _build_expand_query,
    _build_ppr_cypher_fallback,
    _build_ppr_gds_query,
    _build_shortest_path_query,
    blend_scores,
)
from agentic_graph_rag.vector.base import VectorSearchResult, VectorStore


def _mock_graph_db() -> MagicMock:
    mock = MagicMock(spec=GraphDatabase)
    mock.execute = AsyncMock()
    return mock


def _mock_vector_store() -> MagicMock:
    mock = MagicMock(spec=VectorStore)
    mock.search = AsyncMock()
    return mock


def _mock_llm() -> MagicMock:
    mock = MagicMock(spec=LLMClient)
    mock.embed = AsyncMock()
    return mock


@pytest.fixture
def graph_db() -> MagicMock:
    """Mock graph database."""
    return _mock_graph_db()


@pytest.fixture
def vector_store() -> MagicMock:
    """Mock vector store."""
    return _mock_vector_store()


@pytest.fixture
def llm_client() -> MagicMock:
    """Mock LLM client."""
    return _mock_llm()


@pytest.fixture
def retriever(
    graph_db: MagicMock,
    vector_store: MagicMock,
    llm_client: MagicMock,
) -> HybridRetriever:
    """HybridRetriever instance with mocked dependencies."""
    return HybridRetriever(
        graph_db=graph_db,
        vector_store=vector_store,
        llm_client=llm_client,
        uuid_property="uuid",
    )


def test_strategy_returns_hybrid(retriever: HybridRetriever) -> None:
    """strategy property returns RetrievalStrategy.HYBRID."""
    assert retriever.strategy == RetrievalStrategy.HYBRID


@pytest.mark.anyio
async def test_vector_search_returns_results(
    retriever: HybridRetriever,
    vector_store: MagicMock,
    llm_client: MagicMock,
) -> None:
    """vector_search runs query variants and returns fused RRF-ranked seeds."""
    llm_client.embed.side_effect = [[0.1], [0.2], [0.3]]
    vector_store.search.side_effect = [
        [
            VectorSearchResult(
                id="node-1", score=0.95, payload={"title": "The Matrix"}
            ),
            VectorSearchResult(id="node-2", score=0.90, payload={"title": "Neo"}),
        ],
        [
            VectorSearchResult(id="node-2", score=0.99, payload={"title": "Neo"}),
        ],
        [
            VectorSearchResult(id="node-2", score=0.93, payload={"title": "Neo"}),
            VectorSearchResult(id="node-3", score=0.80, payload={"title": "Morpheus"}),
        ],
    ]

    result = await retriever.retrieve(
        "matrix",
        {"action": "vector_search", "limit": 3, "filters": {"must": []}},
    )

    assert result.success is True
    assert [record["uuid"] for record in result.data] == [
        "node-2",
        "node-1",
        "node-3",
    ]
    assert "provenance" in result.data[0]
    assert {item["variant"] for item in result.data[0]["provenance"]} == {
        "original",
        "entity_focused",
        "hypothesis_style",
    }
    assert [call.args[0] for call in llm_client.embed.await_args_list] == [
        "matrix",
        "Entity-focused rewrite: matrix",
        "Hypothesis-style rewrite: matrix",
    ]
    assert vector_store.search.await_count == 3
    for call in vector_store.search.await_args_list:
        assert call.kwargs["limit"] == 3
        assert call.kwargs["filter"] == {"must": []}


@pytest.mark.anyio
async def test_vector_search_rrf_tie_breaks_by_uuid_for_determinism(
    retriever: HybridRetriever,
    vector_store: MagicMock,
    llm_client: MagicMock,
) -> None:
    """Tied RRF scores are ordered by UUID to keep ranking deterministic."""
    llm_client.embed.side_effect = [[0.1], [0.2], [0.3]]
    vector_store.search.side_effect = [
        [VectorSearchResult(id="node-b", score=0.8, payload={})],
        [VectorSearchResult(id="node-a", score=0.9, payload={})],
        [],
    ]

    result = await retriever.retrieve("tie", {"action": "vector_search", "limit": 2})

    assert result.success is True
    assert [record["uuid"] for record in result.data] == ["node-a", "node-b"]


@pytest.mark.anyio
async def test_vector_search_includes_query_variant_provenance(
    retriever: HybridRetriever,
    vector_store: MagicMock,
    llm_client: MagicMock,
) -> None:
    """Each fused seed includes provenance with query-variant metadata."""
    llm_client.embed.side_effect = [[0.1], [0.2], [0.3]]
    vector_store.search.side_effect = [
        [VectorSearchResult(id="node-1", score=0.9, payload={})],
        [VectorSearchResult(id="node-1", score=0.8, payload={})],
        [],
    ]

    result = await retriever.retrieve(
        "graph retrieval",
        {"action": "vector_search", "limit": 5},
    )

    assert result.success is True
    assert len(result.data) == 1
    assert result.data[0]["uuid"] == "node-1"
    assert result.data[0]["provenance"] == [
        {
            "variant": "original",
            "query": "graph retrieval",
            "rank": 1,
            "vector_score": 0.9,
        },
        {
            "variant": "entity_focused",
            "query": "Entity-focused rewrite: graph retrieval",
            "rank": 1,
            "vector_score": 0.8,
        },
    ]


@pytest.mark.anyio
async def test_vector_search_handles_error(
    retriever: HybridRetriever,
    llm_client: MagicMock,
) -> None:
    """vector_search returns error when embedding fails."""
    llm_client.embed.side_effect = RuntimeError("boom")

    result = await retriever.retrieve("query", {"action": "vector_search"})

    assert result.success is False
    assert "Vector search failed" in result.message
    assert result.steps[0].error is not None


@pytest.mark.anyio
async def test_expand_node_executes_cypher(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """expand_node executes Cypher with UUID, direction, and max_paths."""
    graph_db.execute.return_value = QueryResult(
        records=[
            {
                "start_uuid": "node-1",
                "end_uuid": "node-2",
                "node_labels": ["Movie"],
                "path_length": 1,
                "path_nodes": [
                    {"uuid": "node-1", "labels": ["Person"], "name": "Keanu"},
                    {"uuid": "node-2", "labels": ["Movie"], "name": "The Matrix"},
                ],
                "path_rels": [
                    {"type": "ACTED_IN", "from_uuid": "node-1", "to_uuid": "node-2"},
                ],
            }
        ],
        summary={},
        error=None,
    )

    result = await retriever.retrieve(
        "node-1",
        {
            "action": "expand_node",
            "node_id": "node-1",
            "relationship_types": ["ACTED_IN"],
            "depth": 2,
        },
    )

    assert result.success is True
    assert len(result.data) == 1
    assert result.data[0]["start_uuid"] == "node-1"
    assert result.data[0]["end_uuid"] == "node-2"
    assert result.data[0]["path_length"] == 1

    cypher, params = graph_db.execute.await_args[0]
    assert "uuid: $uuid" in cypher
    assert "*1..2" in cypher
    assert "LIMIT $max_paths" in cypher
    assert "path_length" in cypher
    assert "path_nodes" in cypher
    assert "path_rels" in cypher
    assert params["uuid"] == "node-1"
    assert params["relationship_types"] == ["ACTED_IN"]
    assert params["max_paths"] == 20


@pytest.mark.anyio
async def test_expand_node_missing_uuid_returns_error(
    retriever: HybridRetriever,
) -> None:
    """expand_node returns error when node UUID is missing."""
    result = await retriever.retrieve("", {"action": "expand_node"})

    assert result.success is False
    assert "Missing node UUID" in result.message


@pytest.mark.anyio
async def test_expand_node_handles_graph_error(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """expand_node returns error when graph execution fails."""
    graph_db.execute.return_value = QueryResult(
        records=[],
        summary={},
        error="Graph error",
    )

    result = await retriever.retrieve(
        "node-1",
        {"action": "expand_node", "node_id": "node-1"},
    )

    assert result.success is False
    assert "Graph expansion failed" in result.message


@pytest.mark.anyio
async def test_unknown_action_returns_error(
    retriever: HybridRetriever,
) -> None:
    """Unknown hybrid actions return errors."""
    result = await retriever.retrieve("query", {"action": "invalid"})

    assert result.success is False
    assert "Unknown hybrid action" in result.message


# --- _build_expand_query tests ---


def test_build_query_default_direction_is_undirected() -> None:
    """Default direction 'both' produces an undirected pattern."""
    query = _build_expand_query("uuid", 1, None)
    assert "-[*1..1]-" in query
    assert "->" not in query
    assert "<-" not in query


def test_build_query_direction_out() -> None:
    """Direction 'out' produces an outgoing pattern."""
    query = _build_expand_query("uuid", 2, None, direction="out")
    assert "-[*1..2]->" in query


def test_build_query_direction_in() -> None:
    """Direction 'in' produces an incoming pattern."""
    query = _build_expand_query("uuid", 1, None, direction="in")
    assert "<-[*1..1]-" in query


def test_build_query_includes_path_return_fields() -> None:
    """Query returns start_uuid, end_uuid, node_labels, path_length, path_nodes, path_rels."""
    query = _build_expand_query("uuid", 1, None)
    assert "start_uuid" in query
    assert "end_uuid" in query
    assert "node_labels" in query
    assert "path_length" in query
    assert "path_nodes" in query
    assert "path_rels" in query


def test_build_query_includes_max_paths_limit() -> None:
    """Query includes LIMIT $max_paths."""
    query = _build_expand_query("uuid", 1, None, max_paths=10)
    assert "LIMIT $max_paths" in query


def test_build_query_includes_order_by_path_length() -> None:
    """Query orders results by path_length."""
    query = _build_expand_query("uuid", 1, None)
    assert "ORDER BY path_length" in query


def test_build_query_with_relationship_filter() -> None:
    """Query includes WHERE clause when relationship_types are provided."""
    query = _build_expand_query("uuid", 1, ["ACTED_IN"])
    assert "WHERE ALL(rel IN relationships(p)" in query
    assert "$relationship_types" in query


def test_build_query_without_relationship_filter() -> None:
    """Query omits WHERE clause when no relationship_types."""
    query = _build_expand_query("uuid", 1, None)
    assert "WHERE" not in query


def test_build_query_path_nodes_include_name() -> None:
    """path_nodes include coalesce(n.name, n.title, null) for display."""
    query = _build_expand_query("uuid", 1, None)
    assert "coalesce(n.name, n.title, null)" in query


def test_build_query_path_rels_include_direction_metadata() -> None:
    """path_rels include from_uuid and to_uuid for direction reconstruction."""
    query = _build_expand_query("uuid", 1, None)
    assert "from_uuid: startNode(rel).uuid" in query
    assert "to_uuid: endNode(rel).uuid" in query


def test_build_query_uses_custom_uuid_property() -> None:
    """Query uses the configured uuid_property throughout."""
    query = _build_expand_query("node_id", 1, None)
    assert "node_id: $uuid" in query
    assert "start.node_id AS start_uuid" in query
    assert "end_node.node_id AS end_uuid" in query
    assert "n.node_id" in query
    assert "startNode(rel).node_id" in query
    assert "endNode(rel).node_id" in query


def test_build_query_uses_path_variable() -> None:
    """Query uses MATCH p = ... pattern for path access."""
    query = _build_expand_query("uuid", 1, None)
    assert "MATCH p = " in query


# --- expand_node direction / max_paths wiring tests ---


@pytest.mark.anyio
async def test_expand_node_passes_direction_out(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """expand_node with direction='out' produces outgoing Cypher pattern."""
    graph_db.execute.return_value = QueryResult(records=[], summary={}, error=None)

    await retriever.retrieve(
        "n1",
        {"action": "expand_node", "node_id": "n1", "direction": "out"},
    )

    cypher = graph_db.execute.await_args[0][0]
    assert "->" in cypher


@pytest.mark.anyio
async def test_expand_node_passes_direction_in(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """expand_node with direction='in' produces incoming Cypher pattern."""
    graph_db.execute.return_value = QueryResult(records=[], summary={}, error=None)

    await retriever.retrieve(
        "n1",
        {"action": "expand_node", "node_id": "n1", "direction": "in"},
    )

    cypher = graph_db.execute.await_args[0][0]
    assert "<-" in cypher


@pytest.mark.anyio
async def test_expand_node_invalid_direction_falls_back_to_both(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """expand_node with invalid direction falls back to 'both'."""
    graph_db.execute.return_value = QueryResult(records=[], summary={}, error=None)

    await retriever.retrieve(
        "n1",
        {"action": "expand_node", "node_id": "n1", "direction": "invalid"},
    )

    cypher = graph_db.execute.await_args[0][0]
    assert "->" not in cypher
    assert "<-" not in cypher


@pytest.mark.anyio
async def test_expand_node_passes_max_paths(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """expand_node passes max_paths parameter to the query."""
    graph_db.execute.return_value = QueryResult(records=[], summary={}, error=None)

    await retriever.retrieve(
        "n1",
        {"action": "expand_node", "node_id": "n1", "max_paths": 5},
    )

    params = graph_db.execute.await_args[0][1]
    assert params["max_paths"] == 5


@pytest.mark.anyio
async def test_expand_node_default_max_paths_is_20(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """expand_node defaults max_paths to 20 when not specified."""
    graph_db.execute.return_value = QueryResult(records=[], summary={}, error=None)

    await retriever.retrieve(
        "n1",
        {"action": "expand_node", "node_id": "n1"},
    )

    params = graph_db.execute.await_args[0][1]
    assert params["max_paths"] == 20


@pytest.mark.anyio
async def test_expand_node_step_records_direction_and_max_paths(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """expand_node step input includes direction and max_paths."""
    graph_db.execute.return_value = QueryResult(records=[], summary={}, error=None)

    result = await retriever.retrieve(
        "n1",
        {
            "action": "expand_node",
            "node_id": "n1",
            "direction": "out",
            "max_paths": 10,
        },
    )

    step = result.steps[0]
    assert step.input["direction"] == "out"
    assert step.input["max_paths"] == 10


@pytest.mark.anyio
async def test_expand_node_message_includes_path_count(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """expand_node success message reports the number of paths."""
    graph_db.execute.return_value = QueryResult(
        records=[{"start_uuid": "a", "end_uuid": "b"}] * 3,
        summary={},
        error=None,
    )

    result = await retriever.retrieve(
        "n1",
        {"action": "expand_node", "node_id": "n1"},
    )

    assert "3 paths" in result.message


# --- _build_expand_query end_uuid rename tests ---


def test_build_query_returns_end_uuid_field() -> None:
    """Query returns end_uuid (not node_uuid) for the expanded node."""
    query = _build_expand_query("uuid", 1, None)
    assert "AS end_uuid" in query
    assert "AS node_uuid" not in query


# --- max_branching tests ---


def _make_path_record(
    start: str,
    end: str,
    path_nodes: list[dict[str, str]],
    path_length: int = 1,
) -> dict:
    return {
        "start_uuid": start,
        "end_uuid": end,
        "node_labels": ["Node"],
        "path_length": path_length,
        "path_nodes": [
            {"uuid": n["uuid"], "labels": ["Node"], "name": None} for n in path_nodes
        ],
        "path_rels": [],
    }


def test_apply_max_branching_limits_children() -> None:
    """Paths are filtered when a parent exceeds max_branching distinct children."""
    records = [
        _make_path_record("A", "B", [{"uuid": "A"}, {"uuid": "B"}]),
        _make_path_record("A", "C", [{"uuid": "A"}, {"uuid": "C"}]),
        _make_path_record("A", "D", [{"uuid": "A"}, {"uuid": "D"}]),
    ]
    result = _apply_max_branching(records, max_branching=2)
    assert len(result) == 2
    assert {r["end_uuid"] for r in result} == {"B", "C"}


def test_apply_max_branching_none_keeps_all() -> None:
    """None max_branching keeps all records unchanged."""
    records = [
        _make_path_record("A", "B", [{"uuid": "A"}, {"uuid": "B"}]),
        _make_path_record("A", "C", [{"uuid": "A"}, {"uuid": "C"}]),
        _make_path_record("A", "D", [{"uuid": "A"}, {"uuid": "D"}]),
    ]
    result = _apply_max_branching(records, max_branching=None)
    assert len(result) == 3


def test_apply_max_branching_preserves_order() -> None:
    """Shortest paths are kept first when max_branching limits expansion."""
    records = [
        _make_path_record("A", "B", [{"uuid": "A"}, {"uuid": "B"}], path_length=1),
        _make_path_record("A", "C", [{"uuid": "A"}, {"uuid": "C"}], path_length=2),
        _make_path_record("A", "D", [{"uuid": "A"}, {"uuid": "D"}], path_length=3),
    ]
    result = _apply_max_branching(records, max_branching=2)
    assert [r["path_length"] for r in result] == [1, 2]


@pytest.mark.anyio
async def test_expand_node_max_branching_filters_records(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """expand_node with max_branching filters paths after query execution."""
    graph_db.execute.return_value = QueryResult(
        records=[
            _make_path_record("A", "B", [{"uuid": "A"}, {"uuid": "B"}]),
            _make_path_record("A", "C", [{"uuid": "A"}, {"uuid": "C"}]),
            _make_path_record("A", "D", [{"uuid": "A"}, {"uuid": "D"}]),
        ],
        summary={},
        error=None,
    )

    result = await retriever.retrieve(
        "A",
        {"action": "expand_node", "node_id": "A", "max_branching": 2},
    )

    assert result.success is True
    assert len(result.data) == 2


@pytest.mark.anyio
async def test_expand_node_step_records_max_branching(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """expand_node step input includes max_branching."""
    graph_db.execute.return_value = QueryResult(records=[], summary={}, error=None)

    result = await retriever.retrieve(
        "n1",
        {"action": "expand_node", "node_id": "n1", "max_branching": 3},
    )

    step = result.steps[0]
    assert step.input["max_branching"] == 3


@pytest.mark.anyio
async def test_expand_node_max_branching_none_by_default(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """expand_node defaults max_branching to None when not specified."""
    graph_db.execute.return_value = QueryResult(records=[], summary={}, error=None)

    result = await retriever.retrieve(
        "n1",
        {"action": "expand_node", "node_id": "n1"},
    )

    step = result.steps[0]
    assert step.input["max_branching"] is None


# --- blend_scores tests ---


def _make_candidate(uuid: str, score: float, payload: dict | None = None) -> dict:
    return {
        "uuid": uuid,
        "score": score,
        "payload": payload or {},
        "provenance": [
            {"variant": "original", "query": "q", "rank": 1, "vector_score": score}
        ],
    }


def _make_graph_path(
    end_uuid: str,
    path_length: int = 1,
    path_rels: list[dict] | None = None,
    path_nodes: list[dict] | None = None,
) -> dict:
    return {
        "start_uuid": "root",
        "end_uuid": end_uuid,
        "node_labels": ["Node"],
        "path_length": path_length,
        "path_nodes": path_nodes or [],
        "path_rels": path_rels or [],
    }


def test_blend_scores_empty_candidates() -> None:
    """Empty input returns empty list."""
    assert blend_scores([]) == []


def test_blend_scores_vector_only() -> None:
    """Without graph paths, ranking follows vector score order."""
    candidates = [
        _make_candidate("A", 0.5),
        _make_candidate("B", 0.9),
        _make_candidate("C", 0.7),
    ]
    result = blend_scores(candidates, graph_paths=None)
    assert [r["uuid"] for r in result] == ["B", "C", "A"]
    for r in result:
        assert "blended_score" in r
        assert "score_components" in r
        assert r["score_components"]["graph"] == 0.0


def test_blend_scores_with_graph_paths() -> None:
    """Graph paths influence ranking — shorter path boosts candidate."""
    candidates = [
        _make_candidate("A", 0.9),
        _make_candidate("B", 0.8),
    ]
    graph_paths = [
        _make_graph_path("B", path_length=1, path_rels=[{"type": "ACTED_IN"}]),
    ]
    result = blend_scores(candidates, graph_paths=graph_paths)
    # B gets a graph boost; check that its graph component is non-zero
    b_result = next(r for r in result if r["uuid"] == "B")
    assert b_result["score_components"]["graph"] > 0.0
    a_result = next(r for r in result if r["uuid"] == "A")
    assert a_result["score_components"]["graph"] == 0.0


def test_blend_scores_hub_penalty() -> None:
    """Hub nodes appearing in many paths get penalized."""
    candidates = [
        _make_candidate("hub", 0.9),
        _make_candidate("leaf", 0.85),
    ]
    graph_paths = [
        _make_graph_path("hub", path_length=1, path_rels=[{"type": "R1"}]),
        _make_graph_path("hub", path_length=2, path_rels=[{"type": "R2"}]),
        _make_graph_path("hub", path_length=3, path_rels=[{"type": "R3"}]),
        _make_graph_path("leaf", path_length=1, path_rels=[{"type": "R1"}]),
    ]
    result = blend_scores(candidates, graph_paths=graph_paths)
    hub_result = next(r for r in result if r["uuid"] == "hub")
    assert hub_result["score_components"]["hub_penalty"] > 0.0
    leaf_result = next(r for r in result if r["uuid"] == "leaf")
    assert leaf_result["score_components"]["hub_penalty"] == 0.0


def test_blend_scores_novelty_bonus() -> None:
    """Candidates with unique labels get a novelty bonus."""
    candidates = [
        _make_candidate("A", 0.9, payload={"labels": ["Movie"]}),
        _make_candidate("B", 0.8, payload={"labels": ["Person"]}),
    ]
    result = blend_scores(candidates, graph_paths=None)
    # First candidate processed gets novelty (all labels are new)
    a_result = next(r for r in result if r["uuid"] == "A")
    assert a_result["score_components"]["novelty"] > 0.0


def test_blend_scores_deterministic() -> None:
    """Same input always produces same output."""
    candidates = [
        _make_candidate("A", 0.5),
        _make_candidate("B", 0.5),
    ]
    paths = [_make_graph_path("A", path_length=1, path_rels=[{"type": "R"}])]
    result1 = blend_scores(candidates, graph_paths=paths)
    result2 = blend_scores(candidates, graph_paths=paths)
    assert [r["uuid"] for r in result1] == [r["uuid"] for r in result2]
    assert [r["blended_score"] for r in result1] == [
        r["blended_score"] for r in result2
    ]


def test_blend_scores_custom_weights() -> None:
    """Custom HybridScoreWeights changes ranking outcome."""
    candidates = [
        _make_candidate("A", 0.9),
        _make_candidate("B", 0.3),
    ]
    paths = [
        _make_graph_path(
            "B", path_length=1, path_rels=[{"type": "R1"}, {"type": "R2"}]
        ),
    ]
    # Heavy graph weight should boost B despite lower vector score
    heavy_graph = HybridScoreWeights(
        vector_weight=0.1,
        graph_weight=0.8,
        novelty_weight=0.0,
        relation_prior_weight=0.1,
        hub_penalty_factor=0.0,
        path_length_decay=0.2,
    )
    result = blend_scores(candidates, graph_paths=paths, weights=heavy_graph)
    assert result[0]["uuid"] == "B"


# --- _build_shortest_path_query tests ---


def test_build_sp_query_uses_shortestPath_by_default() -> None:
    """Default uses shortestPath (not allShortestPaths)."""
    query = _build_shortest_path_query("uuid", None, 10, all_shortest=False)
    assert "shortestPath(" in query
    assert "allShortestPaths" not in query


def test_build_sp_query_uses_allShortestPaths_when_flag_set() -> None:
    """all_shortest=True uses allShortestPaths."""
    query = _build_shortest_path_query("uuid", None, 10, all_shortest=True)
    assert "allShortestPaths(" in query


def test_build_sp_query_includes_max_length() -> None:
    """Query includes the max_length cap in path pattern."""
    query = _build_shortest_path_query("uuid", None, 5, all_shortest=False)
    assert "*..5" in query


def test_build_sp_query_with_relationship_filter() -> None:
    """Query includes relationship type filter in pattern."""
    query = _build_shortest_path_query(
        "uuid", ["ACTED_IN", "DIRECTED"], 10, all_shortest=False
    )
    assert "ACTED_IN|DIRECTED" in query


def test_build_sp_query_without_relationship_filter() -> None:
    """Query has no rel type filter when relationship_types is None."""
    query = _build_shortest_path_query("uuid", None, 10, all_shortest=False)
    assert "[*..10]" in query


def test_build_sp_query_returns_path_fields() -> None:
    """Query returns start_uuid, end_uuid, path_length, path_nodes, path_rels."""
    query = _build_shortest_path_query("uuid", None, 10, all_shortest=False)
    assert "start_uuid" in query
    assert "end_uuid" in query
    assert "path_length" in query
    assert "path_nodes" in query
    assert "path_rels" in query


def test_build_sp_query_uses_custom_uuid_property() -> None:
    """Query uses the configured uuid_property throughout."""
    query = _build_shortest_path_query("node_id", None, 10, all_shortest=False)
    assert "node_id: $source_uuid" in query
    assert "node_id: $target_uuid" in query
    assert "n.node_id" in query
    assert "startNode(rel).node_id" in query
    assert "endNode(rel).node_id" in query


# --- shortest_path action handler tests ---


@pytest.mark.anyio
async def test_shortest_path_executes_cypher(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """shortest_path executes Cypher with source and target UUIDs."""
    graph_db.execute.return_value = QueryResult(
        records=[
            {
                "start_uuid": "A",
                "end_uuid": "B",
                "path_length": 2,
                "path_nodes": [
                    {"uuid": "A", "labels": ["Person"], "name": "Alice"},
                    {"uuid": "X", "labels": ["Movie"], "name": "M1"},
                    {"uuid": "B", "labels": ["Person"], "name": "Bob"},
                ],
                "path_rels": [
                    {"type": "ACTED_IN", "from_uuid": "A", "to_uuid": "X"},
                    {"type": "ACTED_IN", "from_uuid": "B", "to_uuid": "X"},
                ],
            }
        ],
        summary={},
        error=None,
    )

    result = await retriever.retrieve(
        "A",
        {
            "action": "shortest_path",
            "source_id": "A",
            "target_id": "B",
            "max_length": 5,
        },
    )

    assert result.success is True
    assert len(result.data) == 1
    assert result.data[0]["start_uuid"] == "A"
    assert result.data[0]["end_uuid"] == "B"
    cypher = graph_db.execute.await_args[0][0]
    assert "shortestPath" in cypher
    assert "*..5" in cypher


@pytest.mark.anyio
async def test_shortest_path_missing_source_returns_error(
    retriever: HybridRetriever,
) -> None:
    """shortest_path returns error when source_id is missing."""
    result = await retriever.retrieve(
        "",
        {"action": "shortest_path", "target_id": "B"},
    )
    assert result.success is False
    assert "source_id" in result.message or "Missing" in result.message


@pytest.mark.anyio
async def test_shortest_path_missing_target_returns_error(
    retriever: HybridRetriever,
) -> None:
    """shortest_path returns error when target_id is missing."""
    result = await retriever.retrieve(
        "A",
        {"action": "shortest_path", "source_id": "A"},
    )
    assert result.success is False
    assert "target_id" in result.message or "Missing" in result.message


@pytest.mark.anyio
async def test_shortest_path_handles_graph_error(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """shortest_path returns error when graph execution fails."""
    graph_db.execute.return_value = QueryResult(
        records=[], summary={}, error="Graph error"
    )

    result = await retriever.retrieve(
        "A",
        {"action": "shortest_path", "source_id": "A", "target_id": "B"},
    )

    assert result.success is False
    assert "failed" in result.message


@pytest.mark.anyio
async def test_shortest_path_all_shortest_flag(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """shortest_path with all_shortest=True uses allShortestPaths."""
    graph_db.execute.return_value = QueryResult(records=[], summary={}, error=None)

    await retriever.retrieve(
        "A",
        {
            "action": "shortest_path",
            "source_id": "A",
            "target_id": "B",
            "all_shortest": True,
        },
    )

    cypher = graph_db.execute.await_args[0][0]
    assert "allShortestPaths" in cypher


@pytest.mark.anyio
async def test_shortest_path_step_records_input(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """shortest_path step input includes all parameters."""
    graph_db.execute.return_value = QueryResult(records=[], summary={}, error=None)

    result = await retriever.retrieve(
        "A",
        {
            "action": "shortest_path",
            "source_id": "A",
            "target_id": "B",
            "max_length": 7,
            "all_shortest": True,
        },
    )

    step = result.steps[0]
    assert step.input["source_id"] == "A"
    assert step.input["target_id"] == "B"
    assert step.input["max_length"] == 7
    assert step.input["all_shortest"] is True


# --- _build_ppr_gds_query tests ---


def test_build_ppr_gds_query_structure() -> None:
    """GDS PPR query includes sourceNodes, dampingFactor, and limit."""
    query = _build_ppr_gds_query("uuid", None)
    assert "gds.pageRank.stream" in query
    assert "$source_uuids" in query
    assert "dampingFactor: $damping" in query
    assert "LIMIT $limit" in query
    assert "ppr_score" in query


def test_build_ppr_gds_query_with_rel_filter() -> None:
    """GDS PPR query includes relationship type projection."""
    query = _build_ppr_gds_query("uuid", ["ACTED_IN", "DIRECTED"])
    assert "ACTED_IN" in query
    assert "DIRECTED" in query


def test_build_ppr_gds_query_uses_custom_uuid() -> None:
    """GDS PPR query uses configured uuid_property."""
    query = _build_ppr_gds_query("node_id", None)
    assert "node_id" in query
    assert "n.node_id AS uuid" in query


# --- _build_ppr_cypher_fallback tests ---


def test_build_ppr_fallback_query_structure() -> None:
    """Cypher fallback PPR query includes damping, path counting, limit."""
    query = _build_ppr_cypher_fallback("uuid", 3, None)
    assert "$source_uuids" in query
    assert "$damping" in query
    assert "*1..3" in query
    assert "ppr_score" in query
    assert "min_distance" in query
    assert "path_count" in query
    assert "LIMIT $limit" in query


def test_build_ppr_fallback_with_rel_filter() -> None:
    """Cypher fallback PPR query includes relationship type filter."""
    query = _build_ppr_cypher_fallback("uuid", 3, ["ACTED_IN"])
    assert "ACTED_IN" in query


def test_build_ppr_fallback_uses_custom_uuid() -> None:
    """Cypher fallback PPR query uses configured uuid_property."""
    query = _build_ppr_cypher_fallback("node_id", 3, None)
    assert "node_id" in query
    assert "target.node_id AS uuid" in query


# --- pagerank action handler tests ---


@pytest.mark.anyio
async def test_pagerank_missing_source_ids_returns_error(
    retriever: HybridRetriever,
) -> None:
    """pagerank returns error when source_ids is empty."""
    result = await retriever.retrieve(
        "",
        {"action": "pagerank", "source_ids": []},
    )
    assert result.success is False
    assert "source_ids" in result.message or "Missing" in result.message


@pytest.mark.anyio
async def test_pagerank_uses_cypher_fallback_when_no_gds(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """pagerank uses Cypher fallback when has_gds() returns False."""
    graph_db.has_gds = AsyncMock(return_value=False)
    graph_db.execute.return_value = QueryResult(
        records=[
            {
                "uuid": "X",
                "labels": ["Movie"],
                "name": "Matrix",
                "ppr_score": 0.42,
                "min_distance": 1,
                "path_count": 3,
            }
        ],
        summary={},
        error=None,
    )

    result = await retriever.retrieve(
        "A",
        {"action": "pagerank", "source_ids": ["A"]},
    )

    assert result.success is True
    assert len(result.data) == 1
    assert result.data[0]["uuid"] == "X"
    assert result.steps[0].output["backend"] == "cypher_fallback"


@pytest.mark.anyio
async def test_pagerank_uses_gds_when_available(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """pagerank uses GDS path when has_gds() returns True."""
    graph_db.has_gds = AsyncMock(return_value=True)
    graph_db.execute.return_value = QueryResult(
        records=[
            {
                "uuid": "X",
                "labels": ["Movie"],
                "name": "Matrix",
                "ppr_score": 0.75,
            }
        ],
        summary={},
        error=None,
    )

    result = await retriever.retrieve(
        "A",
        {"action": "pagerank", "source_ids": ["A"]},
    )

    assert result.success is True
    assert result.steps[0].output["backend"] == "gds"
    cypher = graph_db.execute.await_args[0][0]
    assert "gds.pageRank.stream" in cypher


@pytest.mark.anyio
async def test_pagerank_handles_graph_error(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """pagerank returns error when both GDS and fallback fail."""
    graph_db.has_gds = AsyncMock(return_value=False)
    graph_db.execute.return_value = QueryResult(
        records=[], summary={}, error="Graph error"
    )

    result = await retriever.retrieve(
        "A",
        {"action": "pagerank", "source_ids": ["A"]},
    )

    assert result.success is False
    assert "failed" in result.message


@pytest.mark.anyio
async def test_pagerank_step_records_input(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """pagerank step input includes all parameters."""
    graph_db.has_gds = AsyncMock(return_value=False)
    graph_db.execute.return_value = QueryResult(records=[], summary={}, error=None)

    result = await retriever.retrieve(
        "A",
        {
            "action": "pagerank",
            "source_ids": ["A", "B"],
            "damping": 0.9,
            "limit": 10,
            "max_depth": 4,
        },
    )

    step = result.steps[0]
    assert step.input["source_ids"] == ["A", "B"]
    assert step.input["damping"] == 0.9
    assert step.input["limit"] == 10
    assert step.input["max_depth"] == 4


@pytest.mark.anyio
async def test_pagerank_message_includes_count_and_backend(
    retriever: HybridRetriever,
    graph_db: MagicMock,
) -> None:
    """pagerank success message reports node count and backend."""
    graph_db.has_gds = AsyncMock(return_value=False)
    graph_db.execute.return_value = QueryResult(
        records=[{"uuid": "X"}] * 5,
        summary={},
        error=None,
    )

    result = await retriever.retrieve(
        "A",
        {"action": "pagerank", "source_ids": ["A"]},
    )

    assert "5 nodes" in result.message
    assert "cypher_fallback" in result.message
