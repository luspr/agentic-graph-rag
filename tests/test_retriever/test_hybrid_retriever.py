"""Unit tests for HybridRetriever with mocked dependencies."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_graph_rag.graph.base import GraphDatabase, QueryResult
from agentic_graph_rag.llm.base import LLMClient
from agentic_graph_rag.retriever.base import RetrievalStrategy
from agentic_graph_rag.retriever.hybrid_retriever import (
    HybridRetriever,
    _build_expand_query,
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
    """vector_search returns formatted results with UUIDs."""
    llm_client.embed.return_value = [0.1, 0.2, 0.3]
    vector_store.search.return_value = [
        VectorSearchResult(id="node-1", score=0.9, payload={"title": "The Matrix"})
    ]

    result = await retriever.retrieve(
        "matrix",
        {"action": "vector_search", "limit": 3, "filters": {"must": []}},
    )

    assert result.success is True
    assert result.data == [
        {"uuid": "node-1", "score": 0.9, "payload": {"title": "The Matrix"}}
    ]
    llm_client.embed.assert_awaited_once_with("matrix")
    vector_store.search.assert_awaited_once_with(
        [0.1, 0.2, 0.3],
        limit=3,
        filter={"must": []},
    )


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
                "node_uuid": "node-2",
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
    assert result.data[0]["node_uuid"] == "node-2"
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
    """Query returns start_uuid, node_uuid, node_labels, path_length, path_nodes, path_rels."""
    query = _build_expand_query("uuid", 1, None)
    assert "start_uuid" in query
    assert "node_uuid" in query
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
    assert "end_node.node_id AS node_uuid" in query
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
        records=[{"start_uuid": "a", "node_uuid": "b"}] * 3,
        summary={},
        error=None,
    )

    result = await retriever.retrieve(
        "n1",
        {"action": "expand_node", "node_id": "n1"},
    )

    assert "3 paths" in result.message
