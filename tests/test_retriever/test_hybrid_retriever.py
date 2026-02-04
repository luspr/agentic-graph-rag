"""Unit tests for HybridRetriever with mocked dependencies."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_graph_rag.graph.base import GraphDatabase, QueryResult
from agentic_graph_rag.llm.base import LLMClient
from agentic_graph_rag.retriever.base import RetrievalStrategy
from agentic_graph_rag.retriever.hybrid_retriever import HybridRetriever
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
    """expand_node executes Cypher with UUID and relationship filter."""
    graph_db.execute.return_value = QueryResult(
        records=[{"node_uuid": "node-1"}],
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
    assert result.data == [{"node_uuid": "node-1"}]
    cypher, params = graph_db.execute.await_args[0]
    assert "uuid: $uuid" in cypher
    assert "*1..2" in cypher
    assert params["uuid"] == "node-1"
    assert params["relationship_types"] == ["ACTED_IN"]


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
