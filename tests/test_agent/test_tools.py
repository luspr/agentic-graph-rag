"""Unit tests for agent tool definitions and ToolRouter."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_graph_rag.agent.tools import AGENT_TOOLS, ToolRouter
from agentic_graph_rag.llm.base import ToolCall
from agentic_graph_rag.retriever.base import RetrievalResult, RetrievalStep, Retriever


def _make_mock_retriever() -> MagicMock:
    """Create a mock Retriever for testing."""
    mock = MagicMock(spec=Retriever)
    mock.retrieve = AsyncMock()
    return mock


def _success_result(data: list[dict] | None = None) -> RetrievalResult:
    """Create a successful RetrievalResult."""
    return RetrievalResult(
        data=data or [{"title": "The Matrix"}],
        steps=[RetrievalStep(action="test", input={}, output={}, error=None)],
        success=True,
        message="Retrieved 1 records",
    )


def _error_result() -> RetrievalResult:
    """Create a failed RetrievalResult."""
    return RetrievalResult(
        data=[],
        steps=[RetrievalStep(action="test", input={}, output={}, error="Query failed")],
        success=False,
        message="Query execution failed: Query failed",
    )


# --- AGENT_TOOLS definition tests ---


def test_agent_tools_has_three_definitions() -> None:
    """AGENT_TOOLS contains exactly 3 tool definitions."""
    assert len(AGENT_TOOLS) == 3


def test_agent_tools_names() -> None:
    """AGENT_TOOLS contains the expected tool names."""
    names = {tool.name for tool in AGENT_TOOLS}
    assert names == {
        "execute_cypher",
        "expand_node",
        "submit_answer",
    }


def test_execute_cypher_requires_query_and_reasoning() -> None:
    """execute_cypher tool requires 'query' and 'reasoning' parameters."""
    tool = next(t for t in AGENT_TOOLS if t.name == "execute_cypher")
    assert tool.parameters["required"] == ["query", "reasoning"]
    assert "query" in tool.parameters["properties"]
    assert "reasoning" in tool.parameters["properties"]


def test_expand_node_requires_node_id() -> None:
    """expand_node tool requires 'node_id' with optional relationship_types and depth."""
    tool = next(t for t in AGENT_TOOLS if t.name == "expand_node")
    assert tool.parameters["required"] == ["node_id"]
    assert "node_id" in tool.parameters["properties"]
    assert "relationship_types" in tool.parameters["properties"]
    assert "depth" in tool.parameters["properties"]
    assert tool.parameters["properties"]["depth"]["default"] == 1


def test_submit_answer_requires_all_fields() -> None:
    """submit_answer tool requires 'answer', 'confidence', and 'supporting_evidence'."""
    tool = next(t for t in AGENT_TOOLS if t.name == "submit_answer")
    assert tool.parameters["required"] == [
        "answer",
        "confidence",
        "supporting_evidence",
    ]
    assert "answer" in tool.parameters["properties"]
    assert "confidence" in tool.parameters["properties"]
    assert "supporting_evidence" in tool.parameters["properties"]


# --- ToolRouter fixtures ---


@pytest.fixture
def mock_cypher_retriever() -> MagicMock:
    """Create a mock CypherRetriever."""
    return _make_mock_retriever()


@pytest.fixture
def mock_hybrid_retriever() -> MagicMock:
    """Create a mock HybridRetriever."""
    return _make_mock_retriever()


@pytest.fixture
def router(
    mock_cypher_retriever: MagicMock,
    mock_hybrid_retriever: MagicMock,
) -> ToolRouter:
    """Create a ToolRouter with both retrievers."""
    return ToolRouter(mock_cypher_retriever, mock_hybrid_retriever)


@pytest.fixture
def router_cypher_only(mock_cypher_retriever: MagicMock) -> ToolRouter:
    """Create a ToolRouter with only the CypherRetriever."""
    return ToolRouter(mock_cypher_retriever)


# --- route() dispatch tests ---


@pytest.mark.anyio
async def test_route_unknown_tool_returns_error(router: ToolRouter) -> None:
    """route() returns an error dict for unknown tool names."""
    tool_call = ToolCall(id="1", name="nonexistent_tool", arguments={})

    result = await router.route(tool_call)

    assert result["success"] is False
    assert "nonexistent_tool" in result["error"]


# --- execute_cypher handler tests ---


@pytest.mark.anyio
async def test_execute_cypher_calls_retriever(
    router: ToolRouter,
    mock_cypher_retriever: MagicMock,
) -> None:
    """execute_cypher handler passes the query to the CypherRetriever."""
    mock_cypher_retriever.retrieve.return_value = _success_result()
    tool_call = ToolCall(
        id="1",
        name="execute_cypher",
        arguments={
            "query": "MATCH (m:Movie) RETURN m.title",
            "reasoning": "Find all movies",
        },
    )

    await router.route(tool_call)

    mock_cypher_retriever.retrieve.assert_awaited_once_with(
        "MATCH (m:Movie) RETURN m.title"
    )


@pytest.mark.anyio
async def test_execute_cypher_returns_success(
    router: ToolRouter,
    mock_cypher_retriever: MagicMock,
) -> None:
    """execute_cypher handler returns success result with data and message."""
    expected_data = [{"title": "The Matrix"}, {"title": "Inception"}]
    mock_cypher_retriever.retrieve.return_value = _success_result(expected_data)
    tool_call = ToolCall(
        id="1",
        name="execute_cypher",
        arguments={
            "query": "MATCH (m:Movie) RETURN m.title AS title",
            "reasoning": "Get movie titles",
        },
    )

    result = await router.route(tool_call)

    assert result["success"] is True
    assert result["data"] == expected_data
    assert "message" in result


@pytest.mark.anyio
async def test_execute_cypher_returns_error_from_retriever(
    router: ToolRouter,
    mock_cypher_retriever: MagicMock,
) -> None:
    """execute_cypher handler propagates retriever errors without raising."""
    mock_cypher_retriever.retrieve.return_value = _error_result()
    tool_call = ToolCall(
        id="1",
        name="execute_cypher",
        arguments={
            "query": "INVALID CYPHER",
            "reasoning": "Testing error handling",
        },
    )

    result = await router.route(tool_call)

    assert result["success"] is False
    assert result["data"] == []
    assert "failed" in result["message"].lower()


# --- expand_node handler tests ---


@pytest.mark.anyio
async def test_expand_node_calls_hybrid_retriever(
    router: ToolRouter,
    mock_hybrid_retriever: MagicMock,
) -> None:
    """expand_node handler passes node_id and context to HybridRetriever."""
    mock_hybrid_retriever.retrieve.return_value = _success_result()
    tool_call = ToolCall(
        id="3",
        name="expand_node",
        arguments={
            "node_id": "node_42",
            "relationship_types": ["ACTED_IN"],
            "depth": 2,
        },
    )

    await router.route(tool_call)

    mock_hybrid_retriever.retrieve.assert_awaited_once_with(
        "node_42",
        {
            "action": "expand_node",
            "node_id": "node_42",
            "relationship_types": ["ACTED_IN"],
            "depth": 2,
        },
    )


@pytest.mark.anyio
async def test_expand_node_defaults(
    router: ToolRouter,
    mock_hybrid_retriever: MagicMock,
) -> None:
    """expand_node handler defaults depth to 1 and relationship_types to None."""
    mock_hybrid_retriever.retrieve.return_value = _success_result()
    tool_call = ToolCall(
        id="3",
        name="expand_node",
        arguments={"node_id": "node_42"},
    )

    await router.route(tool_call)

    _, context = mock_hybrid_retriever.retrieve.await_args[0]
    assert context["depth"] == 1
    assert context["relationship_types"] is None


@pytest.mark.anyio
async def test_expand_node_returns_success(
    router: ToolRouter,
    mock_hybrid_retriever: MagicMock,
) -> None:
    """expand_node handler returns success result with connected nodes."""
    expected_data = [
        {"id": "node_1", "label": "Person", "name": "Keanu Reeves"},
        {"id": "node_2", "label": "Movie", "title": "Speed"},
    ]
    mock_hybrid_retriever.retrieve.return_value = _success_result(expected_data)
    tool_call = ToolCall(
        id="3",
        name="expand_node",
        arguments={"node_id": "node_42"},
    )

    result = await router.route(tool_call)

    assert result["success"] is True
    assert result["data"] == expected_data


@pytest.mark.anyio
async def test_expand_node_unavailable_without_hybrid_retriever(
    router_cypher_only: ToolRouter,
) -> None:
    """expand_node returns error when hybrid retriever is not configured."""
    tool_call = ToolCall(
        id="3",
        name="expand_node",
        arguments={"node_id": "node_42"},
    )

    result = await router_cypher_only.route(tool_call)

    assert result["success"] is False
    assert "not available" in result["error"]


# --- submit_answer handler tests ---


@pytest.mark.anyio
async def test_submit_answer_returns_structured_result(
    router: ToolRouter,
) -> None:
    """submit_answer handler returns the answer with confidence and evidence."""
    tool_call = ToolCall(
        id="4",
        name="submit_answer",
        arguments={
            "answer": "The Matrix was directed by the Wachowskis.",
            "confidence": 0.95,
            "supporting_evidence": "Found DIRECTED relationship from Wachowski to The Matrix.",
        },
    )

    result = await router.route(tool_call)

    assert result["success"] is True
    assert result["answer"] == "The Matrix was directed by the Wachowskis."
    assert result["confidence"] == 0.95
    assert "Wachowski" in result["supporting_evidence"]


@pytest.mark.anyio
async def test_submit_answer_low_confidence(router: ToolRouter) -> None:
    """submit_answer handler accepts low confidence values."""
    tool_call = ToolCall(
        id="4",
        name="submit_answer",
        arguments={
            "answer": "I'm not sure.",
            "confidence": 0.1,
            "supporting_evidence": "Insufficient data found in the graph.",
        },
    )

    result = await router.route(tool_call)

    assert result["success"] is True
    assert result["confidence"] == 0.1
