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


def test_agent_tools_has_six_definitions() -> None:
    """AGENT_TOOLS contains exactly 6 tool definitions."""
    assert len(AGENT_TOOLS) == 6


def test_agent_tools_names() -> None:
    """AGENT_TOOLS contains the expected tool names."""
    names = {tool.name for tool in AGENT_TOOLS}
    assert names == {
        "execute_cypher",
        "vector_search",
        "expand_node",
        "shortest_path",
        "pagerank",
        "submit_answer",
    }


def test_execute_cypher_requires_query_and_reasoning() -> None:
    """execute_cypher tool requires 'query' and 'reasoning' parameters."""
    tool = next(t for t in AGENT_TOOLS if t.name == "execute_cypher")
    assert tool.parameters["required"] == ["query", "reasoning"]
    assert "query" in tool.parameters["properties"]
    assert "reasoning" in tool.parameters["properties"]


def test_expand_node_requires_node_id() -> None:
    """expand_node tool requires 'node_id' with optional params including max_branching."""
    tool = next(t for t in AGENT_TOOLS if t.name == "expand_node")
    assert tool.parameters["required"] == ["node_id"]
    assert "node_id" in tool.parameters["properties"]
    assert "relationship_types" in tool.parameters["properties"]
    assert "depth" in tool.parameters["properties"]
    assert tool.parameters["properties"]["depth"]["default"] == 1
    assert "direction" in tool.parameters["properties"]
    assert tool.parameters["properties"]["direction"]["default"] == "both"
    assert tool.parameters["properties"]["direction"]["enum"] == ["out", "in", "both"]
    assert "max_paths" in tool.parameters["properties"]
    assert tool.parameters["properties"]["max_paths"]["default"] == 20
    assert "max_branching" in tool.parameters["properties"]
    assert tool.parameters["properties"]["max_branching"]["type"] == "integer"


def test_vector_search_requires_query() -> None:
    """vector_search tool requires 'query' with optional limit and filters."""
    tool = next(t for t in AGENT_TOOLS if t.name == "vector_search")
    assert tool.parameters["required"] == ["query"]
    assert "query" in tool.parameters["properties"]
    assert "limit" in tool.parameters["properties"]
    assert "filters" in tool.parameters["properties"]
    assert tool.parameters["properties"]["limit"]["default"] == 5


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


# --- vector_search handler tests ---


@pytest.mark.anyio
async def test_vector_search_calls_hybrid_retriever(
    router: ToolRouter,
    mock_hybrid_retriever: MagicMock,
) -> None:
    """vector_search handler passes query and context to HybridRetriever."""
    mock_hybrid_retriever.retrieve.return_value = _success_result()
    tool_call = ToolCall(
        id="2",
        name="vector_search",
        arguments={"query": "movies about AI", "limit": 3, "filters": {"must": []}},
    )

    await router.route(tool_call)

    mock_hybrid_retriever.retrieve.assert_awaited_once_with(
        "movies about AI",
        {"action": "vector_search", "limit": 3, "filters": {"must": []}},
    )


@pytest.mark.anyio
async def test_vector_search_defaults(
    router: ToolRouter,
    mock_hybrid_retriever: MagicMock,
) -> None:
    """vector_search handler defaults limit to 5 and filters to None."""
    mock_hybrid_retriever.retrieve.return_value = _success_result()
    tool_call = ToolCall(
        id="2",
        name="vector_search",
        arguments={"query": "find movies"},
    )

    await router.route(tool_call)

    _, context = mock_hybrid_retriever.retrieve.await_args[0]
    assert context["limit"] == 5
    assert context["filters"] is None


@pytest.mark.anyio
async def test_vector_search_returns_success(
    router: ToolRouter,
    mock_hybrid_retriever: MagicMock,
) -> None:
    """vector_search handler returns success result with data and message."""
    expected_data = [
        {"uuid": "node-1", "score": 0.9},
        {"uuid": "node-2", "score": 0.8},
    ]
    mock_hybrid_retriever.retrieve.return_value = _success_result(expected_data)
    tool_call = ToolCall(
        id="2",
        name="vector_search",
        arguments={"query": "find movies"},
    )

    result = await router.route(tool_call)

    assert result["success"] is True
    assert result["data"] == expected_data
    assert "message" in result


@pytest.mark.anyio
async def test_vector_search_unavailable_without_hybrid_retriever(
    router_cypher_only: ToolRouter,
) -> None:
    """vector_search returns error when hybrid retriever is not configured."""
    tool_call = ToolCall(
        id="2",
        name="vector_search",
        arguments={"query": "find movies"},
    )

    result = await router_cypher_only.route(tool_call)

    assert result["success"] is False
    assert "not available" in result["error"]


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
            "direction": "both",
            "max_paths": 20,
            "max_branching": None,
        },
    )


@pytest.mark.anyio
async def test_expand_node_defaults(
    router: ToolRouter,
    mock_hybrid_retriever: MagicMock,
) -> None:
    """expand_node handler defaults depth, direction, max_paths, relationship_types, max_branching."""
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
    assert context["direction"] == "both"
    assert context["max_paths"] == 20
    assert context["max_branching"] is None


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
async def test_expand_node_passes_direction_and_max_paths(
    router: ToolRouter,
    mock_hybrid_retriever: MagicMock,
) -> None:
    """expand_node handler forwards direction and max_paths to the retriever."""
    mock_hybrid_retriever.retrieve.return_value = _success_result()
    tool_call = ToolCall(
        id="3",
        name="expand_node",
        arguments={
            "node_id": "node_42",
            "direction": "out",
            "max_paths": 5,
        },
    )

    await router.route(tool_call)

    _, context = mock_hybrid_retriever.retrieve.await_args[0]
    assert context["direction"] == "out"
    assert context["max_paths"] == 5


@pytest.mark.anyio
async def test_expand_node_forwards_max_branching(
    router: ToolRouter,
    mock_hybrid_retriever: MagicMock,
) -> None:
    """expand_node handler forwards max_branching to the retriever context."""
    mock_hybrid_retriever.retrieve.return_value = _success_result()
    tool_call = ToolCall(
        id="3",
        name="expand_node",
        arguments={"node_id": "node_42", "max_branching": 3},
    )

    await router.route(tool_call)

    _, context = mock_hybrid_retriever.retrieve.await_args[0]
    assert context["max_branching"] == 3


@pytest.mark.anyio
async def test_expand_node_max_branching_defaults_to_none(
    router: ToolRouter,
    mock_hybrid_retriever: MagicMock,
) -> None:
    """expand_node handler defaults max_branching to None when not provided."""
    mock_hybrid_retriever.retrieve.return_value = _success_result()
    tool_call = ToolCall(
        id="3",
        name="expand_node",
        arguments={"node_id": "node_42"},
    )

    await router.route(tool_call)

    _, context = mock_hybrid_retriever.retrieve.await_args[0]
    assert context["max_branching"] is None


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


# --- shortest_path tool definition tests ---


def test_shortest_path_requires_source_and_target() -> None:
    """shortest_path tool requires source_id and target_id."""
    tool = next(t for t in AGENT_TOOLS if t.name == "shortest_path")
    assert tool.parameters["required"] == ["source_id", "target_id"]
    assert "source_id" in tool.parameters["properties"]
    assert "target_id" in tool.parameters["properties"]
    assert "max_length" in tool.parameters["properties"]
    assert tool.parameters["properties"]["max_length"]["default"] == 10
    assert "all_shortest" in tool.parameters["properties"]
    assert tool.parameters["properties"]["all_shortest"]["default"] is False


# --- shortest_path handler tests ---


@pytest.mark.anyio
async def test_shortest_path_calls_hybrid_retriever(
    router: ToolRouter,
    mock_hybrid_retriever: MagicMock,
) -> None:
    """shortest_path handler passes context to HybridRetriever."""
    mock_hybrid_retriever.retrieve.return_value = _success_result()
    tool_call = ToolCall(
        id="5",
        name="shortest_path",
        arguments={
            "source_id": "A",
            "target_id": "B",
            "max_length": 5,
        },
    )

    await router.route(tool_call)

    mock_hybrid_retriever.retrieve.assert_awaited_once_with(
        "A",
        {
            "action": "shortest_path",
            "source_id": "A",
            "target_id": "B",
            "relationship_types": None,
            "max_length": 5,
            "all_shortest": False,
        },
    )


@pytest.mark.anyio
async def test_shortest_path_defaults(
    router: ToolRouter,
    mock_hybrid_retriever: MagicMock,
) -> None:
    """shortest_path handler defaults max_length and all_shortest."""
    mock_hybrid_retriever.retrieve.return_value = _success_result()
    tool_call = ToolCall(
        id="5",
        name="shortest_path",
        arguments={"source_id": "A", "target_id": "B"},
    )

    await router.route(tool_call)

    _, context = mock_hybrid_retriever.retrieve.await_args[0]
    assert context["max_length"] == 10
    assert context["all_shortest"] is False
    assert context["relationship_types"] is None


@pytest.mark.anyio
async def test_shortest_path_unavailable_without_hybrid(
    router_cypher_only: ToolRouter,
) -> None:
    """shortest_path returns error when hybrid retriever is not configured."""
    tool_call = ToolCall(
        id="5",
        name="shortest_path",
        arguments={"source_id": "A", "target_id": "B"},
    )

    result = await router_cypher_only.route(tool_call)

    assert result["success"] is False
    assert "not available" in result["error"]


# --- pagerank tool definition tests ---


def test_pagerank_requires_source_ids() -> None:
    """pagerank tool requires source_ids."""
    tool = next(t for t in AGENT_TOOLS if t.name == "pagerank")
    assert tool.parameters["required"] == ["source_ids"]
    assert "source_ids" in tool.parameters["properties"]
    assert "damping" in tool.parameters["properties"]
    assert tool.parameters["properties"]["damping"]["default"] == 0.85
    assert "limit" in tool.parameters["properties"]
    assert tool.parameters["properties"]["limit"]["default"] == 20
    assert "max_depth" in tool.parameters["properties"]
    assert tool.parameters["properties"]["max_depth"]["default"] == 3


# --- pagerank handler tests ---


@pytest.mark.anyio
async def test_pagerank_calls_hybrid_retriever(
    router: ToolRouter,
    mock_hybrid_retriever: MagicMock,
) -> None:
    """pagerank handler passes context to HybridRetriever."""
    mock_hybrid_retriever.retrieve.return_value = _success_result()
    tool_call = ToolCall(
        id="6",
        name="pagerank",
        arguments={
            "source_ids": ["A", "B"],
            "damping": 0.9,
            "limit": 10,
        },
    )

    await router.route(tool_call)

    mock_hybrid_retriever.retrieve.assert_awaited_once_with(
        "A,B",
        {
            "action": "pagerank",
            "source_ids": ["A", "B"],
            "damping": 0.9,
            "limit": 10,
            "max_depth": 3,
            "relationship_types": None,
        },
    )


@pytest.mark.anyio
async def test_pagerank_defaults(
    router: ToolRouter,
    mock_hybrid_retriever: MagicMock,
) -> None:
    """pagerank handler defaults damping, limit, max_depth."""
    mock_hybrid_retriever.retrieve.return_value = _success_result()
    tool_call = ToolCall(
        id="6",
        name="pagerank",
        arguments={"source_ids": ["A"]},
    )

    await router.route(tool_call)

    _, context = mock_hybrid_retriever.retrieve.await_args[0]
    assert context["damping"] == 0.85
    assert context["limit"] == 20
    assert context["max_depth"] == 3
    assert context["relationship_types"] is None


@pytest.mark.anyio
async def test_pagerank_unavailable_without_hybrid(
    router_cypher_only: ToolRouter,
) -> None:
    """pagerank returns error when hybrid retriever is not configured."""
    tool_call = ToolCall(
        id="6",
        name="pagerank",
        arguments={"source_ids": ["A"]},
    )

    result = await router_cypher_only.route(tool_call)

    assert result["success"] is False
    assert "not available" in result["error"]
