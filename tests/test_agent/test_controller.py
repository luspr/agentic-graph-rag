"""Unit tests for AgentController."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_graph_rag.agent.controller import AgentController
from agentic_graph_rag.agent.state import (
    AgentConfig,
    AgentResult,
    AgentState,
    AgentStatus,
)
from agentic_graph_rag.agent.tools import ToolRouter
from agentic_graph_rag.graph.base import GraphDatabase, GraphSchema, NodeType
from agentic_graph_rag.llm.base import LLMClient, LLMResponse, ToolCall
from agentic_graph_rag.prompts.manager import PromptManager
from agentic_graph_rag.retriever.base import RetrievalStrategy


# --- Fixtures ---


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLMClient."""
    mock = MagicMock(spec=LLMClient)
    mock.complete = AsyncMock()
    return mock


@pytest.fixture
def mock_graph_db() -> MagicMock:
    """Create a mock GraphDatabase."""
    mock = MagicMock(spec=GraphDatabase)
    mock.get_schema = AsyncMock(
        return_value=GraphSchema(
            node_types=[
                NodeType(label="Movie", properties={"title": "STRING"}, count=10)
            ],
            relationship_types=[],
        )
    )
    return mock


@pytest.fixture
def mock_tool_router() -> MagicMock:
    """Create a mock ToolRouter."""
    mock = MagicMock(spec=ToolRouter)
    mock.route = AsyncMock()
    return mock


@pytest.fixture
def mock_prompt_manager() -> PromptManager:
    """Create a real PromptManager (it has no external dependencies)."""
    return PromptManager()


@pytest.fixture
def mock_tracer() -> MagicMock:
    """Create a mock Tracer."""
    mock = MagicMock()
    mock.log_event = MagicMock()
    return mock


@pytest.fixture
def controller(
    mock_llm_client: MagicMock,
    mock_graph_db: MagicMock,
    mock_tool_router: MagicMock,
    mock_prompt_manager: PromptManager,
) -> AgentController:
    """Create an AgentController with mocked dependencies."""
    return AgentController(
        llm_client=mock_llm_client,
        graph_db=mock_graph_db,
        tool_router=mock_tool_router,
        prompt_manager=mock_prompt_manager,
    )


@pytest.fixture
def controller_with_tracer(
    mock_llm_client: MagicMock,
    mock_graph_db: MagicMock,
    mock_tool_router: MagicMock,
    mock_prompt_manager: PromptManager,
    mock_tracer: MagicMock,
) -> AgentController:
    """Create an AgentController with mocked dependencies and tracer."""
    return AgentController(
        llm_client=mock_llm_client,
        graph_db=mock_graph_db,
        tool_router=mock_tool_router,
        prompt_manager=mock_prompt_manager,
        tracer=mock_tracer,
    )


def _make_llm_response(
    tool_calls: list[ToolCall] | None = None,
    content: str | None = None,
) -> LLMResponse:
    """Create an LLMResponse for testing."""
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage={"prompt_tokens": 100, "completion_tokens": 50},
        finish_reason="tool_calls" if tool_calls else "stop",
    )


def _submit_answer_tool_call(
    answer: str = "Test answer",
    confidence: float = 0.9,
    evidence: str = "Test evidence",
) -> ToolCall:
    """Create a submit_answer ToolCall."""
    return ToolCall(
        id="call_1",
        name="submit_answer",
        arguments={
            "answer": answer,
            "confidence": confidence,
            "supporting_evidence": evidence,
        },
    )


def _execute_cypher_tool_call(query: str = "MATCH (m:Movie) RETURN m") -> ToolCall:
    """Create an execute_cypher ToolCall."""
    return ToolCall(
        id="call_1",
        name="execute_cypher",
        arguments={"query": query, "reasoning": "Test reasoning"},
    )


# --- State dataclass tests ---


def test_agent_status_values() -> None:
    """AgentStatus has the expected values."""
    assert AgentStatus.RUNNING.value == "running"
    assert AgentStatus.COMPLETED.value == "completed"
    assert AgentStatus.MAX_ITERATIONS.value == "max_iterations"
    assert AgentStatus.ERROR.value == "error"


def test_agent_state_defaults() -> None:
    """AgentState has expected defaults."""
    state = AgentState(iteration=0, status=AgentStatus.RUNNING, history=[])
    assert state.current_answer is None
    assert state.confidence is None


def test_agent_config_defaults() -> None:
    """AgentConfig has expected defaults."""
    config = AgentConfig()
    assert config.max_iterations == 10
    assert config.strategy == RetrievalStrategy.CYPHER


def test_agent_config_custom_values() -> None:
    """AgentConfig accepts custom values."""
    config = AgentConfig(max_iterations=5, strategy=RetrievalStrategy.HYBRID)
    assert config.max_iterations == 5
    assert config.strategy == RetrievalStrategy.HYBRID


def test_agent_result_defaults() -> None:
    """AgentResult has expected defaults."""
    result = AgentResult(answer="Test", status=AgentStatus.COMPLETED, iterations=1)
    assert result.history == []
    assert result.confidence is None


# --- should_stop() tests ---


def test_should_stop_on_completed(controller: AgentController) -> None:
    """should_stop returns True when status is COMPLETED."""
    state = AgentState(iteration=1, status=AgentStatus.COMPLETED, history=[])
    assert controller.should_stop(state) is True


def test_should_stop_on_error(controller: AgentController) -> None:
    """should_stop returns True when status is ERROR."""
    state = AgentState(iteration=1, status=AgentStatus.ERROR, history=[])
    assert controller.should_stop(state) is True


def test_should_stop_on_max_iterations(
    mock_llm_client: MagicMock,
    mock_graph_db: MagicMock,
    mock_tool_router: MagicMock,
    mock_prompt_manager: PromptManager,
) -> None:
    """should_stop returns True and sets status when max iterations reached."""
    config = AgentConfig(max_iterations=5)
    controller = AgentController(
        llm_client=mock_llm_client,
        graph_db=mock_graph_db,
        tool_router=mock_tool_router,
        prompt_manager=mock_prompt_manager,
        config=config,
    )
    state = AgentState(iteration=5, status=AgentStatus.RUNNING, history=[])

    result = controller.should_stop(state)

    assert result is True
    assert state.status == AgentStatus.MAX_ITERATIONS


def test_should_not_stop_while_running(controller: AgentController) -> None:
    """should_stop returns False when status is RUNNING and below max iterations."""
    state = AgentState(iteration=3, status=AgentStatus.RUNNING, history=[])
    assert controller.should_stop(state) is False


# --- run() tests ---


@pytest.mark.anyio
async def test_run_completes_on_submit_answer(
    controller: AgentController,
    mock_llm_client: MagicMock,
    mock_tool_router: MagicMock,
) -> None:
    """run() returns result when LLM calls submit_answer."""
    mock_llm_client.complete.return_value = _make_llm_response(
        tool_calls=[_submit_answer_tool_call()]
    )
    mock_tool_router.route.return_value = {
        "success": True,
        "answer": "The Matrix is a movie",
        "confidence": 0.95,
        "supporting_evidence": "Found Movie node",
    }

    result = await controller.run("What is The Matrix?")

    assert result.status == AgentStatus.COMPLETED
    assert result.answer == "The Matrix is a movie"
    assert result.confidence == 0.95
    assert result.iterations == 1


@pytest.mark.anyio
async def test_run_iterates_until_answer(
    controller: AgentController,
    mock_llm_client: MagicMock,
    mock_tool_router: MagicMock,
) -> None:
    """run() iterates multiple times before getting final answer."""
    # First call: execute_cypher, second call: submit_answer
    mock_llm_client.complete.side_effect = [
        _make_llm_response(tool_calls=[_execute_cypher_tool_call()]),
        _make_llm_response(tool_calls=[_submit_answer_tool_call()]),
    ]
    mock_tool_router.route.side_effect = [
        {
            "success": True,
            "data": [{"title": "The Matrix"}],
            "message": "Found 1 record",
        },
        {
            "success": True,
            "answer": "The Matrix",
            "confidence": 0.9,
            "supporting_evidence": "Found in query",
        },
    ]

    result = await controller.run("What movie?")

    assert result.status == AgentStatus.COMPLETED
    assert result.iterations == 2
    assert len(result.history) == 2


@pytest.mark.anyio
async def test_run_stops_on_max_iterations(
    mock_llm_client: MagicMock,
    mock_graph_db: MagicMock,
    mock_tool_router: MagicMock,
    mock_prompt_manager: PromptManager,
) -> None:
    """run() stops and returns MAX_ITERATIONS status when limit reached."""
    config = AgentConfig(max_iterations=2)
    controller = AgentController(
        llm_client=mock_llm_client,
        graph_db=mock_graph_db,
        tool_router=mock_tool_router,
        prompt_manager=mock_prompt_manager,
        config=config,
    )

    # Always return execute_cypher (never submit_answer)
    mock_llm_client.complete.return_value = _make_llm_response(
        tool_calls=[_execute_cypher_tool_call()]
    )
    mock_tool_router.route.return_value = {
        "success": True,
        "data": [{"title": "Movie"}],
        "message": "Found",
    }

    result = await controller.run("Query")

    assert result.status == AgentStatus.MAX_ITERATIONS
    assert result.iterations == 2
    assert "Maximum iterations" in result.answer


@pytest.mark.anyio
async def test_run_handles_llm_error(
    controller: AgentController,
    mock_llm_client: MagicMock,
) -> None:
    """run() returns ERROR status when LLM raises exception."""
    mock_llm_client.complete.side_effect = Exception("API error")

    result = await controller.run("Query")

    assert result.status == AgentStatus.ERROR
    assert "Error communicating with LLM" in result.answer


@pytest.mark.anyio
async def test_run_handles_no_tool_calls(
    controller: AgentController,
    mock_llm_client: MagicMock,
) -> None:
    """run() returns ERROR status when LLM returns no tool calls."""
    mock_llm_client.complete.return_value = _make_llm_response(
        content="I don't know",
        tool_calls=[],
    )

    result = await controller.run("Query")

    assert result.status == AgentStatus.ERROR
    assert result.iterations == 1


@pytest.mark.anyio
async def test_run_records_history(
    controller: AgentController,
    mock_llm_client: MagicMock,
    mock_tool_router: MagicMock,
) -> None:
    """run() records all tool calls in history."""
    mock_llm_client.complete.return_value = _make_llm_response(
        tool_calls=[_submit_answer_tool_call()]
    )
    mock_tool_router.route.return_value = {
        "success": True,
        "answer": "Answer",
        "confidence": 0.8,
        "supporting_evidence": "Evidence",
    }

    result = await controller.run("Query")

    assert len(result.history) == 1
    assert result.history[0].action == "submit_answer"
    assert result.history[0].output["answer"] == "Answer"


@pytest.mark.anyio
async def test_run_gets_schema_at_start(
    controller: AgentController,
    mock_llm_client: MagicMock,
    mock_tool_router: MagicMock,
    mock_graph_db: MagicMock,
) -> None:
    """run() fetches graph schema at the start."""
    mock_llm_client.complete.return_value = _make_llm_response(
        tool_calls=[_submit_answer_tool_call()]
    )
    mock_tool_router.route.return_value = {
        "success": True,
        "answer": "Answer",
        "confidence": 0.8,
        "supporting_evidence": "Evidence",
    }

    await controller.run("Query")

    mock_graph_db.get_schema.assert_awaited_once()


# --- Tracer integration tests ---


@pytest.mark.anyio
async def test_run_logs_query_start(
    controller_with_tracer: AgentController,
    mock_llm_client: MagicMock,
    mock_tool_router: MagicMock,
    mock_tracer: MagicMock,
) -> None:
    """run() logs query_start event via tracer."""
    mock_llm_client.complete.return_value = _make_llm_response(
        tool_calls=[_submit_answer_tool_call()]
    )
    mock_tool_router.route.return_value = {
        "success": True,
        "answer": "Answer",
        "confidence": 0.8,
        "supporting_evidence": "Evidence",
    }

    await controller_with_tracer.run("Test query")

    # Find query_start event
    calls = mock_tracer.log_event.call_args_list
    event_types = [call[0][0] for call in calls]
    assert "query_start" in event_types


@pytest.mark.anyio
async def test_run_logs_tool_events(
    controller_with_tracer: AgentController,
    mock_llm_client: MagicMock,
    mock_tool_router: MagicMock,
    mock_tracer: MagicMock,
) -> None:
    """run() logs tool_call and tool_result events via tracer."""
    mock_llm_client.complete.return_value = _make_llm_response(
        tool_calls=[_submit_answer_tool_call()]
    )
    mock_tool_router.route.return_value = {
        "success": True,
        "answer": "Answer",
        "confidence": 0.8,
        "supporting_evidence": "Evidence",
    }

    await controller_with_tracer.run("Query")

    calls = mock_tracer.log_event.call_args_list
    event_types = [call[0][0] for call in calls]
    assert "tool_call" in event_types
    assert "tool_result" in event_types


@pytest.mark.anyio
async def test_run_logs_complete_event(
    controller_with_tracer: AgentController,
    mock_llm_client: MagicMock,
    mock_tool_router: MagicMock,
    mock_tracer: MagicMock,
) -> None:
    """run() logs complete event when answer is submitted."""
    mock_llm_client.complete.return_value = _make_llm_response(
        tool_calls=[_submit_answer_tool_call()]
    )
    mock_tool_router.route.return_value = {
        "success": True,
        "answer": "Answer",
        "confidence": 0.8,
        "supporting_evidence": "Evidence",
    }

    await controller_with_tracer.run("Query")

    calls = mock_tracer.log_event.call_args_list
    event_types = [call[0][0] for call in calls]
    assert "complete" in event_types


@pytest.mark.anyio
async def test_run_logs_error_event_on_exception(
    controller_with_tracer: AgentController,
    mock_llm_client: MagicMock,
    mock_tracer: MagicMock,
) -> None:
    """run() logs error event when LLM raises exception."""
    mock_llm_client.complete.side_effect = Exception("API error")

    await controller_with_tracer.run("Query")

    calls = mock_tracer.log_event.call_args_list
    event_types = [call[0][0] for call in calls]
    assert "error" in event_types


@pytest.mark.anyio
async def test_run_logs_max_iterations_event(
    mock_llm_client: MagicMock,
    mock_graph_db: MagicMock,
    mock_tool_router: MagicMock,
    mock_prompt_manager: PromptManager,
    mock_tracer: MagicMock,
) -> None:
    """run() logs max_iterations event when limit reached."""
    config = AgentConfig(max_iterations=1)
    controller = AgentController(
        llm_client=mock_llm_client,
        graph_db=mock_graph_db,
        tool_router=mock_tool_router,
        prompt_manager=mock_prompt_manager,
        config=config,
        tracer=mock_tracer,
    )
    mock_llm_client.complete.return_value = _make_llm_response(
        tool_calls=[_execute_cypher_tool_call()]
    )
    mock_tool_router.route.return_value = {"success": True, "data": [], "message": "OK"}

    await controller.run("Query")

    calls = mock_tracer.log_event.call_args_list
    event_types = [call[0][0] for call in calls]
    assert "max_iterations" in event_types


# --- step() tests ---


@pytest.mark.anyio
async def test_step_raises_without_run(controller: AgentController) -> None:
    """step() raises RuntimeError if called before run()."""
    with pytest.raises(RuntimeError, match="Cannot call step"):
        await controller.step()


@pytest.mark.anyio
async def test_step_increments_iteration(
    controller: AgentController,
    mock_llm_client: MagicMock,
    mock_tool_router: MagicMock,
    mock_graph_db: MagicMock,
) -> None:
    """step() increments the iteration counter."""
    # Initialize via run() first
    mock_llm_client.complete.return_value = _make_llm_response(
        tool_calls=[_submit_answer_tool_call()]
    )
    mock_tool_router.route.return_value = {
        "success": True,
        "answer": "Answer",
        "confidence": 0.8,
        "supporting_evidence": "Evidence",
    }

    result = await controller.run("Query")

    assert result.iterations == 1


# --- Multiple tool calls in single response ---


@pytest.mark.anyio
async def test_run_handles_multiple_tool_calls(
    controller: AgentController,
    mock_llm_client: MagicMock,
    mock_tool_router: MagicMock,
) -> None:
    """run() processes multiple tool calls in a single response."""
    mock_llm_client.complete.return_value = _make_llm_response(
        tool_calls=[
            _execute_cypher_tool_call(),
            _submit_answer_tool_call(),
        ]
    )
    mock_tool_router.route.side_effect = [
        {"success": True, "data": [{"title": "Movie"}], "message": "OK"},
        {
            "success": True,
            "answer": "Answer",
            "confidence": 0.9,
            "supporting_evidence": "Evidence",
        },
    ]

    result = await controller.run("Query")

    # Should complete on submit_answer
    assert result.status == AgentStatus.COMPLETED
    # Both tool calls should be in history
    assert len(result.history) == 2
