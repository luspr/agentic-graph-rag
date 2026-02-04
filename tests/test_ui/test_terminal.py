"""Unit tests for TerminalUI, UITracer, and TraceInspector."""

from io import StringIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console, Group

from agentic_graph_rag.agent.state import AgentConfig, AgentResult, AgentStatus
from agentic_graph_rag.agent.tools import ToolRouter
from agentic_graph_rag.graph.base import GraphDatabase, GraphSchema, NodeType
from agentic_graph_rag.llm.base import LLMClient
from agentic_graph_rag.prompts.manager import PromptManager
from agentic_graph_rag.retriever.base import RetrievalStep
from agentic_graph_rag.ui.terminal import (
    TerminalUI,
    TraceInspector,
    UITracer,
    _format_data_pretty,
)


# --- Fixtures ---


@pytest.fixture
def console() -> Console:
    """Create a console that writes to a StringIO for testing."""
    return Console(file=StringIO(), force_terminal=True)


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
def prompt_manager() -> PromptManager:
    """Create a real PromptManager."""
    return PromptManager()


@pytest.fixture
def ui_tracer(console: Console) -> UITracer:
    """Create a UITracer for testing."""
    return UITracer(console)


@pytest.fixture
def terminal_ui(
    mock_llm_client: MagicMock,
    mock_graph_db: MagicMock,
    mock_tool_router: MagicMock,
    prompt_manager: PromptManager,
) -> TerminalUI:
    """Create a TerminalUI with mocked dependencies."""
    return TerminalUI(
        llm_client=mock_llm_client,
        graph_db=mock_graph_db,
        tool_router=mock_tool_router,
        prompt_manager=prompt_manager,
    )


# --- UITracer tests ---


def test_ui_tracer_inherits_from_tracer(ui_tracer: UITracer) -> None:
    """UITracer inherits from Tracer."""
    from agentic_graph_rag.agent.tracer import Tracer

    assert isinstance(ui_tracer, Tracer)


def test_ui_tracer_set_max_iterations(ui_tracer: UITracer) -> None:
    """UITracer can set max iterations."""
    ui_tracer.set_max_iterations(20)
    assert ui_tracer._max_iterations == 20


def test_ui_tracer_log_event_calls_parent(ui_tracer: UITracer) -> None:
    """UITracer.log_event calls parent method."""
    trace = ui_tracer.start_trace("test query")
    ui_tracer.log_event("test_event", {"key": "value"})

    # Check event was logged
    assert len(trace.events) == 2  # query_start + test_event
    assert trace.events[1].event_type == "test_event"
    assert trace.events[1].data == {"key": "value"}


def test_ui_tracer_update_progress_without_context(ui_tracer: UITracer) -> None:
    """UITracer handles log_event without live context."""
    # Should not raise even without live context
    ui_tracer.start_trace("test")
    ui_tracer.log_event("iteration_start", {"iteration": 1})


def test_ui_tracer_update_progress_with_context(ui_tracer: UITracer) -> None:
    """UITracer updates progress with live context."""
    mock_progress = MagicMock()
    mock_live = MagicMock()
    task_id = "task_1"

    ui_tracer.set_live_context(mock_live, mock_progress, task_id)
    ui_tracer.set_max_iterations(10)
    ui_tracer.start_trace("test")
    ui_tracer.log_event("iteration_start", {"iteration": 3})

    mock_progress.update.assert_called()
    call_args = mock_progress.update.call_args
    assert "Iteration 3/10" in call_args[1]["description"]


def test_ui_tracer_updates_on_tool_call(ui_tracer: UITracer) -> None:
    """UITracer updates progress description on tool_call event."""
    mock_progress = MagicMock()
    mock_live = MagicMock()
    task_id = "task_1"

    ui_tracer.set_live_context(mock_live, mock_progress, task_id)
    ui_tracer.set_max_iterations(10)
    ui_tracer.start_trace("test")

    # Set current iteration first
    ui_tracer.log_event("iteration_start", {"iteration": 2})
    mock_progress.reset_mock()

    # Then tool call
    ui_tracer.log_event("tool_call", {"tool_name": "execute_cypher"})

    mock_progress.update.assert_called()
    call_args = mock_progress.update.call_args
    assert "execute_cypher" in call_args[1]["description"]


def test_ui_tracer_updates_on_llm_request(ui_tracer: UITracer) -> None:
    """UITracer updates progress description on llm_request event."""
    mock_progress = MagicMock()
    mock_live = MagicMock()
    task_id = "task_1"

    ui_tracer.set_live_context(mock_live, mock_progress, task_id)
    ui_tracer.start_trace("test")
    ui_tracer.log_event("llm_request", {"messages_count": 5})

    mock_progress.update.assert_called()
    call_args = mock_progress.update.call_args
    assert "Querying LLM" in call_args[1]["description"]


# --- TerminalUI command tests ---


def test_terminal_ui_commands_defined(terminal_ui: TerminalUI) -> None:
    """TerminalUI has expected commands."""
    assert "/quit" in TerminalUI.COMMANDS
    assert "/clear" in TerminalUI.COMMANDS
    assert "/trace" in TerminalUI.COMMANDS
    assert "/help" in TerminalUI.COMMANDS


@pytest.mark.anyio
async def test_handle_command_quit(terminal_ui: TerminalUI) -> None:
    """_handle_command returns False for /quit."""
    result = await terminal_ui._handle_command("/quit")
    assert result is False


@pytest.mark.anyio
async def test_handle_command_clear(terminal_ui: TerminalUI) -> None:
    """_handle_command returns True for /clear and clears session."""
    # Create a session first
    terminal_ui._session = terminal_ui._session_manager.create_session()
    terminal_ui._session_manager.add_message(terminal_ui._session, "user", "test")

    result = await terminal_ui._handle_command("/clear")

    assert result is True
    assert len(terminal_ui._session.messages) == 0


@pytest.mark.anyio
async def test_handle_command_help(terminal_ui: TerminalUI) -> None:
    """_handle_command returns True for /help."""
    result = await terminal_ui._handle_command("/help")
    assert result is True


@pytest.mark.anyio
async def test_handle_command_trace_no_trace(terminal_ui: TerminalUI) -> None:
    """_handle_command returns True for /trace even without a trace."""
    result = await terminal_ui._handle_command("/trace")
    assert result is True


@pytest.mark.anyio
async def test_handle_command_unknown(terminal_ui: TerminalUI) -> None:
    """_handle_command returns True for unknown commands."""
    result = await terminal_ui._handle_command("/unknown")
    assert result is True


@pytest.mark.anyio
async def test_handle_command_case_insensitive(terminal_ui: TerminalUI) -> None:
    """_handle_command is case insensitive."""
    result_upper = await terminal_ui._handle_command("/QUIT")
    result_mixed = await terminal_ui._handle_command("/QuIt")

    assert result_upper is False
    assert result_mixed is False


# --- TerminalUI display tests ---


def test_display_result_completed(terminal_ui: TerminalUI) -> None:
    """_display_result handles COMPLETED status."""
    result = AgentResult(
        answer="The answer is 42",
        status=AgentStatus.COMPLETED,
        iterations=3,
        confidence=0.85,
        history=[],
    )

    # Should not raise
    terminal_ui._display_result(result)


def test_display_result_max_iterations(terminal_ui: TerminalUI) -> None:
    """_display_result handles MAX_ITERATIONS status."""
    result = AgentResult(
        answer="Partial answer",
        status=AgentStatus.MAX_ITERATIONS,
        iterations=10,
        confidence=None,
        history=[],
    )

    # Should not raise
    terminal_ui._display_result(result)


def test_display_result_error(terminal_ui: TerminalUI) -> None:
    """_display_result handles ERROR status."""
    result = AgentResult(
        answer="Error occurred",
        status=AgentStatus.ERROR,
        iterations=1,
        confidence=None,
        history=[],
    )

    # Should not raise
    terminal_ui._display_result(result)


def test_display_result_with_evidence(terminal_ui: TerminalUI) -> None:
    """_display_result shows supporting evidence from submit_answer."""
    result = AgentResult(
        answer="The answer",
        status=AgentStatus.COMPLETED,
        iterations=1,
        confidence=0.9,
        history=[
            RetrievalStep(
                action="submit_answer",
                input={"answer": "The answer"},
                output={
                    "answer": "The answer",
                    "confidence": 0.9,
                    "supporting_evidence": "Found in database",
                    "success": True,
                },
                error=None,
            )
        ],
    )

    # Should not raise
    terminal_ui._display_result(result)


# --- TerminalUI trace display tests ---


def test_show_trace_no_trace(terminal_ui: TerminalUI) -> None:
    """_show_trace handles no trace available."""
    terminal_ui._last_trace = None
    # Should not raise
    terminal_ui._show_trace()


def test_show_trace_with_trace(terminal_ui: TerminalUI) -> None:
    """_show_trace displays trace information."""
    # Create a trace
    trace = terminal_ui._tracer.start_trace("test query")
    terminal_ui._tracer.log_event("tool_call", {"tool_name": "test"})
    result = AgentResult(
        answer="Answer",
        status=AgentStatus.COMPLETED,
        iterations=1,
        confidence=0.8,
    )
    terminal_ui._tracer.end_trace(trace, result)
    terminal_ui._last_trace = trace

    # Should not raise
    terminal_ui._show_trace()


# --- TerminalUI format_event_details tests ---


def test_format_event_details_query_start(terminal_ui: TerminalUI) -> None:
    """_format_event_details formats query_start event."""
    event = {"event_type": "query_start", "data": {"query": "test query"}}
    result = terminal_ui._format_event_details(event)
    assert "test query" in result


def test_format_event_details_iteration_start(terminal_ui: TerminalUI) -> None:
    """_format_event_details formats iteration_start event."""
    event = {"event_type": "iteration_start", "data": {"iteration": 3}}
    result = terminal_ui._format_event_details(event)
    assert "3" in result


def test_format_event_details_tool_call_cypher(terminal_ui: TerminalUI) -> None:
    """_format_event_details formats execute_cypher tool_call."""
    event = {
        "event_type": "tool_call",
        "data": {
            "tool_name": "execute_cypher",
            "arguments": {"query": "MATCH (n) RETURN n"},
        },
    }
    result = terminal_ui._format_event_details(event)
    assert "execute_cypher" in result
    assert "MATCH" in result


def test_format_event_details_tool_call_submit(terminal_ui: TerminalUI) -> None:
    """_format_event_details formats submit_answer tool_call."""
    event = {
        "event_type": "tool_call",
        "data": {
            "tool_name": "submit_answer",
            "arguments": {"confidence": 0.9},
        },
    }
    result = terminal_ui._format_event_details(event)
    assert "submit_answer" in result
    assert "0.9" in result


def test_format_event_details_tool_result(terminal_ui: TerminalUI) -> None:
    """_format_event_details formats tool_result event."""
    event_success = {"event_type": "tool_result", "data": {"success": True}}
    event_fail = {"event_type": "tool_result", "data": {"success": False}}

    assert terminal_ui._format_event_details(event_success) == "success"
    assert terminal_ui._format_event_details(event_fail) == "failed"


def test_format_event_details_llm_request(terminal_ui: TerminalUI) -> None:
    """_format_event_details formats llm_request event."""
    event = {"event_type": "llm_request", "data": {"messages_count": 5}}
    result = terminal_ui._format_event_details(event)
    assert "5" in result


def test_format_event_details_llm_response(terminal_ui: TerminalUI) -> None:
    """_format_event_details formats llm_response event."""
    event = {"event_type": "llm_response", "data": {"tool_calls_count": 2}}
    result = terminal_ui._format_event_details(event)
    assert "2" in result


def test_format_event_details_error(terminal_ui: TerminalUI) -> None:
    """_format_event_details formats error event."""
    event = {"event_type": "error", "data": {"error": "Connection failed"}}
    result = terminal_ui._format_event_details(event)
    assert "Connection failed" in result


def test_format_event_details_complete(terminal_ui: TerminalUI) -> None:
    """_format_event_details formats complete event."""
    event = {"event_type": "complete", "data": {"status": "completed"}}
    result = terminal_ui._format_event_details(event)
    assert "completed" in result


def test_format_event_details_max_iterations(terminal_ui: TerminalUI) -> None:
    """_format_event_details formats max_iterations event."""
    event = {"event_type": "max_iterations", "data": {"iterations": 10, "max": 10}}
    result = terminal_ui._format_event_details(event)
    assert "10" in result


def test_format_event_details_unknown(terminal_ui: TerminalUI) -> None:
    """_format_event_details handles unknown event type."""
    event = {"event_type": "unknown_event", "data": {"foo": "bar"}}
    result = terminal_ui._format_event_details(event)
    # Should return JSON representation
    assert "bar" in result or "foo" in result


def test_format_event_details_empty_data(terminal_ui: TerminalUI) -> None:
    """_format_event_details handles empty data."""
    event = {"event_type": "unknown_event", "data": {}}
    result = terminal_ui._format_event_details(event)
    assert result == "-"


# --- TerminalUI initialization tests ---


def test_terminal_ui_creates_session_manager(terminal_ui: TerminalUI) -> None:
    """TerminalUI creates a SessionManager."""
    assert terminal_ui._session_manager is not None


def test_terminal_ui_creates_tracer(terminal_ui: TerminalUI) -> None:
    """TerminalUI creates a UITracer."""
    assert terminal_ui._tracer is not None
    assert isinstance(terminal_ui._tracer, UITracer)


def test_terminal_ui_uses_config(
    mock_llm_client: MagicMock,
    mock_graph_db: MagicMock,
    mock_tool_router: MagicMock,
    prompt_manager: PromptManager,
) -> None:
    """TerminalUI uses provided config."""
    config = AgentConfig(max_iterations=5, max_history_messages=4)
    ui = TerminalUI(
        llm_client=mock_llm_client,
        graph_db=mock_graph_db,
        tool_router=mock_tool_router,
        prompt_manager=prompt_manager,
        config=config,
    )
    assert ui._config.max_iterations == 5
    assert ui._config.max_history_messages == 4


def test_terminal_ui_default_config(terminal_ui: TerminalUI) -> None:
    """TerminalUI uses default config if not provided."""
    assert terminal_ui._config.max_iterations == 10
    assert terminal_ui._config.max_history_messages == 10


# --- TerminalUI session handling ---


def test_clear_session(terminal_ui: TerminalUI) -> None:
    """_clear_session clears session and trace."""
    terminal_ui._session = terminal_ui._session_manager.create_session()
    terminal_ui._session_manager.add_message(terminal_ui._session, "user", "test")
    terminal_ui._last_trace = terminal_ui._tracer.start_trace("test")

    terminal_ui._clear_session()

    assert len(terminal_ui._session.messages) == 0
    assert terminal_ui._last_trace is None


def test_clear_session_without_session(terminal_ui: TerminalUI) -> None:
    """_clear_session handles no session."""
    terminal_ui._session = None
    # Should not raise
    terminal_ui._clear_session()


# --- TerminalUI print helpers ---


def test_print_welcome(terminal_ui: TerminalUI) -> None:
    """_print_welcome prints welcome message."""
    # Should not raise
    terminal_ui._print_welcome()


def test_print_goodbye(terminal_ui: TerminalUI) -> None:
    """_print_goodbye prints goodbye message."""
    # Should not raise
    terminal_ui._print_goodbye()


def test_show_help(terminal_ui: TerminalUI) -> None:
    """_show_help prints help message."""
    # Should not raise
    terminal_ui._show_help()


# --- TraceInspector tests ---


@pytest.fixture
def sample_trace_data() -> dict:
    """Create sample trace data for testing."""
    return {
        "trace_id": "test-trace-123",
        "query": "What movies did Tom Hanks act in?",
        "started_at": "2024-01-01T12:00:00",
        "completed_at": "2024-01-01T12:00:05",
        "duration_ms": 5000.0,
        "events": [
            {
                "event_type": "query_start",
                "timestamp": "2024-01-01T12:00:00",
                "data": {"query": "What movies did Tom Hanks act in?"},
                "duration_ms": None,
            },
            {
                "event_type": "iteration_start",
                "timestamp": "2024-01-01T12:00:01",
                "data": {"iteration": 1},
                "duration_ms": None,
            },
            {
                "event_type": "tool_call",
                "timestamp": "2024-01-01T12:00:02",
                "data": {
                    "tool_name": "execute_cypher",
                    "arguments": {
                        "query": "MATCH (p:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie) "
                        "RETURN m.title",
                        "reasoning": "Find movies Tom Hanks acted in",
                    },
                },
                "duration_ms": 150.0,
            },
            {
                "event_type": "tool_result",
                "timestamp": "2024-01-01T12:00:03",
                "data": {"success": True, "result": ["Forrest Gump", "Cast Away"]},
                "duration_ms": None,
            },
            {
                "event_type": "complete",
                "timestamp": "2024-01-01T12:00:05",
                "data": {"status": "completed"},
                "duration_ms": None,
            },
        ],
        "result": {
            "answer": "Tom Hanks acted in Forrest Gump and Cast Away.",
            "status": "completed",
            "iterations": 1,
            "confidence": 0.95,
        },
    }


def test_trace_inspector_init(console: Console, sample_trace_data: dict) -> None:
    """TraceInspector initializes with correct defaults."""
    inspector = TraceInspector(console, sample_trace_data)

    assert inspector._trace_data == sample_trace_data
    assert len(inspector._events) == 5
    assert inspector._selected_index == 0
    assert not inspector._detail_view
    assert inspector._running


def test_trace_inspector_empty_events(console: Console) -> None:
    """TraceInspector handles empty events gracefully."""
    trace_data = {"trace_id": "test", "query": "test", "events": []}
    inspector = TraceInspector(console, trace_data)

    # Should not raise
    inspector.run()


def test_trace_inspector_non_interactive_mode(
    console: Console, sample_trace_data: dict
) -> None:
    """TraceInspector renders once in non-interactive mode."""
    inspector = TraceInspector(console, sample_trace_data, interactive=False)

    # Should render and return immediately
    inspector.run()


def test_trace_inspector_move_selection(
    console: Console, sample_trace_data: dict
) -> None:
    """TraceInspector selection navigation works correctly."""
    inspector = TraceInspector(console, sample_trace_data, interactive=False)

    assert inspector._selected_index == 0

    # Move down
    inspector._move_selection(1)
    assert inspector._selected_index == 1

    inspector._move_selection(1)
    assert inspector._selected_index == 2

    # Move up
    inspector._move_selection(-1)
    assert inspector._selected_index == 1

    # Can't go below 0
    inspector._move_selection(-10)
    assert inspector._selected_index == 0

    # Can't go above max
    inspector._move_selection(100)
    assert inspector._selected_index == 4  # 5 events, max index is 4


def test_trace_inspector_toggle_detail_view(
    console: Console, sample_trace_data: dict
) -> None:
    """TraceInspector toggles between list and detail view."""
    inspector = TraceInspector(console, sample_trace_data, interactive=False)

    assert not inspector._detail_view

    inspector._toggle_detail_view()
    assert inspector._detail_view

    inspector._toggle_detail_view()
    assert not inspector._detail_view


def test_trace_inspector_format_event_summary_query_start(
    console: Console, sample_trace_data: dict
) -> None:
    """TraceInspector formats query_start event summary."""
    inspector = TraceInspector(console, sample_trace_data)
    event = sample_trace_data["events"][0]

    summary = inspector._format_event_summary(event)
    assert "What movies did Tom Hanks act in?" in summary


def test_trace_inspector_format_event_summary_tool_call(
    console: Console, sample_trace_data: dict
) -> None:
    """TraceInspector formats tool_call event summary."""
    inspector = TraceInspector(console, sample_trace_data)
    event = sample_trace_data["events"][2]

    summary = inspector._format_event_summary(event)
    assert "execute_cypher" in summary
    assert "MATCH" in summary


def test_trace_inspector_format_event_summary_complete(
    console: Console, sample_trace_data: dict
) -> None:
    """TraceInspector formats complete event summary."""
    inspector = TraceInspector(console, sample_trace_data)
    event = sample_trace_data["events"][4]

    summary = inspector._format_event_summary(event)
    assert "completed" in summary


def test_trace_inspector_render_list_view(
    console: Console, sample_trace_data: dict
) -> None:
    """TraceInspector renders list view without error."""
    inspector = TraceInspector(console, sample_trace_data, interactive=False)
    inspector._detail_view = False

    # Should not raise
    inspector._render_list_view()


def test_trace_inspector_render_detail_view(
    console: Console, sample_trace_data: dict
) -> None:
    """TraceInspector renders detail view without error."""
    inspector = TraceInspector(console, sample_trace_data, interactive=False)
    inspector._detail_view = True
    inspector._selected_index = 2  # Select tool_call event

    # Should not raise
    inspector._render_detail_view()


def test_trace_inspector_build_renderable(
    console: Console, sample_trace_data: dict
) -> None:
    """TraceInspector builds a combined renderable."""
    inspector = TraceInspector(console, sample_trace_data, interactive=False)
    renderable = inspector._build_renderable()

    assert isinstance(renderable, Group)


def test_trace_inspector_detail_overflows(sample_trace_data: dict) -> None:
    """TraceInspector detects when detail view exceeds terminal height."""
    console = Console(file=StringIO(), force_terminal=True, width=40, height=5)
    inspector = TraceInspector(console, sample_trace_data, interactive=False)

    assert inspector._detail_overflows()


# --- _format_data_pretty tests ---


def test_format_data_pretty_string() -> None:
    """_format_data_pretty formats simple strings."""
    result = _format_data_pretty("hello")
    assert '"hello"' in result.plain


def test_format_data_pretty_long_string() -> None:
    """_format_data_pretty formats long strings."""
    long_str = "a" * 100
    result = _format_data_pretty(long_str)
    assert long_str in result.plain


def test_format_data_pretty_multiline_string() -> None:
    """_format_data_pretty formats multi-line strings."""
    result = _format_data_pretty("line1\nline2\nline3")
    assert "line1" in result.plain
    assert "line2" in result.plain


def test_format_data_pretty_integer() -> None:
    """_format_data_pretty formats integers."""
    result = _format_data_pretty(42)
    assert "42" in result.plain


def test_format_data_pretty_float() -> None:
    """_format_data_pretty formats floats."""
    result = _format_data_pretty(3.14)
    assert "3.14" in result.plain


def test_format_data_pretty_bool_true() -> None:
    """_format_data_pretty formats boolean True."""
    result = _format_data_pretty(True)
    assert "true" in result.plain


def test_format_data_pretty_bool_false() -> None:
    """_format_data_pretty formats boolean False."""
    result = _format_data_pretty(False)
    assert "false" in result.plain


def test_format_data_pretty_none() -> None:
    """_format_data_pretty formats None."""
    result = _format_data_pretty(None)
    assert "null" in result.plain


def test_format_data_pretty_empty_dict() -> None:
    """_format_data_pretty formats empty dict."""
    result = _format_data_pretty({})
    assert "{}" in result.plain


def test_format_data_pretty_simple_dict() -> None:
    """_format_data_pretty formats simple dict."""
    result = _format_data_pretty({"key": "value"})
    assert '"key"' in result.plain
    assert '"value"' in result.plain


def test_format_data_pretty_nested_dict() -> None:
    """_format_data_pretty formats nested dict."""
    result = _format_data_pretty({"outer": {"inner": "value"}})
    assert '"outer"' in result.plain
    assert '"inner"' in result.plain
    assert '"value"' in result.plain


def test_format_data_pretty_empty_list() -> None:
    """_format_data_pretty formats empty list."""
    result = _format_data_pretty([])
    assert "[]" in result.plain


def test_format_data_pretty_simple_list() -> None:
    """_format_data_pretty formats simple list."""
    result = _format_data_pretty([1, 2, 3])
    assert "1" in result.plain
    assert "2" in result.plain
    assert "3" in result.plain


def test_format_data_pretty_complex_structure() -> None:
    """_format_data_pretty formats complex nested structures."""
    data = {
        "name": "test",
        "values": [1, 2, 3],
        "nested": {"a": True, "b": None},
    }
    result = _format_data_pretty(data)
    plain = result.plain

    assert '"name"' in plain
    assert '"test"' in plain
    assert '"values"' in plain
    assert "1" in plain
    assert '"nested"' in plain
    assert '"a"' in plain
    assert "true" in plain
    assert "null" in plain
