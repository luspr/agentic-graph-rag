"""Unit tests for the Tracer module."""

from datetime import datetime


from agentic_graph_rag.agent.state import AgentResult, AgentStatus
from agentic_graph_rag.agent.tracer import Trace, TraceEvent, Tracer


# --- TraceEvent tests ---


def test_trace_event_has_required_fields() -> None:
    """TraceEvent has timestamp, event_type, data fields."""
    now = datetime.now()
    event = TraceEvent(
        timestamp=now,
        event_type="tool_call",
        data={"tool": "execute_cypher"},
    )

    assert event.timestamp == now
    assert event.event_type == "tool_call"
    assert event.data == {"tool": "execute_cypher"}


def test_trace_event_duration_defaults_to_none() -> None:
    """TraceEvent duration_ms defaults to None."""
    event = TraceEvent(
        timestamp=datetime.now(),
        event_type="test",
        data={},
    )

    assert event.duration_ms is None


def test_trace_event_accepts_duration() -> None:
    """TraceEvent accepts duration_ms parameter."""
    event = TraceEvent(
        timestamp=datetime.now(),
        event_type="llm_request",
        data={"model": "gpt-4"},
        duration_ms=1500.5,
    )

    assert event.duration_ms == 1500.5


# --- Trace tests ---


def test_trace_has_required_fields() -> None:
    """Trace has trace_id, query, started_at fields."""
    now = datetime.now()
    trace = Trace(
        trace_id="abc-123",
        query="What is The Matrix?",
        started_at=now,
    )

    assert trace.trace_id == "abc-123"
    assert trace.query == "What is The Matrix?"
    assert trace.started_at == now


def test_trace_events_default_empty() -> None:
    """Trace events defaults to empty list."""
    trace = Trace(
        trace_id="test",
        query="test",
        started_at=datetime.now(),
    )

    assert trace.events == []


def test_trace_completed_at_defaults_none() -> None:
    """Trace completed_at defaults to None."""
    trace = Trace(
        trace_id="test",
        query="test",
        started_at=datetime.now(),
    )

    assert trace.completed_at is None


def test_trace_result_defaults_none() -> None:
    """Trace result defaults to None."""
    trace = Trace(
        trace_id="test",
        query="test",
        started_at=datetime.now(),
    )

    assert trace.result is None


# --- Tracer.start_trace() tests ---


def test_start_trace_returns_trace() -> None:
    """start_trace returns a Trace object."""
    tracer = Tracer()

    trace = tracer.start_trace("What movies did Tom Hanks act in?")

    assert isinstance(trace, Trace)
    assert trace.query == "What movies did Tom Hanks act in?"


def test_start_trace_generates_unique_ids() -> None:
    """start_trace generates unique trace IDs."""
    tracer = Tracer()

    trace1 = tracer.start_trace("Query 1")
    trace2 = tracer.start_trace("Query 2")

    assert trace1.trace_id != trace2.trace_id


def test_start_trace_sets_started_at() -> None:
    """start_trace sets the started_at timestamp."""
    tracer = Tracer()
    before = datetime.now()

    trace = tracer.start_trace("Query")

    after = datetime.now()
    assert before <= trace.started_at <= after


def test_start_trace_logs_query_start_event() -> None:
    """start_trace automatically logs a query_start event."""
    tracer = Tracer()

    trace = tracer.start_trace("Test query")

    assert len(trace.events) == 1
    assert trace.events[0].event_type == "query_start"
    assert trace.events[0].data["query"] == "Test query"


def test_start_trace_sets_current_trace() -> None:
    """start_trace sets the tracer's current trace."""
    tracer = Tracer()

    trace = tracer.start_trace("Query")

    assert tracer.get_current_trace() == trace


# --- Tracer.log_event() tests ---


def test_log_event_adds_event_to_current_trace() -> None:
    """log_event adds event to the current active trace."""
    tracer = Tracer()
    trace = tracer.start_trace("Query")

    tracer.log_event("tool_call", {"tool": "execute_cypher"})

    # 2 events: query_start + tool_call
    assert len(trace.events) == 2
    assert trace.events[1].event_type == "tool_call"


def test_log_event_does_nothing_without_active_trace() -> None:
    """log_event is a no-op when there's no active trace."""
    tracer = Tracer()

    # Should not raise
    tracer.log_event("orphan_event", {"data": "value"})


def test_log_event_sets_timestamp() -> None:
    """log_event sets the event timestamp."""
    tracer = Tracer()
    tracer.start_trace("Query")
    before = datetime.now()

    tracer.log_event("test", {})

    after = datetime.now()
    event = tracer.get_current_trace().events[-1]  # type: ignore[union-attr]
    assert before <= event.timestamp <= after


def test_log_event_preserves_data() -> None:
    """log_event preserves the event data dict."""
    tracer = Tracer()
    tracer.start_trace("Query")
    data = {"key": "value", "nested": {"a": 1}}

    tracer.log_event("test", data)

    event = tracer.get_current_trace().events[-1]  # type: ignore[union-attr]
    assert event.data == data


# --- Tracer.log_event_to_trace() tests ---


def test_log_event_to_trace_logs_to_specific_trace() -> None:
    """log_event_to_trace logs to a specific trace object."""
    tracer = Tracer()
    trace1 = tracer.start_trace("Query 1")
    trace2 = tracer.start_trace("Query 2")

    tracer.log_event_to_trace(trace1, "test", {"target": "trace1"})

    # trace1 should have query_start + test event
    assert len(trace1.events) == 2
    assert trace1.events[1].data["target"] == "trace1"
    # trace2 should only have query_start
    assert len(trace2.events) == 1


# --- Duration tracking tests ---


def test_start_timed_event_tracks_duration() -> None:
    """start_timed_event enables duration tracking for matching event."""
    tracer = Tracer()
    tracer.start_trace("Query")

    tracer.start_timed_event("call_123")
    tracer.log_event("tool_result", {"tool_id": "call_123"})

    event = tracer.get_current_trace().events[-1]  # type: ignore[union-attr]
    assert event.duration_ms is not None
    assert event.duration_ms >= 0


def test_duration_tracking_is_removed_after_use() -> None:
    """Duration tracking entry is removed after being used."""
    tracer = Tracer()
    tracer.start_trace("Query")

    tracer.start_timed_event("call_123")
    tracer.log_event("first", {"tool_id": "call_123"})
    tracer.log_event("second", {"tool_id": "call_123"})

    first_event = tracer.get_current_trace().events[-2]  # type: ignore[union-attr]
    second_event = tracer.get_current_trace().events[-1]  # type: ignore[union-attr]

    assert first_event.duration_ms is not None  # First gets the duration
    assert second_event.duration_ms is None  # Second doesn't


# --- Tracer.end_trace() tests ---


def test_end_trace_sets_completed_at() -> None:
    """end_trace sets the completed_at timestamp."""
    tracer = Tracer()
    trace = tracer.start_trace("Query")
    result = AgentResult(answer="Test", status=AgentStatus.COMPLETED, iterations=1)

    tracer.end_trace(trace, result)

    assert trace.completed_at is not None
    assert trace.completed_at >= trace.started_at


def test_end_trace_sets_result() -> None:
    """end_trace sets the result on the trace."""
    tracer = Tracer()
    trace = tracer.start_trace("Query")
    result = AgentResult(
        answer="Test answer",
        status=AgentStatus.COMPLETED,
        iterations=3,
        confidence=0.95,
    )

    tracer.end_trace(trace, result)

    assert trace.result == result


def test_end_trace_logs_complete_event() -> None:
    """end_trace logs a complete event."""
    tracer = Tracer()
    trace = tracer.start_trace("Query")
    result = AgentResult(
        answer="Test",
        status=AgentStatus.COMPLETED,
        iterations=2,
        confidence=0.9,
    )

    tracer.end_trace(trace, result)

    complete_events = [e for e in trace.events if e.event_type == "complete"]
    assert len(complete_events) == 1
    assert complete_events[0].data["status"] == "completed"
    assert complete_events[0].data["iterations"] == 2
    assert complete_events[0].data["confidence"] == 0.9


def test_end_trace_clears_current_trace() -> None:
    """end_trace clears the current trace if it matches."""
    tracer = Tracer()
    trace = tracer.start_trace("Query")
    result = AgentResult(answer="Test", status=AgentStatus.COMPLETED, iterations=1)

    tracer.end_trace(trace, result)

    assert tracer.get_current_trace() is None


def test_end_trace_does_not_clear_different_current_trace() -> None:
    """end_trace doesn't clear current trace if ending a different trace."""
    tracer = Tracer()
    trace1 = tracer.start_trace("Query 1")
    trace2 = tracer.start_trace("Query 2")  # This becomes current
    result = AgentResult(answer="Test", status=AgentStatus.COMPLETED, iterations=1)

    tracer.end_trace(trace1, result)  # End trace1, not current

    assert tracer.get_current_trace() == trace2


# --- Tracer.export() tests ---


def test_export_returns_dict() -> None:
    """export returns a JSON-serializable dict."""
    tracer = Tracer()
    trace = tracer.start_trace("Test query")

    exported = tracer.export(trace)

    assert isinstance(exported, dict)


def test_export_contains_trace_fields() -> None:
    """export contains all required trace fields."""
    tracer = Tracer()
    trace = tracer.start_trace("What is The Matrix?")

    exported = tracer.export(trace)

    assert exported["trace_id"] == trace.trace_id
    assert exported["query"] == "What is The Matrix?"
    assert "started_at" in exported
    assert "events" in exported


def test_export_timestamps_are_iso_format() -> None:
    """export converts timestamps to ISO format strings."""
    tracer = Tracer()
    trace = tracer.start_trace("Query")
    result = AgentResult(answer="Test", status=AgentStatus.COMPLETED, iterations=1)
    tracer.end_trace(trace, result)

    exported = tracer.export(trace)

    # ISO format should be parseable
    datetime.fromisoformat(exported["started_at"])
    datetime.fromisoformat(exported["completed_at"])


def test_export_includes_events() -> None:
    """export includes all traced events."""
    tracer = Tracer()
    trace = tracer.start_trace("Query")
    tracer.log_event("tool_call", {"tool": "cypher"})
    tracer.log_event("tool_result", {"success": True})

    exported = tracer.export(trace)

    assert len(exported["events"]) == 3  # query_start + 2 events


def test_export_events_have_correct_structure() -> None:
    """export event dicts have timestamp, event_type, data, duration_ms."""
    tracer = Tracer()
    trace = tracer.start_trace("Query")
    tracer.log_event("test", {"key": "value"})

    exported = tracer.export(trace)
    event = exported["events"][-1]

    assert "timestamp" in event
    assert event["event_type"] == "test"
    assert event["data"] == {"key": "value"}
    assert "duration_ms" in event


def test_export_includes_result_when_completed() -> None:
    """export includes result when trace is completed."""
    tracer = Tracer()
    trace = tracer.start_trace("Query")
    result = AgentResult(
        answer="The Matrix is a movie",
        status=AgentStatus.COMPLETED,
        iterations=2,
        confidence=0.95,
    )
    tracer.end_trace(trace, result)

    exported = tracer.export(trace)

    assert exported["result"] is not None
    assert exported["result"]["answer"] == "The Matrix is a movie"
    assert exported["result"]["status"] == "completed"
    assert exported["result"]["iterations"] == 2
    assert exported["result"]["confidence"] == 0.95


def test_export_result_is_none_when_not_completed() -> None:
    """export result is None when trace is not completed."""
    tracer = Tracer()
    trace = tracer.start_trace("Query")

    exported = tracer.export(trace)

    assert exported["result"] is None


def test_export_includes_duration_ms() -> None:
    """export includes total duration when trace is completed."""
    tracer = Tracer()
    trace = tracer.start_trace("Query")
    result = AgentResult(answer="Test", status=AgentStatus.COMPLETED, iterations=1)
    tracer.end_trace(trace, result)

    exported = tracer.export(trace)

    assert exported["duration_ms"] is not None
    assert exported["duration_ms"] >= 0


def test_export_duration_is_none_when_not_completed() -> None:
    """export duration_ms is None when trace is not completed."""
    tracer = Tracer()
    trace = tracer.start_trace("Query")

    exported = tracer.export(trace)

    assert exported["duration_ms"] is None


# --- Tracer.get_trace() tests ---


def test_get_trace_returns_trace_by_id() -> None:
    """get_trace retrieves trace by ID."""
    tracer = Tracer()
    trace = tracer.start_trace("Query")

    retrieved = tracer.get_trace(trace.trace_id)

    assert retrieved == trace


def test_get_trace_returns_none_for_unknown_id() -> None:
    """get_trace returns None for unknown IDs."""
    tracer = Tracer()

    assert tracer.get_trace("nonexistent") is None


# --- Event type coverage ---


def test_all_event_types_can_be_logged() -> None:
    """All expected event types can be logged."""
    tracer = Tracer()
    tracer.start_trace("Query")

    event_types = [
        "query_start",  # Already logged by start_trace
        "tool_call",
        "tool_result",
        "llm_request",
        "llm_response",
        "error",
        "complete",
    ]

    for event_type in event_types[1:]:  # Skip query_start
        tracer.log_event(event_type, {"type": event_type})

    trace = tracer.get_current_trace()
    assert trace is not None
    logged_types = {e.event_type for e in trace.events}
    assert logged_types == set(event_types)
