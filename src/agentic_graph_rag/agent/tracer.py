"""Structured tracer for recording agent execution events."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import uuid

from agentic_graph_rag.agent.state import AgentResult


@dataclass
class TraceEvent:
    """A single traced event during agent execution."""

    timestamp: datetime
    event_type: str
    data: dict[str, Any]
    duration_ms: float | None = None


@dataclass
class Trace:
    """A complete trace for one agent run."""

    trace_id: str
    run_id: str
    query: str
    started_at: datetime
    events: list[TraceEvent] = field(default_factory=list)
    completed_at: datetime | None = None
    result: AgentResult | None = None


class Tracer:
    """Records and exports agent execution events for debugging and analysis.

    The tracer supports both explicit trace management (start_trace, log_event with trace,
    end_trace) and a simpler interface where events are logged to the current active trace.

    When log_file is provided, events are written to a JSONL file (one JSON object per line)
    as they occur, providing an append-only log of all execution events.

    Example:
        tracer = Tracer(log_file="logs/trace_20240101_120000.jsonl")
        trace = tracer.start_trace("What movies did Tom Hanks act in?")
        tracer.log_event("tool_call", {"tool": "execute_cypher", "query": "MATCH..."})
        tracer.end_trace(trace, result)
        export = tracer.export(trace)
    """

    def __init__(self, log_file: Path | str | None = None) -> None:
        """Initialize the tracer.

        Args:
            log_file: Optional path to a JSONL file for persisting events.
                     If provided, events will be written as they occur.
        """
        self._traces: dict[str, Trace] = {}
        self._current_trace: Trace | None = None
        self._event_start_times: dict[str, datetime] = {}
        self._log_file = Path(log_file) if log_file else None
        self._log_file_handle = None

        # Ensure log directory exists and open file for appending
        if self._log_file:
            try:
                self._log_file.parent.mkdir(parents=True, exist_ok=True)
                self._log_file_handle = open(self._log_file, "a", encoding="utf-8")
            except OSError as e:
                # Log error but continue without file logging
                print(f"Warning: Could not open trace log file {self._log_file}: {e}")

    def start_trace(self, query: str) -> Trace:
        """Start a new trace for a query.

        Args:
            query: The user's query being traced.

        Returns:
            A new Trace object with a unique trace_id and run_id.
        """
        trace_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        trace = Trace(
            trace_id=trace_id,
            run_id=run_id,
            query=query,
            started_at=datetime.now(),
        )
        self._traces[trace_id] = trace
        self._current_trace = trace

        # Log the initial query_start event
        self._add_event(trace, "query_start", {"query": query})

        return trace

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log an event to the current active trace.

        This method provides compatibility with the simple Tracer Protocol
        used by AgentController.

        Args:
            event_type: Type of event (e.g., "tool_call", "llm_response").
            data: Event data to record.
        """
        if self._current_trace is None:
            return

        self._add_event(self._current_trace, event_type, data)

    def log_event_to_trace(
        self, trace: Trace, event_type: str, data: dict[str, Any]
    ) -> None:
        """Log an event to a specific trace.

        Args:
            trace: The trace to log to.
            event_type: Type of event.
            data: Event data to record.
        """
        self._add_event(trace, event_type, data)

    def start_timed_event(self, event_id: str) -> None:
        """Start timing an event for duration tracking.

        Call this before a potentially long operation (LLM request, tool execution).
        Then call log_event with the same event_id in the data to record duration.

        Args:
            event_id: Unique identifier for the timed operation.
        """
        self._event_start_times[event_id] = datetime.now()

    def end_trace(self, trace: Trace, result: AgentResult) -> None:
        """Complete a trace with the final result.

        Args:
            trace: The trace to complete.
            result: The final AgentResult from the agent run.
        """
        trace.completed_at = datetime.now()
        trace.result = result

        # Log the completion event
        self._add_event(
            trace,
            "complete",
            {
                "status": result.status.value,
                "iterations": result.iterations,
                "confidence": result.confidence,
            },
        )

        # Clear current trace if it matches
        if (
            self._current_trace is not None
            and self._current_trace.trace_id == trace.trace_id
        ):
            self._current_trace = None

    def export(self, trace: Trace) -> dict[str, Any]:
        """Export trace as a JSON-serializable dict.

        Args:
            trace: The trace to export.

        Returns:
            A dictionary containing all trace data in JSON-serializable format.
        """
        return {
            "trace_id": trace.trace_id,
            "query": trace.query,
            "started_at": trace.started_at.isoformat(),
            "completed_at": trace.completed_at.isoformat()
            if trace.completed_at
            else None,
            "duration_ms": self._calculate_trace_duration(trace),
            "events": [self._export_event(event) for event in trace.events],
            "result": self._export_result(trace.result) if trace.result else None,
        }

    def get_trace(self, trace_id: str) -> Trace | None:
        """Get a trace by ID.

        Args:
            trace_id: The trace ID to retrieve.

        Returns:
            The Trace if found, None otherwise.
        """
        return self._traces.get(trace_id)

    def get_current_trace(self) -> Trace | None:
        """Get the current active trace.

        Returns:
            The current active Trace if one exists, None otherwise.
        """
        return self._current_trace

    def _add_event(self, trace: Trace, event_type: str, data: dict[str, Any]) -> None:
        """Add an event to a trace with optional duration tracking."""
        duration_ms: float | None = None
        timestamp = datetime.now()

        # Check if we have timing data for this event
        event_id = data.get("tool_id") or data.get("event_id")
        if event_id and event_id in self._event_start_times:
            start_time = self._event_start_times.pop(event_id)
            delta = timestamp - start_time
            duration_ms = delta.total_seconds() * 1000

        event = TraceEvent(
            timestamp=timestamp,
            event_type=event_type,
            data=data,
            duration_ms=duration_ms,
        )
        trace.events.append(event)

        # Write event to JSONL file if configured
        self._write_event_to_file(trace, event)

    def _calculate_trace_duration(self, trace: Trace) -> float | None:
        """Calculate total trace duration in milliseconds."""
        if trace.completed_at is None:
            return None
        delta = trace.completed_at - trace.started_at
        return delta.total_seconds() * 1000

    def _export_event(self, event: TraceEvent) -> dict[str, Any]:
        """Export a single event as a JSON-serializable dict."""
        return {
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "data": event.data,
            "duration_ms": event.duration_ms,
        }

    def _export_result(self, result: AgentResult) -> dict[str, Any]:
        """Export an AgentResult as a JSON-serializable dict."""
        return {
            "answer": result.answer,
            "status": result.status.value,
            "iterations": result.iterations,
            "confidence": result.confidence,
            "history_steps": len(result.history),
        }

    def _write_event_to_file(self, trace: Trace, event: TraceEvent) -> None:
        """Write an event to the JSONL log file.

        Args:
            trace: The trace this event belongs to.
            event: The event to write.
        """
        if not self._log_file_handle:
            return

        try:
            event_record: dict[str, Any] = {
                "run_id": trace.run_id,
                "trace_id": trace.trace_id,
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "data": event.data,
            }

            if event.duration_ms is not None:
                event_record["duration_ms"] = event.duration_ms

            json.dump(event_record, self._log_file_handle)
            self._log_file_handle.write("\n")
            self._log_file_handle.flush()
        except (OSError, TypeError, ValueError) as e:
            # Gracefully handle file write errors without crashing
            print(f"Warning: Failed to write event to trace log: {e}")

    def close(self) -> None:
        """Close the log file if open."""
        if self._log_file_handle:
            try:
                self._log_file_handle.close()
            except OSError:
                pass
            self._log_file_handle = None

    def __del__(self) -> None:
        """Ensure log file is closed on deletion."""
        self.close()
