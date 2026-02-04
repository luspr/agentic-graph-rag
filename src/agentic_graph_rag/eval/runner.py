"""Evaluation runner for benchmark datasets."""

from __future__ import annotations

import json
import statistics
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import anyio

from agentic_graph_rag.agent.state import AgentResult, AgentStatus
from agentic_graph_rag.eval.context import build_evidence_context
from agentic_graph_rag.eval.judge import judge_answer
from agentic_graph_rag.eval.metrics import EmbeddingCache, score_answer
from agentic_graph_rag.eval.types import (
    AggregateScore,
    BenchmarkExample,
    EvalConfig,
    ExampleRun,
)
from agentic_graph_rag.llm.base import LLMClient
from agentic_graph_rag.runner import HeadlessRunner


class JSONLWriter:
    """Thread-safe JSONL writer for concurrent tasks."""

    def __init__(self, path: Path) -> None:
        """Open a JSONL file for appending."""
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        """Write a record as a JSONL line."""
        with self._lock:
            json.dump(record, self._handle)
            self._handle.write("\n")
            self._handle.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        with self._lock:
            try:
                self._handle.close()
            except OSError:
                pass


class RunEventLogger:
    """Minimal tracer implementation for JSONL batch logs."""

    def __init__(self, writer: JSONLWriter, trace_id: str, run_id: str) -> None:
        """Initialize the logger with shared writer and IDs."""
        self._writer = writer
        self._trace_id = trace_id
        self._run_id = run_id

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log a trace event to the shared JSONL file."""
        event = {
            "run_id": self._run_id,
            "trace_id": self._trace_id,
            "event_type": event_type,
            "timestamp": time.time(),
            "data": data,
        }
        self._writer.write(event)


async def run_evaluation(
    *,
    examples: list[BenchmarkExample],
    runner: HeadlessRunner,
    config: EvalConfig,
    result_writer: JSONLWriter | None = None,
    trace_writer: JSONLWriter | None = None,
    judge_client: LLMClient | None = None,
    embedding_cache: EmbeddingCache | None = None,
) -> tuple[list[ExampleRun], AggregateScore]:
    """Run evaluation for a list of benchmark examples."""
    if config.judge_enabled and judge_client is None:
        raise ValueError("judge_client is required when judge_enabled is True.")
    if config.embedding_enabled and embedding_cache is None:
        raise ValueError("embedding_cache is required when embedding_enabled is True.")

    if config.max_examples is not None:
        examples = examples[: config.max_examples]

    results: list[ExampleRun] = []
    results_lock = anyio.Lock()
    limiter = anyio.CapacityLimiter(max(1, config.concurrency))

    async def _process_example(example: BenchmarkExample) -> None:
        async with limiter:
            run_id = str(uuid.uuid4())
            if trace_writer is not None:
                trace_id = str(uuid.uuid4())
                tracer = RunEventLogger(trace_writer, trace_id, run_id)
            else:
                trace_id = None
                tracer = None

            start = time.monotonic()
            try:
                agent_result = await runner.run_query(example.question, tracer=tracer)
            except Exception as exc:  # noqa: BLE001
                agent_result = AgentResult(
                    answer=f"Error during run: {exc}",
                    status=AgentStatus.ERROR,
                    iterations=0,
                )
            latency_ms = (time.monotonic() - start) * 1000

            metrics = await score_answer(
                answer=agent_result.answer,
                ground_truth=example.ground_truth,
                nuggets=example.nuggets,
                embedding_cache=embedding_cache if config.embedding_enabled else None,
                nugget_threshold=config.nugget_threshold,
            )

            judge_score = None
            if config.judge_enabled and judge_client is not None:
                evidence = build_evidence_context(agent_result)
                judge_score = await judge_answer(
                    client=judge_client,
                    question=example.question,
                    ground_truth=example.ground_truth,
                    answer=agent_result.answer,
                    evidence=evidence,
                )

            run = ExampleRun(
                example_id=example.example_id,
                question=example.question,
                ground_truth=example.ground_truth,
                answer=agent_result.answer,
                status=agent_result.status.value,
                iterations=agent_result.iterations,
                confidence=agent_result.confidence,
                latency_ms=latency_ms,
                strategy=config.strategy.value,
                metrics=metrics,
                judge=judge_score,
                run_id=run_id,
                trace_id=trace_id,
            )

            if result_writer is not None:
                result_writer.write(run.to_dict())

            async with results_lock:
                results.append(run)

    async with anyio.create_task_group() as task_group:
        for example in examples:
            task_group.start_soon(_process_example, example)

    summary = _summarize(results)
    return results, summary


def _summarize(results: list[ExampleRun]) -> AggregateScore:
    """Aggregate metrics from example results."""
    total = len(results)
    status_counts = Counter(run.status for run in results)
    success_rate = (
        status_counts.get(AgentStatus.COMPLETED.value, 0) / total if total else 0.0
    )

    avg_iterations = _mean_or_none(run.iterations for run in results)
    avg_latency_ms = _mean_or_none(
        run.latency_ms for run in results if run.latency_ms is not None
    )

    metrics_fields = list(results[0].metrics.to_dict().keys()) if results else []
    metrics_mean: dict[str, float | None] = {}
    metrics_median: dict[str, float | None] = {}
    for field in metrics_fields:
        values = [
            getattr(run.metrics, field)
            for run in results
            if getattr(run.metrics, field) is not None
        ]
        metrics_mean[field] = _mean_or_none(values)
        metrics_median[field] = _median_or_none(values)

    judge_fields = ["correctness", "completeness", "faithfulness", "overall"]
    judge_mean: dict[str, float | None] = {}
    for field in judge_fields:
        values = [
            getattr(run.judge, field)
            for run in results
            if run.judge is not None and getattr(run.judge, field) is not None
        ]
        judge_mean[field] = _mean_or_none(values)

    return AggregateScore(
        total=total,
        success_rate=success_rate,
        status_counts=dict(status_counts),
        metrics_mean=metrics_mean,
        metrics_median=metrics_median,
        judge_mean=judge_mean,
        avg_iterations=avg_iterations,
        avg_latency_ms=avg_latency_ms,
    )


def _mean_or_none(values: Any) -> float | None:
    """Return mean for a list of numeric values or None."""
    values_list = list(values)
    if not values_list:
        return None
    return float(statistics.mean(values_list))


def _median_or_none(values: Any) -> float | None:
    """Return median for a list of numeric values or None."""
    values_list = list(values)
    if not values_list:
        return None
    return float(statistics.median(values_list))
