"""Shared datatypes for evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_graph_rag.retriever.base import RetrievalStrategy


@dataclass(slots=True)
class BenchmarkNuggets:
    """Nugget annotations for an evaluation example."""

    vital: list[str] = field(default_factory=list)
    okay: list[str] = field(default_factory=list)
    trivial: list[str] = field(default_factory=list)

    def all(self) -> list[str]:
        """Return all nuggets as a single list."""
        return [*self.vital, *self.okay, *self.trivial]


@dataclass(slots=True)
class BenchmarkExample:
    """A single benchmark example."""

    example_id: str
    question: str
    ground_truth: str
    nuggets: BenchmarkNuggets | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExampleScore:
    """Scores for a single answer."""

    exact_match: float
    token_f1: float
    rouge_l_f1: float
    embedding_cosine: float | None = None
    nugget_recall_vital: float | None = None
    nugget_recall_overall: float | None = None
    nugget_precision_vital: float | None = None
    nugget_precision_overall: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "exact_match": self.exact_match,
            "token_f1": self.token_f1,
            "rouge_l_f1": self.rouge_l_f1,
            "embedding_cosine": self.embedding_cosine,
            "nugget_recall_vital": self.nugget_recall_vital,
            "nugget_recall_overall": self.nugget_recall_overall,
            "nugget_precision_vital": self.nugget_precision_vital,
            "nugget_precision_overall": self.nugget_precision_overall,
        }


@dataclass(slots=True)
class JudgeScore:
    """LLM judge scores for a single answer."""

    correctness: float | None
    completeness: float | None
    faithfulness: float | None
    overall: float | None
    rationale: str | None = None
    error: str | None = None
    raw_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "correctness": self.correctness,
            "completeness": self.completeness,
            "faithfulness": self.faithfulness,
            "overall": self.overall,
            "rationale": self.rationale,
            "error": self.error,
            "raw_response": self.raw_response,
        }


@dataclass(slots=True)
class ExampleRun:
    """Result of running the agent for a benchmark example."""

    example_id: str
    question: str
    ground_truth: str
    answer: str
    status: str
    iterations: int
    confidence: float | None
    latency_ms: float | None
    strategy: str
    metrics: ExampleScore
    judge: JudgeScore | None
    run_id: str
    trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "example_id": self.example_id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "answer": self.answer,
            "status": self.status,
            "iterations": self.iterations,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "strategy": self.strategy,
            "metrics": self.metrics.to_dict(),
            "judge": self.judge.to_dict() if self.judge else None,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
        }


@dataclass(slots=True)
class AggregateScore:
    """Aggregate metrics for an evaluation run."""

    total: int
    success_rate: float
    status_counts: dict[str, int]
    metrics_mean: dict[str, float | None]
    metrics_median: dict[str, float | None]
    judge_mean: dict[str, float | None]
    avg_iterations: float | None
    avg_latency_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "total": self.total,
            "success_rate": self.success_rate,
            "status_counts": self.status_counts,
            "metrics_mean": self.metrics_mean,
            "metrics_median": self.metrics_median,
            "judge_mean": self.judge_mean,
            "avg_iterations": self.avg_iterations,
            "avg_latency_ms": self.avg_latency_ms,
        }


@dataclass(slots=True)
class EvalConfig:
    """Configuration for evaluation runs."""

    strategy: RetrievalStrategy
    concurrency: int
    judge_enabled: bool = True
    embedding_enabled: bool = True
    nugget_threshold: float = 0.78
    max_examples: int | None = None
