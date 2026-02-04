from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_graph_rag.agent.state import AgentResult, AgentStatus
from typing import Any, cast

from agentic_graph_rag.eval.metrics import EmbeddingCache
from agentic_graph_rag.eval.runner import (
    JSONLWriter,
    RunEventLogger,
    _summarize,
    run_evaluation,
)
from agentic_graph_rag.eval.types import (
    AggregateScore,
    BenchmarkExample,
    BenchmarkNuggets,
    EvalConfig,
    ExampleRun,
    ExampleScore,
    JudgeScore,
)
from agentic_graph_rag.llm.base import LLMClient, LLMResponse, ToolDefinition
from agentic_graph_rag.retriever.base import RetrievalStep, RetrievalStrategy
from agentic_graph_rag.runner import HeadlessRunner


class _FakeLLMClient(LLMClient):
    async def complete(
        self,
        messages: list[dict[str, Any]],  # noqa: ARG002
        tools: list[ToolDefinition] | None = None,  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
        reasoning_effort: str | None = None,  # noqa: ARG002
    ) -> LLMResponse:
        return LLMResponse(
            content=None,
            tool_calls=[],
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            finish_reason="stop",
        )

    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0] if text else [0.0, 1.0]


class _FakeRunner:
    async def run_query(self, user_query: str, **_: object) -> AgentResult:
        step = RetrievalStep(
            action="submit_answer",
            input={},
            output={"supporting_evidence": "Evidence"},
        )
        return AgentResult(
            answer=f"Answer for {user_query}",
            status=AgentStatus.COMPLETED,
            iterations=1,
            history=[step],
            confidence=0.9,
        )


def test_jsonl_writer_and_logger(tmp_path: Path) -> None:
    log_path = tmp_path / "trace.jsonl"
    writer = JSONLWriter(log_path)
    logger = RunEventLogger(writer, trace_id="trace", run_id="run")
    logger.log_event("event", {"ok": True})
    writer.close()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event_type"] == "event"


def test_summarize_results() -> None:
    metrics = ExampleScore(
        exact_match=1.0,
        token_f1=0.5,
        rouge_l_f1=0.5,
        embedding_cosine=0.9,
    )
    judge = JudgeScore(
        correctness=4.0,
        completeness=3.0,
        faithfulness=5.0,
        overall=4.0,
    )
    runs = [
        ExampleRun(
            example_id="1",
            question="Q",
            ground_truth="GT",
            answer="A",
            status=AgentStatus.COMPLETED.value,
            iterations=1,
            confidence=0.8,
            latency_ms=10.0,
            strategy=RetrievalStrategy.CYPHER.value,
            metrics=metrics,
            judge=judge,
            run_id="run-1",
        ),
        ExampleRun(
            example_id="2",
            question="Q2",
            ground_truth="GT2",
            answer="A2",
            status=AgentStatus.ERROR.value,
            iterations=2,
            confidence=None,
            latency_ms=20.0,
            strategy=RetrievalStrategy.CYPHER.value,
            metrics=metrics,
            judge=judge,
            run_id="run-2",
        ),
    ]

    summary = _summarize(runs)
    assert isinstance(summary, AggregateScore)
    assert summary.total == 2
    assert summary.status_counts[AgentStatus.COMPLETED.value] == 1
    assert summary.metrics_mean["exact_match"] == pytest.approx(1.0)


@pytest.mark.anyio
async def test_run_evaluation_basic(tmp_path: Path) -> None:
    examples = [
        BenchmarkExample(example_id="1", question="Q1", ground_truth="GT1"),
        BenchmarkExample(example_id="2", question="Q2", ground_truth="GT2"),
    ]
    runner = _FakeRunner()
    cache = EmbeddingCache(_FakeLLMClient())

    result_writer = JSONLWriter(tmp_path / "results.jsonl")
    trace_writer = JSONLWriter(tmp_path / "trace.jsonl")
    config = EvalConfig(
        strategy=RetrievalStrategy.CYPHER,
        concurrency=1,
        judge_enabled=False,
        embedding_enabled=True,
        nugget_threshold=0.9,
    )

    results, summary = await run_evaluation(
        examples=examples,
        runner=cast(HeadlessRunner, runner),
        config=config,
        result_writer=result_writer,
        trace_writer=trace_writer,
        embedding_cache=cache,
    )

    result_writer.close()
    trace_writer.close()

    assert len(results) == 2
    assert summary.total == 2


@pytest.mark.anyio
async def test_run_evaluation_with_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    examples = [
        BenchmarkExample(
            example_id="1",
            question="Q1",
            ground_truth="GT1",
            nuggets=BenchmarkNuggets(vital=["V1"]),
        )
    ]
    runner = _FakeRunner()
    cache = EmbeddingCache(_FakeLLMClient())

    async def _fake_judge(**_: object) -> JudgeScore:
        return JudgeScore(
            correctness=5.0,
            completeness=4.0,
            faithfulness=4.5,
            overall=4.8,
        )

    monkeypatch.setattr("agentic_graph_rag.eval.runner.judge_answer", _fake_judge)

    config = EvalConfig(
        strategy=RetrievalStrategy.CYPHER,
        concurrency=1,
        judge_enabled=True,
        embedding_enabled=True,
    )

    results, summary = await run_evaluation(
        examples=examples,
        runner=cast(HeadlessRunner, runner),
        config=config,
        embedding_cache=cache,
        judge_client=_FakeLLMClient(),
    )

    assert results[0].judge is not None
    assert summary.judge_mean["overall"] == pytest.approx(4.8)
