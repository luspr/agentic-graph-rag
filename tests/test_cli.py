from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from agentic_graph_rag.agent.state import AgentResult, AgentStatus
from agentic_graph_rag.cli import _run_eval, _run_single, main
from agentic_graph_rag.config import Settings
from agentic_graph_rag.eval.types import (
    AggregateScore,
    ExampleRun,
    ExampleScore,
    JudgeScore,
)
from agentic_graph_rag.retriever.base import RetrievalStrategy


class _DummyLLM:
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0] if text else [0.0, 1.0]


class _DummyRunner:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.llm_client = _DummyLLM()

    async def __aenter__(self) -> "_DummyRunner":
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def run_query(self, *_: Any, **__: Any) -> AgentResult:
        return AgentResult(
            answer="Answer",
            status=AgentStatus.COMPLETED,
            iterations=1,
            confidence=0.9,
        )


def _dummy_settings() -> Settings:
    return Settings(
        openai_api_key="key",
        openai_model="model",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="pass",
    )


@pytest.mark.anyio
async def test_run_single_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr("agentic_graph_rag.cli.HeadlessRunner", _DummyRunner)
    monkeypatch.setattr("agentic_graph_rag.cli._load_settings", _dummy_settings)

    args = argparse.Namespace(
        query="Hello",
        strategy="cypher",
        json=True,
        trace_log=None,
    )
    result = await _run_single(args)
    assert result == 0

    out = capsys.readouterr().out
    payload = json.loads(out.strip())
    assert payload["answer"] == "Answer"


@pytest.mark.anyio
async def test_run_eval_writes_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("agentic_graph_rag.cli.HeadlessRunner", _DummyRunner)
    monkeypatch.setattr("agentic_graph_rag.cli._load_settings", _dummy_settings)

    example_metrics = ExampleScore(
        exact_match=1.0,
        token_f1=1.0,
        rouge_l_f1=1.0,
    )
    example_judge = JudgeScore(
        correctness=5.0,
        completeness=5.0,
        faithfulness=5.0,
        overall=5.0,
    )
    example_run = ExampleRun(
        example_id="1",
        question="Q",
        ground_truth="GT",
        answer="A",
        status=AgentStatus.COMPLETED.value,
        iterations=1,
        confidence=0.9,
        latency_ms=1.0,
        strategy=RetrievalStrategy.CYPHER.value,
        metrics=example_metrics,
        judge=example_judge,
        run_id="run-1",
    )
    summary = AggregateScore(
        total=1,
        success_rate=1.0,
        status_counts={AgentStatus.COMPLETED.value: 1},
        metrics_mean={"exact_match": 1.0},
        metrics_median={"exact_match": 1.0},
        judge_mean={"overall": 5.0},
        avg_iterations=1.0,
        avg_latency_ms=1.0,
    )

    async def _fake_run_eval(**_: Any):
        return [example_run], summary

    monkeypatch.setattr("agentic_graph_rag.cli.run_evaluation", _fake_run_eval)

    args = argparse.Namespace(
        input=tmp_path / "bench.jsonl",
        format="sr_rag",
        question_field="question",
        ground_truth_field="ground_truth",
        strategy="cypher",
        concurrency=1,
        max_examples=None,
        output_dir=tmp_path / "out",
        trace_log=None,
        no_judge=False,
        no_embedding_eval=False,
        nugget_threshold=0.78,
    )
    args.input.write_text("", encoding="utf-8")

    result = await _run_eval(args)
    assert result == 0

    summary_path = args.output_dir / "summary.json"
    assert summary_path.exists()


def test_main_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(func, args):
        assert func.__name__ in {"_run_single", "_run_eval"}
        return 0

    monkeypatch.setattr("agentic_graph_rag.cli.anyio.run", _fake_run)
    monkeypatch.setattr(sys, "argv", ["prog", "run", "--query", "Hi"])
    assert main() == 0
