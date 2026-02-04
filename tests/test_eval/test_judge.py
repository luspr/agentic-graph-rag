from typing import Any

import pytest

from agentic_graph_rag.eval.judge import (
    _extract_json_object,
    _parse_judge_payload,
    judge_answer,
)
from agentic_graph_rag.llm.base import LLMClient, LLMResponse, ToolDefinition


def test_extract_json_object() -> None:
    payload = '{"correctness": 5, "completeness": 4, "faithfulness": 3, "overall": 4}'
    assert _extract_json_object(payload) == {
        "correctness": 5,
        "completeness": 4,
        "faithfulness": 3,
        "overall": 4,
    }


def test_extract_json_from_code_fence() -> None:
    payload = '```json\n{"correctness": 5}\n```'
    assert _extract_json_object(payload) == {"correctness": 5}


def test_parse_judge_payload_missing_scores() -> None:
    result = _parse_judge_payload({"correctness": 5})
    assert result.error is not None


class _FakeClient(LLMClient):
    async def complete(
        self,
        messages: list[dict[str, Any]],  # noqa: ARG002
        tools: list[ToolDefinition] | None = None,  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
        reasoning_effort: str | None = None,  # noqa: ARG002
    ) -> LLMResponse:
        return LLMResponse(
            content=(
                '{"correctness": 5, "completeness": 4, "faithfulness": 4, "overall": 5, '
                '"rationale": "ok"}'
            ),
            tool_calls=[],
            usage={"prompt_tokens": 1, "completion_tokens": 1},
            finish_reason="stop",
        )

    async def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.0]


@pytest.mark.anyio
async def test_judge_answer_success() -> None:
    result = await judge_answer(
        client=_FakeClient(),
        question="Q",
        ground_truth="GT",
        answer="A",
        evidence="E",
    )
    assert result.overall == pytest.approx(5.0)
    assert result.error is None


class _FailingClient(LLMClient):
    async def complete(
        self,
        messages: list[dict[str, Any]],  # noqa: ARG002
        tools: list[ToolDefinition] | None = None,  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
        reasoning_effort: str | None = None,  # noqa: ARG002
    ) -> LLMResponse:
        raise RuntimeError("boom")

    async def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.0]


@pytest.mark.anyio
async def test_judge_answer_error() -> None:
    result = await judge_answer(
        client=_FailingClient(),
        question="Q",
        ground_truth="GT",
        answer="A",
        evidence="E",
    )
    assert result.error == "boom"
