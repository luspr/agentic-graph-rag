from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_graph_rag.eval.metrics import (
    EmbeddingCache,
    cosine_similarity,
    exact_match,
    rouge_l_f1,
    score_answer,
    split_sentences,
    token_f1,
)
from agentic_graph_rag.eval.types import BenchmarkNuggets
from typing import Any

from agentic_graph_rag.llm.base import LLMClient, LLMResponse, ToolDefinition


def test_exact_match_normalized() -> None:
    assert exact_match("The cat", "cat") == 1.0


def test_token_f1() -> None:
    assert token_f1("alpha beta gamma", "alpha beta") == pytest.approx(0.8)


def test_rouge_l_f1() -> None:
    assert rouge_l_f1("alpha beta gamma", "alpha gamma") == pytest.approx(0.8)


def test_cosine_similarity() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_split_sentences() -> None:
    assert split_sentences("A. B? C!") == ["A", "B", "C!"]


def test_token_f1_empty() -> None:
    assert token_f1("", "") == 1.0


def test_rouge_l_f1_empty() -> None:
    assert rouge_l_f1("", "text") == 0.0


@pytest.mark.anyio
async def test_embedding_cache_reuses_embeddings() -> None:
    mock_client = MagicMock(spec=LLMClient)
    mock_client.embed = AsyncMock(return_value=[1.0, 0.0])
    cache = EmbeddingCache(mock_client)

    first = await cache.get("alpha")
    second = await cache.get("alpha")

    assert first == [1.0, 0.0]
    assert second == [1.0, 0.0]
    mock_client.embed.assert_awaited_once()


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
        lowered = text.lower()
        if "alpha" in lowered:
            return [1.0, 0.0]
        if "beta" in lowered:
            return [0.0, 1.0]
        return [1.0, 1.0]


@pytest.mark.anyio
async def test_score_answer_with_embeddings_and_nuggets() -> None:
    cache = EmbeddingCache(_FakeLLMClient())
    nuggets = BenchmarkNuggets(vital=["Alpha sentence"], okay=["Gamma sentence"])
    score = await score_answer(
        answer="Alpha sentence. Beta sentence.",
        ground_truth="Alpha sentence.",
        nuggets=nuggets,
        embedding_cache=cache,
        nugget_threshold=0.95,
    )

    assert score.embedding_cosine is not None
    assert score.nugget_recall_vital == pytest.approx(1.0)
    assert score.nugget_recall_overall == pytest.approx(0.5)
    assert score.nugget_precision_vital == pytest.approx(0.5)


@pytest.mark.anyio
async def test_score_answer_with_empty_answer_nuggets() -> None:
    cache = EmbeddingCache(_FakeLLMClient())
    nuggets = BenchmarkNuggets(vital=["Alpha sentence"])
    score = await score_answer(
        answer="",
        ground_truth="",
        nuggets=nuggets,
        embedding_cache=cache,
        nugget_threshold=0.95,
    )
    assert score.nugget_recall_vital == pytest.approx(0.0)
    assert score.nugget_precision_vital is None
