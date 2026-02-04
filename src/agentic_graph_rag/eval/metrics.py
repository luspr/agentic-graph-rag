"""Evaluation metrics for QA-style benchmarks."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Iterable

import anyio

from agentic_graph_rag.eval.types import BenchmarkNuggets, ExampleScore
from agentic_graph_rag.llm.base import LLMClient

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]\s+")


def normalize_text(text: str) -> str:
    """Normalize text for string-based metrics."""
    lowered = text.lower()
    without_articles = _ARTICLE_RE.sub(" ", lowered)
    without_punct = without_articles.translate(
        str.maketrans("", "", string.punctuation)
    )
    collapsed = _WHITESPACE_RE.sub(" ", without_punct).strip()
    return collapsed


def tokenize(text: str) -> list[str]:
    """Tokenize text for overlap-based metrics."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split()


def exact_match(prediction: str, reference: str) -> float:
    """Compute exact match after normalization."""
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 (SQuAD-style)."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 using token LCS."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs_length = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs_length / len(pred_tokens)
    recall = lcs_length / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute LCS length between two token lists."""
    if not a or not b:
        return 0

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, token_a in enumerate(a, start=1):
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_list = list(vec_a)
    b_list = list(vec_b)
    if not a_list or not b_list:
        return 0.0

    length = min(len(a_list), len(b_list))
    dot = sum(a_list[i] * b_list[i] for i in range(length))
    norm_a = math.sqrt(sum(a_list[i] ** 2 for i in range(length)))
    norm_b = math.sqrt(sum(b_list[i] ** 2 for i in range(length)))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for nugget precision estimates."""
    cleaned = text.strip()
    if not cleaned:
        return []
    parts = _SENTENCE_SPLIT_RE.split(cleaned)
    return [part.strip() for part in parts if part.strip()]


class EmbeddingCache:
    """Async embedding cache for repeated texts."""

    def __init__(self, client: LLMClient) -> None:
        """Initialize the cache with an LLM client."""
        self._client = client
        self._cache: dict[str, list[float]] = {}
        self._lock = anyio.Lock()

    async def get(self, text: str) -> list[float]:
        """Fetch or compute the embedding for a text."""
        key = text.strip()
        if not key:
            return []
        async with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        embedding = await self._client.embed(key)
        async with self._lock:
            self._cache[key] = embedding
        return embedding


async def score_answer(
    answer: str,
    ground_truth: str,
    *,
    nuggets: BenchmarkNuggets | None = None,
    embedding_cache: EmbeddingCache | None = None,
    nugget_threshold: float = 0.78,
) -> ExampleScore:
    """Compute evaluation metrics for a single answer."""
    base_score = ExampleScore(
        exact_match=exact_match(answer, ground_truth),
        token_f1=token_f1(answer, ground_truth),
        rouge_l_f1=rouge_l_f1(answer, ground_truth),
    )

    if embedding_cache is not None:
        answer_vec = await embedding_cache.get(answer)
        gt_vec = await embedding_cache.get(ground_truth)
        base_score.embedding_cosine = cosine_similarity(answer_vec, gt_vec)

    if nuggets is not None and embedding_cache is not None:
        nugget_metrics = await _compute_nugget_metrics(
            answer=answer,
            nuggets=nuggets,
            embedding_cache=embedding_cache,
            threshold=nugget_threshold,
        )
        base_score.nugget_recall_vital = nugget_metrics["vital_recall"]
        base_score.nugget_recall_overall = nugget_metrics["overall_recall"]
        base_score.nugget_precision_vital = nugget_metrics["vital_precision"]
        base_score.nugget_precision_overall = nugget_metrics["overall_precision"]

    return base_score


async def _compute_nugget_metrics(
    answer: str,
    nuggets: BenchmarkNuggets,
    embedding_cache: EmbeddingCache,
    threshold: float,
) -> dict[str, float | None]:
    """Compute nugget recall and precision using embedding similarity."""
    answer_sentences = split_sentences(answer)
    sentence_embeddings = [
        await embedding_cache.get(sentence) for sentence in answer_sentences
    ]

    vital = nuggets.vital
    overall = nuggets.all()

    vital_metrics = await _coverage_metrics(
        sentence_embeddings=sentence_embeddings,
        answer_sentences=answer_sentences,
        nuggets=vital,
        embedding_cache=embedding_cache,
        threshold=threshold,
    )
    overall_metrics = await _coverage_metrics(
        sentence_embeddings=sentence_embeddings,
        answer_sentences=answer_sentences,
        nuggets=overall,
        embedding_cache=embedding_cache,
        threshold=threshold,
    )

    return {
        "vital_recall": vital_metrics["recall"],
        "vital_precision": vital_metrics["precision"],
        "overall_recall": overall_metrics["recall"],
        "overall_precision": overall_metrics["precision"],
    }


async def _coverage_metrics(
    *,
    sentence_embeddings: list[list[float]],
    answer_sentences: list[str],
    nuggets: list[str],
    embedding_cache: EmbeddingCache,
    threshold: float,
) -> dict[str, float | None]:
    """Compute coverage recall and precision for a nugget set."""
    if not nuggets:
        return {"recall": None, "precision": None}

    nugget_embeddings = [await embedding_cache.get(nugget) for nugget in nuggets]

    matched_nuggets = 0
    for nugget_embedding in nugget_embeddings:
        if _max_similarity(nugget_embedding, sentence_embeddings) >= threshold:
            matched_nuggets += 1

    recall = matched_nuggets / len(nuggets) if nuggets else None

    if not answer_sentences:
        precision = None
    else:
        matched_sentences = 0
        for sentence_embedding in sentence_embeddings:
            if _max_similarity(sentence_embedding, nugget_embeddings) >= threshold:
                matched_sentences += 1
        precision = matched_sentences / len(answer_sentences)

    return {"recall": recall, "precision": precision}


def _max_similarity(
    vector: list[float],
    candidates: list[list[float]],
) -> float:
    """Return the max cosine similarity between a vector and candidates."""
    if not vector or not candidates:
        return 0.0
    return max(cosine_similarity(vector, candidate) for candidate in candidates)
