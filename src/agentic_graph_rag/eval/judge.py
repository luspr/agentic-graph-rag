"""LLM judge utilities for QA evaluation."""

from __future__ import annotations

import json
from typing import Any

from agentic_graph_rag.eval.types import JudgeScore
from agentic_graph_rag.llm.base import LLMClient

_SYSTEM_PROMPT = (
    "You are a strict QA evaluator. Score the answer against the ground truth and "
    "the provided evidence context. Return ONLY a JSON object with the keys: "
    "correctness, completeness, faithfulness, overall, rationale. Use a 0-5 scale "
    "for the scores where 5 is best. If evidence is missing or does not support the "
    "answer, reduce faithfulness accordingly."
)


async def judge_answer(
    *,
    client: LLMClient,
    question: str,
    ground_truth: str,
    answer: str,
    evidence: str,
) -> JudgeScore:
    """Score an answer using an LLM judge."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Ground Truth:\n{ground_truth}\n\n"
                f"Answer:\n{answer}\n\n"
                f"Evidence:\n{evidence or 'No evidence provided.'}\n\n"
                "Respond with JSON only."
            ),
        },
    ]

    try:
        response = await client.complete(messages)
    except Exception as exc:  # noqa: BLE001
        return JudgeScore(
            correctness=None,
            completeness=None,
            faithfulness=None,
            overall=None,
            error=str(exc),
        )

    raw_text = response.content or ""
    payload = _extract_json_object(raw_text)
    if payload is None:
        return JudgeScore(
            correctness=None,
            completeness=None,
            faithfulness=None,
            overall=None,
            error="Failed to parse judge response as JSON.",
            raw_response=raw_text,
        )

    parsed = _parse_judge_payload(payload)
    if parsed.error:
        parsed.raw_response = raw_text
    return parsed


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from a response string."""
    cleaned = _strip_code_fences(text.strip())
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = cleaned[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _strip_code_fences(text: str) -> str:
    """Remove surrounding Markdown code fences if present."""
    if "```" not in text:
        return text
    parts = text.split("```")
    if len(parts) >= 3:
        return parts[1].strip()
    return text


def _parse_judge_payload(payload: dict[str, Any]) -> JudgeScore:
    """Validate and coerce the judge payload to a JudgeScore."""
    correctness = _coerce_score(payload.get("correctness"))
    completeness = _coerce_score(payload.get("completeness"))
    faithfulness = _coerce_score(payload.get("faithfulness"))
    overall = _coerce_score(payload.get("overall"))
    rationale = payload.get("rationale")
    if rationale is not None:
        rationale = str(rationale)

    error = None
    if any(
        value is None for value in (correctness, completeness, faithfulness, overall)
    ):
        error = "Missing or invalid judge scores."

    return JudgeScore(
        correctness=correctness,
        completeness=completeness,
        faithfulness=faithfulness,
        overall=overall,
        rationale=rationale,
        error=error,
    )


def _coerce_score(value: Any) -> float | None:
    """Coerce a numeric judge score to float."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None
