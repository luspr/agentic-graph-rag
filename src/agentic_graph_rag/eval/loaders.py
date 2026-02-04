"""Benchmark dataset loaders for evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic_graph_rag.eval.types import BenchmarkExample, BenchmarkNuggets


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of records."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_sr_rag_jsonl(path: Path) -> list[BenchmarkExample]:
    """Load SR-RAG benchmark examples from JSONL."""
    records = _load_jsonl(path)
    examples: list[BenchmarkExample] = []

    for index, record in enumerate(records):
        question = record.get("question_en")
        ground_truth = record.get("ground_truth")
        if question is None or ground_truth is None:
            raise ValueError(
                "SR-RAG records must include question_en and ground_truth."
            )

        nuggets_data = record.get("nuggets") or {}
        nuggets = BenchmarkNuggets(
            vital=list(nuggets_data.get("vital", [])),
            okay=list(nuggets_data.get("okay", [])),
            trivial=list(nuggets_data.get("trivial", [])),
        )
        example_id = str(
            record.get("id") or record.get("question_id") or record.get("qid") or index
        )
        meta = dict(record)
        for key in ("question_en", "ground_truth", "nuggets"):
            meta.pop(key, None)

        examples.append(
            BenchmarkExample(
                example_id=example_id,
                question=str(question),
                ground_truth=str(ground_truth),
                nuggets=nuggets,
                meta=meta,
            )
        )

    return examples


def load_generic_jsonl(
    path: Path,
    question_field: str = "question",
    ground_truth_field: str = "ground_truth",
) -> list[BenchmarkExample]:
    """Load generic JSONL benchmarks using configurable field names."""
    records = _load_jsonl(path)
    examples: list[BenchmarkExample] = []

    for index, record in enumerate(records):
        if question_field not in record or ground_truth_field not in record:
            raise ValueError(
                f"Missing required fields: {question_field}, {ground_truth_field}"
            )
        question = record[question_field]
        ground_truth = record[ground_truth_field]
        example_id = str(
            record.get("id") or record.get("question_id") or record.get("qid") or index
        )
        meta = dict(record)
        meta.pop(question_field, None)
        meta.pop(ground_truth_field, None)

        examples.append(
            BenchmarkExample(
                example_id=example_id,
                question=str(question),
                ground_truth=str(ground_truth),
                meta=meta,
            )
        )

    return examples
