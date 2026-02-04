import json
from pathlib import Path

from agentic_graph_rag.eval.loaders import load_generic_jsonl, load_sr_rag_jsonl


def test_load_sr_rag_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "sr.jsonl"
    record = {
        "question_en": "What is Graph RAG?",
        "ground_truth": "Graph RAG uses a graph database for retrieval.",
        "nuggets": {"vital": ["graph database"], "okay": [], "trivial": []},
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    examples = load_sr_rag_jsonl(path)
    assert len(examples) == 1
    example = examples[0]
    assert example.question == record["question_en"]
    assert example.ground_truth == record["ground_truth"]
    assert example.nuggets is not None
    assert example.nuggets.vital == ["graph database"]


def test_load_generic_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "generic.jsonl"
    record = {"question": "Who?", "ground_truth": "Someone"}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    examples = load_generic_jsonl(path)
    assert len(examples) == 1
    example = examples[0]
    assert example.question == "Who?"
    assert example.ground_truth == "Someone"
