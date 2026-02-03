"""Unit tests for the SR-RAG dataset loader script."""

import json
from pathlib import Path

import pytest

from agentic_graph_rag.scripts.load_sr_rag_datasets import (
    _edge_row,
    _load_meta_counts,
    _node_row,
)


def test_node_row_maps_fields() -> None:
    """_node_row maps raw fields into Neo4j properties."""
    raw = {
        "id": "node-1",
        "label": "entity",
        "level": 2,
        "name": "Test Node",
        "chunk_id": "chunk-9",
        "schema_type": "population",
    }
    row = _node_row(raw)

    assert row["id"] == "node-1"
    assert row["label"] == "population"
    assert row["properties"]["node_label"] == "entity"
    assert row["properties"]["level"] == 2
    assert row["properties"]["name"] == "Test Node"
    assert row["properties"]["chunk_id"] == "chunk-9"
    assert row["properties"]["schema_type"] == "population"
    assert row["properties"]["schema_type_label"] == "population"


def test_edge_row_maps_fields() -> None:
    """_edge_row maps raw fields into relation properties."""
    raw = {
        "source": "node-a",
        "target": "node-b",
        "relation": "has_attribute",
        "key": 3,
    }
    row = _edge_row(raw)

    assert row["source"] == "node-a"
    assert row["target"] == "node-b"
    assert row["relation"] == "has_attribute"
    assert row["relation_type"] == "has_attribute"
    assert row["key"] == 3


def test_edge_row_requires_core_fields() -> None:
    """_edge_row raises when required fields are missing."""
    with pytest.raises(ValueError, match="missing source, target, or relation"):
        _edge_row({"source": "node-a"})


def test_node_row_flattens_structured_name() -> None:
    """_node_row flattens structured name into text."""
    raw = {
        "id": "attr-1",
        "label": "attribute",
        "name": {"key": "frequency", "value": "2", "unit": "d/wk", "cohort": "adult"},
    }
    row = _node_row(raw)

    assert row["label"] == "attribute"
    assert row["properties"]["name"] == "frequency 2 d/wk adult"


def test_node_row_sanitizes_schema_type_label() -> None:
    """_node_row sanitizes schema_type into a valid label."""
    raw = {"id": "node-2", "label": "entity", "schema_type": "risk factor"}
    row = _node_row(raw)

    assert row["label"] == "risk_factor"
    assert row["properties"]["schema_type_label"] == "risk_factor"


def test_load_meta_counts(tmp_path: Path) -> None:
    """_load_meta_counts returns counts from meta.json."""
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(
        json.dumps({"node_count": 12, "edge_count": "34"}), encoding="utf-8"
    )

    node_count, edge_count = _load_meta_counts(meta_path)

    assert node_count == 12
    assert edge_count == 34
