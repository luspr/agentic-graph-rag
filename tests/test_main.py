from __future__ import annotations

from typing import Any

import anyio
import pytest

from agentic_graph_rag.graph.base import GraphSchema, NodeType, RelationshipType


class _DummySettings:
    openai_api_key = "key"
    openai_model = "model"
    openai_embedding_model = "text-embedding-3-small"
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "pass"
    qdrant_collection = "graph_nodes"
    embedding_dim = 1536
    qdrant_vector_name = "default"
    node_uuid_property = "uuid"
    max_iterations = 2
    max_history_messages = 2
    trace_log_dir = "logs"
    trace_logging_enabled = False


class _DummyLLM:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _DummyGraph:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.closed = False

    async def __aenter__(self) -> "_DummyGraph":
        return self

    async def __aexit__(self, *_: Any) -> None:
        self.closed = True

    async def get_schema(self) -> GraphSchema:
        return GraphSchema(
            node_types=[
                NodeType(label="Movie", properties={"title": "STRING"}, count=1)
            ],
            relationship_types=[
                RelationshipType(
                    type="ACTED_IN",
                    start_label="Person",
                    end_label="Movie",
                    properties={},
                )
            ],
        )


class _DummyUI:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.ran = False

    async def run(self) -> None:
        self.ran = True


def test_main_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentic_graph_rag.main.Settings", lambda: _DummySettings())
    monkeypatch.setattr("agentic_graph_rag.main.OpenAILLMClient", _DummyLLM)
    monkeypatch.setattr("agentic_graph_rag.main.Neo4jClient", _DummyGraph)
    monkeypatch.setattr(
        "agentic_graph_rag.main.QdrantVectorStore", lambda **_: object()
    )
    monkeypatch.setattr("agentic_graph_rag.main.TerminalUI", _DummyUI)

    from agentic_graph_rag.main import main

    exit_code = anyio.run(main)
    assert exit_code == 0
