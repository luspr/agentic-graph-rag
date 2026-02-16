from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agentic_graph_rag.agent.state import AgentResult, AgentStatus
from agentic_graph_rag.retriever.base import RetrievalStrategy
from agentic_graph_rag.runner import HeadlessRunner, _select_tools


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


@dataclass
class _DummyController:
    tools: list[Any]

    async def run(self, *_: Any, **__: Any) -> AgentResult:
        return AgentResult(
            answer="ok",
            status=AgentStatus.COMPLETED,
            iterations=1,
            confidence=0.8,
        )


def test_select_tools_for_cypher() -> None:
    tools = _select_tools(RetrievalStrategy.CYPHER)
    tool_names = {tool.name for tool in tools}
    assert tool_names == {"execute_cypher", "submit_answer"}


@pytest.mark.anyio
async def test_headless_runner_run_query(monkeypatch: pytest.MonkeyPatch) -> None:
    from agentic_graph_rag.config import Settings

    dummy_settings = Settings(
        openai_api_key="key",
        openai_model="model",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="pass",
    )

    dummy_controller = _DummyController(tools=[])

    def _controller_factory(*_: Any, **kwargs: Any) -> _DummyController:
        dummy_controller.tools = list(kwargs.get("tools", []))
        return dummy_controller

    monkeypatch.setattr("agentic_graph_rag.runner.OpenAILLMClient", _DummyLLM)
    monkeypatch.setattr("agentic_graph_rag.runner.Neo4jClient", _DummyGraph)
    monkeypatch.setattr("agentic_graph_rag.runner.AgentController", _controller_factory)

    async with HeadlessRunner(
        settings=dummy_settings, strategy=RetrievalStrategy.CYPHER
    ) as runner:
        result = await runner.run_query("Question")

    assert result.status == AgentStatus.COMPLETED
    assert {tool.name for tool in dummy_controller.tools} == {
        "execute_cypher",
        "submit_answer",
    }


@pytest.mark.anyio
async def test_headless_runner_hybrid_initializes_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentic_graph_rag.config import Settings

    dummy_settings = Settings(
        openai_api_key="key",
        openai_model="model",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="pass",
    )

    created: dict[str, Any] = {}

    class _DummyVector:
        def __init__(self, *_, **kwargs: Any) -> None:
            created["vector_kwargs"] = kwargs

    class _DummyHybrid:
        def __init__(self, *_, **__: Any) -> None:
            created["hybrid"] = True

    monkeypatch.setattr("agentic_graph_rag.runner.OpenAILLMClient", _DummyLLM)
    monkeypatch.setattr("agentic_graph_rag.runner.Neo4jClient", _DummyGraph)
    monkeypatch.setattr("agentic_graph_rag.runner.QdrantVectorStore", _DummyVector)
    monkeypatch.setattr("agentic_graph_rag.runner.HybridRetriever", _DummyHybrid)

    async with HeadlessRunner(
        settings=dummy_settings, strategy=RetrievalStrategy.HYBRID
    ):
        pass

    assert created["vector_kwargs"]["vector_size"] == dummy_settings.embedding_dim
    assert created.get("hybrid") is True


@pytest.mark.anyio
async def test_headless_runner_passes_embedding_model_to_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentic_graph_rag.config import Settings

    dummy_settings = Settings(
        openai_api_key="key",
        openai_model="model",
        openai_embedding_model="text-embedding-3-large",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="pass",
    )

    captured: dict[str, Any] = {}

    class _CapturingLLM:
        def __init__(self, *_: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("agentic_graph_rag.runner.OpenAILLMClient", _CapturingLLM)
    monkeypatch.setattr("agentic_graph_rag.runner.Neo4jClient", _DummyGraph)

    async with HeadlessRunner(
        settings=dummy_settings, strategy=RetrievalStrategy.CYPHER
    ):
        pass

    assert captured["embedding_model"] == "text-embedding-3-large"
    assert captured["embedding_dimensions"] == dummy_settings.embedding_dim
