"""Headless runner for executing agent queries without the terminal UI."""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Any

from agentic_graph_rag.agent import (
    AgentConfig,
    AgentController,
    AgentResult,
    AGENT_TOOLS,
)
from agentic_graph_rag.agent.controller import Tracer
from agentic_graph_rag.agent.tools import ToolRouter
from agentic_graph_rag.config import Settings
from agentic_graph_rag.graph.neo4j_client import Neo4jClient
from agentic_graph_rag.llm.base import LLMClient, ToolDefinition
from agentic_graph_rag.llm.openai_client import OpenAILLMClient
from agentic_graph_rag.prompts.manager import PromptManager
from agentic_graph_rag.retriever.base import RetrievalStrategy
from agentic_graph_rag.retriever.cypher_retriever import CypherRetriever
from agentic_graph_rag.retriever.hybrid_retriever import HybridRetriever
from agentic_graph_rag.vector.qdrant_client import QdrantVectorStore


def _select_tools(strategy: RetrievalStrategy) -> list[ToolDefinition]:
    """Select the tool set to expose for a given retrieval strategy."""
    if strategy == RetrievalStrategy.CYPHER:
        allowed = {"execute_cypher", "submit_answer"}
        return [tool for tool in AGENT_TOOLS if tool.name in allowed]
    return list(AGENT_TOOLS)


@dataclass(slots=True)
class RunnerComponents:
    """Container for shared components used by the headless runner."""

    llm_client: LLMClient
    graph_db: Neo4jClient
    tool_router: ToolRouter
    prompt_manager: PromptManager
    config: AgentConfig
    tools: list[ToolDefinition]


class HeadlessRunner:
    """Headless runner for batch or programmatic agent execution."""

    def __init__(
        self,
        settings: Settings,
        strategy: RetrievalStrategy = RetrievalStrategy.CYPHER,
    ) -> None:
        """Initialize the runner.

        Args:
            settings: Application settings.
            strategy: Retrieval strategy to expose (cypher or hybrid).
        """
        self._settings = settings
        self._strategy = strategy
        self._components: RunnerComponents | None = None
        self._graph_db: Neo4jClient | None = None
        self._llm_client: OpenAILLMClient | None = None

    async def __aenter__(self) -> "HeadlessRunner":
        self._llm_client = OpenAILLMClient(
            api_key=self._settings.openai_api_key,
            model=self._settings.openai_model,
            embedding_model=self._settings.openai_embedding_model,
            embedding_dimensions=self._settings.embedding_dim,
        )
        self._graph_db = Neo4jClient(self._settings)
        await self._graph_db.__aenter__()

        cypher_retriever = CypherRetriever(self._graph_db)
        hybrid_retriever = None
        if self._strategy == RetrievalStrategy.HYBRID:
            vector_store = QdrantVectorStore(
                settings=self._settings,
                collection_name=self._settings.qdrant_collection,
                vector_size=self._settings.embedding_dim,
                vector_name=self._settings.qdrant_vector_name,
            )
            hybrid_retriever = HybridRetriever(
                graph_db=self._graph_db,
                vector_store=vector_store,
                llm_client=self._llm_client,
                uuid_property=self._settings.node_uuid_property,
            )

        tool_router = ToolRouter(
            cypher_retriever=cypher_retriever,
            hybrid_retriever=hybrid_retriever,
        )
        prompt_manager = PromptManager()
        config = AgentConfig(
            max_iterations=self._settings.max_iterations,
            max_history_messages=self._settings.max_history_messages,
            strategy=self._strategy,
        )
        self._components = RunnerComponents(
            llm_client=self._llm_client,
            graph_db=self._graph_db,
            tool_router=tool_router,
            prompt_manager=prompt_manager,
            config=config,
            tools=_select_tools(self._strategy),
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._graph_db is not None:
            await self._graph_db.__aexit__(exc_type, exc, exc_tb)
        if self._llm_client is not None:
            await self._llm_client.aclose()

    @property
    def llm_client(self) -> LLMClient:
        """Return the shared LLM client."""
        if self._components is None:
            raise RuntimeError(
                "HeadlessRunner must be used as an async context manager."
            )
        return self._components.llm_client

    async def run_query(
        self,
        user_query: str,
        history_messages: list[dict[str, Any]] | None = None,
        tracer: Tracer | None = None,
    ) -> AgentResult:
        """Run a single query through the agent.

        Args:
            user_query: The question to answer.
            history_messages: Optional prior conversation messages.
            tracer: Optional tracer for execution events.

        Returns:
            The agent result for the query.
        """
        if self._components is None:
            raise RuntimeError(
                "HeadlessRunner must be used as an async context manager."
            )

        controller = AgentController(
            llm_client=self._components.llm_client,
            graph_db=self._components.graph_db,
            tool_router=self._components.tool_router,
            prompt_manager=self._components.prompt_manager,
            config=self._components.config,
            tracer=tracer,
            tools=self._components.tools,
        )
        return await controller.run(user_query, history_messages=history_messages)
