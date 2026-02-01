"""Agent controller for running the iterative retrieval loop."""

from typing import Any, Protocol

from agentic_graph_rag.agent.state import (
    AgentConfig,
    AgentResult,
    AgentState,
    AgentStatus,
)
from agentic_graph_rag.agent.tools import AGENT_TOOLS, ToolRouter
from agentic_graph_rag.graph.base import GraphDatabase
from agentic_graph_rag.llm.base import LLMClient, ToolCall
from agentic_graph_rag.prompts.manager import PromptContext, PromptManager
from agentic_graph_rag.retriever.base import RetrievalStep


class Tracer(Protocol):
    """Protocol for tracing agent execution events."""

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log an event during agent execution."""
        ...


class AgentController:
    """Controls the agentic retrieval loop.

    The controller runs an iterative loop where it:
    1. Builds prompts with the current state and history
    2. Sends requests to the LLM with available tools
    3. Executes tool calls and records results
    4. Continues until an answer is submitted or max iterations reached
    """

    def __init__(
        self,
        llm_client: LLMClient,
        graph_db: GraphDatabase,
        tool_router: ToolRouter,
        prompt_manager: PromptManager,
        config: AgentConfig | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize the AgentController.

        Args:
            llm_client: Client for LLM completions.
            graph_db: Graph database for schema retrieval.
            tool_router: Router for dispatching tool calls.
            prompt_manager: Manager for building prompts.
            config: Agent configuration. Defaults to AgentConfig().
            tracer: Optional tracer for logging events.
        """
        self._llm_client = llm_client
        self._graph_db = graph_db
        self._tool_router = tool_router
        self._prompt_manager = prompt_manager
        self._config = config or AgentConfig()
        self._tracer = tracer

        self._state: AgentState | None = None
        self._user_query: str = ""
        self._messages: list[dict[str, Any]] = []

    async def run(self, user_query: str) -> AgentResult:
        """Run the agent until completion or max iterations.

        Args:
            user_query: The user's question to answer.

        Returns:
            AgentResult containing the answer and execution metadata.
        """
        self._user_query = user_query
        self._state = AgentState(
            iteration=0,
            status=AgentStatus.RUNNING,
            history=[],
        )

        self._log_event("query_start", {"query": user_query})

        # Get schema and build system prompt
        schema = await self._graph_db.get_schema()
        system_prompt = self._prompt_manager.build_system_prompt(schema)

        self._messages = [{"role": "system", "content": system_prompt}]

        # Build initial context for the LLM
        context = PromptContext(
            user_query=user_query,
            schema=schema,
            history=[],
        )
        initial_prompt = self._prompt_manager.build_retrieval_prompt(context)
        self._messages.append({"role": "user", "content": initial_prompt})

        while not self.should_stop(self._state):
            self._state = await self.step()

        return self._build_result()

    async def step(self) -> AgentState:
        """Execute a single iteration step.

        Returns:
            Updated AgentState after the step.
        """
        if self._state is None:
            raise RuntimeError("Cannot call step() before run()")

        self._state.iteration += 1
        self._log_event("iteration_start", {"iteration": self._state.iteration})

        # Send to LLM
        self._log_event("llm_request", {"messages_count": len(self._messages)})

        try:
            response = await self._llm_client.complete(
                messages=self._messages,
                tools=AGENT_TOOLS,
            )
        except Exception as e:
            self._log_event("error", {"error": str(e), "type": "llm_error"})
            self._state.status = AgentStatus.ERROR
            self._state.current_answer = f"Error communicating with LLM: {e}"
            return self._state

        self._log_event(
            "llm_response",
            {
                "content": response.content,
                "tool_calls_count": len(response.tool_calls),
                "finish_reason": response.finish_reason,
            },
        )

        # Add assistant message to conversation
        if response.content:
            self._messages.append({"role": "assistant", "content": response.content})

        # Handle no tool calls (shouldn't happen with proper prompting)
        if not response.tool_calls:
            self._log_event(
                "error", {"error": "No tool calls in response", "type": "no_tools"}
            )
            self._state.status = AgentStatus.ERROR
            self._state.current_answer = response.content or "No action taken by agent."
            return self._state

        # Process each tool call
        for tool_call in response.tool_calls:
            await self._process_tool_call(tool_call)

            # Check if we got an answer
            if self._state.status == AgentStatus.COMPLETED:
                break

        return self._state

    async def _process_tool_call(self, tool_call: ToolCall) -> None:
        """Process a single tool call and update state."""
        self._log_event(
            "tool_call",
            {
                "tool_id": tool_call.id,
                "tool_name": tool_call.name,
                "arguments": tool_call.arguments,
            },
        )

        # Execute the tool
        result = await self._tool_router.route(tool_call)

        self._log_event(
            "tool_result",
            {
                "tool_id": tool_call.id,
                "tool_name": tool_call.name,
                "success": result.get("success", False),
            },
        )

        # Record step in history
        step = RetrievalStep(
            action=tool_call.name,
            input=tool_call.arguments,
            output=result,
            error=result.get("error"),
        )
        if self._state is not None:
            self._state.history.append(step)

        # Handle submit_answer
        if tool_call.name == "submit_answer" and result.get("success"):
            if self._state is not None:
                self._state.status = AgentStatus.COMPLETED
                self._state.current_answer = result.get("answer", "")
                self._state.confidence = result.get("confidence")
            self._log_event("complete", {"answer": result.get("answer")})
            return

        # Add tool result to conversation for next iteration
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": self._prompt_manager.format_results(
                result.get("data", []) if isinstance(result.get("data"), list) else []
            ),
        }
        self._messages.append(tool_message)

    def should_stop(self, state: AgentState) -> bool:
        """Determine if the agent should stop iterating.

        Args:
            state: Current agent state.

        Returns:
            True if agent should stop, False otherwise.
        """
        # Stop if completed (answer submitted)
        if state.status == AgentStatus.COMPLETED:
            return True

        # Stop if error occurred
        if state.status == AgentStatus.ERROR:
            return True

        # Stop if max iterations reached
        if state.iteration >= self._config.max_iterations:
            state.status = AgentStatus.MAX_ITERATIONS
            self._log_event(
                "max_iterations",
                {"iterations": state.iteration, "max": self._config.max_iterations},
            )
            return True

        return False

    def _build_result(self) -> AgentResult:
        """Build the final AgentResult from current state."""
        if self._state is None:
            return AgentResult(
                answer="Agent was not initialized",
                status=AgentStatus.ERROR,
                iterations=0,
            )

        answer = self._state.current_answer or ""
        if self._state.status == AgentStatus.MAX_ITERATIONS and not answer:
            answer = "Maximum iterations reached without finding a complete answer."

        return AgentResult(
            answer=answer,
            status=self._state.status,
            iterations=self._state.iteration,
            history=self._state.history,
            confidence=self._state.confidence,
        )

    def _log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log an event via the tracer if available."""
        if self._tracer is not None:
            self._tracer.log_event(event_type, data)
