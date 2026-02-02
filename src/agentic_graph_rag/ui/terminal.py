"""Terminal UI using prompt_toolkit for input and rich for output."""

import json
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from agentic_graph_rag.agent import (
    AgentConfig,
    AgentController,
    AgentResult,
    AgentStatus,
    Session,
    SessionManager,
    Trace,
    Tracer,
)
from agentic_graph_rag.agent.tools import ToolRouter
from agentic_graph_rag.graph.base import GraphDatabase
from agentic_graph_rag.llm.base import LLMClient
from agentic_graph_rag.prompts.manager import PromptManager


class UITracer(Tracer):
    """Extended tracer that also updates the UI during agent execution."""

    def __init__(self, console: Console) -> None:
        """Initialize the UI tracer.

        Args:
            console: Rich console for output.
        """
        super().__init__()
        self._console = console
        self._current_iteration = 0
        self._max_iterations = 10
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._task_id: Any = None

    def set_max_iterations(self, max_iterations: int) -> None:
        """Set the maximum iterations for progress display."""
        self._max_iterations = max_iterations

    def set_live_context(self, live: Live, progress: Progress, task_id: Any) -> None:
        """Set the live context for updating progress display."""
        self._live = live
        self._progress = progress
        self._task_id = task_id

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log an event and update the UI progress."""
        super().log_event(event_type, data)
        self._update_progress(event_type, data)

    def _update_progress(self, event_type: str, data: dict[str, Any]) -> None:
        """Update the progress display based on event type."""
        if self._progress is None or self._task_id is None:
            return

        if event_type == "iteration_start":
            self._current_iteration = data.get("iteration", 0)
            desc = f"Iteration {self._current_iteration}/{self._max_iterations}: Thinking..."
            self._progress.update(self._task_id, description=desc)

        elif event_type == "tool_call":
            tool_name = data.get("tool_name", "unknown")
            desc = (
                f"Iteration {self._current_iteration}/{self._max_iterations}: "
                f"Calling {tool_name}..."
            )
            self._progress.update(self._task_id, description=desc)

        elif event_type == "llm_request":
            desc = (
                f"Iteration {self._current_iteration}/{self._max_iterations}: "
                f"Querying LLM..."
            )
            self._progress.update(self._task_id, description=desc)


class TerminalUI:
    """Interactive terminal UI for Agentic Graph RAG."""

    COMMANDS = {
        "/quit": "Exit the application",
        "/clear": "Clear the current session and start fresh",
        "/trace": "Show details of the last query trace",
        "/help": "Show available commands",
    }

    def __init__(
        self,
        llm_client: LLMClient,
        graph_db: GraphDatabase,
        tool_router: ToolRouter,
        prompt_manager: PromptManager,
        config: AgentConfig | None = None,
    ) -> None:
        """Initialize the terminal UI.

        Args:
            llm_client: Client for LLM completions.
            graph_db: Graph database for queries.
            tool_router: Router for dispatching tool calls.
            prompt_manager: Manager for building prompts.
            config: Agent configuration.
        """
        self._llm_client = llm_client
        self._graph_db = graph_db
        self._tool_router = tool_router
        self._prompt_manager = prompt_manager
        self._config = config or AgentConfig()

        self._console = Console()
        self._session_manager = SessionManager()
        self._session: Session | None = None
        self._tracer = UITracer(self._console)
        self._tracer.set_max_iterations(self._config.max_iterations)
        self._last_trace: Trace | None = None

        self._prompt_session: PromptSession[str] = PromptSession(
            history=InMemoryHistory()
        )

    async def run(self) -> None:
        """Run the terminal UI main loop."""
        self._print_welcome()
        self._session = self._session_manager.create_session()

        while True:
            try:
                user_input = await self._get_input()

                if user_input is None:
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    should_continue = await self._handle_command(user_input)
                    if not should_continue:
                        break
                    continue

                await self._handle_query(user_input)

            except KeyboardInterrupt:
                self._console.print("\n[dim]Use /quit to exit[/dim]")
            except EOFError:
                break

        self._print_goodbye()

    async def _get_input(self) -> str | None:
        """Get input from the user using prompt_toolkit."""
        try:
            # Use a simple prompt without markup (prompt_toolkit doesn't render rich markup)
            self._console.print()
            return await self._prompt_session.prompt_async(
                "> ",
            )
        except (KeyboardInterrupt, EOFError):
            return None

    async def _handle_command(self, command: str) -> bool:
        """Handle a slash command.

        Args:
            command: The command string starting with '/'.

        Returns:
            True to continue the main loop, False to exit.
        """
        cmd = command.lower().split()[0]

        if cmd == "/quit":
            return False

        elif cmd == "/clear":
            self._clear_session()
            return True

        elif cmd == "/trace":
            self._show_trace()
            return True

        elif cmd == "/help":
            self._show_help()
            return True

        else:
            self._console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
            self._console.print("[dim]Type /help for available commands[/dim]")
            return True

    async def _handle_query(self, query: str) -> None:
        """Handle a user query by running the agent."""
        if self._session is None:
            self._session = self._session_manager.create_session()

        self._session_manager.add_message(self._session, "user", query)

        # Start trace
        trace = self._tracer.start_trace(query)

        # Create agent controller
        controller = AgentController(
            llm_client=self._llm_client,
            graph_db=self._graph_db,
            tool_router=self._tool_router,
            prompt_manager=self._prompt_manager,
            config=self._config,
            tracer=self._tracer,
        )

        # Run with progress display
        result = await self._run_with_progress(controller, query)

        # End trace
        self._tracer.end_trace(trace, result)
        self._last_trace = trace

        # Display result
        self._display_result(result)

        # Add to session
        self._session_manager.add_message(
            self._session, "assistant", result.answer, trace_id=trace.trace_id
        )

    async def _run_with_progress(
        self, controller: AgentController, query: str
    ) -> AgentResult:
        """Run the agent with a live progress display."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self._console,
            transient=True,
        )

        with Live(progress, console=self._console, refresh_per_second=4) as live:
            task_id = progress.add_task("Starting...", total=None)
            self._tracer.set_live_context(live, progress, task_id)

            result = await controller.run(query)

            self._tracer.set_live_context(None, None, None)  # type: ignore[arg-type]

        return result

    def _display_result(self, result: AgentResult) -> None:
        """Display the agent result with formatting."""
        # Status color mapping
        status_colors = {
            AgentStatus.COMPLETED: "green",
            AgentStatus.MAX_ITERATIONS: "yellow",
            AgentStatus.ERROR: "red",
            AgentStatus.RUNNING: "blue",
        }
        status_color = status_colors.get(result.status, "white")

        # Build answer panel
        answer_text = Text()
        answer_text.append(result.answer)

        panel_title = f"[{status_color}]Answer[/{status_color}]"
        if result.status != AgentStatus.COMPLETED:
            panel_title = (
                f"[{status_color}]{result.status.value.title()}[/{status_color}]"
            )

        panel = Panel(
            answer_text,
            title=panel_title,
            border_style=status_color,
            padding=(1, 2),
        )
        self._console.print(panel)

        # Show metadata
        meta_parts = [f"[dim]Iterations: {result.iterations}[/dim]"]
        if result.confidence is not None:
            confidence_pct = result.confidence * 100
            confidence_color = (
                "green"
                if confidence_pct >= 70
                else "yellow"
                if confidence_pct >= 40
                else "red"
            )
            meta_parts.append(
                f"[{confidence_color}]Confidence: {confidence_pct:.0f}%[/{confidence_color}]"
            )

        self._console.print(" | ".join(meta_parts))

        # Show supporting evidence if present in the last tool call
        if result.history:
            last_step = result.history[-1]
            if last_step.action == "submit_answer":
                evidence = last_step.output.get("supporting_evidence")
                if evidence:
                    self._console.print()
                    self._console.print("[bold]Supporting Evidence:[/bold]")
                    self._console.print(f"[dim]{evidence}[/dim]")

    def _clear_session(self) -> None:
        """Clear the current session."""
        if self._session:
            self._session_manager.clear_session(self._session)
        self._last_trace = None
        self._console.print("[green]Session cleared.[/green]")

    def _show_trace(self) -> None:
        """Show details of the last trace."""
        if self._last_trace is None:
            self._console.print(
                "[yellow]No trace available. Run a query first.[/yellow]"
            )
            return

        trace_data = self._tracer.export(self._last_trace)

        # Header
        self._console.print()
        self._console.print(f"[bold]Trace ID:[/bold] {trace_data['trace_id']}")
        self._console.print(f"[bold]Query:[/bold] {trace_data['query']}")

        if trace_data.get("duration_ms"):
            duration_sec = trace_data["duration_ms"] / 1000
            self._console.print(f"[bold]Duration:[/bold] {duration_sec:.2f}s")

        # Events table
        if trace_data.get("events"):
            self._console.print()
            self._console.print("[bold]Events:[/bold]")

            table = Table(show_header=True, header_style="bold")
            table.add_column("Type", style="cyan")
            table.add_column("Details")
            table.add_column("Duration", justify="right")

            for event in trace_data["events"]:
                event_type = event["event_type"]
                details = self._format_event_details(event)
                duration = (
                    f"{event['duration_ms']:.0f}ms" if event.get("duration_ms") else "-"
                )
                table.add_row(event_type, details, duration)

            self._console.print(table)

        # Result summary
        if trace_data.get("result"):
            result = trace_data["result"]
            self._console.print()
            self._console.print(f"[bold]Result:[/bold] {result['status']}")
            self._console.print(f"[bold]Iterations:[/bold] {result['iterations']}")
            if result.get("confidence"):
                self._console.print(
                    f"[bold]Confidence:[/bold] {result['confidence']:.0%}"
                )

    def _format_event_details(self, event: dict[str, Any]) -> str:
        """Format event details for display."""
        data = event.get("data", {})
        event_type = event["event_type"]

        if event_type == "query_start":
            return f"Query: {data.get('query', '')[:50]}..."

        elif event_type == "iteration_start":
            return f"Iteration {data.get('iteration', '?')}"

        elif event_type == "tool_call":
            tool_name = data.get("tool_name", "unknown")
            args = data.get("arguments", {})
            if tool_name == "execute_cypher":
                query = args.get("query", "")[:40]
                return f"{tool_name}: {query}..."
            elif tool_name == "submit_answer":
                return f"{tool_name}: confidence={args.get('confidence', '?')}"
            return f"{tool_name}"

        elif event_type == "tool_result":
            success = data.get("success", False)
            return "success" if success else "failed"

        elif event_type == "llm_request":
            return f"{data.get('messages_count', '?')} messages"

        elif event_type == "llm_response":
            return f"{data.get('tool_calls_count', 0)} tool calls"

        elif event_type == "error":
            return data.get("error", "Unknown error")[:50]

        elif event_type == "complete":
            return data.get("status", "completed")

        elif event_type == "max_iterations":
            return f"Reached {data.get('iterations', '?')}/{data.get('max', '?')}"

        return json.dumps(data)[:50] if data else "-"

    def _show_help(self) -> None:
        """Show available commands."""
        self._console.print()
        self._console.print("[bold]Available Commands:[/bold]")
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Command", style="cyan")
        table.add_column("Description")

        for cmd, desc in self.COMMANDS.items():
            table.add_row(cmd, desc)

        self._console.print(table)
        self._console.print()
        self._console.print(
            "[dim]Type your question to query the knowledge graph.[/dim]"
        )

    def _print_welcome(self) -> None:
        """Print the welcome message."""
        self._console.print()
        self._console.print(
            Panel(
                "[bold]Agentic Graph RAG[/bold]\n\n"
                "Ask questions about the knowledge graph.\n"
                "The agent will query the database iteratively to find answers.\n\n"
                "[dim]Type /help for available commands.[/dim]",
                border_style="blue",
            )
        )

    def _print_goodbye(self) -> None:
        """Print the goodbye message."""
        self._console.print()
        self._console.print("[dim]Goodbye![/dim]")
