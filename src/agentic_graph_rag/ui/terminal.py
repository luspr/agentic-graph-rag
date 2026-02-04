"""Terminal UI using prompt_toolkit for input and rich for output."""

import json
import sys
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console, Group
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


def _read_key() -> str | None:
    """Read a single keypress, handling arrow keys and special keys.

    Returns:
        String representing the key pressed:
        - 'up', 'down', 'left', 'right' for arrow keys
        - 'enter' for Enter key
        - 'q', 'j', 'k', etc. for letter keys
        - None if reading fails
    """
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)

            # Handle escape sequences (arrow keys, etc.)
            if ch == "\x1b":
                # Read more characters for escape sequence
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 == "A":
                        return "up"
                    elif ch3 == "B":
                        return "down"
                    elif ch3 == "C":
                        return "right"
                    elif ch3 == "D":
                        return "left"
                return "escape"

            # Handle special keys
            if ch == "\r" or ch == "\n":
                return "enter"
            if ch == "\x03":  # Ctrl+C
                return "ctrl-c"
            if ch == "\x04":  # Ctrl+D
                return "ctrl-d"

            return ch

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    except (ImportError, OSError):
        # Fallback for systems without termios
        return None
    except Exception:
        # Handle termios.error which may not be defined
        return None


def _format_data_pretty(data: Any, indent: int = 0) -> Text:
    """Format data structures for pretty display with colors.

    Args:
        data: The data to format (dict, list, str, etc.)
        indent: Current indentation level.

    Returns:
        Rich Text object with colored formatting.
    """
    text = Text()
    prefix = "  " * indent

    if isinstance(data, dict):
        if not data:
            text.append("{}", style="dim")
            return text
        text.append("{\n", style="dim")
        items = list(data.items())
        for i, (key, value) in enumerate(items):
            text.append(f"{prefix}  ")
            text.append(f'"{key}"', style="cyan")
            text.append(": ", style="dim")
            text.append_text(_format_data_pretty(value, indent + 1))
            if i < len(items) - 1:
                text.append(",", style="dim")
            text.append("\n")
        text.append(f"{prefix}}}", style="dim")

    elif isinstance(data, list):
        if not data:
            text.append("[]", style="dim")
            return text
        text.append("[\n", style="dim")
        for i, item in enumerate(data):
            text.append(f"{prefix}  ")
            text.append_text(_format_data_pretty(item, indent + 1))
            if i < len(data) - 1:
                text.append(",", style="dim")
            text.append("\n")
        text.append(f"{prefix}]", style="dim")

    elif isinstance(data, str):
        # For multi-line strings, display them nicely
        if "\n" in data or len(data) > 80:
            lines = data.split("\n")
            if len(lines) > 1:
                text.append('"""', style="green")
                text.append("\n")
                for line in lines:
                    text.append(f"{prefix}  ")
                    text.append(line, style="green")
                    text.append("\n")
                text.append(f'{prefix}"""', style="green")
            else:
                # Long single line - wrap it
                text.append(f'"{data}"', style="green")
        else:
            text.append(f'"{data}"', style="green")

    elif isinstance(data, bool):
        text.append(str(data).lower(), style="yellow")

    elif isinstance(data, (int, float)):
        text.append(str(data), style="magenta")

    elif data is None:
        text.append("null", style="dim italic")

    else:
        text.append(str(data), style="white")

    return text


class TraceInspector:
    """Interactive trace inspector for exploring trace events."""

    def __init__(
        self,
        console: Console,
        trace_data: dict[str, Any],
        interactive: bool = True,
    ) -> None:
        """Initialize the trace inspector.

        Args:
            console: Rich console for output.
            trace_data: Exported trace data to inspect.
            interactive: If False, just display the trace without interaction.
        """
        self._console = console
        self._trace_data = trace_data
        self._events: list[dict[str, Any]] = trace_data.get("events", [])
        self._selected_index = 0
        self._detail_view = False
        self._running = True
        self._interactive = interactive
        self._pager_requested = False

    def run(self) -> None:
        """Run the interactive trace inspector."""
        if not self._events:
            self._console.print("[yellow]No events in trace.[/yellow]")
            return

        # Non-interactive mode: just render once
        if not self._interactive:
            self._console.print(self._render_list_view())
            return

        # Check if we can use raw key input (termios + TTY)
        try:
            import termios  # noqa: F401

            can_use_raw = sys.stdin.isatty()
        except ImportError:
            can_use_raw = False

        self._running = True
        if can_use_raw:
            with Live(
                self._build_renderable(),
                console=self._console,
                screen=False,
                auto_refresh=False,
            ) as live:
                while self._running:
                    live.update(self._build_renderable(), refresh=True)
                    key = _read_key()
                    if key is None:
                        self._running = False
                    else:
                        self._process_key(key)
                    if self._pager_requested:
                        live.stop()
                        self._open_detail_pager()
                        self._pager_requested = False
                        live.start()
        else:
            while self._running:
                self._console.print(self._build_renderable())
                try:
                    self._console.print()
                    self._console.print("[cyan]>[/cyan] ", end="")
                    user_input = input().strip().lower()
                    self._process_input_fallback(user_input)
                except (EOFError, KeyboardInterrupt, OSError):
                    self._running = False
                if self._pager_requested:
                    self._open_detail_pager()
                    self._pager_requested = False

    def _build_renderable(self) -> Group:
        """Build the full renderable for the current view."""
        if self._detail_view:
            body = self._render_detail_view()
        else:
            body = self._render_list_view()
        if self._detail_view:
            footer = Text.from_markup(
                "[dim]  [cyan]q[/cyan]/[cyan]Esc[/cyan] back  "
                "[cyan]j[/cyan]/[cyan]↓[/cyan] next  "
                "[cyan]k[/cyan]/[cyan]↑[/cyan] prev  "
                "[cyan]p[/cyan] page[/dim]"
            )
        else:
            footer = Text.from_markup(
                "[dim]  [cyan]q[/cyan]/[cyan]Esc[/cyan] exit  "
                "[cyan]j[/cyan]/[cyan]↓[/cyan] down  "
                "[cyan]k[/cyan]/[cyan]↑[/cyan] up  "
                "[cyan]Enter[/cyan]/[cyan]v[/cyan] view  "
                "[cyan]p[/cyan] page  "
                "[cyan]1-9[/cyan] jump[/dim]"
            )
        return Group(body, Text(""), footer)

    def _process_key(self, key: str) -> None:
        """Process a single keypress."""
        if key in ("q", "Q", "escape"):
            if self._detail_view:
                self._detail_view = False
            else:
                self._running = False
        elif key in ("j", "J", "down"):
            self._move_selection(1)
        elif key in ("k", "K", "up"):
            self._move_selection(-1)
        elif key in ("enter", "v", "V"):
            if self._detail_view:
                self._toggle_detail_view()
            elif self._detail_overflows():
                self._request_pager()
            else:
                self._toggle_detail_view()
        elif key in ("p", "P"):
            self._request_pager()
        elif key in ("b", "B", "left"):
            if self._detail_view:
                self._detail_view = False
        elif key in ("ctrl-c", "ctrl-d"):
            self._running = False
        elif key.isdigit() and key != "0":
            # Jump to specific event by number
            idx = int(key) - 1
            if 0 <= idx < len(self._events):
                self._selected_index = idx
                self._detail_view = True

    def _process_input_fallback(self, user_input: str) -> None:
        """Process user input in fallback mode (using input())."""
        if user_input in ("q", "quit", "exit", "b", "back"):
            if self._detail_view:
                self._detail_view = False
            else:
                self._running = False
        elif user_input in ("j", "down", "n", "next"):
            self._move_selection(1)
        elif user_input in ("k", "up", "prev"):
            self._move_selection(-1)
        elif user_input in ("p", "page"):
            self._request_pager()
        elif user_input in ("", "enter", "v", "view"):
            if self._detail_view:
                self._toggle_detail_view()
            elif self._detail_overflows():
                self._request_pager()
            else:
                self._toggle_detail_view()
        elif user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(self._events):
                self._selected_index = idx
                self._detail_view = True

    def _render_list_view(self) -> Group:
        """Render the event list view."""
        parts: list[Any] = [
            Panel(Text("Trace Inspector", style="bold"), border_style="blue"),
            Text.from_markup(
                f"[bold]Trace ID:[/bold] {self._trace_data['trace_id'][:36]}"
            ),
        ]
        query = self._trace_data["query"]
        if len(query) > 70:
            query = query[:67] + "..."
        parts.append(Text.from_markup(f"[bold]Query:[/bold] {query}"))
        if self._trace_data.get("duration_ms"):
            duration_sec = self._trace_data["duration_ms"] / 1000
            parts.append(
                Text.from_markup(f"[bold]Duration:[/bold] {duration_sec:.2f}s")
            )
        parts.append(Text(""))

        # Events table
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("#", style="dim", width=3, justify="right")
        table.add_column("Type", style="cyan", width=16)
        table.add_column("Details", no_wrap=False, max_width=50)
        table.add_column("Time", justify="right", width=8)

        for i, event in enumerate(self._events):
            event_type = event["event_type"]
            details = self._format_event_summary(event)
            duration = (
                f"{event['duration_ms']:.0f}ms" if event.get("duration_ms") else "-"
            )

            # Highlight selected row with marker
            if i == self._selected_index:
                marker = ">"
                num_style = "bold cyan"
                type_style = "bold cyan"
                detail_style = "bold"
                time_style = "bold"
            else:
                marker = " "
                num_style = "dim"
                type_style = "cyan"
                detail_style = ""
                time_style = ""

            table.add_row(
                Text(f"{marker}{i + 1}", style=num_style),
                Text(event_type, style=type_style),
                Text(details[:48], style=detail_style),
                Text(duration, style=time_style),
            )

        parts.append(table)

        # Result summary
        if self._trace_data.get("result"):
            result = self._trace_data["result"]
            confidence_str = (
                f" | Confidence: {result['confidence']:.0%}"
                if result.get("confidence")
                else ""
            )
            parts.append(Text(""))
            parts.append(
                Text.from_markup(
                    f"[bold]Result:[/bold] {result['status']} | "
                    f"Iterations: {result['iterations']}{confidence_str}"
                )
            )

        return Group(*parts)

    def _render_detail_view(self) -> Group:
        """Render the detail view for the selected event."""
        event = self._events[self._selected_index]

        # Header
        title = f"Event {self._selected_index + 1}/{len(self._events)}: {event['event_type']}"
        parts: list[Any] = [Panel(Text(title, style="bold"), border_style="cyan")]

        # Event metadata
        parts.append(
            Text.from_markup(f"[bold]Type:[/bold] [cyan]{event['event_type']}[/cyan]")
        )
        parts.append(Text.from_markup(f"[bold]Timestamp:[/bold] {event['timestamp']}"))
        if event.get("duration_ms"):
            parts.append(
                Text.from_markup(f"[bold]Duration:[/bold] {event['duration_ms']:.0f}ms")
            )
        parts.append(Text(""))

        # Data payload
        parts.append(Text.from_markup("[bold]Data:[/bold]"))
        data = event.get("data", {})
        if data:
            formatted = _format_data_pretty(data)
            parts.append(Panel(formatted, border_style="dim", padding=(0, 1)))
        else:
            parts.append(Text.from_markup("[dim](no data)[/dim]"))

        # Special handling for llm_request - show message previews
        if event["event_type"] == "llm_request" and "messages" in data:
            parts.append(Text(""))
            parts.append(Text.from_markup("[bold]Messages Preview:[/bold]"))
            for i, msg in enumerate(data.get("messages", [])[:5]):
                role = msg.get("role", "unknown")
                content = str(msg.get("content", ""))[:80]
                if len(str(msg.get("content", ""))) > 80:
                    content += "..."
                role_colors = {
                    "system": "yellow",
                    "user": "green",
                    "assistant": "blue",
                    "tool": "magenta",
                }
                color = role_colors.get(role, "white")
                parts.append(
                    Text.from_markup(f"  [{color}]{i + 1}. {role}:[/{color}] {content}")
                )

        return Group(*parts)

    def _format_event_summary(self, event: dict[str, Any]) -> str:
        """Format event details as a brief summary for list view."""
        data = event.get("data", {})
        event_type = event["event_type"]

        if event_type == "query_start":
            query = data.get("query", "")
            return (
                f'Query: "{query[:45]}..."' if len(query) > 45 else f'Query: "{query}"'
            )

        elif event_type == "iteration_start":
            return f"Iteration {data.get('iteration', '?')}"

        elif event_type == "tool_call":
            tool_name = data.get("tool_name", "unknown")
            args = data.get("arguments", {})
            if tool_name == "execute_cypher":
                query = args.get("query", "")[:35]
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

        return json.dumps(data)[:45] + "..." if data else "-"

    def _move_selection(self, delta: int) -> None:
        """Move the selection by delta."""
        self._selected_index = max(
            0, min(len(self._events) - 1, self._selected_index + delta)
        )

    def _toggle_detail_view(self) -> None:
        """Toggle between list and detail view."""
        self._detail_view = not self._detail_view

    def _detail_overflows(self) -> bool:
        """Return True if the detail view would exceed the terminal height."""
        renderable = self._render_detail_view()
        measure_console = Console(
            width=self._console.size.width,
            record=True,
            force_terminal=True,
        )
        measure_console.print(renderable)
        lines = measure_console.export_text().splitlines()
        available_lines = max(1, self._console.size.height - 2)
        return len(lines) > available_lines

    def _request_pager(self) -> None:
        """Request opening the pager for the selected event."""
        self._pager_requested = True

    def _open_detail_pager(self) -> None:
        """Open a pager for the selected event details."""
        renderable = self._render_detail_view()
        with self._console.pager(styles=True):
            self._console.print(renderable)


class UITracer(Tracer):
    """Extended tracer that also updates the UI during agent execution."""

    def __init__(self, console: Console, log_file: Path | str | None = None) -> None:
        """Initialize the UI tracer.

        Args:
            console: Rich console for output.
            log_file: Optional path to a JSONL file for persisting events.
        """
        super().__init__(log_file=log_file)
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
        "/trace": "Open interactive trace inspector for the last query",
        "/help": "Show available commands",
    }

    def __init__(
        self,
        llm_client: LLMClient,
        graph_db: GraphDatabase,
        tool_router: ToolRouter,
        prompt_manager: PromptManager,
        config: AgentConfig | None = None,
        trace_log_file: Path | str | None = None,
    ) -> None:
        """Initialize the terminal UI.

        Args:
            llm_client: Client for LLM completions.
            graph_db: Graph database for queries.
            tool_router: Router for dispatching tool calls.
            prompt_manager: Manager for building prompts.
            config: Agent configuration.
            trace_log_file: Optional path to a JSONL file for trace persistence.
        """
        self._llm_client = llm_client
        self._graph_db = graph_db
        self._tool_router = tool_router
        self._prompt_manager = prompt_manager
        self._config = config or AgentConfig()

        self._console = Console()
        self._session_manager = SessionManager()
        self._session: Session | None = None
        self._tracer = UITracer(self._console, log_file=trace_log_file)
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

        history_messages = self._session_manager.get_context_messages(
            self._session, max_messages=self._config.max_history_messages
        )
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
        result = await self._run_with_progress(
            controller, query, history_messages=history_messages
        )

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
        self,
        controller: AgentController,
        query: str,
        history_messages: list[dict[str, Any]] | None = None,
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

            result = await controller.run(query, history_messages=history_messages)

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
        """Show interactive trace inspector for the last trace."""
        if self._last_trace is None:
            self._console.print(
                "[yellow]No trace available. Run a query first.[/yellow]"
            )
            return

        trace_data = self._tracer.export(self._last_trace)
        inspector = TraceInspector(self._console, trace_data)
        inspector.run()

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
