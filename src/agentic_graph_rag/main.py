"""Entry point for Agentic Graph RAG application."""

import sys
from datetime import datetime
from pathlib import Path

import anyio
from pydantic import ValidationError
from rich.console import Console

from agentic_graph_rag.agent import AgentConfig, ToolRouter
from agentic_graph_rag.config import Settings
from agentic_graph_rag.graph.neo4j_client import Neo4jClient
from agentic_graph_rag.llm.openai_client import OpenAILLMClient
from agentic_graph_rag.prompts.manager import PromptManager
from agentic_graph_rag.retriever.cypher_retriever import CypherRetriever
from agentic_graph_rag.ui import TerminalUI


async def main() -> int:
    """Main entry point for the application."""
    console = Console()

    # Load configuration
    try:
        settings = Settings()
    except ValidationError as e:
        console.print("[red]Configuration error:[/red]")
        for error in e.errors():
            field = error.get("loc", ("unknown",))[0]
            msg = error.get("msg", "Invalid value")
            console.print(f"  [yellow]{field}[/yellow]: {msg}")
        console.print(
            "\n[dim]Set required environment variables or create a .env file.[/dim]"
        )
        return 1

    # Initialize components
    llm_client = OpenAILLMClient(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )

    async with Neo4jClient(settings) as graph_db:
        # Test database connection
        try:
            await graph_db.get_schema()
        except Exception as e:
            console.print(f"[red]Failed to connect to Neo4j:[/red] {e}")
            console.print(
                f"\n[dim]Check your Neo4j connection settings:[/dim]\n"
                f"  URI: {settings.neo4j_uri}\n"
                f"  User: {settings.neo4j_user}"
            )
            await llm_client.aclose()
            return 1

        # Create retriever and tool router
        cypher_retriever = CypherRetriever(graph_db)
        tool_router = ToolRouter(cypher_retriever=cypher_retriever)

        # Create prompt manager and config
        prompt_manager = PromptManager()
        config = AgentConfig(max_iterations=settings.max_iterations)

        # Set up trace logging if enabled
        trace_log_file = None
        if settings.trace_logging_enabled:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_log_file = Path(settings.trace_log_dir) / f"trace_{timestamp}.jsonl"

        # Create and run UI
        ui = TerminalUI(
            llm_client=llm_client,
            graph_db=graph_db,
            tool_router=tool_router,
            prompt_manager=prompt_manager,
            config=config,
            trace_log_file=trace_log_file,
        )

        try:
            await ui.run()
        finally:
            await llm_client.aclose()

    return 0


def run() -> None:
    """Run the application."""
    exit_code = anyio.run(main)
    sys.exit(exit_code)


if __name__ == "__main__":
    run()
