"""Tool definitions and router for dispatching LLM tool calls."""

from collections.abc import Callable, Coroutine
from typing import Any

from agentic_graph_rag.llm.base import ToolCall, ToolDefinition
from agentic_graph_rag.retriever.base import Retriever

_Handler = Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]

AGENT_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="execute_cypher",
        description="Execute a Cypher query against the Neo4j knowledge graph",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The Cypher query to execute",
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Explanation of why this query will help answer the question"
                    ),
                },
            },
            "required": ["query", "reasoning"],
        },
    ),
    ToolDefinition(
        name="vector_search",
        description=(
            "Search for nodes semantically similar to the given text. Returned IDs "
            "are Neo4j elementId values."
        ),
        parameters={
            "type": "object",
            "properties": {
                "search_text": {
                    "type": "string",
                    "description": "Text to find similar nodes for",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 10,
                },
            },
            "required": ["search_text"],
        },
    ),
    ToolDefinition(
        name="expand_node",
        description=(
            "Expand from a node to find connected nodes and relationships using the "
            "node's Neo4j elementId."
        ),
        parameters={
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "The ID of the node to expand from",
                },
                "relationship_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": ("Filter to specific relationship types (optional)"),
                },
                "depth": {
                    "type": "integer",
                    "description": "How many hops to traverse",
                    "default": 1,
                },
            },
            "required": ["node_id"],
        },
    ),
    ToolDefinition(
        name="submit_answer",
        description=(
            "Submit the final answer when confident the question has been answered"
        ),
        parameters={
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer to the user's question",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence level from 0.0 to 1.0",
                },
                "supporting_evidence": {
                    "type": "string",
                    "description": (
                        "Summary of evidence from the graph that supports this answer"
                    ),
                },
            },
            "required": ["answer", "confidence", "supporting_evidence"],
        },
    ),
]


class ToolRouter:
    """Routes LLM tool calls to the appropriate handler functions."""

    def __init__(
        self,
        cypher_retriever: Retriever,
        hybrid_retriever: Retriever | None = None,
    ) -> None:
        """Initialize the ToolRouter.

        Args:
            cypher_retriever: Retriever for executing Cypher queries.
            hybrid_retriever: Optional retriever for vector search and graph expansion.
        """
        self._cypher_retriever = cypher_retriever
        self._hybrid_retriever = hybrid_retriever
        self._handlers: dict[str, _Handler] = {
            "execute_cypher": self._handle_execute_cypher,
            "vector_search": self._handle_vector_search,
            "expand_node": self._handle_expand_node,
            "submit_answer": self._handle_submit_answer,
        }

    async def route(self, tool_call: ToolCall) -> dict[str, Any]:
        """Dispatch a tool call to the appropriate handler.

        Args:
            tool_call: The tool call from the LLM containing name and arguments.

        Returns:
            Dict result suitable for LLM consumption.
        """
        handler = self._handlers.get(tool_call.name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_call.name}", "success": False}
        return await handler(tool_call.arguments)

    async def _handle_execute_cypher(self, args: dict[str, Any]) -> dict[str, Any]:
        """Validate and execute a Cypher query via the CypherRetriever."""
        query: str = args["query"]
        result = await self._cypher_retriever.retrieve(query)
        return {
            "success": result.success,
            "data": result.data,
            "message": result.message,
        }

    async def _handle_vector_search(self, args: dict[str, Any]) -> dict[str, Any]:
        """Search for semantically similar nodes via the HybridRetriever."""
        if self._hybrid_retriever is None:
            return {
                "error": "Vector search is not available. Hybrid retriever not configured.",
                "success": False,
            }
        search_text: str = args["search_text"]
        context: dict[str, Any] = {
            "action": "vector_search",
            "search_text": search_text,
            "limit": args.get("limit", 10),
        }
        result = await self._hybrid_retriever.retrieve(search_text, context)
        return {
            "success": result.success,
            "data": result.data,
            "message": result.message,
        }

    async def _handle_expand_node(self, args: dict[str, Any]) -> dict[str, Any]:
        """Traverse the graph from a node via the HybridRetriever."""
        if self._hybrid_retriever is None:
            return {
                "error": "Graph expansion is not available. Hybrid retriever not configured.",
                "success": False,
            }
        node_id: str = args["node_id"]
        context: dict[str, Any] = {
            "action": "expand_node",
            "node_id": node_id,
            "relationship_types": args.get("relationship_types"),
            "depth": args.get("depth", 1),
        }
        result = await self._hybrid_retriever.retrieve(node_id, context)
        return {
            "success": result.success,
            "data": result.data,
            "message": result.message,
        }

    async def _handle_submit_answer(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return the final answer with confidence and supporting evidence."""
        return {
            "answer": args["answer"],
            "confidence": args["confidence"],
            "supporting_evidence": args["supporting_evidence"],
            "success": True,
        }
