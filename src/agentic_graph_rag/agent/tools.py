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
            "Search the vector database for semantically similar nodes and return "
            "matching node UUIDs."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural language query to embed and search with",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of vector results to return",
                    "default": 10,
                },
                "filters": {
                    "type": "object",
                    "description": (
                        "Optional vector store filter (Qdrant filter JSON structure)"
                    ),
                },
            },
            "required": ["query"],
        },
    ),
    ToolDefinition(
        name="expand_node",
        description=(
            "Expand from a node to find connected nodes and relationships using the "
            "node's UUID property. Returns one record per path with ordered "
            "path_nodes and path_rels preserving direction."
        ),
        parameters={
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "The UUID of the node to expand from",
                },
                "relationship_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter to specific relationship types (optional)",
                },
                "depth": {
                    "type": "integer",
                    "description": "How many hops to traverse",
                    "default": 1,
                },
                "direction": {
                    "type": "string",
                    "enum": ["out", "in", "both"],
                    "description": "Traversal direction from the start node",
                    "default": "both",
                },
                "max_paths": {
                    "type": "integer",
                    "description": "Maximum number of paths to return",
                    "default": 20,
                },
                "max_branching": {
                    "type": "integer",
                    "description": (
                        "Max distinct neighbor nodes to explore from any single node"
                    ),
                },
            },
            "required": ["node_id"],
        },
    ),
    ToolDefinition(
        name="shortest_path",
        description=(
            "Find the shortest path(s) between two known nodes by UUID. "
            "Uses built-in Cypher shortestPath — no GDS required."
        ),
        parameters={
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "UUID of the source node",
                },
                "target_id": {
                    "type": "string",
                    "description": "UUID of the target node",
                },
                "relationship_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": ("Filter to specific relationship types (optional)"),
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum path length to consider",
                    "default": 10,
                },
                "all_shortest": {
                    "type": "boolean",
                    "description": ("Return all shortest paths instead of just one"),
                    "default": False,
                },
            },
            "required": ["source_id", "target_id"],
        },
    ),
    ToolDefinition(
        name="pagerank",
        description=(
            "Run Personalized PageRank from seed node UUIDs to find "
            "structurally important nodes around them. Uses GDS when "
            "available, falls back to a Cypher heuristic."
        ),
        parameters={
            "type": "object",
            "properties": {
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of seed node UUIDs to rank from",
                },
                "damping": {
                    "type": "number",
                    "description": "Damping factor for PageRank",
                    "default": 0.85,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of ranked nodes to return",
                    "default": 20,
                },
                "max_depth": {
                    "type": "integer",
                    "description": ("Maximum traversal depth for Cypher fallback"),
                    "default": 3,
                },
                "relationship_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": ("Filter to specific relationship types (optional)"),
                },
            },
            "required": ["source_ids"],
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
            "shortest_path": self._handle_shortest_path,
            "pagerank": self._handle_pagerank,
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
        """Execute vector search via the HybridRetriever."""
        if self._hybrid_retriever is None:
            return {
                "error": "Vector search is not available. Hybrid retriever not configured.",
                "success": False,
            }
        query: str = args["query"]
        limit = args.get("limit", 5)
        filters = args.get("filters")
        context: dict[str, Any] = {
            "action": "vector_search",
            "limit": limit,
            "filters": filters,
        }
        result = await self._hybrid_retriever.retrieve(query, context)
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
            "direction": args.get("direction", "both"),
            "max_paths": args.get("max_paths", 20),
            "max_branching": args.get("max_branching"),
        }
        result = await self._hybrid_retriever.retrieve(node_id, context)
        return {
            "success": result.success,
            "data": result.data,
            "message": result.message,
        }

    async def _handle_shortest_path(self, args: dict[str, Any]) -> dict[str, Any]:
        """Find shortest path(s) between two nodes via the HybridRetriever."""
        if self._hybrid_retriever is None:
            return {
                "error": (
                    "Shortest path is not available. Hybrid retriever not configured."
                ),
                "success": False,
            }
        source_id: str = args["source_id"]
        context: dict[str, Any] = {
            "action": "shortest_path",
            "source_id": source_id,
            "target_id": args["target_id"],
            "relationship_types": args.get("relationship_types"),
            "max_length": args.get("max_length", 10),
            "all_shortest": args.get("all_shortest", False),
        }
        result = await self._hybrid_retriever.retrieve(source_id, context)
        return {
            "success": result.success,
            "data": result.data,
            "message": result.message,
        }

    async def _handle_pagerank(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run Personalized PageRank via the HybridRetriever."""
        if self._hybrid_retriever is None:
            return {
                "error": (
                    "PageRank is not available. Hybrid retriever not configured."
                ),
                "success": False,
            }
        source_ids: list[str] = args["source_ids"]
        context: dict[str, Any] = {
            "action": "pagerank",
            "source_ids": source_ids,
            "damping": args.get("damping", 0.85),
            "limit": args.get("limit", 20),
            "max_depth": args.get("max_depth", 3),
            "relationship_types": args.get("relationship_types"),
        }
        result = await self._hybrid_retriever.retrieve(",".join(source_ids), context)
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
