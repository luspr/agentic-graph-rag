"""Prompt manager for constructing system and user prompts."""

import json
from dataclasses import dataclass
from typing import Any

from agentic_graph_rag.graph.base import GraphSchema, NodeType, RelationshipType
from agentic_graph_rag.prompts.templates import (
    HYBRID_SYSTEM_PROMPT_TEMPLATE,
    NO_HISTORY_MESSAGE,
    NO_RESULTS_MESSAGE,
    RETRIEVAL_PROMPT_TEMPLATE,
    SCHEMA_NODE_TEMPLATE,
    SCHEMA_RELATIONSHIP_TEMPLATE,
    STEP_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
)
from agentic_graph_rag.retriever.base import RetrievalStep, RetrievalStrategy


@dataclass
class PromptContext:
    """Context for building prompts."""

    user_query: str
    schema: GraphSchema
    history: list[RetrievalStep]
    examples: list[dict[str, Any]] | None = None


class PromptManager:
    """Manages prompt construction and formatting."""

    def build_system_prompt(
        self,
        schema: GraphSchema,
        strategy: RetrievalStrategy = RetrievalStrategy.CYPHER,
    ) -> str:
        """Build the system prompt with schema information.

        Args:
            schema: The graph schema containing node types and relationships.
            strategy: The retrieval strategy, determines which prompt template to use.

        Returns:
            The formatted system prompt.
        """
        schema_text = self._format_schema(schema)
        template = (
            HYBRID_SYSTEM_PROMPT_TEMPLATE
            if strategy == RetrievalStrategy.HYBRID
            else SYSTEM_PROMPT_TEMPLATE
        )
        return template.format(schema=schema_text)

    def build_retrieval_prompt(self, context: PromptContext) -> str:
        """Build prompt for retrieval iteration.

        Args:
            context: The context containing user query, history, and schema.

        Returns:
            The formatted retrieval prompt.
        """
        history_text = self._format_history(context.history)
        current_results = self._get_current_results(context.history)

        return RETRIEVAL_PROMPT_TEMPLATE.format(
            user_query=context.user_query,
            history=history_text,
            current_results=current_results,
        )

    def format_results(self, results: list[dict[str, Any]]) -> str:
        """Format query results for LLM consumption.

        Args:
            results: List of result dictionaries from a query.

        Returns:
            A formatted string representation of the results.
        """
        if not results:
            return NO_RESULTS_MESSAGE

        formatted_lines = []
        for i, record in enumerate(results, 1):
            formatted_lines.append(f"Record {i}:")
            for key, value in record.items():
                formatted_value = self._format_value(value)
                formatted_lines.append(f"  {key}: {formatted_value}")

        return "\n".join(formatted_lines)

    def _format_schema(self, schema: GraphSchema) -> str:
        """Format the graph schema as text."""
        parts = []

        if schema.node_types:
            parts.append("## Node Types\n")
            for node_type in schema.node_types:
                parts.append(self._format_node_type(node_type))

        if schema.relationship_types:
            parts.append("\n## Relationship Types\n")
            for rel_type in schema.relationship_types:
                parts.append(self._format_relationship_type(rel_type))

        return "".join(parts) if parts else "No schema information available."

    def _format_node_type(self, node_type: NodeType) -> str:
        """Format a single node type."""
        properties = ", ".join(
            f"{name}: {dtype}" for name, dtype in node_type.properties.items()
        )
        if not properties:
            properties = "none"

        return SCHEMA_NODE_TEMPLATE.format(
            label=node_type.label,
            count=node_type.count,
            properties=properties,
        )

    def _format_relationship_type(self, rel_type: RelationshipType) -> str:
        """Format a single relationship type."""
        properties = ", ".join(
            f"{name}: {dtype}" for name, dtype in rel_type.properties.items()
        )
        if not properties:
            properties = "none"

        return SCHEMA_RELATIONSHIP_TEMPLATE.format(
            type=rel_type.type,
            start_label=rel_type.start_label,
            end_label=rel_type.end_label,
            properties=properties,
        )

    def _format_history(self, history: list[RetrievalStep]) -> str:
        """Format the retrieval history."""
        if not history:
            return NO_HISTORY_MESSAGE

        parts = []
        for i, step in enumerate(history, 1):
            error_text = f"Error: {step.error}" if step.error else ""
            input_text = self._format_dict(step.input)
            output_text = self._format_dict(step.output)

            parts.append(
                STEP_TEMPLATE.format(
                    step_num=i,
                    action=step.action,
                    input=input_text,
                    output=output_text,
                    error=error_text,
                )
            )

        return "\n\n".join(parts)

    def _get_current_results(self, history: list[RetrievalStep]) -> str:
        """Get the most recent successful results from history."""
        if not history:
            return NO_RESULTS_MESSAGE

        # Get the last successful step with output data
        for step in reversed(history):
            if step.error is None and step.output:
                data = step.output.get("data", step.output.get("records", []))
                if isinstance(data, list):
                    return self.format_results(data)
                return self._format_dict(step.output)

        return NO_RESULTS_MESSAGE

    def _format_dict(self, d: dict[str, Any]) -> str:
        """Format a dictionary for display."""
        if not d:
            return "{}"
        try:
            return json.dumps(d, indent=2, default=str)
        except (TypeError, ValueError):
            return str(d)

    def _format_value(self, value: Any) -> str:
        """Format a single value for display."""
        if isinstance(value, dict):
            return self._format_dict(value)
        if isinstance(value, list):
            if len(value) <= 5:
                return str(value)
            return f"[{len(value)} items]"
        return str(value)
