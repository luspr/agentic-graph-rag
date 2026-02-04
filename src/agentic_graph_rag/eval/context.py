"""Utilities for extracting evaluation context from agent results."""

from __future__ import annotations

from agentic_graph_rag.agent.state import AgentResult


def build_evidence_context(result: AgentResult) -> str:
    """Extract supporting evidence context from an agent result."""
    for step in reversed(result.history):
        if step.action != "submit_answer":
            continue
        evidence = step.output.get("supporting_evidence")
        if isinstance(evidence, str) and evidence.strip():
            return evidence.strip()
    return ""
