from agentic_graph_rag.agent.state import AgentResult, AgentStatus
from agentic_graph_rag.eval.context import build_evidence_context
from agentic_graph_rag.retriever.base import RetrievalStep


def test_build_evidence_context() -> None:
    step = RetrievalStep(
        action="submit_answer",
        input={},
        output={"supporting_evidence": "Evidence text"},
    )
    result = AgentResult(
        answer="Answer",
        status=AgentStatus.COMPLETED,
        iterations=1,
        history=[step],
        confidence=0.9,
    )
    assert build_evidence_context(result) == "Evidence text"


def test_build_evidence_context_empty() -> None:
    result = AgentResult(
        answer="Answer",
        status=AgentStatus.COMPLETED,
        iterations=1,
        history=[],
        confidence=0.9,
    )
    assert build_evidence_context(result) == ""
