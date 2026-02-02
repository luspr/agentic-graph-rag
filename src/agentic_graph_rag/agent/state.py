"""Agent state management types."""

from dataclasses import dataclass, field
from enum import Enum

from agentic_graph_rag.retriever.base import RetrievalStep, RetrievalStrategy


class AgentStatus(Enum):
    """Status of the agent during execution."""

    RUNNING = "running"
    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    ERROR = "error"


@dataclass
class AgentState:
    """Current state of the agent during execution."""

    iteration: int
    status: AgentStatus
    history: list[RetrievalStep]
    current_answer: str | None = None
    confidence: float | None = None


@dataclass
class AgentConfig:
    """Configuration for the agent."""

    max_iterations: int = 10
    strategy: RetrievalStrategy = RetrievalStrategy.CYPHER


@dataclass
class AgentResult:
    """Final result from agent execution."""

    answer: str
    status: AgentStatus
    iterations: int
    history: list[RetrievalStep] = field(default_factory=list)
    confidence: float | None = None
