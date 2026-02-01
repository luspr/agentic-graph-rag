from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class RetrievalStrategy(Enum):
    """Strategy for retrieving data from the knowledge graph."""

    CYPHER = "cypher"
    HYBRID = "hybrid"


@dataclass
class RetrievalStep:
    """A single step in the retrieval process."""

    action: str  # e.g., "cypher_query", "vector_search", "graph_expand"
    input: dict[str, Any]
    output: dict[str, Any]
    error: str | None = None


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""

    data: list[dict[str, Any]]
    steps: list[RetrievalStep]
    success: bool
    message: str


class Retriever(ABC):
    """Abstract interface for retrieval operations."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Execute a retrieval based on query.

        Args:
            query: The natural language query to retrieve data for.
            context: Optional context to guide the retrieval.

        Returns:
            RetrievalResult containing the retrieved data and metadata.
        """
        ...

    @property
    @abstractmethod
    def strategy(self) -> RetrievalStrategy:
        """Return the retrieval strategy type."""
        ...
