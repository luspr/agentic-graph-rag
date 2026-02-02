"""Vector store interfaces and data models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class VectorSearchResult:
    """A single result from vector search."""

    id: str
    score: float
    payload: dict[str, Any]


class VectorStore(ABC):
    """Abstract interface for vector storage and search.

    Contract:
        Vector IDs must match Neo4j node IDs (use Neo4j elementId as the canonical
        identifier). This ensures vector search results can be used directly for
        graph expansion and Cypher lookups.
    """

    @abstractmethod
    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors.

        Args:
            embedding: Vector embedding to search with.
            limit: Maximum number of results.
            filter: Optional Qdrant filter expressed as a dict.

        Returns:
            List of vector search results.
        """
        ...

    @abstractmethod
    async def upsert(
        self,
        id: str,
        embedding: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Insert or update a vector.

        Args:
            id: Vector ID matching the Neo4j elementId for the node.
            embedding: Vector embedding to store.
            payload: Metadata payload for the vector.
        """
        ...
