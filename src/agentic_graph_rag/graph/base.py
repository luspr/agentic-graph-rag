from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class QueryResult:
    """Result from executing a Cypher query."""

    records: list[dict[str, Any]]
    summary: dict[str, Any]
    error: str | None = None


@dataclass
class NodeType:
    """Description of a node class (exact label combination) in the schema."""

    labels: tuple[str, ...]
    label_expression: str
    properties: dict[str, str]  # property_name -> type
    count: int


@dataclass
class RelationshipType:
    """Description of a relationship type in the schema."""

    type: str
    start_labels: tuple[str, ...]
    end_labels: tuple[str, ...]
    start_label_expression: str
    end_label_expression: str
    properties: dict[str, str]


@dataclass
class GraphSchema:
    """Schema information about the graph."""

    node_types: list[NodeType]
    relationship_types: list[RelationshipType]


class GraphDatabase(ABC):
    """Abstract interface for graph database operations."""

    @abstractmethod
    async def execute(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> QueryResult:
        """Execute a Cypher query."""
        ...

    @abstractmethod
    async def get_schema(self) -> GraphSchema:
        """Retrieve the graph schema."""
        ...

    @abstractmethod
    async def validate_query(self, cypher: str) -> tuple[bool, str | None]:
        """Validate a Cypher query without executing.

        Returns:
            Tuple of (is_valid, error_message).
        """
        ...

    async def has_gds(self) -> bool:
        """Check whether the Neo4j Graph Data Science library is available.

        Returns:
            True if GDS is installed and callable, False otherwise.
        """
        return False
