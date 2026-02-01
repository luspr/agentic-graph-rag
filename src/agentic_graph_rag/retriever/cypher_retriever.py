"""Cypher-based retriever for executing Cypher queries against Neo4j."""

from typing import Any

from agentic_graph_rag.graph.base import GraphDatabase
from agentic_graph_rag.retriever.base import (
    RetrievalResult,
    RetrievalStep,
    RetrievalStrategy,
    Retriever,
)


class CypherRetriever(Retriever):
    """Retriever that executes Cypher queries against a Neo4j database.

    This retriever takes a Cypher query as input and executes it against the
    configured graph database, returning the results as a list of dictionaries.
    """

    def __init__(self, graph_db: GraphDatabase) -> None:
        """Initialize the Cypher retriever.

        Args:
            graph_db: The graph database client to execute queries against.
        """
        self._graph_db = graph_db

    async def retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Execute a Cypher query and return the results.

        Args:
            query: The Cypher query to execute.
            context: Optional context containing query parameters under the 'params' key.

        Returns:
            RetrievalResult containing the query results and execution metadata.
        """
        params = context.get("params") if context else None

        step = RetrievalStep(
            action="cypher_query",
            input={"query": query, "params": params},
            output={},
            error=None,
        )

        result = await self._graph_db.execute(query, params)

        if result.error:
            step.error = result.error
            step.output = {"records": [], "summary": result.summary}
            return RetrievalResult(
                data=[],
                steps=[step],
                success=False,
                message=f"Query execution failed: {result.error}",
            )

        step.output = {"records": result.records, "summary": result.summary}
        return RetrievalResult(
            data=result.records,
            steps=[step],
            success=True,
            message=f"Retrieved {len(result.records)} records",
        )

    @property
    def strategy(self) -> RetrievalStrategy:
        """Return the retrieval strategy type."""
        return RetrievalStrategy.CYPHER
