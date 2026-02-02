"""Integration tests for CypherRetriever against a live Neo4j database.

These tests require a running Neo4j instance loaded with the movies dataset
(see docker-compose.yaml). They are excluded from the default test run.

Run explicitly:
    uv run pytest -m integration
"""

import pytest

from agentic_graph_rag.config import Settings
from agentic_graph_rag.graph.neo4j_client import Neo4jClient
from agentic_graph_rag.retriever.base import RetrievalStrategy
from agentic_graph_rag.retriever.cypher_retriever import CypherRetriever


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings pointing at the docker-compose Neo4j instance."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-placeholder")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secretpassword")
    return Settings(_env_file=None)


# ---------------------------------------------------------------------------
# strategy property
# ---------------------------------------------------------------------------


def test_strategy_returns_cypher(settings: Settings) -> None:
    """strategy property returns RetrievalStrategy.CYPHER."""
    # We don't need a connection to check the strategy property
    client = Neo4jClient(settings)
    retriever = CypherRetriever(client)
    assert retriever.strategy == RetrievalStrategy.CYPHER


# ---------------------------------------------------------------------------
# retrieve() - successful queries
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_retrieve_returns_movie_records(settings: Settings) -> None:
    """retrieve() executes query and returns movie records."""
    async with Neo4jClient(settings) as client:
        retriever = CypherRetriever(client)
        result = await retriever.retrieve(
            "MATCH (m:Movie) RETURN m.title AS title, m.year AS year LIMIT 5"
        )

    assert result.success is True
    assert len(result.data) == 5
    for row in result.data:
        assert "title" in row
        assert "year" in row
        assert isinstance(row["title"], str)


@pytest.mark.integration
@pytest.mark.anyio
async def test_retrieve_with_params_in_context(settings: Settings) -> None:
    """retrieve() passes params from context to the query."""
    async with Neo4jClient(settings) as client:
        retriever = CypherRetriever(client)
        result = await retriever.retrieve(
            "MATCH (m:Movie) WHERE m.year >= $min_year RETURN m.title LIMIT 5",
            context={"params": {"min_year": 2000}},
        )

    assert result.success is True
    assert isinstance(result.data, list)


@pytest.mark.integration
@pytest.mark.anyio
async def test_retrieve_empty_result_is_success(settings: Settings) -> None:
    """retrieve() returns success with empty data when query matches nothing."""
    async with Neo4jClient(settings) as client:
        retriever = CypherRetriever(client)
        result = await retriever.retrieve(
            "MATCH (m:Movie {title: $title}) RETURN m",
            context={"params": {"title": "__nonexistent_movie_xyzzy__"}},
        )

    assert result.success is True
    assert result.data == []
    assert "0 records" in result.message


@pytest.mark.integration
@pytest.mark.anyio
async def test_retrieve_relationship_traversal(settings: Settings) -> None:
    """retrieve() handles relationship traversal queries."""
    async with Neo4jClient(settings) as client:
        retriever = CypherRetriever(client)
        result = await retriever.retrieve(
            "MATCH (a:Actor)-[r:ACTED_IN]->(m:Movie) "
            "RETURN a.name AS actor, m.title AS movie LIMIT 3"
        )

    assert result.success is True
    assert len(result.data) == 3
    for row in result.data:
        assert row["actor"] is not None
        assert row["movie"] is not None


# ---------------------------------------------------------------------------
# retrieve() - step recording
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_retrieve_records_step_with_query(settings: Settings) -> None:
    """retrieve() records step with the executed query."""
    query = "MATCH (m:Movie) RETURN m.title LIMIT 1"
    async with Neo4jClient(settings) as client:
        retriever = CypherRetriever(client)
        result = await retriever.retrieve(query)

    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.action == "cypher_query"
    assert step.input["query"] == query
    assert step.error is None


@pytest.mark.integration
@pytest.mark.anyio
async def test_retrieve_step_output_contains_records(settings: Settings) -> None:
    """retrieve() step output contains the query records."""
    async with Neo4jClient(settings) as client:
        retriever = CypherRetriever(client)
        result = await retriever.retrieve(
            "MATCH (m:Movie) RETURN m.title AS title LIMIT 2"
        )

    step = result.steps[0]
    assert "records" in step.output
    assert len(step.output["records"]) == 2
    assert step.output["records"] == result.data


@pytest.mark.integration
@pytest.mark.anyio
async def test_retrieve_step_output_contains_summary(settings: Settings) -> None:
    """retrieve() step output contains the query summary."""
    async with Neo4jClient(settings) as client:
        retriever = CypherRetriever(client)
        result = await retriever.retrieve("MATCH (m:Movie) RETURN m LIMIT 1")

    step = result.steps[0]
    assert "summary" in step.output
    assert "query_type" in step.output["summary"]


# ---------------------------------------------------------------------------
# retrieve() - error handling
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_retrieve_captures_syntax_error(settings: Settings) -> None:
    """retrieve() captures syntax errors without raising exceptions."""
    async with Neo4jClient(settings) as client:
        retriever = CypherRetriever(client)
        result = await retriever.retrieve("THIS IS INVALID CYPHER SYNTAX !!!")

    assert result.success is False
    assert result.data == []
    assert len(result.steps) == 1
    assert result.steps[0].error is not None
    assert "failed" in result.message.lower()


@pytest.mark.integration
@pytest.mark.anyio
async def test_retrieve_captures_semantic_error(settings: Settings) -> None:
    """retrieve() captures semantic errors (e.g., unknown function)."""
    async with Neo4jClient(settings) as client:
        retriever = CypherRetriever(client)
        result = await retriever.retrieve(
            "MATCH (m:Movie) RETURN unknownFunction(m.title)"
        )

    assert result.success is False
    assert result.steps[0].error is not None


# ---------------------------------------------------------------------------
# retrieve() - message content
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_retrieve_message_indicates_record_count(settings: Settings) -> None:
    """retrieve() message indicates number of records retrieved."""
    async with Neo4jClient(settings) as client:
        retriever = CypherRetriever(client)
        result = await retriever.retrieve("MATCH (m:Movie) RETURN m.title LIMIT 7")

    assert result.success is True
    assert "7 records" in result.message
