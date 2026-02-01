"""Unit tests for CypherRetriever with mocked GraphDatabase."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_graph_rag.graph.base import GraphDatabase, GraphSchema, QueryResult
from agentic_graph_rag.retriever.base import RetrievalStrategy
from agentic_graph_rag.retriever.cypher_retriever import CypherRetriever


def create_mock_graph_db() -> MagicMock:
    """Create a mock GraphDatabase for testing."""
    mock = MagicMock(spec=GraphDatabase)
    mock.execute = AsyncMock()
    mock.get_schema = AsyncMock(
        return_value=GraphSchema(node_types=[], relationship_types=[])
    )
    mock.validate_query = AsyncMock(return_value=(True, None))
    return mock


@pytest.fixture
def mock_graph_db() -> MagicMock:
    """Create a mock graph database."""
    return create_mock_graph_db()


@pytest.fixture
def retriever(mock_graph_db: MagicMock) -> CypherRetriever:
    """Create a CypherRetriever with mock database."""
    return CypherRetriever(mock_graph_db)


# --- strategy property tests ---


def test_strategy_returns_cypher(retriever: CypherRetriever) -> None:
    """strategy property returns RetrievalStrategy.CYPHER."""
    assert retriever.strategy == RetrievalStrategy.CYPHER


# --- retrieve() tests ---


@pytest.mark.anyio
async def test_retrieve_returns_records(
    retriever: CypherRetriever,
    mock_graph_db: MagicMock,
) -> None:
    """retrieve() returns records from successful query execution."""
    expected_records = [{"name": "Alice"}, {"name": "Bob"}]
    mock_graph_db.execute.return_value = QueryResult(
        records=expected_records,
        summary={"query_type": "r"},
        error=None,
    )

    result = await retriever.retrieve("MATCH (n:Person) RETURN n.name AS name")

    assert result.success is True
    assert result.data == expected_records
    assert len(result.steps) == 1
    assert result.steps[0].action == "cypher_query"
    assert result.steps[0].error is None
    mock_graph_db.execute.assert_awaited_once_with(
        "MATCH (n:Person) RETURN n.name AS name", None
    )


@pytest.mark.anyio
async def test_retrieve_passes_params_from_context(
    retriever: CypherRetriever,
    mock_graph_db: MagicMock,
) -> None:
    """retrieve() passes params from context to execute()."""
    mock_graph_db.execute.return_value = QueryResult(
        records=[{"name": "Alice"}],
        summary={},
        error=None,
    )
    params: dict[str, Any] = {"name": "Alice"}
    context = {"params": params}

    await retriever.retrieve("MATCH (n {name: $name}) RETURN n", context)

    mock_graph_db.execute.assert_awaited_once_with(
        "MATCH (n {name: $name}) RETURN n", params
    )


@pytest.mark.anyio
async def test_retrieve_handles_empty_context(
    retriever: CypherRetriever,
    mock_graph_db: MagicMock,
) -> None:
    """retrieve() handles empty context correctly."""
    mock_graph_db.execute.return_value = QueryResult(
        records=[],
        summary={},
        error=None,
    )

    result = await retriever.retrieve("MATCH (n) RETURN n", {})

    assert result.success is True
    mock_graph_db.execute.assert_awaited_once_with("MATCH (n) RETURN n", None)


@pytest.mark.anyio
async def test_retrieve_handles_none_context(
    retriever: CypherRetriever,
    mock_graph_db: MagicMock,
) -> None:
    """retrieve() handles None context correctly."""
    mock_graph_db.execute.return_value = QueryResult(
        records=[],
        summary={},
        error=None,
    )

    result = await retriever.retrieve("MATCH (n) RETURN n", None)

    assert result.success is True
    mock_graph_db.execute.assert_awaited_once_with("MATCH (n) RETURN n", None)


@pytest.mark.anyio
async def test_retrieve_captures_error_without_raising(
    retriever: CypherRetriever,
    mock_graph_db: MagicMock,
) -> None:
    """retrieve() captures errors in result without raising exceptions."""
    mock_graph_db.execute.return_value = QueryResult(
        records=[],
        summary={},
        error="Invalid syntax at position 0",
    )

    result = await retriever.retrieve("INVALID CYPHER")

    assert result.success is False
    assert result.data == []
    assert "Invalid syntax" in result.message
    assert len(result.steps) == 1
    assert result.steps[0].error == "Invalid syntax at position 0"


@pytest.mark.anyio
async def test_retrieve_records_step_with_input_and_output(
    retriever: CypherRetriever,
    mock_graph_db: MagicMock,
) -> None:
    """retrieve() records step with query input and result output."""
    records = [{"title": "The Matrix"}]
    summary = {"query_type": "r", "counters": {}}
    mock_graph_db.execute.return_value = QueryResult(
        records=records,
        summary=summary,
        error=None,
    )
    params: dict[str, Any] = {"year": 1999}
    context = {"params": params}

    result = await retriever.retrieve(
        "MATCH (m:Movie {year: $year}) RETURN m.title AS title",
        context,
    )

    step = result.steps[0]
    assert step.action == "cypher_query"
    assert step.input == {
        "query": "MATCH (m:Movie {year: $year}) RETURN m.title AS title",
        "params": params,
    }
    assert step.output == {"records": records, "summary": summary}
    assert step.error is None


@pytest.mark.anyio
async def test_retrieve_message_indicates_record_count(
    retriever: CypherRetriever,
    mock_graph_db: MagicMock,
) -> None:
    """retrieve() message indicates number of records retrieved."""
    records = [{"a": 1}, {"a": 2}, {"a": 3}]
    mock_graph_db.execute.return_value = QueryResult(
        records=records,
        summary={},
        error=None,
    )

    result = await retriever.retrieve("MATCH (n) RETURN n")

    assert "3 records" in result.message


@pytest.mark.anyio
async def test_retrieve_empty_result_is_success(
    retriever: CypherRetriever,
    mock_graph_db: MagicMock,
) -> None:
    """retrieve() with no matching records is still a success."""
    mock_graph_db.execute.return_value = QueryResult(
        records=[],
        summary={},
        error=None,
    )

    result = await retriever.retrieve("MATCH (n:NonExistent) RETURN n")

    assert result.success is True
    assert result.data == []
    assert "0 records" in result.message
