"""Unit tests for Neo4jClient with mocked driver."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j.exceptions import Neo4jError

from agentic_graph_rag.config import Settings
from agentic_graph_rag.graph.base import QueryResult
from agentic_graph_rag.graph.neo4j_client import Neo4jClient

_PATCH_TARGET = "agentic_graph_rag.graph.neo4j_client.Neo4jAsyncGraphDatabase"


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Minimal Settings for Neo4j client tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    return Settings(_env_file=None)


def _build_mock_driver(
    records: list[dict[str, Any]] | None = None,
    query_type: str = "r",
) -> tuple[MagicMock, AsyncMock, AsyncMock]:
    """Create a mock driver → session → result chain.

    Returns:
        Tuple of (mock_driver, mock_session, mock_result).
    """
    mock_result = AsyncMock()
    mock_result.data.return_value = records or []
    mock_result.consume.return_value = MagicMock(query_type=query_type)

    mock_session = AsyncMock()
    mock_session.run.return_value = mock_result

    session_cm = AsyncMock()
    session_cm.__aenter__.return_value = mock_session

    mock_driver = MagicMock()
    mock_driver.session.return_value = session_cm
    mock_driver.close = AsyncMock()

    return mock_driver, mock_session, mock_result


# --- execute() tests ---


@pytest.mark.anyio
async def test_execute_returns_records(settings: Settings) -> None:
    """execute() returns QueryResult with records and summary."""
    expected = [{"name": "Alice"}, {"name": "Bob"}]
    mock_driver, _, _ = _build_mock_driver(records=expected)

    with patch(_PATCH_TARGET) as mock_gdb:
        mock_gdb.driver.return_value = mock_driver
        async with Neo4jClient(settings) as client:
            result = await client.execute("MATCH (n:Person) RETURN n.name AS name")

    assert result.records == expected
    assert result.error is None
    assert result.summary["query_type"] == "r"


@pytest.mark.anyio
async def test_execute_passes_params(settings: Settings) -> None:
    """execute() forwards parameters to the driver."""
    mock_driver, mock_session, _ = _build_mock_driver()
    params = {"name": "Alice"}

    with patch(_PATCH_TARGET) as mock_gdb:
        mock_gdb.driver.return_value = mock_driver
        async with Neo4jClient(settings) as client:
            await client.execute("MATCH (n {name: $name}) RETURN n", params)

    mock_session.run.assert_awaited_once_with(
        "MATCH (n {name: $name}) RETURN n", params
    )


@pytest.mark.anyio
async def test_execute_defaults_params_to_empty_dict(
    settings: Settings,
) -> None:
    """execute() uses empty dict when no params provided."""
    mock_driver, mock_session, _ = _build_mock_driver()

    with patch(_PATCH_TARGET) as mock_gdb:
        mock_gdb.driver.return_value = mock_driver
        async with Neo4jClient(settings) as client:
            await client.execute("MATCH (n) RETURN n")

    mock_session.run.assert_awaited_once_with("MATCH (n) RETURN n", {})


@pytest.mark.anyio
async def test_execute_captures_neo4j_error(settings: Settings) -> None:
    """execute() catches Neo4jError and stores it in QueryResult.error."""
    mock_driver, mock_session, _ = _build_mock_driver()
    mock_session.run.side_effect = Neo4jError("syntax error")

    with patch(_PATCH_TARGET) as mock_gdb:
        mock_gdb.driver.return_value = mock_driver
        async with Neo4jClient(settings) as client:
            result = await client.execute("BAD CYPHER")

    assert result.error is not None
    assert "syntax error" in result.error
    assert result.records == []


# --- validate_query() tests ---


@pytest.mark.anyio
async def test_validate_query_valid(settings: Settings) -> None:
    """validate_query() returns (True, None) for valid Cypher."""
    mock_driver, mock_session, _ = _build_mock_driver()

    with patch(_PATCH_TARGET) as mock_gdb:
        mock_gdb.driver.return_value = mock_driver
        async with Neo4jClient(settings) as client:
            is_valid, error = await client.validate_query("MATCH (n) RETURN n")

    assert is_valid is True
    assert error is None
    mock_session.run.assert_awaited_once_with("EXPLAIN MATCH (n) RETURN n")


@pytest.mark.anyio
async def test_validate_query_invalid(settings: Settings) -> None:
    """validate_query() returns (False, error_msg) for invalid Cypher."""
    mock_driver, mock_session, _ = _build_mock_driver()
    mock_session.run.side_effect = Neo4jError("syntax error at position 0")

    with patch(_PATCH_TARGET) as mock_gdb:
        mock_gdb.driver.return_value = mock_driver
        async with Neo4jClient(settings) as client:
            is_valid, error = await client.validate_query("NOT VALID")

    assert is_valid is False
    assert error is not None


# --- connection lifecycle tests ---


@pytest.mark.anyio
async def test_context_manager_closes_driver(settings: Settings) -> None:
    """Exiting the async context manager closes the driver."""
    mock_driver = MagicMock()
    mock_driver.close = AsyncMock()

    with patch(_PATCH_TARGET) as mock_gdb:
        mock_gdb.driver.return_value = mock_driver
        async with Neo4jClient(settings):
            pass

    mock_driver.close.assert_awaited_once()


@pytest.mark.anyio
async def test_execute_without_connect_raises(settings: Settings) -> None:
    """Calling execute without connecting raises RuntimeError."""
    client = Neo4jClient(settings)
    with pytest.raises(RuntimeError):
        await client.execute("MATCH (n) RETURN n")


# --- get_schema() tests ---


@pytest.mark.anyio
async def test_get_schema_returns_graph_schema(settings: Settings) -> None:
    """get_schema() assembles GraphSchema from node and relationship metadata."""
    client = Neo4jClient(settings)

    responses: dict[str, QueryResult] = {
        "nodeTypeProperties": QueryResult(
            records=[
                {
                    "nodeLabels": ["Movie"],
                    "propertyName": "title",
                    "propertyTypes": ["String"],
                },
                {
                    "nodeLabels": ["Movie"],
                    "propertyName": "year",
                    "propertyTypes": ["Integer"],
                },
                {
                    "nodeLabels": ["Person"],
                    "propertyName": "name",
                    "propertyTypes": ["String"],
                },
            ],
            summary={},
        ),
        "count(*)": QueryResult(
            records=[
                {"labels": ["Movie"], "count": 10},
                {"labels": ["Person"], "count": 5},
            ],
            summary={},
        ),
        "start_labels": QueryResult(
            records=[
                {
                    "type": "ACTED_IN",
                    "start_labels": ["Person"],
                    "end_labels": ["Movie"],
                },
            ],
            summary={},
        ),
        "relationshipTypeProperties": QueryResult(
            records=[
                {
                    "relationshipType": "ACTED_IN",
                    "propertyName": "role",
                    "propertyTypes": ["String"],
                },
            ],
            summary={},
        ),
    }

    async def _mock_execute(
        cypher: str, params: dict[str, Any] | None = None
    ) -> QueryResult:
        for key, result in responses.items():
            if key in cypher:
                return result
        return QueryResult(records=[], summary={})

    client.execute = _mock_execute  # type: ignore[assignment]

    schema = await client.get_schema()

    # Verify node types
    assert len(schema.node_types) == 2
    movie = next(n for n in schema.node_types if n.label_expression == "Movie")
    assert movie.labels == ("Movie",)
    assert movie.properties == {"title": "String", "year": "Integer"}
    assert movie.count == 10

    person = next(n for n in schema.node_types if n.label_expression == "Person")
    assert person.labels == ("Person",)
    assert person.properties == {"name": "String"}
    assert person.count == 5

    # Verify relationship types
    assert len(schema.relationship_types) == 1
    rel = schema.relationship_types[0]
    assert rel.type == "ACTED_IN"
    assert rel.start_labels == ("Person",)
    assert rel.end_labels == ("Movie",)
    assert rel.start_label_expression == "Person"
    assert rel.end_label_expression == "Movie"
    assert rel.properties == {"role": "String"}


@pytest.mark.anyio
async def test_get_schema_preserves_multi_label_combinations(
    settings: Settings,
) -> None:
    """get_schema() keeps full label combinations for node and relationship endpoints."""
    client = Neo4jClient(settings)

    responses: dict[str, QueryResult] = {
        "nodeTypeProperties": QueryResult(
            records=[
                {
                    "nodeLabels": ["Base", "Account", "MSGraphAccount"],
                    "propertyName": "user_principal_name",
                    "propertyTypes": ["String"],
                },
                {
                    "nodeLabels": ["Base", "Account", "EmailAccount"],
                    "propertyName": "registered_email",
                    "propertyTypes": ["String"],
                },
            ],
            summary={},
        ),
        "count(*)": QueryResult(
            records=[
                {"labels": ["Base", "Account", "MSGraphAccount"], "count": 12},
                {"labels": ["Base", "Account", "EmailAccount"], "count": 7},
            ],
            summary={},
        ),
        "start_labels": QueryResult(
            records=[
                {
                    "type": "SENT_BY",
                    "start_labels": ["Base", "Message", "MSTeamsMessage"],
                    "end_labels": ["Base", "Account", "MSGraphAccount"],
                },
            ],
            summary={},
        ),
        "relationshipTypeProperties": QueryResult(records=[], summary={}),
        "valueType": QueryResult(records=[], summary={}),
    }

    async def _mock_execute(
        cypher: str, params: dict[str, Any] | None = None
    ) -> QueryResult:
        for key, result in responses.items():
            if key in cypher:
                return result
        return QueryResult(records=[], summary={})

    client.execute = _mock_execute  # type: ignore[assignment]

    schema = await client.get_schema()

    assert len(schema.node_types) == 2
    ms_graph_account = next(
        node_type
        for node_type in schema.node_types
        if node_type.labels == ("Account", "Base", "MSGraphAccount")
    )
    assert ms_graph_account.label_expression == "Account:Base:MSGraphAccount"
    assert ms_graph_account.count == 12
    assert ms_graph_account.properties == {"user_principal_name": "String"}

    rel = next(r for r in schema.relationship_types if r.type == "SENT_BY")
    assert rel.start_labels == ("Base", "MSTeamsMessage", "Message")
    assert rel.end_labels == ("Account", "Base", "MSGraphAccount")
    assert rel.start_label_expression == "Base:MSTeamsMessage:Message"
    assert rel.end_label_expression == "Account:Base:MSGraphAccount"


@pytest.mark.anyio
async def test_get_schema_relationship_properties_fallback(
    settings: Settings,
) -> None:
    """get_schema() falls back to data-derived relationship properties."""
    client = Neo4jClient(settings)

    responses: dict[str, QueryResult] = {
        "nodeTypeProperties": QueryResult(
            records=[
                {
                    "nodeLabels": ["Entity"],
                    "propertyName": "name",
                    "propertyTypes": ["String"],
                },
            ],
            summary={},
        ),
        "count(*)": QueryResult(
            records=[
                {"labels": ["Entity"], "count": 2},
            ],
            summary={},
        ),
        "start_labels": QueryResult(
            records=[
                {
                    "type": "INVESTIGATED",
                    "start_labels": ["Entity"],
                    "end_labels": ["Entity"],
                },
            ],
            summary={},
        ),
        "relationshipTypeProperties": QueryResult(
            records=[],
            summary={},
        ),
        "valueType": QueryResult(
            records=[
                {
                    "relationshipType": "INVESTIGATED",
                    "propertyName": "summary",
                    "propertyTypes": ["String"],
                },
                {
                    "relationshipType": "INVESTIGATED",
                    "propertyName": "source_document",
                    "propertyTypes": ["String"],
                },
            ],
            summary={},
        ),
    }

    async def _mock_execute(
        cypher: str, params: dict[str, Any] | None = None
    ) -> QueryResult:
        for key, result in responses.items():
            if key in cypher:
                return result
        return QueryResult(records=[], summary={})

    client.execute = _mock_execute  # type: ignore[assignment]

    schema = await client.get_schema()

    rel = next(r for r in schema.relationship_types if r.type == "INVESTIGATED")
    assert rel.properties == {
        "summary": "String",
        "source_document": "String",
    }


@pytest.mark.anyio
async def test_get_schema_returns_empty_on_error(
    settings: Settings,
) -> None:
    """get_schema() returns empty schema when metadata queries fail."""
    client = Neo4jClient(settings)

    async def _failing_execute(
        cypher: str, params: dict[str, Any] | None = None
    ) -> QueryResult:
        return QueryResult(records=[], summary={}, error="procedure not found")

    client.execute = _failing_execute  # type: ignore[assignment]

    schema = await client.get_schema()
    assert schema.node_types == []
    assert schema.relationship_types == []


# --- has_gds() tests ---


@pytest.mark.anyio
async def test_has_gds_returns_true_when_gds_available(
    settings: Settings,
) -> None:
    """has_gds() returns True when gds.version() succeeds."""
    client = Neo4jClient(settings)

    async def _mock_execute(
        cypher: str, params: dict[str, Any] | None = None
    ) -> QueryResult:
        if "gds.version()" in cypher:
            return QueryResult(records=[{"version": "2.6.0"}], summary={})
        return QueryResult(records=[], summary={})

    client.execute = _mock_execute  # type: ignore[assignment]

    assert await client.has_gds() is True


@pytest.mark.anyio
async def test_has_gds_returns_false_when_gds_unavailable(
    settings: Settings,
) -> None:
    """has_gds() returns False when gds.version() fails."""
    client = Neo4jClient(settings)

    async def _mock_execute(
        cypher: str, params: dict[str, Any] | None = None
    ) -> QueryResult:
        return QueryResult(records=[], summary={}, error="Unknown function")

    client.execute = _mock_execute  # type: ignore[assignment]

    assert await client.has_gds() is False


@pytest.mark.anyio
async def test_has_gds_caches_result(settings: Settings) -> None:
    """has_gds() caches after first probe and does not re-execute."""
    client = Neo4jClient(settings)
    call_count = 0

    async def _counting_execute(
        cypher: str, params: dict[str, Any] | None = None
    ) -> QueryResult:
        nonlocal call_count
        call_count += 1
        return QueryResult(records=[{"version": "2.6.0"}], summary={})

    client.execute = _counting_execute  # type: ignore[assignment]

    assert await client.has_gds() is True
    assert await client.has_gds() is True
    assert call_count == 1
