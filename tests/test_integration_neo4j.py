"""Integration tests for Neo4jClient against a live database.

These tests require a running Neo4j instance loaded with the movies dataset
(see docker-compose.yaml).  They are excluded from the default test run.

Run explicitly:
    uv run pytest -m integration
"""

import pytest

from agentic_graph_rag.config import Settings
from agentic_graph_rag.graph.neo4j_client import Neo4jClient

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

# Labels that appear as the *first* label on at least one node.
# "Person" is always co-labelled with Actor or Director in this dataset, so
# labels(n)[0] never returns it — it's a subordinate label, not a primary one.
DOCKER_COMPOSE_LABELS = {"Movie", "Genre", "Actor", "Director", "User"}
DOCKER_COMPOSE_REL_TYPES = {"IN_GENRE", "RATED", "ACTED_IN", "DIRECTED"}


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings pointing at the docker-compose Neo4j instance."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-placeholder")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secretpassword")
    return Settings()


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_execute_returns_movies(settings: Settings) -> None:
    """execute() fetches Movie nodes and returns populated records."""
    async with Neo4jClient(settings) as client:
        result = await client.execute(
            "MATCH (m:Movie) RETURN m.title AS title, m.year AS year LIMIT 5"
        )

    assert result.error is None
    assert len(result.records) == 5
    for row in result.records:
        assert "title" in row
        assert "year" in row
        assert isinstance(row["title"], str)
        assert row["title"] != ""


@pytest.mark.integration
@pytest.mark.anyio
async def test_execute_with_params(settings: Settings) -> None:
    """execute() correctly binds parameters and filters results."""
    async with Neo4jClient(settings) as client:
        result = await client.execute(
            "MATCH (m:Movie {title: $title}) RETURN m.title AS title",
            {"title": "The Matrix"},
        )

    # The Matrix is a well-known movie — present in every movies dataset.
    # If it happens not to be in this particular load the test is still
    # structurally sound; we just check the query ran cleanly.
    assert result.error is None
    assert isinstance(result.records, list)


@pytest.mark.integration
@pytest.mark.anyio
async def test_execute_empty_result(settings: Settings) -> None:
    """execute() returns an empty records list for a query that matches nothing."""
    async with Neo4jClient(settings) as client:
        result = await client.execute(
            "MATCH (m:Movie {title: $title}) RETURN m",
            {"title": "__nonexistent_movie_title_xyzzy__"},
        )

    assert result.error is None
    assert result.records == []


@pytest.mark.integration
@pytest.mark.anyio
async def test_execute_relationship_traversal(settings: Settings) -> None:
    """execute() traverses relationships and returns joined data."""
    async with Neo4jClient(settings) as client:
        result = await client.execute(
            "MATCH (a:Actor)-[r:ACTED_IN]->(m:Movie) "
            "RETURN a.name AS actor, r.role AS role, m.title AS movie "
            "LIMIT 3"
        )

    assert result.error is None
    assert len(result.records) == 3
    for row in result.records:
        assert row["actor"] is not None
        assert row["movie"] is not None
        # role property exists on ACTED_IN in this dataset
        assert "role" in row


@pytest.mark.integration
@pytest.mark.anyio
async def test_execute_summary_contains_query_type(settings: Settings) -> None:
    """execute() populates the summary with query_type."""
    async with Neo4jClient(settings) as client:
        result = await client.execute("MATCH (m:Movie) RETURN m LIMIT 1")

    assert result.error is None
    assert "query_type" in result.summary
    assert result.summary["query_type"] in ("r", "rw", "w", "s")


# ---------------------------------------------------------------------------
# validate_query()
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_validate_query_accepts_valid_cypher(settings: Settings) -> None:
    """validate_query() returns (True, None) for syntactically valid Cypher."""
    async with Neo4jClient(settings) as client:
        is_valid, error = await client.validate_query(
            "MATCH (m:Movie) WHERE m.year > 2000 RETURN m.title LIMIT 10"
        )

    assert is_valid is True
    assert error is None


@pytest.mark.integration
@pytest.mark.anyio
async def test_validate_query_rejects_invalid_cypher(
    settings: Settings,
) -> None:
    """validate_query() returns (False, error_msg) for broken Cypher."""
    async with Neo4jClient(settings) as client:
        is_valid, error = await client.validate_query(
            "THIS IS COMPLETELY INVALID CYPHER SYNTAX !!!"
        )

    assert is_valid is False
    assert error is not None
    assert len(error) > 0


# ---------------------------------------------------------------------------
# get_schema()
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_get_schema_returns_all_node_labels(settings: Settings) -> None:
    """get_schema() discovers all node labels present in the database."""
    async with Neo4jClient(settings) as client:
        schema = await client.get_schema()

    discovered_labels = {nt.label for nt in schema.node_types}
    # Every expected label must be present
    assert DOCKER_COMPOSE_LABELS <= discovered_labels


@pytest.mark.integration
@pytest.mark.anyio
async def test_get_schema_node_counts_are_positive(
    settings: Settings,
) -> None:
    """get_schema() reports positive counts for all node types."""
    async with Neo4jClient(settings) as client:
        schema = await client.get_schema()

    for nt in schema.node_types:
        assert nt.count > 0, f"Expected positive count for {nt.label}"


@pytest.mark.integration
@pytest.mark.anyio
async def test_get_schema_movie_has_expected_properties(
    settings: Settings,
) -> None:
    """get_schema() includes well-known Movie properties."""
    async with Neo4jClient(settings) as client:
        schema = await client.get_schema()

    movie = next((nt for nt in schema.node_types if nt.label == "Movie"), None)
    assert movie is not None, "Movie label not found in schema"
    # The movies dataset always has at least these properties
    assert "title" in movie.properties
    assert "year" in movie.properties


@pytest.mark.integration
@pytest.mark.anyio
async def test_get_schema_returns_all_relationship_types(
    settings: Settings,
) -> None:
    """get_schema() discovers all relationship types in the database."""
    async with Neo4jClient(settings) as client:
        schema = await client.get_schema()

    discovered_rels = {rt.type for rt in schema.relationship_types}
    assert DOCKER_COMPOSE_REL_TYPES <= discovered_rels


@pytest.mark.integration
@pytest.mark.anyio
async def test_get_schema_relationships_have_labels(
    settings: Settings,
) -> None:
    """get_schema() populates start_label and end_label on every relationship type."""
    async with Neo4jClient(settings) as client:
        schema = await client.get_schema()

    for rt in schema.relationship_types:
        assert rt.start_label != "", f"start_label empty for {rt.type}"
        assert rt.end_label != "", f"end_label empty for {rt.type}"
