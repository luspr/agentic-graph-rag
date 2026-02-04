"""Unit tests for QdrantVectorStore with mocked Qdrant client."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import Headers
from qdrant_client.http import exceptions as qdrant_exceptions
from qdrant_client.http import models

from agentic_graph_rag.config import Settings
from agentic_graph_rag.vector.qdrant_client import QdrantVectorStore

_PATCH_TARGET = "agentic_graph_rag.vector.qdrant_client.AsyncQdrantClient"


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Minimal Settings for Qdrant client tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    return Settings(_env_file=None)


def _mock_query_response() -> models.QueryResponse:
    """Build a sample QueryResponse."""
    return models.QueryResponse(
        points=[
            models.ScoredPoint(
                id="node-1",
                version=1,
                score=0.9,
                payload={"title": "The Matrix"},
            )
        ]
    )


def _mock_named_collection_info() -> MagicMock:
    info = MagicMock()
    info.config = MagicMock()
    info.config.params = MagicMock()
    info.config.params.vectors = {
        "vector": models.VectorParams(size=3, distance=models.Distance.COSINE)
    }
    return info


@pytest.mark.anyio
async def test_search_returns_results(settings: Settings) -> None:
    """search() returns VectorSearchResult entries."""
    mock_client = MagicMock()
    mock_client.get_collection = AsyncMock(return_value=MagicMock())
    mock_client.query_points = AsyncMock(return_value=_mock_query_response())

    with patch(_PATCH_TARGET, return_value=mock_client):
        store = QdrantVectorStore(settings, "movies", vector_size=3)
        results = await store.search([0.1, 0.2, 0.3], limit=1)

    assert len(results) == 1
    assert results[0].id == "node-1"
    assert results[0].score == 0.9
    assert results[0].payload["title"] == "The Matrix"


@pytest.mark.anyio
async def test_search_passes_filter_dict(settings: Settings) -> None:
    """search() converts filter dict to Qdrant Filter."""
    mock_client = MagicMock()
    mock_client.get_collection = AsyncMock(return_value=MagicMock())
    mock_client.query_points = AsyncMock(return_value=_mock_query_response())

    filter_dict = {"must": []}

    with patch(_PATCH_TARGET, return_value=mock_client):
        store = QdrantVectorStore(settings, "movies", vector_size=3)
        await store.search([0.1, 0.2, 0.3], limit=5, filter=filter_dict)

    await_args = mock_client.query_points.await_args
    assert await_args is not None
    _, kwargs = await_args
    assert isinstance(kwargs["query_filter"], models.Filter)


@pytest.mark.anyio
async def test_search_uses_vector_name_when_collection_named(settings: Settings) -> None:
    """search() sets using when collection has named vector."""
    mock_client = MagicMock()
    mock_client.get_collection = AsyncMock(return_value=_mock_named_collection_info())
    mock_client.query_points = AsyncMock(return_value=_mock_query_response())

    with patch(_PATCH_TARGET, return_value=mock_client):
        store = QdrantVectorStore(settings, "movies", vector_size=3)
        await store.search([0.1, 0.2, 0.3], limit=1)

    await_args = mock_client.query_points.await_args
    assert await_args is not None
    _, kwargs = await_args
    assert kwargs["using"] == "vector"


@pytest.mark.anyio
async def test_search_auto_creates_collection(settings: Settings) -> None:
    """search() creates collection when missing."""
    mock_client = MagicMock()
    not_found = qdrant_exceptions.UnexpectedResponse(
        status_code=404,
        reason_phrase="Not Found",
        content=b"",
        headers=Headers({}),
    )
    mock_client.get_collection = AsyncMock(side_effect=not_found)
    mock_client.create_collection = AsyncMock(return_value=True)
    mock_client.query_points = AsyncMock(return_value=_mock_query_response())

    with patch(_PATCH_TARGET, return_value=mock_client):
        store = QdrantVectorStore(settings, "movies", vector_size=3)
        await store.search([0.1, 0.2, 0.3], limit=1)

    mock_client.create_collection.assert_awaited_once()


@pytest.mark.anyio
async def test_upsert_inserts_point(settings: Settings) -> None:
    """upsert() inserts a point with payload."""
    mock_client = MagicMock()
    mock_client.get_collection = AsyncMock(return_value=MagicMock())
    mock_client.upsert = AsyncMock()

    with patch(_PATCH_TARGET, return_value=mock_client):
        store = QdrantVectorStore(settings, "movies", vector_size=3)
        await store.upsert(
            id="node-1",
            embedding=[0.1, 0.2, 0.3],
            payload={"title": "The Matrix"},
        )

    mock_client.upsert.assert_awaited_once()
    await_args = mock_client.upsert.await_args
    assert await_args is not None
    _, kwargs = await_args
    assert kwargs["collection_name"] == "movies"
    assert len(kwargs["points"]) == 1
    assert kwargs["points"][0].id == "node-1"


@pytest.mark.anyio
async def test_upsert_uses_named_vector_when_collection_named(settings: Settings) -> None:
    """upsert() wraps vector with name when collection uses named vectors."""
    mock_client = MagicMock()
    mock_client.get_collection = AsyncMock(return_value=_mock_named_collection_info())
    mock_client.upsert = AsyncMock()

    with patch(_PATCH_TARGET, return_value=mock_client):
        store = QdrantVectorStore(settings, "movies", vector_size=3)
        await store.upsert(
            id="node-1",
            embedding=[0.1, 0.2, 0.3],
            payload={"title": "The Matrix"},
        )

    await_args = mock_client.upsert.await_args
    assert await_args is not None
    _, kwargs = await_args
    point = kwargs["points"][0]
    assert point.vector == {"vector": [0.1, 0.2, 0.3]}


@pytest.mark.anyio
async def test_upsert_rejects_non_string_id(settings: Settings) -> None:
    """upsert() rejects non-string IDs."""
    mock_client = MagicMock()
    mock_client.get_collection = AsyncMock(return_value=MagicMock())

    with patch(_PATCH_TARGET, return_value=mock_client):
        store = QdrantVectorStore(settings, "movies", vector_size=3)
        with pytest.raises(ValueError, match="Vector id"):
            bad_id: Any = 123
            await store.upsert(
                id=bad_id,
                embedding=[0.1, 0.2, 0.3],
                payload={"title": "The Matrix"},
            )


@pytest.mark.anyio
async def test_embedding_size_mismatch_raises(settings: Settings) -> None:
    """Embedding size mismatches raise ValueError."""
    mock_client = MagicMock()
    mock_client.get_collection = AsyncMock(return_value=MagicMock())

    with patch(_PATCH_TARGET, return_value=mock_client):
        store = QdrantVectorStore(settings, "movies", vector_size=3)
        with pytest.raises(ValueError, match="Embedding size"):
            await store.search([0.1, 0.2], limit=1)
