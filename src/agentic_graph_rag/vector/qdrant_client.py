"""Qdrant vector store implementation."""

from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import exceptions as qdrant_exceptions
from qdrant_client.http import models

from agentic_graph_rag.config import Settings
from agentic_graph_rag.vector.base import VectorSearchResult, VectorStore


class QdrantVectorStore(VectorStore):
    """Qdrant-based vector store implementation."""

    def __init__(
        self,
        settings: Settings,
        collection_name: str,
        vector_size: int,
        distance: models.Distance = models.Distance.COSINE,
    ) -> None:
        """Initialize the Qdrant vector store.

        Args:
            settings: Application settings with Qdrant connection info.
            collection_name: Qdrant collection name for vectors.
            vector_size: Dimensionality of embeddings stored in the collection.
            distance: Distance metric for similarity search.
        """
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._distance = distance
        self._collection_ready = False

    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filter: dict[str, Any] | models.Filter | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors in Qdrant."""
        self._validate_embedding(embedding)
        await self._ensure_collection()

        query_filter = self._normalize_filter(filter)
        response = await self._client.query_points(
            collection_name=self._collection_name,
            query=embedding,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return [
            VectorSearchResult(
                id=str(point.id),
                score=point.score,
                payload=point.payload or {},
            )
            for point in response.points
        ]

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Insert or update a vector in Qdrant."""
        if not isinstance(id, str) or not id:
            raise ValueError("Vector id must be a non-empty string.")

        self._validate_embedding(embedding)
        await self._ensure_collection()

        point = models.PointStruct(id=id, vector=embedding, payload=payload)
        await self._client.upsert(
            collection_name=self._collection_name,
            points=[point],
        )

    def _normalize_filter(
        self, filter: dict[str, Any] | models.Filter | None
    ) -> models.Filter | None:
        """Normalize filter input to a Qdrant Filter object."""
        if filter is None:
            return None
        if isinstance(filter, models.Filter):
            return filter
        return models.Filter.model_validate(filter)

    async def _ensure_collection(self) -> None:
        """Ensure the Qdrant collection exists before operations."""
        if self._collection_ready:
            return

        try:
            await self._client.get_collection(self._collection_name)
            self._collection_ready = True
            return
        except qdrant_exceptions.UnexpectedResponse as exc:
            if exc.status_code != 404:
                raise

        await self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=self._vector_size,
                distance=self._distance,
            ),
        )
        self._collection_ready = True

    def _validate_embedding(self, embedding: list[float]) -> None:
        """Validate embedding dimensionality."""
        if len(embedding) != self._vector_size:
            raise ValueError(
                "Embedding size does not match collection vector size: "
                f"{len(embedding)} != {self._vector_size}"
            )
