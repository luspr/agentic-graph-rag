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
        vector_name: str | None = None,
    ) -> None:
        """Initialize the Qdrant vector store.

        Args:
            settings: Application settings with Qdrant connection info.
            collection_name: Qdrant collection name for vectors.
            vector_size: Dimensionality of embeddings stored in the collection.
            distance: Distance metric for similarity search.
            vector_name: Optional named vector to use for multi-vector collections.
        """
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._distance = distance
        if vector_name is not None and not vector_name.strip():
            raise ValueError("Vector name must be a non-empty string.")
        self._vector_name = vector_name
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
            using=self._vector_name,
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

        vector_struct = self._build_vector_struct(embedding)
        point = models.PointStruct(id=id, vector=vector_struct, payload=payload)
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

    def _build_vector_struct(self, embedding: list[float]) -> models.VectorStruct:
        """Build the vector struct for Qdrant APIs."""
        if self._vector_name is None:
            return embedding
        return {self._vector_name: embedding}

    async def _ensure_collection(self) -> None:
        """Ensure the Qdrant collection exists before operations."""
        if self._collection_ready:
            return

        try:
            info = await self._client.get_collection(self._collection_name)
            vectors = self._extract_vectors_config(info)
            self._vector_name = self._resolve_vector_name(vectors, self._vector_name)
            self._collection_ready = True
            return
        except qdrant_exceptions.UnexpectedResponse as exc:
            if exc.status_code != 404:
                raise

        await self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=self._build_vectors_config(),
        )
        self._collection_ready = True

    def _validate_embedding(self, embedding: list[float]) -> None:
        """Validate embedding dimensionality."""
        if len(embedding) != self._vector_size:
            raise ValueError(
                "Embedding size does not match collection vector size: "
                f"{len(embedding)} != {self._vector_size}"
            )

    def _build_vectors_config(
        self,
    ) -> models.VectorParams | dict[str, models.VectorParams]:
        params = models.VectorParams(
            size=self._vector_size,
            distance=self._distance,
        )
        if self._vector_name is None:
            return params
        return {self._vector_name: params}

    def _extract_vectors_config(
        self,
        info: models.CollectionInfo,
    ) -> dict[str, models.VectorParams] | models.VectorParams | None:
        try:
            return info.config.params.vectors
        except AttributeError:
            return None

    def _resolve_vector_name(
        self,
        vectors: dict[str, models.VectorParams] | models.VectorParams | None,
        requested: str | None,
    ) -> str | None:
        if vectors is None:
            return requested
        if isinstance(vectors, models.VectorParams):
            if requested is not None:
                raise ValueError(
                    "Qdrant collection uses an unnamed vector. "
                    "Unset QDRANT_VECTOR_NAME or create a named collection."
                )
            return None
        if isinstance(vectors, dict):
            if requested is not None:
                if requested not in vectors:
                    available = ", ".join(sorted(vectors))
                    raise ValueError(
                        "Qdrant collection does not include vector name "
                        f"'{requested}'. Available names: {available}"
                    )
                return requested
            if len(vectors) == 1:
                return next(iter(vectors))
            available = ", ".join(sorted(vectors))
            raise ValueError(
                "Qdrant collection defines multiple vector names. "
                f"Set QDRANT_VECTOR_NAME to one of: {available}"
            )
        return requested
