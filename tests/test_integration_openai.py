"""Integration tests for OpenAILLMClient against the live OpenAI API.

These tests require a valid OPENAI_API_KEY environment variable (or .env file)
and consume real API credits.  They are excluded from the default test run.

Run explicitly:
    uv run pytest tests/test_integration_openai.py -m integration -v
    uv run pytest tests/test_integration_openai.py -m integration -k vector -v
"""

import os
import uuid
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import exceptions as qdrant_exceptions
from qdrant_client.http import models

from agentic_graph_rag.config import Settings
from agentic_graph_rag.agent.tools import ToolRouter
from agentic_graph_rag.llm.base import ToolCall, ToolDefinition
from agentic_graph_rag.llm.openai_client import OpenAILLMClient
from agentic_graph_rag.retriever.base import Retriever
from agentic_graph_rag.retriever.hybrid_retriever import HybridRetriever
from agentic_graph_rag.vector.qdrant_client import QdrantVectorStore

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 1536  # text-embedding-3-small dimensionality


@pytest.fixture
async def client() -> AsyncIterator[OpenAILLMClient]:
    """OpenAILLMClient backed by a real API key, properly closed after each test."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    async with OpenAILLMClient(api_key=api_key, model="gpt-5.2") as c:
        yield c


@pytest.fixture
def qdrant_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings fixture for Qdrant integration."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    collection = os.environ.get("QDRANT_COLLECTION")
    if not collection:
        pytest.skip("QDRANT_COLLECTION not set; requires dedicated test collection.")
    if "test" not in collection.lower():
        pytest.skip("QDRANT_COLLECTION must reference a dedicated test collection.")

    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    monkeypatch.setenv(
        "NEO4J_URI",
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    )
    monkeypatch.setenv("NEO4J_USER", os.environ.get("NEO4J_USER", "neo4j"))
    monkeypatch.setenv("NEO4J_PASSWORD", os.environ.get("NEO4J_PASSWORD", "secret"))
    monkeypatch.setenv("QDRANT_HOST", os.environ.get("QDRANT_HOST", "localhost"))
    monkeypatch.setenv("QDRANT_PORT", os.environ.get("QDRANT_PORT", "6333"))
    monkeypatch.setenv("QDRANT_COLLECTION", collection)
    embedding_dim = os.environ.get("EMBEDDING_DIM")
    if embedding_dim:
        monkeypatch.setenv("EMBEDDING_DIM", embedding_dim)
    uuid_property = os.environ.get("NODE_UUID_PROPERTY")
    if uuid_property:
        monkeypatch.setenv("NODE_UUID_PROPERTY", uuid_property)
    return Settings(_env_file=None)


async def _ensure_test_collection(
    settings: Settings,
) -> tuple[AsyncQdrantClient, bool]:
    client = AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
    try:
        info = await client.get_collection(settings.qdrant_collection)
    except qdrant_exceptions.UnexpectedResponse as exc:
        if exc.status_code != 404:
            await client.close()
            raise
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=models.VectorParams(
                size=settings.embedding_dim,
                distance=models.Distance.COSINE,
            ),
        )
        return client, True
    except Exception:
        await client.close()
        pytest.skip("Qdrant collection missing or unavailable; skipping vector test.")
        raise

    size = _extract_collection_size(info)
    if size is not None and size != settings.embedding_dim:
        await client.close()
        pytest.skip(
            "Qdrant collection vector size does not match EMBEDDING_DIM; skipping."
        )
    return client, False


def _extract_collection_size(info: models.CollectionInfo) -> int | None:
    try:
        vectors = info.config.params.vectors
        if isinstance(vectors, models.VectorParams):
            return vectors.size
    except AttributeError:
        return None
    return None


# ---------------------------------------------------------------------------
# complete() — basic
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_complete_returns_text(client: OpenAILLMClient) -> None:
    """complete() returns a non-empty text response for a simple prompt."""
    messages = [{"role": "user", "content": "Reply with exactly one word: hello"}]

    response = await client.complete(messages)

    assert response.content is not None
    assert len(response.content.strip()) > 0
    assert response.finish_reason == "stop"
    assert response.usage["prompt_tokens"] > 0
    assert response.usage["completion_tokens"] > 0


@pytest.mark.integration
@pytest.mark.anyio
async def test_complete_respects_system_message(client: OpenAILLMClient) -> None:
    """complete() honours a system message that constrains the response."""
    messages = [
        {
            "role": "system",
            "content": "You are a robot. Every response must be exactly the single word 'beep'.",
        },
        {"role": "user", "content": "Say something."},
    ]

    response = await client.complete(messages)

    assert response.content is not None
    assert "beep" in response.content.lower()


@pytest.mark.integration
@pytest.mark.anyio
async def test_complete_with_reasoning_effort(client: OpenAILLMClient) -> None:
    """complete() supports non-default reasoning effort settings."""
    messages = [{"role": "user", "content": "Reply with exactly one word: hello"}]

    response = await client.complete(messages, reasoning_effort="high")

    assert response.content is not None
    assert len(response.content.strip()) > 0
    assert response.finish_reason == "stop"
    assert response.usage["prompt_tokens"] > 0
    assert response.usage["completion_tokens"] > 0


# ---------------------------------------------------------------------------
# complete() — tool calling
# ---------------------------------------------------------------------------

CALCULATOR_TOOL = ToolDefinition(
    name="calculator",
    description="Evaluate a simple arithmetic expression and return the numeric result.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A simple math expression, e.g. '2 + 2'.",
            }
        },
        "required": ["expression"],
    },
)


@pytest.mark.integration
@pytest.mark.anyio
async def test_complete_invokes_tool(client: OpenAILLMClient) -> None:
    """complete() returns a tool call when the prompt requests tool use."""
    messages = [
        {
            "role": "user",
            "content": "Use the calculator tool to compute 17 + 25.",
        }
    ]

    response = await client.complete(messages, tools=[CALCULATOR_TOOL])

    assert len(response.tool_calls) >= 1
    assert response.finish_reason == "tool_calls"

    tc = response.tool_calls[0]
    assert tc.name == "calculator"
    assert "expression" in tc.arguments
    assert tc.id != ""


@pytest.mark.integration
@pytest.mark.anyio
async def test_complete_tool_call_arguments_are_parsed(
    client: OpenAILLMClient,
) -> None:
    """Tool call arguments are valid, parsed JSON — not raw strings."""
    messages = [
        {
            "role": "user",
            "content": "Use the calculator tool to compute 100 - 58.",
        }
    ]

    response = await client.complete(messages, tools=[CALCULATOR_TOOL])

    assert len(response.tool_calls) >= 1
    tc = response.tool_calls[0]
    # arguments must be a real dict, not a string
    assert isinstance(tc.arguments, dict)
    assert isinstance(tc.arguments.get("expression"), str)


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_embed_returns_correct_dimensionality(
    client: OpenAILLMClient,
) -> None:
    """embed() returns a vector with the expected dimensionality for text-embedding-3-small."""
    vector = await client.embed("The quick brown fox jumps over the lazy dog.")

    assert len(vector) == EMBEDDING_DIM
    assert all(isinstance(v, float) for v in vector)


@pytest.mark.integration
@pytest.mark.anyio
async def test_embed_different_texts_yield_different_vectors(
    client: OpenAILLMClient,
) -> None:
    """embed() produces distinct vectors for semantically different inputs."""
    vec_a = await client.embed("The capital of France is Paris.")
    vec_b = await client.embed("Quantum mechanics describes subatomic particles.")

    assert vec_a != vec_b


@pytest.mark.integration
@pytest.mark.anyio
async def test_embed_similar_texts_yield_similar_vectors(
    client: OpenAILLMClient,
) -> None:
    """embed() produces similar vectors for paraphrases of the same meaning."""
    vec_a = await client.embed("How do I cook pasta?")
    vec_b = await client.embed("What is the best way to prepare pasta?")

    # Cosine similarity should be meaningfully high for near-paraphrases
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = sum(a * a for a in vec_a) ** 0.5
    mag_b = sum(b * b for b in vec_b) ** 0.5
    cosine_sim = dot / (mag_a * mag_b)

    assert cosine_sim > 0.7, f"Expected cosine similarity > 0.7, got {cosine_sim:.3f}"


# ---------------------------------------------------------------------------
# vector_search() — OpenAI embeddings + Qdrant search (read-only)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.anyio
async def test_vector_search_roundtrip(
    client: OpenAILLMClient,
    qdrant_settings: Settings,
) -> None:
    """Vector search uses real embeddings and queries Qdrant without mutation."""
    qdrant_client, created = await _ensure_test_collection(qdrant_settings)

    vector_store = QdrantVectorStore(
        settings=qdrant_settings,
        collection_name=qdrant_settings.qdrant_collection,
        vector_size=qdrant_settings.embedding_dim,
    )
    retriever = HybridRetriever(
        graph_db=MagicMock(),
        vector_store=vector_store,
        llm_client=client,
        uuid_property=qdrant_settings.node_uuid_property,
    )

    result = await retriever.retrieve(
        "test query for vector search",
        {"action": "vector_search", "limit": 3},
    )

    assert result.success is True
    assert isinstance(result.data, list)

    if created:
        await qdrant_client.delete_collection(qdrant_settings.qdrant_collection)
    await qdrant_client.close()


@pytest.mark.integration
@pytest.mark.anyio
async def test_vector_tool_write_search_delete(
    client: OpenAILLMClient,
    qdrant_settings: Settings,
) -> None:
    """Vector tool can write, search, and delete within a test collection."""
    qdrant_client, created = await _ensure_test_collection(qdrant_settings)
    vector_store = QdrantVectorStore(
        settings=qdrant_settings,
        collection_name=qdrant_settings.qdrant_collection,
        vector_size=qdrant_settings.embedding_dim,
    )

    text_a = "A detective investigates a mysterious disappearance."
    text_b = "An astronaut explores a distant planet."
    embedding_a = await client.embed(text_a)
    embedding_b = await client.embed(text_b)
    id_a = str(uuid.uuid4())
    id_b = str(uuid.uuid4())

    await vector_store.upsert(
        id=id_a,
        embedding=embedding_a,
        payload={"uuid": id_a, "text": text_a},
    )
    await vector_store.upsert(
        id=id_b,
        embedding=embedding_b,
        payload={"uuid": id_b, "text": text_b},
    )

    hybrid_retriever = HybridRetriever(
        graph_db=MagicMock(),
        vector_store=vector_store,
        llm_client=client,
        uuid_property=qdrant_settings.node_uuid_property,
    )
    mock_cypher = MagicMock(spec=Retriever)
    tool_router = ToolRouter(mock_cypher, hybrid_retriever)

    tool_call = ToolCall(
        id="vector-1",
        name="vector_search",
        arguments={"query": text_a, "limit": 2},
    )
    result = await tool_router.route(tool_call)

    assert result["success"] is True
    returned_ids = {row.get("uuid") for row in result["data"]}
    assert id_a in returned_ids

    await qdrant_client.delete(
        collection_name=qdrant_settings.qdrant_collection,
        points_selector=models.PointIdsList(points=[id_a, id_b]),
        wait=True,
    )

    if created:
        await qdrant_client.delete_collection(qdrant_settings.qdrant_collection)
    await qdrant_client.close()
