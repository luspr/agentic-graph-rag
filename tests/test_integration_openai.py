"""Integration tests for OpenAILLMClient against the live OpenAI API.

These tests require a valid OPENAI_API_KEY environment variable (or .env file)
and consume real API credits.  They are excluded from the default test run.

Run explicitly:
    uv run pytest tests/test_integration_openai.py -m integration -v
"""

import os
from collections.abc import AsyncIterator

import pytest

from agentic_graph_rag.llm.base import ToolDefinition
from agentic_graph_rag.llm.openai_client import OpenAILLMClient

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
        {"role": "system", "content": "You are a robot. Every response must be exactly the single word 'beep'."},
        {"role": "user", "content": "Say something."},
    ]

    response = await client.complete(messages)

    assert response.content is not None
    assert "beep" in response.content.lower()


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
