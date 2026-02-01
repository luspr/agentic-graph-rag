from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIConnectionError, RateLimitError

from agentic_graph_rag.llm import (
    LLMClient,
    OpenAIClientError,
    OpenAILLMClient,
    ToolCall,
    ToolDefinition,
)


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Create a mock AsyncOpenAI client."""
    return MagicMock()


@pytest.fixture
def client(mock_openai_client: MagicMock) -> OpenAILLMClient:
    """Create an OpenAILLMClient with a mocked AsyncOpenAI."""
    with patch(
        "agentic_graph_rag.llm.openai_client.AsyncOpenAI",
        return_value=mock_openai_client,
    ):
        return OpenAILLMClient(api_key="test-key", model="gpt-5.2")


def test_openai_client_implements_llm_client() -> None:
    """OpenAILLMClient is an instance of LLMClient."""
    with patch("agentic_graph_rag.llm.openai_client.AsyncOpenAI"):
        client = OpenAILLMClient(api_key="test-key")
        assert isinstance(client, LLMClient)


@pytest.mark.anyio
async def test_complete_basic_message(
    client: OpenAILLMClient, mock_openai_client: MagicMock
) -> None:
    """complete() returns a response for a basic message."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content="Hello!", tool_calls=None),
            finish_reason="stop",
        )
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )

    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

    messages = [{"role": "user", "content": "Hi"}]
    response = await client.complete(messages)

    assert response.content == "Hello!"
    assert response.tool_calls == []
    assert response.finish_reason == "stop"
    assert response.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


@pytest.mark.anyio
async def test_complete_with_tools(
    client: OpenAILLMClient, mock_openai_client: MagicMock
) -> None:
    """complete() parses tool calls from the response."""
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.function.name = "get_weather"
    mock_tool_call.function.arguments = '{"location": "NYC"}'

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content=None, tool_calls=[mock_tool_call]),
            finish_reason="tool_calls",
        )
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=20, completion_tokens=10, total_tokens=30
    )

    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

    tools = [
        ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        )
    ]

    messages = [{"role": "user", "content": "What's the weather in NYC?"}]
    response = await client.complete(messages, tools=tools)

    assert response.content is None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0] == ToolCall(
        id="call_123", name="get_weather", arguments={"location": "NYC"}
    )
    assert response.finish_reason == "tool_calls"


@pytest.mark.anyio
async def test_complete_sends_tools_in_correct_format(
    client: OpenAILLMClient, mock_openai_client: MagicMock
) -> None:
    """complete() converts ToolDefinition to OpenAI format."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="OK", tool_calls=None), finish_reason="stop")
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )

    mock_create = AsyncMock(return_value=mock_response)
    mock_openai_client.chat.completions.create = mock_create

    tools = [
        ToolDefinition(
            name="search",
            description="Search for documents",
            parameters={"type": "object", "properties": {}},
        )
    ]

    await client.complete([{"role": "user", "content": "test"}], tools=tools)

    call_kwargs = mock_create.call_args.kwargs
    assert "tools" in call_kwargs
    assert call_kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for documents",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


@pytest.mark.anyio
async def test_complete_omits_temperature_with_reasoning_effort(
    client: OpenAILLMClient, mock_openai_client: MagicMock
) -> None:
    """complete() omits temperature when reasoning effort is set."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="OK", tool_calls=None), finish_reason="stop")
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )

    mock_create = AsyncMock(return_value=mock_response)
    mock_openai_client.chat.completions.create = mock_create

    await client.complete(
        [{"role": "user", "content": "test"}],
        reasoning_effort="high",
        temperature=0.7,
    )

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["reasoning_effort"] == "high"
    assert "temperature" not in call_kwargs


@pytest.mark.anyio
async def test_complete_allows_temperature_with_reasoning_none(
    client: OpenAILLMClient, mock_openai_client: MagicMock
) -> None:
    """complete() keeps temperature when reasoning effort is none."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="OK", tool_calls=None), finish_reason="stop")
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )

    mock_create = AsyncMock(return_value=mock_response)
    mock_openai_client.chat.completions.create = mock_create

    await client.complete(
        [{"role": "user", "content": "test"}],
        reasoning_effort="none",
        temperature=0.3,
    )

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["reasoning_effort"] == "none"
    assert call_kwargs["temperature"] == 0.3


@pytest.mark.anyio
async def test_complete_handles_invalid_json_arguments(
    client: OpenAILLMClient, mock_openai_client: MagicMock
) -> None:
    """complete() handles malformed JSON in tool call arguments."""
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_456"
    mock_tool_call.function.name = "test_func"
    mock_tool_call.function.arguments = "not valid json"

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content=None, tool_calls=[mock_tool_call]),
            finish_reason="tool_calls",
        )
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )

    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

    response = await client.complete([{"role": "user", "content": "test"}])

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].arguments == {}


@pytest.mark.anyio
async def test_embed_returns_vector(
    client: OpenAILLMClient, mock_openai_client: MagicMock
) -> None:
    """embed() returns an embedding vector."""
    expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=expected_embedding)]

    mock_openai_client.embeddings.create = AsyncMock(return_value=mock_response)

    result = await client.embed("test text")

    assert result == expected_embedding
    mock_openai_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small", input="test text"
    )


@pytest.mark.anyio
async def test_embed_uses_custom_model() -> None:
    """embed() uses custom embedding model when specified."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2])]
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)

    with patch(
        "agentic_graph_rag.llm.openai_client.AsyncOpenAI", return_value=mock_client
    ):
        client = OpenAILLMClient(
            api_key="test-key", embedding_model="text-embedding-3-large"
        )

    await client.embed("test")

    mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-large", input="test"
    )


@pytest.mark.anyio
async def test_retry_on_rate_limit(
    client: OpenAILLMClient, mock_openai_client: MagicMock
) -> None:
    """complete() retries on rate limit errors with exponential backoff."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="OK", tool_calls=None), finish_reason="stop")
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )

    # Fail twice with rate limit, then succeed
    rate_limit_error = RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429),
        body=None,
    )
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=[rate_limit_error, rate_limit_error, mock_response]
    )

    # Patch sleep to avoid actual delays in tests
    with patch("asyncio.sleep", new_callable=AsyncMock):
        response = await client.complete([{"role": "user", "content": "test"}])

    assert response.content == "OK"
    assert mock_openai_client.chat.completions.create.call_count == 3


@pytest.mark.anyio
async def test_retry_on_connection_error(
    client: OpenAILLMClient, mock_openai_client: MagicMock
) -> None:
    """complete() retries on connection errors."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="OK", tool_calls=None), finish_reason="stop")
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )

    connection_error = APIConnectionError(request=MagicMock())
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=[connection_error, mock_response]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock):
        response = await client.complete([{"role": "user", "content": "test"}])

    assert response.content == "OK"
    assert mock_openai_client.chat.completions.create.call_count == 2


@pytest.mark.anyio
async def test_max_retries_exceeded(
    client: OpenAILLMClient, mock_openai_client: MagicMock
) -> None:
    """complete() raises OpenAIClientError after max retries."""
    rate_limit_error = RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429),
        body=None,
    )
    mock_openai_client.chat.completions.create = AsyncMock(side_effect=rate_limit_error)

    with (
        patch("asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(OpenAIClientError) as exc_info,
    ):
        await client.complete([{"role": "user", "content": "test"}])

    assert "failed after 5 retries" in str(exc_info.value)
    assert mock_openai_client.chat.completions.create.call_count == 5
