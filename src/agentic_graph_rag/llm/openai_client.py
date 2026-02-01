import asyncio
import json
import logging
from typing import Any

from openai import APIConnectionError, AsyncOpenAI, RateLimitError

from agentic_graph_rag.llm.base import LLMClient, LLMResponse, ToolCall, ToolDefinition

logger = logging.getLogger(__name__)


class OpenAILLMClient(LLMClient):
    """OpenAI implementation of LLMClient with GPT-5.2 support."""

    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
    MAX_RETRIES = 5
    BASE_DELAY = 1.0  # Base delay in seconds for exponential backoff

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.2",
        embedding_model: str | None = None,
    ) -> None:
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key.
            model: Model to use for completions (default: gpt-5.2).
            embedding_model: Model to use for embeddings (default: text-embedding-3-small).
        """
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._embedding_model = embedding_model or self.DEFAULT_EMBEDDING_MODEL

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate a completion with optional tool calling.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            tools: Optional list of tool definitions the model can call.
            temperature: Sampling temperature (0.0 for deterministic).

        Returns:
            LLMResponse containing the model's response.

        Raises:
            OpenAIClientError: If the API request fails after retries.
        """
        openai_tools = self._convert_tools(tools) if tools else None

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }

        if openai_tools:
            kwargs["tools"] = openai_tools

        response = await self._request_with_retry(
            self._client.chat.completions.create, **kwargs
        )

        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": (
                    response.usage.completion_tokens if response.usage else 0
                ),
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            finish_reason=choice.finish_reason or "unknown",
        )

    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text.

        Args:
            text: Text to generate embedding for.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            OpenAIClientError: If the API request fails after retries.
        """
        response = await self._request_with_retry(
            self._client.embeddings.create,
            model=self._embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _convert_tools(
        self, tools: list[ToolDefinition]
    ) -> list[dict[str, Any]]:
        """Convert ToolDefinition list to OpenAI tools format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    async def _request_with_retry[T](
        self,
        func: Any,
        **kwargs: Any,
    ) -> T:
        """Execute an API request with exponential backoff retry.

        Args:
            func: Async function to call.
            **kwargs: Arguments to pass to the function.

        Returns:
            The result of the function call.

        Raises:
            OpenAIClientError: If all retries are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(self.MAX_RETRIES):
            try:
                return await func(**kwargs)
            except RateLimitError as e:
                last_error = e
                delay = self.BASE_DELAY * (2**attempt)
                logger.warning(
                    "Rate limited (attempt %d/%d), retrying in %.1f seconds",
                    attempt + 1,
                    self.MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
            except APIConnectionError as e:
                last_error = e
                delay = self.BASE_DELAY * (2**attempt)
                logger.warning(
                    "Connection error (attempt %d/%d), retrying in %.1f seconds: %s",
                    attempt + 1,
                    self.MAX_RETRIES,
                    delay,
                    str(e),
                )
                await asyncio.sleep(delay)

        raise OpenAIClientError(
            f"Request failed after {self.MAX_RETRIES} retries"
        ) from last_error


class OpenAIClientError(Exception):
    """Exception raised when OpenAI API requests fail."""

    pass
