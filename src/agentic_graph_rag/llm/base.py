from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """Represents a tool/function call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from LLM completion."""

    content: str | None
    tool_calls: list[ToolCall]
    usage: dict[str, int]
    finish_reason: str


@dataclass
class ToolDefinition:
    """Definition of a tool the LLM can call."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)  # JSON Schema


class LLMClient(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """Generate a completion with optional tool calling."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text."""
        ...
