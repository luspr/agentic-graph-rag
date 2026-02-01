# Agentic Graph RAG - Architecture

This document defines the architecture, interfaces, and design decisions for the Agentic Graph RAG system.

## Overview

The system allows an LLM to iteratively query a knowledge graph (Neo4j) until it reaches a satisfactory answer. It supports two retrieval strategies:
1. **Pure Cypher**: LLM generates and refines Cypher queries directly
2. **Hybrid**: Vector search (Qdrant) finds entry points, then graph traversal expands

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Terminal UI                                     │
│                         (prompt_toolkit + rich)                              │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Agent Controller                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌───────────────────┐  │
│  │ State Mgmt  │  │ Stop Criteria│  │  Reasoner   │  │ Iteration Tracker │  │
│  │ Session Mem │  │    Tracer    │  │             │  │                   │  │
│  └─────────────┘  └──────────────┘  └─────────────┘  └───────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
┌───────────────────────────┐    ┌───────────────────────────────────────────┐
│      Prompt Manager       │    │              LLM Client                    │
│  ┌─────────────────────┐  │    │  ┌─────────────┐  ┌────────────────────┐  │
│  │ Schema Formatter    │  │    │  │ Tool Calling│  │ Embedding Service  │  │
│  │ Context Builder     │  │    │  └─────────────┘  └────────────────────┘  │
│  │ Result Formatter    │  │    └───────────────────────────────────────────┘
│  └─────────────────────┘  │
└───────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
┌───────────────────────────┐    ┌───────────────────────────────────────────┐
│    Cypher Retriever       │    │           Hybrid Retriever                 │
│  ┌─────────────────────┐  │    │  ┌─────────────┐  ┌────────────────────┐  │
│  │ Query Generator     │  │    │  │ Vector Search│ │ Graph Expander     │  │
│  │ Query Executor      │  │    │  │ (Qdrant)     │ │ (Neo4j traversal)  │  │
│  │ Query Validator     │  │    │  └─────────────┘  └────────────────────┘  │
│  └─────────────────────┘  │    └───────────────────────────────────────────┘
└───────────────────────────┘
                    │                           │
                    └───────────┬───────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Data Layer                                         │
│            ┌──────────────────┐      ┌──────────────────┐                   │
│            │      Neo4j       │      │      Qdrant      │                   │
│            │  (Graph Store)   │      │  (Vector Store)  │                   │
│            └──────────────────┘      └──────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Interfaces

### 1. LLM Client Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    parameters: dict[str, Any]  # JSON Schema

class LLMClient(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate a completion with optional tool calling."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text."""
        ...
```

### 2. Graph Database Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class QueryResult:
    """Result from executing a Cypher query."""
    records: list[dict[str, Any]]
    summary: dict[str, Any]
    error: str | None = None

@dataclass
class NodeType:
    """Description of a node label in the schema."""
    label: str
    properties: dict[str, str]  # property_name -> type
    count: int

@dataclass
class RelationshipType:
    """Description of a relationship type in the schema."""
    type: str
    start_label: str
    end_label: str
    properties: dict[str, str]

@dataclass
class GraphSchema:
    """Schema information about the graph."""
    node_types: list[NodeType]
    relationship_types: list[RelationshipType]

class GraphDatabase(ABC):
    """Abstract interface for graph database operations."""

    @abstractmethod
    async def execute(self, cypher: str, params: dict[str, Any] | None = None) -> QueryResult:
        """Execute a Cypher query."""
        ...

    @abstractmethod
    async def get_schema(self) -> GraphSchema:
        """Retrieve the graph schema."""
        ...

    @abstractmethod
    async def validate_query(self, cypher: str) -> tuple[bool, str | None]:
        """Validate a Cypher query without executing. Returns (is_valid, error_message)."""
        ...
```

### 3. Vector Store Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class VectorSearchResult:
    """A single result from vector search."""
    id: str
    score: float
    payload: dict[str, Any]  # Node properties

class VectorStore(ABC):
    """Abstract interface for vector storage and search."""

    @abstractmethod
    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        ...

    @abstractmethod
    async def upsert(
        self,
        id: str,
        embedding: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Insert or update a vector."""
        ...
```

### 4. Retriever Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

class RetrievalStrategy(Enum):
    CYPHER = "cypher"
    HYBRID = "hybrid"

@dataclass
class RetrievalStep:
    """A single step in the retrieval process."""
    action: str  # e.g., "cypher_query", "vector_search", "graph_expand"
    input: dict[str, Any]
    output: dict[str, Any]
    error: str | None = None

@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    data: list[dict[str, Any]]
    steps: list[RetrievalStep]
    success: bool
    message: str

class Retriever(ABC):
    """Abstract interface for retrieval operations."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Execute a retrieval based on query."""
        ...

    @property
    @abstractmethod
    def strategy(self) -> RetrievalStrategy:
        """Return the retrieval strategy type."""
        ...
```

### 5. Prompt Manager Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class PromptContext:
    """Context for building prompts."""
    user_query: str
    schema: GraphSchema
    history: list[RetrievalStep]
    examples: list[dict[str, Any]] | None = None

class PromptManager(ABC):
    """Manages prompt construction and formatting."""

    @abstractmethod
    def build_system_prompt(self, schema: GraphSchema) -> str:
        """Build the system prompt with schema information."""
        ...

    @abstractmethod
    def build_retrieval_prompt(self, context: PromptContext) -> str:
        """Build prompt for retrieval iteration."""
        ...

    @abstractmethod
    def format_results(self, results: list[dict[str, Any]]) -> str:
        """Format query results for LLM consumption."""
        ...
```

### 6. Agent Controller Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class AgentStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    ERROR = "error"

@dataclass
class AgentState:
    """Current state of the agent."""
    iteration: int
    status: AgentStatus
    history: list[RetrievalStep]
    current_answer: str | None
    confidence: float | None  # LLM's self-assessed confidence

@dataclass
class AgentConfig:
    """Configuration for the agent."""
    max_iterations: int = 10
    strategy: RetrievalStrategy = RetrievalStrategy.CYPHER

@dataclass
class AgentResult:
    """Final result from agent execution."""
    answer: str
    status: AgentStatus
    iterations: int
    history: list[RetrievalStep]
    confidence: float | None

class AgentController(ABC):
    """Controls the agentic retrieval loop."""

    @abstractmethod
    async def run(self, user_query: str) -> AgentResult:
        """Run the agent until completion or max iterations."""
        ...

    @abstractmethod
    async def step(self) -> AgentState:
        """Execute a single iteration step."""
        ...

    @abstractmethod
    def should_stop(self, state: AgentState) -> bool:
        """Determine if the agent should stop iterating."""
        ...
```

### 7. Tracer Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass
class TraceEvent:
    """A single traced event."""
    timestamp: datetime
    event_type: str  # "tool_call", "llm_response", "error", etc.
    data: dict[str, Any]
    duration_ms: float | None = None

@dataclass
class Trace:
    """A complete trace for one agent run."""
    trace_id: str
    query: str
    started_at: datetime
    events: list[TraceEvent] = field(default_factory=list)
    completed_at: datetime | None = None
    result: AgentResult | None = None

class Tracer(ABC):
    """Interface for tracing agent execution."""

    @abstractmethod
    def start_trace(self, query: str) -> Trace:
        """Start a new trace for a query."""
        ...

    @abstractmethod
    def log_event(self, trace: Trace, event_type: str, data: dict[str, Any]) -> None:
        """Log an event to the trace."""
        ...

    @abstractmethod
    def end_trace(self, trace: Trace, result: AgentResult) -> None:
        """Complete a trace."""
        ...

    @abstractmethod
    def export(self, trace: Trace) -> dict[str, Any]:
        """Export trace as JSON-serializable dict."""
        ...
```

### 8. Session Memory Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SessionMessage:
    """A message in the session history."""
    role: str  # "user" or "assistant"
    content: str
    trace_id: str | None = None  # Link to detailed trace

@dataclass
class Session:
    """A user session with conversation history."""
    session_id: str
    messages: list[SessionMessage]
    created_at: datetime

class SessionManager(ABC):
    """Manages session state and conversation history."""

    @abstractmethod
    def create_session(self) -> Session:
        """Create a new session."""
        ...

    @abstractmethod
    def add_message(
        self, session: Session, role: str, content: str, trace_id: str | None = None
    ) -> None:
        """Add a message to the session."""
        ...

    @abstractmethod
    def get_context_messages(
        self, session: Session, max_messages: int = 10
    ) -> list[dict[str, str]]:
        """Get recent messages formatted for LLM context."""
        ...
```

---

## Tool Definitions

The agent has access to 4 tools via LLM function calling:

```python
AGENT_TOOLS = [
    ToolDefinition(
        name="execute_cypher",
        description="Execute a Cypher query against the Neo4j knowledge graph",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The Cypher query to execute"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of why this query will help answer the question"
                }
            },
            "required": ["query", "reasoning"]
        }
    ),
    ToolDefinition(
        name="vector_search",
        description="Search for nodes semantically similar to the given text",
        parameters={
            "type": "object",
            "properties": {
                "search_text": {
                    "type": "string",
                    "description": "Text to find similar nodes for"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 10
                }
            },
            "required": ["search_text"]
        }
    ),
    ToolDefinition(
        name="expand_node",
        description="Expand from a node to find connected nodes and relationships",
        parameters={
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "The ID of the node to expand from"
                },
                "relationship_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter to specific relationship types (optional)"
                },
                "depth": {
                    "type": "integer",
                    "description": "How many hops to traverse",
                    "default": 1
                }
            },
            "required": ["node_id"]
        }
    ),
    ToolDefinition(
        name="submit_answer",
        description="Submit the final answer when confident the question has been answered",
        parameters={
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer to the user's question"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence level from 0.0 to 1.0"
                },
                "supporting_evidence": {
                    "type": "string",
                    "description": "Summary of evidence from the graph that supports this answer"
                }
            },
            "required": ["answer", "confidence", "supporting_evidence"]
        }
    )
]
```

---

## Directory Structure

```
agentic-graph-rag/
├── src/
│   └── agentic_graph_rag/
│       ├── __init__.py
│       ├── main.py                    # Entry point
│       ├── config.py                  # Configuration management
│       │
│       ├── llm/                       # LLM Client layer
│       │   ├── __init__.py
│       │   ├── base.py                # LLMClient ABC
│       │   └── openai_client.py       # OpenAI implementation
│       │
│       ├── graph/                     # Graph database layer
│       │   ├── __init__.py
│       │   ├── base.py                # GraphDatabase ABC
│       │   └── neo4j_client.py        # Neo4j implementation
│       │
│       ├── vector/                    # Vector store layer
│       │   ├── __init__.py
│       │   ├── base.py                # VectorStore ABC
│       │   └── qdrant_client.py       # Qdrant implementation
│       │
│       ├── retriever/                 # Retrieval strategies
│       │   ├── __init__.py
│       │   ├── base.py                # Retriever ABC
│       │   ├── cypher_retriever.py    # Pure Cypher approach
│       │   └── hybrid_retriever.py    # Hybrid approach
│       │
│       ├── agent/                     # Agentic controller
│       │   ├── __init__.py
│       │   ├── controller.py          # AgentController
│       │   ├── tools.py               # Tool definitions
│       │   ├── state.py               # State management
│       │   ├── tracer.py              # Execution tracing
│       │   └── session.py             # Session/conversation memory
│       │
│       ├── prompts/                   # Prompt management
│       │   ├── __init__.py
│       │   ├── manager.py             # PromptManager
│       │   └── templates.py           # Prompt templates
│       │
│       └── ui/                        # Terminal UI
│           ├── __init__.py
│           └── terminal.py            # prompt_toolkit + rich UI
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures
│   ├── test_llm/
│   ├── test_graph/
│   ├── test_vector/
│   ├── test_retriever/
│   └── test_agent/
│
├── scripts/
│   └── load_movies_dataset.py         # Load the movies dataset
│
├── pyproject.toml
├── docker-compose.yaml
├── CLAUDE.md
├── ARCHITECTURE.md
├── TODO.md
└── README.md
```

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM Interaction | Tool/function calling | More reliable structured output, mature OpenAI support |
| Vector Store | Qdrant (separate from Neo4j) | Better performance, more features for vector search |
| Stop Criteria | LLM self-assessment + max iterations | Balance between autonomy and resource limits |
| Observability | Rich structured tracing | Essential for debugging agent reasoning |
| Session Memory | In-memory within session | Sufficient for PoC, enables multi-turn conversations |

---

## Flow: Agent Iteration Loop

```
1. User submits query
2. Agent starts trace
3. Build system prompt (with schema)
4. LOOP:
   a. Build retrieval prompt (with history)
   b. Send to LLM with tools
   c. Log LLM response to trace
   d. IF tool_call == "submit_answer":
      - Return answer, end trace
   e. ELSE:
      - Execute tool (cypher/vector_search/expand)
      - Log tool result to trace
      - Append to history
   f. IF iteration >= max_iterations:
      - Return partial answer, end trace
5. Update session with result
6. Display to user
```
