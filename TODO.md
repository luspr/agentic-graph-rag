# Agentic Graph RAG - Implementation Tasks

This document contains all implementation tasks for the Agentic Graph RAG system.
Each task includes a description, files to create/modify, and acceptance criteria.

See `ARCHITECTURE.md` for the full architecture design and interface definitions.

For each task, create a new feature branch (branching from default master), commit intermediate steps, push, and create a PR on github. Default branch is master. PR to that.

---

## Phase 1: Foundation (Core Infrastructure)

### Task 1.1: Set up project structure and dependencies

state: done

**Description:** Create the complete directory structure for the project and configure all required dependencies using uv.

**Acceptance Criteria:**
- [x] Directory structure matches the plan:
  ```
  src/agentic_graph_rag/
  ├── __init__.py
  ├── main.py
  ├── config.py
  ├── llm/
  ├── graph/
  ├── vector/
  ├── retriever/
  ├── agent/
  ├── prompts/
  └── ui/
  ```
- [x] pyproject.toml includes: neo4j, qdrant-client, openai, prompt_toolkit, rich, pydantic, pydantic-settings
- [x] Dev dependencies include: pytest, pytest-anyio, ruff
- [x] `uv sync` completes without errors
- [x] `pyrefly init` configured and `pyrefly check` passes on empty project
- [x] All `__init__.py` files created

---

### Task 1.2: Implement configuration management

state: done

**Description:** Create a centralized configuration system using Pydantic Settings to manage API keys, connection strings, and feature flags from environment variables.

**Files:** `src/agentic_graph_rag/config.py`

**Acceptance Criteria:**
- [x] `Settings` class using pydantic-settings with env file support (.env)
- [x] Configuration for: `OPENAI_API_KEY`, `OPENAI_MODEL` (default: gpt-5.2)
- [x] Configuration for: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- [x] Configuration for: `QDRANT_HOST`, `QDRANT_PORT`
- [x] Configuration for: `MAX_ITERATIONS` (default: 10)
- [x] Validation errors raised for missing required config
- [x] Unit test verifying config loads from env vars
- [x] `pyrefly check` passes

---

### Task 1.3: Implement Neo4j client

state: done

**Description:** Create an async Neo4j client implementing the GraphDatabase interface for query execution and schema introspection.

**Files:**
- `src/agentic_graph_rag/graph/base.py`
- `src/agentic_graph_rag/graph/neo4j_client.py`

**Acceptance Criteria:**
- [x] `GraphDatabase` ABC defined with: `execute()`, `get_schema()`, `validate_query()` methods
- [x] `QueryResult`, `NodeType`, `RelationshipType`, `GraphSchema` dataclasses defined
- [x] `Neo4jClient` class implements `GraphDatabase`
- [x] Connection pool management with async context manager
- [x] `execute()` runs Cypher and returns `QueryResult` with records and summary
- [x] `get_schema()` returns `GraphSchema` with node labels, relationship types, and properties
- [x] `validate_query()` uses EXPLAIN to check query syntax without executing
- [x] Proper error handling - captures Neo4j errors in `QueryResult.error`
- [x] Unit tests with mocked driver
- [x] `pyrefly check` passes
- [x] Fixed `_NODE_COUNTS_QUERY`: Neo4j 5.x drops variables after `WITH`; switched to
      `RETURN labels(n)[0], count(*)` which uses implicit grouping.
- [x] `db.schema.relationshipTypeProperties()` is unavailable on Neo4j community
      edition — existing error-handling path already degrades gracefully (relationships
      returned without properties).
- [x] Integration tests added (`tests/test_integration_neo4j.py`), opt-in via
      `uv run pytest -m integration`. Tests are read-only against the movies dataset.

---

### Task 1.4: Implement Qdrant client

**Description:** Create an async Qdrant client implementing the VectorStore interface for vector similarity search and upsert operations.

**Files:**
- `src/agentic_graph_rag/vector/base.py`
- `src/agentic_graph_rag/vector/qdrant_client.py`

**Acceptance Criteria:**
- [ ] `VectorStore` ABC defined with: `search()`, `upsert()` methods
- [ ] `VectorSearchResult` dataclass defined
- [ ] `QdrantVectorStore` class implements `VectorStore`
- [ ] `search()` performs vector similarity search, returns list of `VectorSearchResult`
- [ ] `search()` supports limit and optional filter parameters
- [ ] `upsert()` inserts or updates vectors with payload
- [ ] Collection auto-creation if it doesn't exist
- [ ] Unit tests with mocked Qdrant client
- [ ] `pyrefly check` passes

---

### Task 1.5: Implement OpenAI client

state: done

**Description:** Create an async OpenAI client implementing the LLMClient interface with support for chat completions with tool calling and embedding generation.

Refer to this for gpt-5.2: https://platform.openai.com/docs/guides/latest-model

**Files:**
- `src/agentic_graph_rag/llm/base.py`
- `src/agentic_graph_rag/llm/openai_client.py`

**Acceptance Criteria:**
- [x] `LLMClient` ABC defined with: `complete()`, `embed()` methods
- [x] `ToolCall`, `LLMResponse`, `ToolDefinition` dataclasses defined
- [x] `OpenAILLMClient` class implements `LLMClient`
- [x] `complete()` sends chat completion request with optional tools
- [x] `complete()` parses tool calls from response into `ToolCall` objects
- [x] `embed()` generates embeddings using text-embedding-3-small
- [x] Retry logic with exponential backoff for rate limits
- [x] Proper error handling for API errors
- [x] Unit tests with mocked OpenAI client
- [x] `pyrefly check` passes

---

## Phase 2: Retrieval Layer

### Task 2.1: Implement base retriever interfaces

state: done

**Description:** Define the abstract Retriever interface and all supporting dataclasses for retrieval operations.

**Files:** `src/agentic_graph_rag/retriever/base.py`

**Acceptance Criteria:**
- [x] `RetrievalStrategy` enum with CYPHER, HYBRID values
- [x] `RetrievalStep` dataclass: action, input, output, error fields
- [x] `RetrievalResult` dataclass: data, steps, success, message fields
- [x] `Retriever` ABC with: `retrieve()` method, `strategy` property
- [x] `pyrefly check` passes

---

### Task 2.2: Implement Cypher retriever

state: done

**Description:** Create a retriever that executes LLM-generated Cypher queries against Neo4j.

**Files:** `src/agentic_graph_rag/retriever/cypher_retriever.py`

**Acceptance Criteria:**
- [x] `CypherRetriever` class implements `Retriever`
- [x] Constructor takes `GraphDatabase` dependency
- [x] `retrieve()` executes provided Cypher query
- [x] Results formatted as list of dicts
- [x] Each query execution recorded as a `RetrievalStep`
- [x] Errors captured without raising exceptions
- [x] `strategy` property returns `RetrievalStrategy.CYPHER`
- [x] Unit tests with mocked GraphDatabase
- [x] `pyrefly check` passes

---

### Task 2.3: Implement Hybrid retriever

**Description:** Create a retriever that combines vector search for entry points with graph traversal for expansion.

**Files:** `src/agentic_graph_rag/retriever/hybrid_retriever.py`

**Acceptance Criteria:**
- [ ] `HybridRetriever` class implements `Retriever`
- [ ] Constructor takes `GraphDatabase`, `VectorStore`, `LLMClient` dependencies
- [ ] `retrieve()` supports two actions: vector_search and expand_node
- [ ] Vector search: generates embedding, queries Qdrant, returns matching nodes
- [ ] Expand node: given node ID, fetches connected nodes via Cypher
- [ ] Each action recorded as a `RetrievalStep`
- [ ] `strategy` property returns `RetrievalStrategy.HYBRID`
- [ ] Unit tests with mocked dependencies
- [ ] `pyrefly check` passes

---

## Phase 3: Agent Layer

### Task 3.1: Implement prompt manager

state: done

**Description:** Create a prompt manager that constructs system and user prompts with schema information, history, and results formatting.

**Files:**
- `src/agentic_graph_rag/prompts/manager.py`
- `src/agentic_graph_rag/prompts/templates.py`

**Acceptance Criteria:**
- [x] `PromptContext` dataclass defined
- [x] `PromptManager` class with: `build_system_prompt()`, `build_retrieval_prompt()`, `format_results()`
- [x] System prompt includes: graph schema (node types, relationships), available tools, instructions
- [x] Retrieval prompt includes: user query, previous steps history, current results
- [x] `format_results()` converts query results to readable text for LLM
- [x] Templates stored as constants in templates.py
- [x] Unit tests verifying prompt structure
- [x] `pyrefly check` passes

---

### Task 3.2: Implement tool definitions and handlers

state: done

**Description:** Define the 4 agent tools (execute_cypher, vector_search, expand_node, submit_answer) and create a tool router to dispatch calls.

**Files:** `src/agentic_graph_rag/agent/tools.py`

**Acceptance Criteria:**
- [x] `AGENT_TOOLS` list with 4 `ToolDefinition` objects:
  - `execute_cypher` - Execute Cypher query against Neo4j
  - `vector_search` - Search for semantically similar nodes
  - `expand_node` - Expand from a node to find connections
  - `submit_answer` - Submit final answer with confidence
- [x] `ToolRouter` class that dispatches tool calls to appropriate handlers
- [x] `execute_cypher` handler: validates and executes Cypher via retriever
- [x] `vector_search` handler: generates embedding and searches via retriever
- [x] `expand_node` handler: traverses graph from node via retriever
- [x] `submit_answer` handler: returns final answer with confidence
- [x] All handlers return dict results suitable for LLM consumption
- [x] Unit tests for each tool handler (20 tests covering all handlers and edge cases)
- [x] `pyrefly check` passes

---

### Task 3.3: Implement agent controller

state: done

**Description:** Create the main agent controller that runs the iterative retrieval loop until the LLM submits an answer or max iterations reached.

**Files:**
- `src/agentic_graph_rag/agent/controller.py`
- `src/agentic_graph_rag/agent/state.py`

**Acceptance Criteria:**
- [x] `AgentStatus` enum: RUNNING, COMPLETED, MAX_ITERATIONS, ERROR
- [x] `AgentState` dataclass: iteration, status, history, current_answer, confidence
- [x] `AgentConfig` dataclass: max_iterations, strategy
- [x] `AgentResult` dataclass: answer, status, iterations, history, confidence
- [x] `AgentController` class with: `run()`, `step()`, `should_stop()` methods
- [x] `run()` loops calling `step()` until `should_stop()` returns True
- [x] `step()` sends prompt to LLM, parses tool calls, executes tools, records history
- [x] `should_stop()` returns True if: submit_answer called, max_iterations reached, or error
- [x] Integrates with tracer to log events (via optional Tracer protocol)
- [x] Unit tests with mocked LLM responses (24 tests)
- [x] `pyrefly check` passes

---

### Task 3.4: Implement tracer

state: done

**Description:** Create a structured tracer for recording agent execution events for debugging and analysis.

**Files:** `src/agentic_graph_rag/agent/tracer.py`

**Acceptance Criteria:**
- [x] `TraceEvent` dataclass: timestamp, event_type, data, duration_ms
- [x] `Trace` dataclass: trace_id, query, started_at, events, completed_at, result
- [x] `Tracer` class with: `start_trace()`, `log_event()`, `end_trace()`, `export()`
- [x] Event types include: "query_start", "tool_call", "tool_result", "llm_request", "llm_response", "error", "complete"
- [x] `export()` returns JSON-serializable dict
- [x] Duration tracking for tool calls and LLM requests
- [x] Unit tests verifying trace structure (36 tests)
- [x] `pyrefly check` passes

---

### Task 3.5: Implement session manager

state: done

**Description:** Create a session manager to maintain conversation history across multiple user queries within a session.

**Files:** `src/agentic_graph_rag/agent/session.py`

**Acceptance Criteria:**
- [x] `SessionMessage` dataclass: role, content, trace_id
- [x] `Session` dataclass: session_id, messages, created_at
- [x] `SessionManager` class with: `create_session()`, `add_message()`, `get_context_messages()`
- [x] `create_session()` generates unique session ID
- [x] `add_message()` appends message to session history
- [x] `get_context_messages()` returns recent messages formatted for LLM (respects max_messages limit)
- [x] Sessions stored in memory (no persistence needed for PoC)
- [x] Unit tests for all methods (36 tests)
- [x] `pyrefly check` passes

---

## Phase 4: User Interface

### Task 4.1: Implement terminal UI

**Description:** Create an interactive terminal UI using prompt_toolkit for input and rich for formatted output.

**Files:**
- `src/agentic_graph_rag/ui/terminal.py`
- `src/agentic_graph_rag/main.py`

**Acceptance Criteria:**
- [ ] Main loop using prompt_toolkit for user input with history
- [ ] Rich console for formatted output
- [ ] Display agent iteration progress (e.g., "Iteration 2/10: Executing Cypher query...")
- [ ] Display final answer with confidence and supporting evidence
- [ ] Command to show trace details (e.g., `/trace`)
- [ ] Command to exit (e.g., `/quit` or Ctrl+C)
- [ ] Command to clear session (e.g., `/clear`)
- [ ] Graceful error handling with user-friendly messages
- [ ] Entry point in main.py that initializes components and starts UI
- [ ] `pyrefly check` passes

---

## Phase 5: Data & Testing

### Task 5.1: Create movies dataset loader script

**Description:** Create a script to load the Neo4j movies recommendation dataset and index node embeddings in Qdrant.

**Files:** `scripts/load_movies_dataset.py`

**Acceptance Criteria:**
- [ ] Script connects to Neo4j and loads movies dataset (built-in `:play movies` or neo4j-graph-examples/recommendations)
- [ ] Generates embeddings for Movie nodes (title + tagline)
- [ ] Generates embeddings for Person nodes (name + bio if available)
- [ ] Indexes embeddings in Qdrant with node properties as payload
- [ ] Payload includes: neo4j_id, label, name/title, other relevant properties
- [ ] Progress indicator during embedding/indexing
- [ ] Idempotent - can be re-run without duplicating data
- [ ] `uv run python scripts/load_movies_dataset.py` works end-to-end
- [ ] `pyrefly check` passes

---

### Task 5.2: Write unit tests

**Description:** Create comprehensive unit tests for all components with mocked dependencies.

**Files:** `tests/` directory

**Acceptance Criteria:**
- [ ] `tests/conftest.py` with shared fixtures (mock LLM, mock GraphDB, mock VectorStore)
- [ ] `tests/test_llm/test_openai_client.py` - tests for OpenAI client
- [ ] `tests/test_graph/test_neo4j_client.py` - tests for Neo4j client
- [ ] `tests/test_vector/test_qdrant_client.py` - tests for Qdrant client
- [ ] `tests/test_retriever/test_cypher_retriever.py` - tests for Cypher retriever
- [ ] `tests/test_retriever/test_hybrid_retriever.py` - tests for Hybrid retriever
- [ ] `tests/test_agent/test_controller.py` - tests for agent controller
- [ ] `tests/test_agent/test_tracer.py` - tests for tracer
- [ ] `tests/test_agent/test_session.py` - tests for session manager
- [ ] All tests pass: `uv run pytest`
- [ ] `pyrefly check` passes on test files

---

### Task 5.3: Write integration tests

**Description:** Create integration tests that verify end-to-end functionality with real or mocked external services.

**Files:** `tests/test_integration.py`

**Acceptance Criteria:**
- [ ] Test full agent loop with mocked LLM responses (predefined tool call sequence)
- [ ] Verify agent correctly executes multiple iterations
- [ ] Verify agent stops on submit_answer tool call
- [ ] Verify agent stops on max_iterations
- [ ] Verify trace captures all events correctly
- [ ] Verify session memory persists across queries
- [ ] All integration tests pass: `uv run pytest tests/test_integration.py`

---

## Verification Checklist

After completing all tasks, verify:

1. **Type checking:** `pyrefly check` passes
2. **Linting:** `uv run ruff check .` passes
3. **Formatting:** `uv run ruff format .` (no changes needed)
4. **Unit tests:** `uv run pytest` - all pass
5. **Integration test:**
   - Start services: `docker compose up -d`
   - Load data: `uv run python scripts/load_movies_dataset.py`
   - Run app: `uv run python -m agentic_graph_rag.main`
   - Test queries work end-to-end
