# Agentic Graph RAG - Implementation Tasks

This document contains all implementation tasks for the Agentic Graph RAG system.
Each task includes a description, files to create/modify, and acceptance criteria.

See `ARCHITECTURE.md` for the full architecture design and interface definitions.

when you start working on a task, create a new feature branch (branching from default master) commit intermediate steps, push, and once you're finished (tests pass! pre-commit successful), create a PR on github. Default branch is master. PR to that.

## Verification Checklist

After completing all tasks, verify:

**Unit tests:** `uv run pytest` - all pass

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
- [x] Fixed `_NODE_COUNTS_QUERY`: Neo4j 5.x drops variables after `WITH`; switched to
      `RETURN labels(n)[0], count(*)` which uses implicit grouping.
- [x] `db.schema.relationshipTypeProperties()` is unavailable on Neo4j community
      edition — existing error-handling path already degrades gracefully (relationships
      returned without properties).
- [x] Integration tests added (`tests/test_integration_neo4j.py`), opt-in via
      `uv run pytest -m integration`. Tests are read-only against the movies dataset.

---

### Task 1.4: Implement Qdrant client

state: done

**Description:** Create an async Qdrant client implementing the VectorStore interface for vector similarity search and upsert operations.

**Files:**
- `src/agentic_graph_rag/vector/base.py`
- `src/agentic_graph_rag/vector/qdrant_client.py`

**Acceptance Criteria:**
- [x] `VectorStore` ABC defined with: `search()`, `upsert()` methods
- [x] `VectorSearchResult` dataclass defined
- [x] `QdrantVectorStore` class implements `VectorStore`
- [x] `search()` performs vector similarity search, returns list of `VectorSearchResult`
- [x] `search()` supports limit and optional filter parameters
- [x] `upsert()` inserts or updates vectors with payload
- [x] Collection auto-creation if it doesn't exist
- [x] Unit tests with mocked Qdrant client

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

---

### Task 2.3: Implement Hybrid retriever

state: done

**Description:** Create a retriever that combines vector search for entry points with graph traversal for expansion.

**Files:** `src/agentic_graph_rag/retriever/hybrid_retriever.py`

**Acceptance Criteria:**
- [x] `HybridRetriever` class implements `Retriever`
- [x] Constructor takes `GraphDatabase`, `VectorStore`, `LLMClient` dependencies
- [x] `retrieve()` supports two actions: vector_search and expand_node
- [x] Vector search: generates embedding, queries Qdrant, returns matching nodes
- [x] Expand node: given node ID, fetches connected nodes via Cypher
- [x] Each action recorded as a `RetrievalStep`
- [x] `strategy` property returns `RetrievalStrategy.HYBRID`
- [x] Unit tests with mocked dependencies

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

---

### Task 3.6: Persist traces to JSONL with full prompt capture

state: done

**Description:** Persist agent traces to an append-only JSON Lines log (one file per run) and include full prompts/messages for each LLM request (including the system prompt).

**Files:**
- `src/agentic_graph_rag/agent/tracer.py`
- `src/agentic_graph_rag/agent/controller.py`
- `src/agentic_graph_rag/config.py`
- `src/agentic_graph_rag/ui/terminal.py`

**Acceptance Criteria:**
- [x] Trace logging writes one JSON line per event to a run-scoped JSONL file (append-only)
- [x] Each event line includes: `run_id`, `trace_id`, `event_type`, `timestamp`, `data`, and optional `duration_ms`
- [x] LLM request events include the full message list sent to the LLM (system + user + tool messages)
- [x] System prompt is included verbatim in the logged messages
- [x] Trace file path is configurable via env/config (e.g., output dir + timestamped filename)
- [x] Logging can be enabled/disabled via config (default: enabled for terminal UI)
- [x] No in-memory trace data loss if file write fails (graceful fallback)
- [x] Unit tests cover JSONL serialization and prompt/message capture

---

## Phase 4: User Interface

### Task 4.1: Implement terminal UI

state: done

**Description:** Create an interactive terminal UI using prompt_toolkit for input and rich for formatted output.

**Files:**
- `src/agentic_graph_rag/ui/terminal.py`
- `src/agentic_graph_rag/main.py`

**Acceptance Criteria:**
- [x] Main loop using prompt_toolkit for user input with history
- [x] Rich console for formatted output
- [x] Display agent iteration progress (e.g., "Iteration 2/10: Executing Cypher query...")
- [x] Display final answer with confidence and supporting evidence
- [x] Command to show trace details (e.g., `/trace`)
- [x] Command to exit (e.g., `/quit` or Ctrl+C)
- [x] Command to clear session (e.g., `/clear`)
- [x] Graceful error handling with user-friendly messages
- [x] Entry point in main.py that initializes components and starts UI
- [x] Unit tests for TerminalUI and UITracer (41 tests)

---

### Task 4.2: Add interactive trace inspector (terminal UI)

state: done

**Description:** Add an interactive trace inspector to the terminal UI that lets users toggle focus to the trace list, navigate events, and open a full-detail view for the selected event.

**Files:**
- `src/agentic_graph_rag/ui/terminal.py`

**Acceptance Criteria:**
- [x] When trace view is focused, Up/Down or `k/j` moves selection through events
- [x] Selected trace row is visually highlighted
- [x] Enter opens a detail pane for the selected event
- [x] Detail pane shows full, untruncated event data: event_type, timestamp, duration, data payload
- [x] Data, including dicts must be rendered nicely: Easily to read and visually pleasant for a human
- [x] Full prompt strings, queries, tool arguments, and results are visible in the detail pane
- [x] Esc closes the detail pane and returns to the trace list, indicate how to close on the bottom of the screen
- [x] `/trace` overview remains truncated in the list, full fidelity only in detail view

**Note:** The Shift+Tab toggle between chat input and trace view was replaced with a simpler design: `/trace` command opens a standalone interactive trace inspector that can be navigated with j/k arrows and exited with q/Esc. This provides a cleaner UX without complex focus management.

---

### Task 4.3: Support multi-turn chat context in the agent

state: done 
**Description:** Persist and inject recent user/assistant messages into the LLM context so follow-up queries are handled as a conversation rather than standalone requests.

**Files:**
- `src/agentic_graph_rag/agent/session.py`
- `src/agentic_graph_rag/agent/controller.py`
- `src/agentic_graph_rag/ui/terminal.py`
- `src/agentic_graph_rag/config.py`

**Acceptance Criteria:**
- [x] Recent session messages (user + assistant) are included in LLM requests
- [x] System prompt remains the first message; chat history is appended before the retrieval prompt
- [x] Configurable max history length (count or token-based)
- [x] Ability to clear conversation context with `/clear`
- [x] No regression for single-turn behavior if history length is set to 0
- [x] Unit tests verify multi-turn context inclusion and ordering

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

## Phase 6: Improvements (Follow-up)

### Task 6.1: Enrich HybridRetriever expand_node output

state: done

**Description:** Improve the hybrid expand query to preserve more graph structure and context for the LLM.

**Current behavior (reference):**
- `src/agentic_graph_rag/retriever/hybrid_retriever.py` uses an undirected pattern:
  - `MATCH (start)-[rels*1..{depth}]-(node)`
- Returns only node + labels + list of relationship types:
  - `RETURN DISTINCT node, node.{uuid} AS node_uuid, labels(node) AS node_labels, [rel IN rels | type(rel)] AS relationship_types`
- This collapses multiple paths into a single node, loses directionality, and does not include path length or edge identity.

Below is a detailed explanation with code references and a comparison to the Cypher retriever.

  Where this is implemented

  - Hybrid retriever expand query: src/agentic_graph_rag/retriever/hybrid_retriever.py
  - Cypher retriever: src/agentic_graph_rag/retriever/cypher_retriever.py
  - Tool entrypoint for expand: src/agentic_graph_rag/agent/tools.py

  ———

  Current expand_node behavior (Hybrid retriever)

  In src/agentic_graph_rag/retriever/hybrid_retriever.py, expand_node builds this query (simplified):

  MATCH (start {uuid: $uuid})
  MATCH (start)-[rels*1..$depth]-(node)
  WHERE ALL(rel IN rels WHERE type(rel) IN $relationship_types)
  RETURN DISTINCT node,
                  node.uuid AS node_uuid,

  Key behaviors:

  1. Undirected traversal
  2. No explicit path returned
      - The query doesn’t return the path, only a list of relationship types.
      - You can’t reconstruct the exact chain of nodes/edges that led to the node.
  3. No path length returned
      - The LLM cannot see whether a returned node was 1 hop away or 3 hops away.
      - That matters for confidence or for deciding whether to expand further.
  4. No edge IDs or relationship identity
      - Only relationship types are returned.
      - If there are multiple relationship types or multiple paths to the same node, the output collapses to DISTINCT
        node and you lose which specific path was used.
      - The list of types can be ambiguous when there are multiple paths (only one path’s rels might be represented).
  5. Start node not returned
      - The output includes the expanded node and metadata but not the start node or its UUID.
      - That makes it harder to present the result as a structured graph step in the trace or in the LLM prompt.

  These outputs are what the LLM sees in the prompt because PromptManager.format_results(...) formats the returned list
  of records. The LLM has no richer structure beyond what the query returns.


**Planned improvements:**
- Return richer structure per **path**, not just per node. Minimum schema:
  - `start_uuid` (uuid of the start node)
  - `node_uuid` (uuid of the expanded node)
  - `node_labels`
  - `relationship_types` (ordered)
  - `path_length`
- Add a **path payload** to preserve structure:
  - `path_nodes`: ordered list of `{uuid, labels}` (optionally name/title if present)
  - `path_rels`: ordered list of `{type, direction, rel_id}` or `{type, from_uuid, to_uuid}`
- Preserve **directionality**:
  - Use directed traversal (e.g., `-[:TYPE*]->`) or capture direction per rel in output.
  - If the tool remains undirected, include a `direction` field per relationship to reconstruct.
- Avoid collapsing multiple paths into a single node:
  - Prefer returning one record per path (`path`), optionally with a `max_paths` limit.
  - If deduplication is needed, dedupe on `(node_uuid, path_length, relationship_types)` not just node.
- Optional controls to add to tool args:
  - `direction`: `"out" | "in" | "both"` (default `"both"` if not specified)
  - `max_paths`: cap the number of returned paths (default reasonable, e.g. 20)
  - `include_properties`: bool to include select node properties in path nodes
- Suggested Cypher shape (example):
  - `MATCH p=(start {uuid: $uuid})-[rels*1..$depth]->(node)`
  - `RETURN p, length(p) AS path_length, ...`
  - Derive `path_nodes` + `path_rels` from `nodes(p)` and `relationships(p)` for output.

**Files:**
- `src/agentic_graph_rag/retriever/hybrid_retriever.py`
- `src/agentic_graph_rag/agent/tools.py` (if output schema changes)
- `src/agentic_graph_rag/prompts/manager.py` (if result formatting needs updates)
- `tests/test_retriever/test_hybrid_retriever.py`
- `tests/test_agent/test_tools.py` (if output schema changes)

**Acceptance Criteria:**
- [x] `expand_node` returns one record per path with `start_uuid`, `node_uuid`, and `path_length`.
- [x] Directionality is preserved (directed pattern or explicit direction metadata).
- [x] Returned data includes ordered `path_nodes` and `path_rels` (or equivalent).
- [x] If `max_paths` / `direction` are added, they are wired into tool args and tested.
- [x] Unit tests validate path ordering, direction, and length.

### Task 6.2: Add Qdrant ingestion/indexing pipeline

state: todo

**Description:** Implement a dataset-aware ingestion pipeline that generates embeddings and upserts vectors into Qdrant. This is required to make `vector_search` useful beyond external/hand-built indexes.

**Current behavior (reference):**
- Vector search exists (`HybridRetriever`) but there is no pipeline to populate Qdrant.
- Only UUID is added during dataset load (`src/agentic_graph_rag/scripts/load_sr_rag_datasets.py`).

**Planned improvements:**
- Add a script (or scripts) that:
  - reads nodes from Neo4j (by label and field selection),
  - builds text for embedding (e.g., `title + tagline` for movies),
  - generates embeddings via `OpenAILLMClient.embed`,
  - upserts vectors into Qdrant with payload containing:
    - `uuid`,
    - label,
    - key properties (name/title),
    - any fields needed for filters.
- Make the script idempotent (re-runs do not duplicate).
- Add progress reporting and basic stats (count, rate).

**Files:**
- `scripts/index_<dataset>.py` (new)
- `src/agentic_graph_rag/llm/openai_client.py` (reuse for embeddings)
- `src/agentic_graph_rag/vector/qdrant_client.py`
- `README.md` (usage documentation)

**Acceptance Criteria:**
- [ ] Script can embed and upsert nodes for at least one dataset.
- [ ] Payload includes `uuid` and label in every Qdrant point.
- [ ] Script is idempotent and reports progress.
- [ ] Documented run command in README.

### Task 6.3: Improve result formatting for vector search

state: todo

**Description:** The LLM currently receives raw vector hits with payloads; this can be too verbose and inconsistent.

**Current behavior (reference):**
- `PromptManager.format_results()` is used for all results, including vector hits.
- Vector payloads can be large and are not summarized.

**Planned improvements:**
- Add a dedicated formatter for vector search results:
  - show `uuid`, `score`, and a short payload summary,
  - truncate payload fields or large lists,
  - prefer consistent top-k summary lines.
- Keep output stable for LLM consumption.

**Files:**
- `src/agentic_graph_rag/prompts/manager.py`
- `src/agentic_graph_rag/agent/controller.py` (if routing needs metadata)
- `tests/test_prompts/test_manager.py`

**Acceptance Criteria:**
- [ ] Vector search results are formatted with a compact, consistent schema.
- [ ] Large payloads are truncated or summarized.
- [ ] Unit tests cover vector formatting.

### Task 6.4: Strengthen config validation

state: todo

**Description:** Validate vector config values early to avoid runtime surprises.

**Current behavior (reference):**
- `Settings` does not validate `EMBEDDING_DIM`, `QDRANT_COLLECTION`, or `NODE_UUID_PROPERTY`.
- `HybridRetriever` validates UUID property, but too late.

**Planned improvements:**
- Add Pydantic validators to enforce:
  - `embedding_dim > 0`,
  - `qdrant_collection` non-empty,
  - `node_uuid_property` matches Neo4j token pattern.
- Fail fast with clear errors.

**Files:**
- `src/agentic_graph_rag/config.py`
- `tests/test_config.py`

**Acceptance Criteria:**
- [ ] Invalid vector config values raise ValidationError with clear messages.
- [ ] Unit tests cover invalid/valid cases.

### Task 6.5: Verify Qdrant collection compatibility at runtime

state: todo

**Description:** Ensure collection vector size matches configured `embedding_dim` and avoid silent mismatches.

**Current behavior (reference):**
- `QdrantVectorStore` auto-creates if missing but does not verify size on connect.

**Planned improvements:**
- On first use, fetch collection info and validate size.
- If size mismatch, raise a clear error (do not proceed).

**Files:**
- `src/agentic_graph_rag/vector/qdrant_client.py`
- `tests/test_vector/test_qdrant_client.py`

**Acceptance Criteria:**
- [ ] Size mismatches raise a descriptive error.
- [ ] Unit tests cover both valid and invalid size cases.

### Task 6.6: Stabilize pre-commit hook execution in CI/dev

state: todo

**Description:** Commits can fail when pre-commit cannot write its cache (read-only filesystem).

**Current behavior (reference):**
- `git commit` failed due to pre-commit cache permissions.

**Planned improvements:**
- Document or set `PRE_COMMIT_HOME` to a writable location (e.g., `.cache/pre-commit` under repo).
- Optionally add a troubleshooting note to `README.md`.

**Files:**
- `README.md` (documentation)
- Optional: `.env` template or developer docs

**Acceptance Criteria:**
- [ ] Pre-commit cache location guidance is documented.
- [ ] Developers can run hooks without cache permission errors.

---

## Phase 7: SOTA Hybrid Agentic Retrieval (Vector + Graph)

### Task 7.1: Extend retrieval result model with hybrid artifacts

state: todo

**Context / Background:**
- `RetrievalResult` currently carries `data`, `steps`, `success`, and `message`.
- Hybrid upgrades need structured metadata for ranking, coverage checks, and traceability.

**Task Description:**
- Extend retrieval models to support optional hybrid artifacts:
  - seed provenance,
  - per-candidate scores,
  - coverage summary,
  - contradiction flags.
- Keep backward compatibility for existing call sites and tests.

**Files:**
- `src/agentic_graph_rag/retriever/base.py`
- `tests/test_retriever/test_cypher_retriever.py`
- `tests/test_retriever/test_hybrid_retriever.py`

**Acceptance Criteria:**
- [ ] `RetrievalResult` supports optional artifact metadata fields.
- [ ] Existing retrievers work without populating new fields.
- [ ] Unit tests validate backward compatibility and artifact behavior.

### Task 7.2: Add configuration knobs for advanced hybrid retrieval

state: todo

**Context / Background:**
- Advanced hybrid behavior needs runtime controls for planner usage, graph rerank backend,
  and retrieval budgets.
- Current settings do not expose these controls.

**Task Description:**
- Add settings for:
  - planner enablement,
  - graph rerank backend (`gds` or fallback),
  - frontier/path/token budgets.
- Add validation with clear error messages and defaults.

**Files:**
- `src/agentic_graph_rag/config.py`
- `tests/test_config.py`
- `README.md`

**Acceptance Criteria:**
- [ ] New settings exist with sane defaults.
- [ ] Invalid values raise clear validation errors.
- [ ] README documents all new settings.

### Task 7.3: Implement multi-query vector seeding with rank fusion

state: done

**Context / Background:**
- Current hybrid retrieval uses one query embedding and one vector search call.
- Multi-query seeding improves recall for ambiguous and compositional questions.

**Task Description:**
- Add query variant generation in `HybridRetriever`:
  - original query,
  - entity-focused rewrite,
  - hypothesis-style rewrite.
- Run vector search for each variant and fuse results with Reciprocal Rank Fusion (RRF).
- Track provenance of which variant retrieved each seed.

**Files:**
- `src/agentic_graph_rag/retriever/hybrid_retriever.py`
- `tests/test_retriever/test_hybrid_retriever.py`

**Acceptance Criteria:**
- [x] Hybrid retrieval executes vector search across multiple query variants.
- [x] Fused ranking is deterministic for fixed inputs.
- [x] Returned seeds include query-variant provenance metadata.

### Task 7.4: Return path-level evidence from graph expansion

state: todo

**Context / Background:**
- Current `expand_node` output is node-collapsed and loses path detail.
- Path-level evidence is required for stronger reasoning and verification.

**Task Description:**
- Update expansion logic to return one record per path with:
  - `start_uuid`,
  - `end_uuid`,
  - `path_length`,
  - ordered `path_nodes`,
  - ordered `path_rels` with direction metadata.
- Add optional controls:
  - `direction`,
  - `max_paths`,
  - `max_branching`.

**Files:**
- `src/agentic_graph_rag/retriever/hybrid_retriever.py`
- `src/agentic_graph_rag/agent/tools.py`
- `tests/test_retriever/test_hybrid_retriever.py`
- `tests/test_agent/test_tools.py`

**Acceptance Criteria:**
- [ ] Expansion returns path-level records with ordered nodes/relationships.
- [ ] Directionality is preserved in returned evidence.
- [ ] Optional traversal controls are wired and tested.
- [ ] Existing calls without new args remain valid.

### Task 7.5: Add hybrid score blending for candidate ranking

state: todo

**Context / Background:**
- Current ranking is dominated by raw vector similarity.
- Better hybrid quality requires graph-aware ranking signals.

**Task Description:**
- Add a ranking blend combining:
  - vector relevance,
  - graph/path quality,
  - relation priors.
- Include simple penalties/bonuses for hubs, novelty, and path length.
- Keep weights configurable via constants/settings.

**Files:**
- `src/agentic_graph_rag/retriever/hybrid_retriever.py`
- `tests/test_retriever/test_hybrid_retriever.py`

**Acceptance Criteria:**
- [ ] Ranking uses blended hybrid scores, not vector score alone.
- [ ] Ranking output is deterministic for fixed input.
- [ ] Unit tests cover scoring behavior and edge cases.

### Task 7.6: Add compact hybrid-specific prompt/result formatting

state: todo

**Context / Background:**
- Generic result formatting is too verbose for hybrid vector/path outputs.
- Prompt quality and cost improve with compact, stable evidence packaging.

**Task Description:**
- Add hybrid-specific formatting in prompt manager:
  - top seeds with score + provenance,
  - top paths with evidence IDs,
  - compact coverage summary.
- Add deterministic truncation for large payloads.

**Files:**
- `src/agentic_graph_rag/prompts/manager.py`
- `src/agentic_graph_rag/agent/controller.py`
- `tests/test_prompts/test_manager.py`
- `tests/test_agent/test_controller.py`

**Acceptance Criteria:**
- [ ] Hybrid results are formatted in a compact, consistent structure.
- [ ] Large payloads are truncated deterministically.
- [ ] Tests validate stability of formatted output.

### Task 7.7: Add optional retrieval strategy `hybrid_planned`

state: todo

**Context / Background:**
- We want planner guidance as an optional strategy while keeping the existing loop model.
- Strategy plumbing must remain backward compatible.

**Task Description:**
- Add `hybrid_planned` as a strategy value in retrieval/config/runner/CLI flow.
- Keep the same iterative tool-calling execution loop in controller.

**Files:**
- `src/agentic_graph_rag/retriever/base.py`
- `src/agentic_graph_rag/agent/state.py`
- `src/agentic_graph_rag/runner.py`
- `src/agentic_graph_rag/cli.py`
- `tests/test_agent/test_controller.py`
- `tests/test_cli.py`

**Acceptance Criteria:**
- [ ] `hybrid_planned` is selectable via existing configuration/CLI surfaces.
- [ ] Existing `cypher` and `hybrid` behavior is unchanged.
- [ ] Controller still uses the same iterative loop pattern.

### Task 7.8: Add light planner intents for hybrid planned mode

state: todo

**Context / Background:**
- Planner-style decomposition improves retrieval direction.
- We explicitly do not want a full DAG executor in this phase.

**Task Description:**
- Implement a light planner that emits retrieval intents:
  - objective,
  - constraints,
  - seed hints,
  - relation hints,
  - stop hints.
- Feed intents into retrieval context and tool selection guidance only.

**Files:**
- `src/agentic_graph_rag/retriever/hybrid_retriever.py`
- `src/agentic_graph_rag/agent/controller.py`
- `src/agentic_graph_rag/prompts/manager.py`
- `tests/test_retriever/test_hybrid_retriever.py`
- `tests/test_agent/test_controller.py`

**Acceptance Criteria:**
- [ ] Planner emits structured intents in `hybrid_planned` mode.
- [ ] Intents influence retrieval behavior/context.
- [ ] No DAG executor is introduced.

### Task 7.9: Add coverage and contradiction checks between iterations

state: todo

**Context / Background:**
- The agent can currently answer with partial or conflicting evidence.
- Iterative quality gates are required for better faithfulness.

**Task Description:**
- Add per-iteration checks for:
  - constraint coverage completeness,
  - evidence contradictions/conflicts.
- Produce structured signals for retry/backtrack/focused expansion.

**Files:**
- `src/agentic_graph_rag/agent/controller.py`
- `src/agentic_graph_rag/retriever/hybrid_retriever.py`
- `src/agentic_graph_rag/agent/state.py`
- `tests/test_agent/test_controller.py`
- `tests/test_retriever/test_hybrid_retriever.py`

**Acceptance Criteria:**
- [ ] Coverage status is computed and exposed per iteration.
- [ ] Contradiction signals are surfaced in state/retrieval artifacts.
- [ ] Tests validate retry/backtrack recommendations.

### Task 7.10: Add graph reranking service (GDS with fallback)

state: todo

**Context / Background:**
- Graph ranking from vector seeds is a major quality lever.
- Preferred backend is Neo4j GDS; fallback is needed for portability.

**Task Description:**
- Add reranking pipeline:
  - use GDS Personalized PageRank when available,
  - fallback to Cypher heuristic ranking when unavailable.
- Integrate reranked frontier into expansion order.

**Files:**
- `src/agentic_graph_rag/retriever/hybrid_retriever.py`
- `src/agentic_graph_rag/graph/neo4j_client.py`
- `src/agentic_graph_rag/config.py`
- `tests/test_retriever/test_hybrid_retriever.py`
- `tests/test_graph/test_neo4j_client.py`

**Acceptance Criteria:**
- [ ] GDS-backed reranking works when available.
- [ ] Fallback path works when GDS is unavailable.
- [ ] Backend selection and fallback behavior are tested.

### Task 7.11: Enforce frontier and token budgets in hybrid retrieval

state: todo

**Context / Background:**
- Retrieval needs explicit budget controls to contain latency and token cost.
- Current flow does not enforce frontier/path/evidence budgets consistently.

**Task Description:**
- Enforce configurable limits for:
  - frontier nodes,
  - total paths,
  - prompt token budget for evidence.
- Prioritize high-value evidence under budget.

**Files:**
- `src/agentic_graph_rag/retriever/hybrid_retriever.py`
- `src/agentic_graph_rag/prompts/manager.py`
- `src/agentic_graph_rag/config.py`
- `tests/test_retriever/test_hybrid_retriever.py`
- `tests/test_prompts/test_manager.py`

**Acceptance Criteria:**
- [ ] Hybrid retrieval enforces configured frontier/path limits.
- [ ] Evidence packaging respects configured prompt budget.
- [ ] Truncation/prioritization order is deterministic and tested.

### Task 7.12: Extend eval with retrieval-quality metrics

state: todo

**Context / Background:**
- Current evaluation is answer-centric.
- Retrieval upgrades require retrieval-centric KPIs for iteration decisions.

**Task Description:**
- Add retrieval metrics:
  - seed recall@k,
  - path evidence precision,
  - constraint coverage rate,
  - iterations-to-convergence.
- Include metrics in per-example outputs and summary aggregates.

**Files:**
- `src/agentic_graph_rag/eval/types.py`
- `src/agentic_graph_rag/eval/metrics.py`
- `src/agentic_graph_rag/eval/runner.py`
- `tests/test_eval/test_metrics.py`
- `tests/test_eval/test_runner.py`
- `README.md`

**Acceptance Criteria:**
- [ ] Retrieval-quality metrics are computed and serialized.
- [ ] Summary aggregates include new retrieval metrics.
- [ ] Existing answer metrics remain backward compatible.

### Task 7.13: Add token-aware chat compaction with rolling summaries

state: todo

**Context / Background:**
- The controller appends messages across iterations and chat history can grow large.
- Long-running sessions risk context-window overflow, degraded responses, or hard API errors.
- Current behavior is not token-aware and does not compact history by budget.

**Task Description:**
- Add token-aware context management for chat/session history:
  - estimate token usage before each LLM call,
  - compact older messages when a configured threshold is reached,
  - replace compacted segments with a rolling summary that preserves key facts,
    decisions, tool outcomes, and unresolved questions.
- Preserve critical context during compaction:
  - system prompt,
  - latest user request,
  - recent turns,
  - high-value retrieval evidence needed for answer faithfulness.
- Add config knobs for compaction thresholds and retained recent-turn window.

**Dependency Note:**
- `blocked_by`: `Task 7.2` (new configuration knobs are required for thresholds/budgets).
- `parallel_with`: `Task 7.6` (formatting work can proceed independently; integration happens here).

**Files:**
- `src/agentic_graph_rag/agent/controller.py`
- `src/agentic_graph_rag/agent/session.py`
- `src/agentic_graph_rag/prompts/manager.py`
- `src/agentic_graph_rag/config.py`
- `tests/test_agent/test_controller.py`
- `tests/test_agent/test_session.py`
- `tests/test_prompts/test_manager.py`

**Acceptance Criteria:**
- [ ] Controller compacts history automatically when token budget threshold is exceeded.
- [ ] Compaction preserves system prompt and most recent conversational context.
- [ ] Rolling summary includes key facts, tool results, and unresolved items.
- [ ] Configurable thresholds control when and how much history is compacted.
- [ ] Unit tests cover compaction trigger, summary injection, and no-compaction paths.
