**Agentic Graph RAG**

Agentic Graph RAG lets an LLM iteratively query a Neo4j knowledge graph. It supports
pure Cypher retrieval and a hybrid flow that starts with vector search (Qdrant) and
expands through graph traversal.

**Requirements**
- Neo4j running with the target dataset loaded.
- Qdrant running if you use hybrid retrieval.
- OpenAI API key for LLM + embeddings.

**CLI**
- Run a single headless query:
  `uv run agentic-graph-rag run --query "What movies did Tom Hanks act in?" --strategy cypher`
- Evaluate a benchmark:
  `uv run agentic-graph-rag eval --input data/sr_rag/benchmark/benchmark_1637.jsonl --format sr_rag --strategy cypher`

**Evaluation How It Works**
The eval pipeline loads JSONL benchmarks, runs each question through the headless
agent, computes metrics against ground truth, and writes per-example results plus a
summary file.

**Benchmark Formats**
- SR-RAG JSONL expects `question_en`, `ground_truth`, and optional `nuggets` with
  `vital`, `okay`, `trivial` lists.
- Generic JSONL expects `question` and `ground_truth` (field names are configurable).

**Metrics**
- Exact match (normalized).
- Token-level F1 (SQuAD-style).
- ROUGE-L F1 (token LCS).
- Embedding cosine similarity (OpenAI embeddings).
- Nugget recall/precision if nuggets are present (embedding similarity).

**LLM Judge**
- Optional rubric scoring for correctness, completeness, and faithfulness on a 0–5
  scale.
- The judge receives the question, ground truth, the agent’s answer, and supporting
  evidence extracted from the agent’s `submit_answer` tool call.

**Outputs**
- `results.jsonl`: one record per example with answer, metrics, and judge scores.
- `summary.json`: aggregated metrics (means/medians), success rates, averages.
- Optional `--trace-log` writes a JSONL stream of tool calls and LLM events.

**Evaluation Options**
- `--format sr_rag|generic`
- `--strategy cypher|hybrid`
- `--concurrency N`
- `--max-examples N`
- `--no-judge`
- `--no-embedding-eval`
- `--nugget-threshold 0.78`

**Configuration**
Set env vars (or `.env`) for the required services:
- `OPENAI_API_KEY`, `OPENAI_MODEL`, optional `OPENAI_JUDGE_MODEL`
- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION`, `EMBEDDING_DIM` (used for OpenAI embedding dimensions and Qdrant vector size)
