"""Command-line interface for headless execution and evaluation."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path

import anyio
from pydantic import ValidationError

from agentic_graph_rag.agent.controller import Tracer
from agentic_graph_rag.config import Settings
from agentic_graph_rag.eval.loaders import load_generic_jsonl, load_sr_rag_jsonl
from agentic_graph_rag.eval.metrics import EmbeddingCache
from agentic_graph_rag.eval.runner import JSONLWriter, run_evaluation
from agentic_graph_rag.eval.types import EvalConfig
from agentic_graph_rag.llm.base import LLMClient
from agentic_graph_rag.llm.openai_client import OpenAILLMClient
from agentic_graph_rag.retriever.base import RetrievalStrategy
from agentic_graph_rag.runner import HeadlessRunner


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic Graph RAG CLI (headless execution + evaluation)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a single query headlessly.")
    run_parser.add_argument("--query", type=str, required=True, help="User query.")
    run_parser.add_argument(
        "--strategy",
        type=str,
        choices=["cypher", "hybrid"],
        default="cypher",
        help="Retrieval strategy to use.",
    )
    run_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit result as JSON to stdout.",
    )
    run_parser.add_argument(
        "--trace-log",
        type=Path,
        default=None,
        help="Optional path for trace JSONL logging.",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate a benchmark dataset.")
    eval_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to JSONL benchmark file.",
    )
    eval_parser.add_argument(
        "--format",
        type=str,
        choices=["sr_rag", "generic"],
        default="sr_rag",
        help="Benchmark format.",
    )
    eval_parser.add_argument(
        "--question-field",
        type=str,
        default="question",
        help="Field name for the question (generic format only).",
    )
    eval_parser.add_argument(
        "--ground-truth-field",
        type=str,
        default="ground_truth",
        help="Field name for the ground truth (generic format only).",
    )
    eval_parser.add_argument(
        "--strategy",
        type=str,
        choices=["cypher", "hybrid"],
        default="cypher",
        help="Retrieval strategy to use.",
    )
    eval_parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent queries to run.",
    )
    eval_parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional max number of examples to evaluate.",
    )
    eval_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write results.jsonl and summary.json.",
    )
    eval_parser.add_argument(
        "--trace-log",
        type=Path,
        default=None,
        help="Optional path for trace JSONL logging.",
    )
    eval_parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable LLM judge scoring.",
    )
    eval_parser.add_argument(
        "--no-embedding-eval",
        action="store_true",
        help="Disable embedding-based metrics.",
    )
    eval_parser.add_argument(
        "--nugget-threshold",
        type=float,
        default=0.78,
        help="Embedding similarity threshold for nugget coverage.",
    )

    return parser.parse_args()


def _parse_strategy(value: str) -> RetrievalStrategy:
    if value == "hybrid":
        return RetrievalStrategy.HYBRID
    return RetrievalStrategy.CYPHER


def _load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        print("Configuration error:")
        for error in exc.errors():
            field = error.get("loc", ("unknown",))[0]
            msg = error.get("msg", "Invalid value")
            print(f"  {field}: {msg}")
        raise


async def _run_single(args: argparse.Namespace) -> int:
    try:
        settings = _load_settings()
    except ValidationError:
        return 1
    strategy = _parse_strategy(args.strategy)

    trace_writer = JSONLWriter(args.trace_log) if args.trace_log else None
    run_id = None
    trace_id = None

    async with HeadlessRunner(settings=settings, strategy=strategy) as runner:
        tracer: Tracer | None = None
        if trace_writer is not None:
            run_id = str(uuid.uuid4())
            trace_id = str(uuid.uuid4())
            tracer = _build_tracer(trace_writer, trace_id, run_id)

        result = await runner.run_query(args.query, tracer=tracer)

    if trace_writer is not None:
        trace_writer.close()

    if args.json:
        payload = {
            "answer": result.answer,
            "status": result.status.value,
            "iterations": result.iterations,
            "confidence": result.confidence,
            "run_id": run_id,
            "trace_id": trace_id,
        }
        print(json.dumps(payload))
        return 0

    print(f"Status: {result.status.value}")
    print(f"Iterations: {result.iterations}")
    if result.confidence is not None:
        print(f"Confidence: {result.confidence}")
    print("Answer:")
    print(result.answer)
    return 0


async def _run_eval(args: argparse.Namespace) -> int:
    try:
        settings = _load_settings()
    except ValidationError:
        return 1
    strategy = _parse_strategy(args.strategy)

    if args.format == "sr_rag":
        examples = load_sr_rag_jsonl(args.input)
    else:
        examples = load_generic_jsonl(
            args.input,
            question_field=args.question_field,
            ground_truth_field=args.ground_truth_field,
        )

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("eval_runs") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    result_writer = JSONLWriter(results_path)
    trace_writer = JSONLWriter(args.trace_log) if args.trace_log else None

    judge_enabled = not args.no_judge
    embedding_enabled = not args.no_embedding_eval

    judge_client: LLMClient | None = None
    judge_owner: OpenAILLMClient | None = None

    async with HeadlessRunner(settings=settings, strategy=strategy) as runner:
        if judge_enabled:
            judge_model = settings.openai_judge_model or settings.openai_model
            if judge_model != settings.openai_model:
                judge_owner = OpenAILLMClient(
                    api_key=settings.openai_api_key,
                    model=judge_model,
                )
                judge_client = judge_owner
            else:
                judge_client = runner.llm_client

        embedding_cache = (
            EmbeddingCache(runner.llm_client) if embedding_enabled else None
        )
        config = EvalConfig(
            strategy=strategy,
            concurrency=args.concurrency,
            judge_enabled=judge_enabled,
            embedding_enabled=embedding_enabled,
            nugget_threshold=args.nugget_threshold,
            max_examples=args.max_examples,
        )

        results, summary = await run_evaluation(
            examples=examples,
            runner=runner,
            config=config,
            result_writer=result_writer,
            trace_writer=trace_writer,
            judge_client=judge_client,
            embedding_cache=embedding_cache,
        )

    if judge_owner is not None:
        await judge_owner.aclose()

    result_writer.close()
    if trace_writer is not None:
        trace_writer.close()

    summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

    print(f"Wrote {len(results)} results to {results_path}")
    print(f"Wrote summary to {summary_path}")
    return 0


def _build_tracer(
    trace_writer: JSONLWriter,
    trace_id: str,
    run_id: str,
) -> Tracer:
    from agentic_graph_rag.eval.runner import RunEventLogger

    return RunEventLogger(trace_writer, trace_id, run_id)


def main() -> int:
    args = _parse_args()
    if args.command == "run":
        return anyio.run(_run_single, args)
    if args.command == "eval":
        return anyio.run(_run_eval, args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
