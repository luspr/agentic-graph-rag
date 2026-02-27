#!/usr/bin/env python3
"""Fetch newest arXiv papers containing specific query phrases."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rich.console import Console

from agentic_graph_rag.scripts.arxiv_client import (
    ArxivApiClient,
    ArxivPaper,
    extract_arxiv_id,
    paper_to_dict,
)

DEFAULT_QUERIES = ("knowledge graph", "graph rag")
DEFAULT_LIMIT = 50
DEFAULT_TIMEOUT = 30.0
DEFAULT_OUTPUT_DIR = Path("data/paper_search")
DEFAULT_CACHE_DIR = Path("data/arxiv/cache")
DEFAULT_PDF_DIRNAME = "pdfs"
DEFAULT_CACHE_TTL_HOURS = 24
DEFAULT_THROTTLE_SECONDS = 3.5
SOURCE_NAME = "arxiv"
USER_AGENT = "agentic-graph-rag/0.1"


@dataclass(frozen=True)
class SearchResult:
    """Search results for a query against arXiv."""

    query: str
    retrieved_at: str
    limit: int
    total_results: int | None
    papers: list[ArxivPaper]
    source: str = SOURCE_NAME

    def to_dict(self) -> dict[str, Any]:
        """Convert the results to a JSON-serializable dict."""
        return {
            "query": self.query,
            "retrieved_at": self.retrieved_at,
            "limit": self.limit,
            "total_results": self.total_results,
            "source": self.source,
            "papers": [paper_to_dict(paper) for paper in self.papers],
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch the newest arXiv papers containing specified phrases and store them as JSON."
        )
    )
    parser.add_argument(
        "--query",
        dest="queries",
        action="append",
        help=(
            "Search phrase to query (repeatable). "
            "Defaults to 'knowledge graph' and 'graph rag'."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum number of papers to fetch per query.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for JSON files.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for cached arXiv responses.",
    )
    parser.add_argument(
        "--cache-ttl-hours",
        type=int,
        default=DEFAULT_CACHE_TTL_HOURS,
        help="Cache TTL in hours for arXiv responses.",
    )
    parser.add_argument(
        "--throttle-seconds",
        type=float,
        default=DEFAULT_THROTTLE_SECONDS,
        help="Minimum delay between arXiv API requests.",
    )
    parser.add_argument(
        "--download-pdfs",
        action="store_true",
        help="Download PDFs for papers that include a PDF link.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Directory to store downloaded PDFs (defaults to <out-dir>/pdfs).",
    )
    parser.add_argument(
        "--overwrite-pdfs",
        action="store_true",
        help="Re-download PDFs even if they already exist.",
    )
    parser.add_argument(
        "--print-titles",
        action="store_true",
        help="Print paper titles to the console.",
    )
    return parser.parse_args()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
    slug = slug.strip("_")
    return slug or "query"


def _write_results(result: SearchResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_slugify(result.query)}.json"
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, ensure_ascii=True, indent=2)
    return path


def _print_summary(console: Console, result: SearchResult, path: Path) -> None:
    total_display = result.total_results if result.total_results is not None else "?"
    console.print(
        f"[bold]{result.query}[/bold]: "
        f"{len(result.papers)} papers (total available: {total_display})."
    )
    console.print(f"Wrote {path}.")


def _print_titles(console: Console, result: SearchResult) -> None:
    console.print(f"[bold]Titles for '{result.query}':[/bold]")
    for paper in result.papers:
        console.print(f"- {paper.title}")


def _pdf_filename(paper: ArxivPaper) -> str:
    candidates = [paper.pdf_url, paper.id_url]
    for candidate in candidates:
        if not candidate:
            continue
        arxiv_id = extract_arxiv_id(candidate)
        if arxiv_id:
            return f"{arxiv_id}.pdf"
    return f"{_slugify(paper.title)}.pdf"


def _download_pdf(url: str, destination: Path, timeout: float, overwrite: bool) -> bool:
    if destination.exists() and not overwrite:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with (
            urlopen(request, timeout=timeout) as response,
            destination.open("wb") as handle,  # noqa: S310
        ):
            handle.write(response.read())
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to download PDF {url}: {exc}") from exc
    return True


def _download_pdfs(
    console: Console,
    result: SearchResult,
    pdf_dir: Path,
    timeout: float,
    overwrite: bool,
) -> None:
    downloaded = 0
    skipped = 0
    for paper in result.papers:
        if not paper.pdf_url:
            skipped += 1
            continue
        filename = _pdf_filename(paper)
        destination = pdf_dir / filename
        if _download_pdf(paper.pdf_url, destination, timeout, overwrite):
            downloaded += 1
        else:
            skipped += 1
    console.print(
        f"PDFs for '{result.query}': downloaded {downloaded}, skipped {skipped}."
    )


async def _fetch_papers(
    client: ArxivApiClient,
    query: str,
    limit: int,
) -> SearchResult:
    page = await client.search(query, start=0, max_results=limit)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    return SearchResult(
        query=query,
        retrieved_at=retrieved_at,
        limit=limit,
        total_results=page.total_results,
        papers=page.papers,
    )


async def _run(args: argparse.Namespace, console: Console) -> int:
    queries = args.queries or list(DEFAULT_QUERIES)
    pdf_dir = args.pdf_dir or (args.out_dir / DEFAULT_PDF_DIRNAME)

    if args.limit <= 0:
        console.print("[red]Limit must be greater than zero.[/red]")
        return 1
    if args.cache_ttl_hours < 0:
        console.print("[red]Cache TTL hours must be >= 0.[/red]")
        return 1
    if args.throttle_seconds < 0:
        console.print("[red]Throttle seconds must be >= 0.[/red]")
        return 1

    client = ArxivApiClient(
        user_agent=USER_AGENT,
        cache_dir=args.cache_dir,
        cache_ttl_seconds=args.cache_ttl_hours * 60 * 60,
        throttle_seconds=args.throttle_seconds,
        timeout_seconds=args.timeout,
    )

    try:
        for query in queries:
            result = await _fetch_papers(client, query, args.limit)
            path = _write_results(result, args.out_dir)
            _print_summary(console, result, path)
            if args.print_titles:
                _print_titles(console, result)
            if args.download_pdfs:
                _download_pdfs(
                    console,
                    result,
                    pdf_dir,
                    args.timeout,
                    args.overwrite_pdfs,
                )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    return 0


def main() -> int:
    """Run the paper fetcher."""
    args = _parse_args()
    console = Console()
    try:
        return asyncio.run(_run(args, console))
    except KeyboardInterrupt:
        console.print("[red]Interrupted.[/red]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
