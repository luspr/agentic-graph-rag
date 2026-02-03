#!/usr/bin/env python3
"""Fetch newest arXiv papers containing specific query phrases."""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from rich.console import Console


ARXIV_API_URL = "https://export.arxiv.org/api/query"
DEFAULT_QUERIES = ("knowledge graph", "graph rag")
DEFAULT_LIMIT = 50
DEFAULT_TIMEOUT = 30.0
DEFAULT_OUTPUT_DIR = Path("data/paper_search")
DEFAULT_PDF_DIRNAME = "pdfs"
SOURCE_NAME = "arxiv"
USER_AGENT = "agentic-graph-rag/0.1"

ATOM_NS = "http://www.w3.org/2005/Atom"
OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"
ARXIV_NS = "http://arxiv.org/schemas/atom"
NS = {"atom": ATOM_NS, "opensearch": OPENSEARCH_NS, "arxiv": ARXIV_NS}


@dataclass(frozen=True)
class Paper:
    """Representation of a single arXiv paper entry."""

    title: str
    authors: list[str]
    published: str
    updated: str
    summary: str
    url: str
    pdf_url: str | None
    primary_category: str | None
    query: str
    source: str = SOURCE_NAME

    def to_dict(self) -> dict[str, Any]:
        """Convert the paper to a JSON-serializable dict."""
        return {
            "title": self.title,
            "authors": self.authors,
            "published": self.published,
            "updated": self.updated,
            "summary": self.summary,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "primary_category": self.primary_category,
            "query": self.query,
            "source": self.source,
        }


@dataclass(frozen=True)
class SearchResult:
    """Search results for a query against arXiv."""

    query: str
    retrieved_at: str
    limit: int
    total_results: int | None
    papers: list[Paper]
    source: str = SOURCE_NAME

    def to_dict(self) -> dict[str, Any]:
        """Convert the results to a JSON-serializable dict."""
        return {
            "query": self.query,
            "retrieved_at": self.retrieved_at,
            "limit": self.limit,
            "total_results": self.total_results,
            "source": self.source,
            "papers": [paper.to_dict() for paper in self.papers],
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


def _build_query_url(query: str, limit: int) -> str:
    params = {
        "search_query": f'all:"{query}"',
        "start": 0,
        "max_results": limit,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    return f"{ARXIV_API_URL}?{urlencode(params)}"


def _fetch_feed(url: str, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.read().decode("utf-8")
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to fetch arXiv feed: {exc}") from exc


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def _text(entry: ElementTree.Element, tag: str) -> str:
    node = entry.find(tag, NS)
    return _normalize_text(node.text if node is not None else None)


def _primary_category(entry: ElementTree.Element) -> str | None:
    node = entry.find("arxiv:primary_category", NS)
    if node is None:
        return None
    term = node.attrib.get("term", "")
    return term or None


def _pdf_link(entry: ElementTree.Element) -> str | None:
    for link in entry.findall("atom:link", NS):
        link_type = link.attrib.get("type")
        if link_type == "application/pdf":
            href = link.attrib.get("href")
            if href:
                return href
        if link.attrib.get("title") == "pdf":
            href = link.attrib.get("href")
            if href:
                return href
    return None


def _authors(entry: ElementTree.Element) -> list[str]:
    authors: list[str] = []
    for author in entry.findall("atom:author", NS):
        name = author.find("atom:name", NS)
        if name is not None and name.text:
            authors.append(_normalize_text(name.text))
    return authors


def _entry_to_paper(entry: ElementTree.Element, query: str) -> Paper:
    title = _text(entry, "atom:title")
    summary = _text(entry, "atom:summary")
    published = _text(entry, "atom:published")
    updated = _text(entry, "atom:updated")
    url = _text(entry, "atom:id")
    return Paper(
        title=title,
        authors=_authors(entry),
        published=published,
        updated=updated,
        summary=summary,
        url=url,
        pdf_url=_pdf_link(entry),
        primary_category=_primary_category(entry),
        query=query,
    )


def _parse_total_results(root: ElementTree.Element) -> int | None:
    total_node = root.find("opensearch:totalResults", NS)
    if total_node is None or not total_node.text:
        return None
    total_text = total_node.text.strip()
    return int(total_text) if total_text.isdigit() else None


def _parse_feed(xml_text: str, query: str) -> tuple[list[Paper], int | None]:
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as exc:
        raise RuntimeError(f"Failed to parse arXiv feed: {exc}") from exc
    papers = [_entry_to_paper(entry, query) for entry in root.findall("atom:entry", NS)]
    return papers, _parse_total_results(root)


def _fetch_papers(query: str, limit: int, timeout: float) -> SearchResult:
    url = _build_query_url(query, limit)
    feed = _fetch_feed(url, timeout)
    papers, total_results = _parse_feed(feed, query)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    return SearchResult(
        query=query,
        retrieved_at=retrieved_at,
        limit=limit,
        total_results=total_results,
        papers=papers,
    )


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


def _extract_arxiv_id(value: str) -> str | None:
    match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", value)
    if match:
        return match.group(0)
    legacy_match = re.search(r"([a-z-]+/\d{7})(v\d+)?", value, re.IGNORECASE)
    if legacy_match:
        return legacy_match.group(0)
    return None


def _pdf_filename(paper: Paper) -> str:
    candidates = [paper.pdf_url, paper.url]
    for candidate in candidates:
        if not candidate:
            continue
        arxiv_id = _extract_arxiv_id(candidate)
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
            destination.open(  # noqa: S310
                "wb"
            ) as handle,
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


def main() -> int:
    """Run the paper fetcher."""
    args = _parse_args()
    console = Console()
    queries = args.queries or list(DEFAULT_QUERIES)
    pdf_dir = args.pdf_dir or (args.out_dir / DEFAULT_PDF_DIRNAME)
    if args.limit <= 0:
        console.print("[red]Limit must be greater than zero.[/red]")
        return 1

    try:
        for query in queries:
            result = _fetch_papers(query, args.limit, args.timeout)
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


if __name__ == "__main__":
    sys.exit(main())
