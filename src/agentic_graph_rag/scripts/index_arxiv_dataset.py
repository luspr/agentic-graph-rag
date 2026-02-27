#!/usr/bin/env python3
"""Fetch arXiv papers and index graph entities into Neo4j and Qdrant."""

from __future__ import annotations

import argparse
import asyncio
import calendar
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from agentic_graph_rag.config import Settings
from agentic_graph_rag.graph.neo4j_client import Neo4jClient
from agentic_graph_rag.llm.openai_client import OpenAILLMClient
from agentic_graph_rag.scripts.arxiv_client import ArxivApiClient, ArxivPaper
from agentic_graph_rag.vector.qdrant_client import QdrantVectorStore

DEFAULT_QUERIES = (
    "knowledge graph",
    "graph rag",
    "retrieval augmented generation",
)
DEFAULT_LIMIT = 200
DEFAULT_PAGE_SIZE = 50
DEFAULT_MONTHS_BACK = 18
DEFAULT_MAX_PAGES_PER_QUERY = 20
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_CACHE_DIR = Path("data/arxiv/cache")
DEFAULT_CACHE_TTL_HOURS = 24
DEFAULT_THROTTLE_SECONDS = 3.5
DEFAULT_BATCH_SIZE = 50
DEFAULT_EMBEDDING_CONCURRENCY = 3
DATASET_NAME = "arxiv"
SOURCE_NAME = "arxiv"
USER_AGENT = "agentic-graph-rag/0.1"
PROPERTY_TOKEN_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class CollectedPaper:
    """A paper plus the set of matched search queries."""

    paper: ArxivPaper
    matched_queries: list[str]


@dataclass(frozen=True)
class VectorPoint:
    """A vector point to embed and upsert into Qdrant."""

    uuid: str
    text: str
    payload: dict[str, Any]


@dataclass
class IngestBundle:
    """All graph rows and vector points derived from fetched papers."""

    papers: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    authors: list[dict[str, Any]]
    universities: list[dict[str, Any]]
    categories: list[dict[str, Any]]
    venues: list[dict[str, Any]]
    authored_edges: list[dict[str, Any]]
    affiliated_edges: list[dict[str, Any]]
    has_chunk_edges: list[dict[str, Any]]
    in_category_edges: list[dict[str, Any]]
    published_in_edges: list[dict[str, Any]]
    vector_points: list[VectorPoint]


@dataclass(frozen=True)
class FetchStats:
    """Stats produced by the fetch and dedupe stage."""

    raw_papers: int
    unique_papers: int
    kept_papers: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch arXiv papers and index Paper/Chunk/Author/University/Category/"
            "Venue entities into Neo4j and Qdrant."
        )
    )
    parser.add_argument(
        "--query",
        dest="queries",
        action="append",
        help="Search phrase to query (repeatable). Uses built-in defaults if omitted.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum number of unique papers to ingest across all queries.",
    )
    parser.add_argument(
        "--months-back",
        type=int,
        default=DEFAULT_MONTHS_BACK,
        help="Recency window in months based on published/updated dates.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help="arXiv API page size.",
    )
    parser.add_argument(
        "--max-pages-per-query",
        type=int,
        default=DEFAULT_MAX_PAGES_PER_QUERY,
        help="Safety cap on arXiv pages fetched per query.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout in seconds for arXiv requests.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for cached arXiv API responses.",
    )
    parser.add_argument(
        "--cache-ttl-hours",
        type=int,
        default=DEFAULT_CACHE_TTL_HOURS,
        help="Cache TTL in hours.",
    )
    parser.add_argument(
        "--throttle-seconds",
        type=float,
        default=DEFAULT_THROTTLE_SECONDS,
        help="Minimum delay between arXiv API requests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for Neo4j upsert queries.",
    )
    parser.add_argument(
        "--embedding-concurrency",
        type=int,
        default=DEFAULT_EMBEDDING_CONCURRENCY,
        help="Concurrent embedding requests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and build records only; skip Neo4j and Qdrant writes.",
    )
    return parser.parse_args()


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _normalize_identifier(value: str) -> str:
    normalized = _normalize_whitespace(value.strip().lower())
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = normalized.strip("_")
    return normalized or "unknown"


def _parse_timestamp(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _subtract_months(reference: datetime, months: int) -> datetime:
    if months <= 0:
        return reference
    year = reference.year
    month = reference.month - months
    while month <= 0:
        month += 12
        year -= 1
    day = min(reference.day, calendar.monthrange(year, month)[1])
    return reference.replace(year=year, month=month, day=day)


def _paper_sort_key(paper: ArxivPaper) -> tuple[datetime, datetime]:
    published = _parse_timestamp(paper.published) or datetime.min.replace(
        tzinfo=timezone.utc
    )
    updated = _parse_timestamp(paper.updated) or datetime.min.replace(
        tzinfo=timezone.utc
    )
    return (published, updated)


def _is_recent(paper: ArxivPaper, cutoff: datetime) -> bool:
    published = _parse_timestamp(paper.published)
    if published is not None and published >= cutoff:
        return True
    updated = _parse_timestamp(paper.updated)
    if updated is not None and updated >= cutoff:
        return True
    return False


def _paper_uuid(arxiv_id_base: str) -> str:
    return f"paper:{arxiv_id_base}"


def _chunk_uuid(arxiv_id_base: str) -> str:
    return f"chunk:{arxiv_id_base}:0"


def _author_uuid(name: str) -> str:
    return f"author:{_normalize_identifier(name)}"


def _university_uuid(name: str) -> str:
    return f"university:{_normalize_identifier(name)}"


def _category_uuid(category: str) -> str:
    return f"category:{_normalize_identifier(category)}"


def _venue_uuid(name: str) -> str:
    return f"venue:{_normalize_identifier(name)}"


def _published_year(paper: ArxivPaper) -> int | None:
    published = _parse_timestamp(paper.published)
    return published.year if published is not None else None


def _build_payload(
    *,
    uuid: str,
    entity_type: str,
    label: str,
    display_name: str,
    source: str = SOURCE_NAME,
    arxiv_id: str | None = None,
    primary_category: str | None = None,
    published_year: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "uuid": uuid,
        "dataset": DATASET_NAME,
        "source": source,
        "entity_type": entity_type,
        "label": label,
        "labels": [label],
        "display_name": display_name,
    }
    if arxiv_id:
        payload["arxiv_id"] = arxiv_id
    if primary_category:
        payload["primary_category"] = primary_category
    if published_year is not None:
        payload["published_year"] = published_year
    return payload


def _paper_embedding_text(paper: ArxivPaper, queries: list[str]) -> str:
    author_names = [author.name for author in paper.authors]
    fields = [
        f"Title: {paper.title}",
        f"Abstract: {paper.summary}",
        f"Authors: {', '.join(author_names)}" if author_names else "",
        f"Primary category: {paper.primary_category}" if paper.primary_category else "",
        f"Categories: {', '.join(paper.categories)}" if paper.categories else "",
        f"DOI: {paper.doi}" if paper.doi else "",
        f"Journal ref: {paper.journal_ref}" if paper.journal_ref else "",
        f"Comment: {paper.comment}" if paper.comment else "",
        f"Matched queries: {', '.join(queries)}" if queries else "",
    ]
    return "\n".join(field for field in fields if field)


def _chunk_embedding_text(paper: ArxivPaper) -> str:
    return "\n".join(
        [
            f"Paper title: {paper.title}",
            f"Abstract chunk 0: {paper.summary}",
        ]
    )


def _author_embedding_text(name: str, affiliations: list[str]) -> str:
    fields = [f"Author: {name}"]
    if affiliations:
        fields.append(f"Affiliations: {', '.join(affiliations)}")
    return "\n".join(fields)


async def _collect_papers_async(
    *,
    client: ArxivApiClient,
    queries: list[str],
    limit: int,
    page_size: int,
    max_pages_per_query: int,
    months_back: int,
    now: datetime,
    console: Console,
) -> tuple[list[CollectedPaper], FetchStats]:
    cutoff = _subtract_months(now, months_back)
    by_id: dict[str, ArxivPaper] = {}
    query_hits: dict[str, set[str]] = {}
    raw_papers = 0

    for query in queries:
        start = 0
        pages_fetched = 0
        while pages_fetched < max_pages_per_query and len(by_id) < limit:
            page = await client.search(query, start=start, max_results=page_size)
            pages_fetched += 1
            if not page.papers:
                break

            raw_papers += len(page.papers)
            for paper in page.papers:
                if not _is_recent(paper, cutoff):
                    continue

                key = paper.arxiv_id_base
                existing = by_id.get(key)
                if existing is None or _paper_sort_key(paper) > _paper_sort_key(
                    existing
                ):
                    by_id[key] = paper
                query_hits.setdefault(key, set()).add(query)

            start += page_size
            if page.total_results is not None and start >= page.total_results:
                break

        console.print(
            f"Fetched {pages_fetched} page(s) for '{query}'. "
            f"Current unique recent papers: {len(by_id)}"
        )
        if len(by_id) >= limit:
            break

    sorted_papers = sorted(
        by_id.values(),
        key=_paper_sort_key,
        reverse=True,
    )
    selected = sorted_papers[:limit]
    selected_ids = {paper.arxiv_id_base for paper in selected}

    collected = [
        CollectedPaper(
            paper=paper,
            matched_queries=sorted(query_hits.get(paper.arxiv_id_base, set())),
        )
        for paper in selected
    ]

    return (
        collected,
        FetchStats(
            raw_papers=raw_papers,
            unique_papers=len(by_id),
            kept_papers=len(selected_ids),
        ),
    )


def build_ingest_bundle(
    papers: list[CollectedPaper],
    *,
    node_uuid_property: str,
) -> IngestBundle:
    """Build graph rows and vector points from collected papers."""
    if not PROPERTY_TOKEN_PATTERN.match(node_uuid_property):
        raise ValueError(f"Invalid UUID property name: {node_uuid_property}")

    paper_rows: dict[str, dict[str, Any]] = {}
    chunk_rows: dict[str, dict[str, Any]] = {}
    author_rows: dict[str, dict[str, Any]] = {}
    university_rows: dict[str, dict[str, Any]] = {}
    category_rows: dict[str, dict[str, Any]] = {}
    venue_rows: dict[str, dict[str, Any]] = {}

    authored_edges: set[tuple[str, str]] = set()
    affiliated_edges: set[tuple[str, str]] = set()
    has_chunk_edges: set[tuple[str, str]] = set()
    in_category_edges: dict[tuple[str, str], bool] = {}
    published_in_edges: set[tuple[str, str]] = set()

    vector_points: dict[str, VectorPoint] = {}

    for item in papers:
        paper = item.paper
        paper_uuid = _paper_uuid(paper.arxiv_id_base)
        chunk_uuid = _chunk_uuid(paper.arxiv_id_base)
        published_year = _published_year(paper)

        paper_props: dict[str, Any] = {
            node_uuid_property: paper_uuid,
            "dataset": DATASET_NAME,
            "source": SOURCE_NAME,
            "entity_type": "paper",
            "arxiv_id": paper.arxiv_id_base,
            "arxiv_id_versioned": paper.arxiv_id,
            "title": paper.title,
            "abstract": paper.summary,
            "published": paper.published,
            "updated": paper.updated,
            "primary_category": paper.primary_category,
            "categories": paper.categories,
            "doi": paper.doi,
            "journal_ref": paper.journal_ref,
            "comment": paper.comment,
            "entry_url": paper.id_url,
            "pdf_url": paper.pdf_url,
            "matched_queries": item.matched_queries,
        }
        paper_rows[paper_uuid] = {"uuid": paper_uuid, "properties": paper_props}

        chunk_props: dict[str, Any] = {
            node_uuid_property: chunk_uuid,
            "dataset": DATASET_NAME,
            "source": SOURCE_NAME,
            "entity_type": "chunk",
            "arxiv_id": paper.arxiv_id_base,
            "chunk_index": 0,
            "text": paper.summary,
            "title": paper.title,
        }
        chunk_rows[chunk_uuid] = {"uuid": chunk_uuid, "properties": chunk_props}
        has_chunk_edges.add((paper_uuid, chunk_uuid))

        paper_vector_payload = _build_payload(
            uuid=paper_uuid,
            entity_type="paper",
            label="Paper",
            display_name=paper.title,
            arxiv_id=paper.arxiv_id_base,
            primary_category=paper.primary_category,
            published_year=published_year,
        )
        vector_points[paper_uuid] = VectorPoint(
            uuid=paper_uuid,
            text=_paper_embedding_text(paper, item.matched_queries),
            payload=paper_vector_payload,
        )

        chunk_vector_payload = _build_payload(
            uuid=chunk_uuid,
            entity_type="chunk",
            label="Chunk",
            display_name=f"Abstract chunk for {paper.title}",
            arxiv_id=paper.arxiv_id_base,
            primary_category=paper.primary_category,
            published_year=published_year,
        )
        vector_points[chunk_uuid] = VectorPoint(
            uuid=chunk_uuid,
            text=_chunk_embedding_text(paper),
            payload=chunk_vector_payload,
        )

        for author in paper.authors:
            author_uuid = _author_uuid(author.name)
            author_row = author_rows.get(author_uuid)
            if author_row is None:
                author_props = {
                    node_uuid_property: author_uuid,
                    "dataset": DATASET_NAME,
                    "source": SOURCE_NAME,
                    "entity_type": "author",
                    "name": author.name,
                    "affiliations": author.affiliations,
                }
                author_rows[author_uuid] = {
                    "uuid": author_uuid,
                    "properties": author_props,
                }
            authored_edges.add((author_uuid, paper_uuid))

            if author_uuid not in vector_points:
                author_vector_payload = _build_payload(
                    uuid=author_uuid,
                    entity_type="author",
                    label="Author",
                    display_name=author.name,
                )
                vector_points[author_uuid] = VectorPoint(
                    uuid=author_uuid,
                    text=_author_embedding_text(author.name, author.affiliations),
                    payload=author_vector_payload,
                )

            for affiliation in author.affiliations:
                affiliation_text = _normalize_whitespace(affiliation)
                if not affiliation_text:
                    continue
                university_uuid = _university_uuid(affiliation_text)
                if university_uuid not in university_rows:
                    university_props = {
                        node_uuid_property: university_uuid,
                        "dataset": DATASET_NAME,
                        "source": SOURCE_NAME,
                        "entity_type": "university",
                        "name": affiliation_text,
                    }
                    university_rows[university_uuid] = {
                        "uuid": university_uuid,
                        "properties": university_props,
                    }
                affiliated_edges.add((author_uuid, university_uuid))

                if university_uuid not in vector_points:
                    university_vector_payload = _build_payload(
                        uuid=university_uuid,
                        entity_type="university",
                        label="University",
                        display_name=affiliation_text,
                    )
                    vector_points[university_uuid] = VectorPoint(
                        uuid=university_uuid,
                        text=f"University affiliation: {affiliation_text}",
                        payload=university_vector_payload,
                    )

        for category in paper.categories:
            category_text = _normalize_whitespace(category)
            if not category_text:
                continue
            category_uuid = _category_uuid(category_text)
            if category_uuid not in category_rows:
                category_props = {
                    node_uuid_property: category_uuid,
                    "dataset": DATASET_NAME,
                    "source": SOURCE_NAME,
                    "entity_type": "category",
                    "code": category_text,
                }
                category_rows[category_uuid] = {
                    "uuid": category_uuid,
                    "properties": category_props,
                }

            is_primary = paper.primary_category == category_text
            current = in_category_edges.get((paper_uuid, category_uuid), False)
            in_category_edges[(paper_uuid, category_uuid)] = current or is_primary

            if category_uuid not in vector_points:
                category_vector_payload = _build_payload(
                    uuid=category_uuid,
                    entity_type="category",
                    label="Category",
                    display_name=category_text,
                )
                vector_points[category_uuid] = VectorPoint(
                    uuid=category_uuid,
                    text=f"arXiv category: {category_text}",
                    payload=category_vector_payload,
                )

        if paper.journal_ref:
            venue_name = _normalize_whitespace(paper.journal_ref)
            if venue_name:
                venue_uuid = _venue_uuid(venue_name)
                if venue_uuid not in venue_rows:
                    venue_props = {
                        node_uuid_property: venue_uuid,
                        "dataset": DATASET_NAME,
                        "source": SOURCE_NAME,
                        "entity_type": "venue",
                        "name": venue_name,
                    }
                    venue_rows[venue_uuid] = {
                        "uuid": venue_uuid,
                        "properties": venue_props,
                    }
                published_in_edges.add((paper_uuid, venue_uuid))

                if venue_uuid not in vector_points:
                    venue_vector_payload = _build_payload(
                        uuid=venue_uuid,
                        entity_type="venue",
                        label="Venue",
                        display_name=venue_name,
                    )
                    vector_points[venue_uuid] = VectorPoint(
                        uuid=venue_uuid,
                        text=f"Publication venue or journal reference: {venue_name}",
                        payload=venue_vector_payload,
                    )

    authored_rows = [
        {"author_uuid": author_uuid, "paper_uuid": paper_uuid}
        for (author_uuid, paper_uuid) in sorted(authored_edges)
    ]
    affiliated_rows = [
        {"author_uuid": author_uuid, "university_uuid": university_uuid}
        for (author_uuid, university_uuid) in sorted(affiliated_edges)
    ]
    has_chunk_rows = [
        {"paper_uuid": paper_uuid, "chunk_uuid": chunk_uuid}
        for (paper_uuid, chunk_uuid) in sorted(has_chunk_edges)
    ]
    in_category_rows = [
        {
            "paper_uuid": paper_uuid,
            "category_uuid": category_uuid,
            "primary": is_primary,
        }
        for (paper_uuid, category_uuid), is_primary in sorted(in_category_edges.items())
    ]
    published_in_rows = [
        {"paper_uuid": paper_uuid, "venue_uuid": venue_uuid}
        for (paper_uuid, venue_uuid) in sorted(published_in_edges)
    ]

    return IngestBundle(
        papers=list(paper_rows.values()),
        chunks=list(chunk_rows.values()),
        authors=list(author_rows.values()),
        universities=list(university_rows.values()),
        categories=list(category_rows.values()),
        venues=list(venue_rows.values()),
        authored_edges=authored_rows,
        affiliated_edges=affiliated_rows,
        has_chunk_edges=has_chunk_rows,
        in_category_edges=in_category_rows,
        published_in_edges=published_in_rows,
        vector_points=list(vector_points.values()),
    )


def _batch(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [
        rows[index : index + batch_size] for index in range(0, len(rows), batch_size)
    ]


def _merge_node_query(label: str, uuid_property: str) -> str:
    return (
        "UNWIND $rows AS row "
        f"MERGE (n:{label} {{{uuid_property}: row.uuid}}) "
        "SET n += row.properties"
    )


async def _execute_required(
    graph_db: Neo4jClient,
    cypher: str,
    params: dict[str, Any],
) -> None:
    result = await graph_db.execute(cypher, params)
    if result.error:
        raise RuntimeError(result.error)


async def _upsert_nodes(
    graph_db: Neo4jClient,
    *,
    label: str,
    rows: list[dict[str, Any]],
    uuid_property: str,
    batch_size: int,
) -> int:
    if not rows:
        return 0
    query = _merge_node_query(label, uuid_property)
    for chunk in _batch(rows, batch_size):
        await _execute_required(graph_db, query, {"rows": chunk})
    return len(rows)


async def _upsert_authored_edges(
    graph_db: Neo4jClient,
    rows: list[dict[str, Any]],
    *,
    uuid_property: str,
    batch_size: int,
) -> int:
    if not rows:
        return 0
    query = (
        "UNWIND $rows AS row "
        f"MATCH (a:Author {{{uuid_property}: row.author_uuid}}) "
        f"MATCH (p:Paper {{{uuid_property}: row.paper_uuid}}) "
        "MERGE (a)-[r:AUTHORED]->(p) "
        "SET r.dataset = $dataset, r.source = $source"
    )
    for chunk in _batch(rows, batch_size):
        await _execute_required(
            graph_db,
            query,
            {"rows": chunk, "dataset": DATASET_NAME, "source": SOURCE_NAME},
        )
    return len(rows)


async def _upsert_affiliated_edges(
    graph_db: Neo4jClient,
    rows: list[dict[str, Any]],
    *,
    uuid_property: str,
    batch_size: int,
) -> int:
    if not rows:
        return 0
    query = (
        "UNWIND $rows AS row "
        f"MATCH (a:Author {{{uuid_property}: row.author_uuid}}) "
        f"MATCH (u:University {{{uuid_property}: row.university_uuid}}) "
        "MERGE (a)-[r:AFFILIATED_WITH]->(u) "
        "SET r.dataset = $dataset, r.source = $source"
    )
    for chunk in _batch(rows, batch_size):
        await _execute_required(
            graph_db,
            query,
            {"rows": chunk, "dataset": DATASET_NAME, "source": SOURCE_NAME},
        )
    return len(rows)


async def _upsert_has_chunk_edges(
    graph_db: Neo4jClient,
    rows: list[dict[str, Any]],
    *,
    uuid_property: str,
    batch_size: int,
) -> int:
    if not rows:
        return 0
    query = (
        "UNWIND $rows AS row "
        f"MATCH (p:Paper {{{uuid_property}: row.paper_uuid}}) "
        f"MATCH (c:Chunk {{{uuid_property}: row.chunk_uuid}}) "
        "MERGE (p)-[r:HAS_CHUNK]->(c) "
        "SET r.dataset = $dataset, r.source = $source"
    )
    for chunk in _batch(rows, batch_size):
        await _execute_required(
            graph_db,
            query,
            {"rows": chunk, "dataset": DATASET_NAME, "source": SOURCE_NAME},
        )
    return len(rows)


async def _upsert_in_category_edges(
    graph_db: Neo4jClient,
    rows: list[dict[str, Any]],
    *,
    uuid_property: str,
    batch_size: int,
) -> int:
    if not rows:
        return 0
    query = (
        "UNWIND $rows AS row "
        f"MATCH (p:Paper {{{uuid_property}: row.paper_uuid}}) "
        f"MATCH (c:Category {{{uuid_property}: row.category_uuid}}) "
        "MERGE (p)-[r:IN_CATEGORY]->(c) "
        "SET r.primary = row.primary, r.dataset = $dataset, r.source = $source"
    )
    for chunk in _batch(rows, batch_size):
        await _execute_required(
            graph_db,
            query,
            {"rows": chunk, "dataset": DATASET_NAME, "source": SOURCE_NAME},
        )
    return len(rows)


async def _upsert_published_in_edges(
    graph_db: Neo4jClient,
    rows: list[dict[str, Any]],
    *,
    uuid_property: str,
    batch_size: int,
) -> int:
    if not rows:
        return 0
    query = (
        "UNWIND $rows AS row "
        f"MATCH (p:Paper {{{uuid_property}: row.paper_uuid}}) "
        f"MATCH (v:Venue {{{uuid_property}: row.venue_uuid}}) "
        "MERGE (p)-[r:PUBLISHED_IN]->(v) "
        "SET r.dataset = $dataset, r.source = $source"
    )
    for chunk in _batch(rows, batch_size):
        await _execute_required(
            graph_db,
            query,
            {"rows": chunk, "dataset": DATASET_NAME, "source": SOURCE_NAME},
        )
    return len(rows)


async def _write_graph(
    settings: Settings,
    *,
    bundle: IngestBundle,
    batch_size: int,
    console: Console,
) -> None:
    async with Neo4jClient(settings) as graph_db:
        uuid_property = settings.node_uuid_property
        paper_count = await _upsert_nodes(
            graph_db,
            label="Paper",
            rows=bundle.papers,
            uuid_property=uuid_property,
            batch_size=batch_size,
        )
        chunk_count = await _upsert_nodes(
            graph_db,
            label="Chunk",
            rows=bundle.chunks,
            uuid_property=uuid_property,
            batch_size=batch_size,
        )
        author_count = await _upsert_nodes(
            graph_db,
            label="Author",
            rows=bundle.authors,
            uuid_property=uuid_property,
            batch_size=batch_size,
        )
        university_count = await _upsert_nodes(
            graph_db,
            label="University",
            rows=bundle.universities,
            uuid_property=uuid_property,
            batch_size=batch_size,
        )
        category_count = await _upsert_nodes(
            graph_db,
            label="Category",
            rows=bundle.categories,
            uuid_property=uuid_property,
            batch_size=batch_size,
        )
        venue_count = await _upsert_nodes(
            graph_db,
            label="Venue",
            rows=bundle.venues,
            uuid_property=uuid_property,
            batch_size=batch_size,
        )

        authored_count = await _upsert_authored_edges(
            graph_db,
            bundle.authored_edges,
            uuid_property=uuid_property,
            batch_size=batch_size,
        )
        affiliated_count = await _upsert_affiliated_edges(
            graph_db,
            bundle.affiliated_edges,
            uuid_property=uuid_property,
            batch_size=batch_size,
        )
        has_chunk_count = await _upsert_has_chunk_edges(
            graph_db,
            bundle.has_chunk_edges,
            uuid_property=uuid_property,
            batch_size=batch_size,
        )
        in_category_count = await _upsert_in_category_edges(
            graph_db,
            bundle.in_category_edges,
            uuid_property=uuid_property,
            batch_size=batch_size,
        )
        published_in_count = await _upsert_published_in_edges(
            graph_db,
            bundle.published_in_edges,
            uuid_property=uuid_property,
            batch_size=batch_size,
        )

    console.print("Neo4j upsert complete:")
    console.print(
        "  Nodes: "
        f"Paper={paper_count}, Chunk={chunk_count}, Author={author_count}, "
        f"University={university_count}, Category={category_count}, Venue={venue_count}"
    )
    console.print(
        "  Edges: "
        f"AUTHORED={authored_count}, AFFILIATED_WITH={affiliated_count}, "
        f"HAS_CHUNK={has_chunk_count}, IN_CATEGORY={in_category_count}, "
        f"PUBLISHED_IN={published_in_count}"
    )


async def _write_vectors(
    settings: Settings,
    *,
    points: list[VectorPoint],
    concurrency: int,
    console: Console,
) -> tuple[int, int]:
    if not points:
        return (0, 0)

    llm_client = OpenAILLMClient(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        embedding_model=settings.openai_embedding_model,
        embedding_dimensions=settings.embedding_dim,
    )
    vector_store = QdrantVectorStore(
        settings=settings,
        collection_name=settings.qdrant_collection,
        vector_size=settings.embedding_dim,
        vector_name=settings.qdrant_vector_name,
    )

    success_count = 0
    error_count = 0
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    async def _process_point(point: VectorPoint, task_id: TaskID) -> None:
        nonlocal success_count
        nonlocal error_count

        async with semaphore:
            try:
                embedding = await llm_client.embed(point.text)
                await vector_store.upsert(
                    id=point.uuid,
                    embedding=embedding,
                    payload=point.payload,
                )
                async with lock:
                    success_count += 1
            except Exception as exc:  # noqa: BLE001
                async with lock:
                    error_count += 1
                console.print(
                    f"[yellow]Vector upsert failed for {point.uuid}: {exc}[/yellow]"
                )
            finally:
                progress.update(task_id, advance=1)

    with progress:
        task_id = progress.add_task(
            "Embedding and upserting vectors", total=len(points)
        )
        await asyncio.gather(*[_process_point(point, task_id) for point in points])

    await llm_client.aclose()
    return (success_count, error_count)


def _validate_args(args: argparse.Namespace) -> None:
    if args.limit <= 0:
        raise ValueError("--limit must be > 0")
    if args.page_size <= 0:
        raise ValueError("--page-size must be > 0")
    if args.months_back < 0:
        raise ValueError("--months-back must be >= 0")
    if args.max_pages_per_query <= 0:
        raise ValueError("--max-pages-per-query must be > 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.cache_ttl_hours < 0:
        raise ValueError("--cache-ttl-hours must be >= 0")
    if args.throttle_seconds < 0:
        raise ValueError("--throttle-seconds must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.embedding_concurrency <= 0:
        raise ValueError("--embedding-concurrency must be > 0")


async def _run(args: argparse.Namespace) -> int:
    _validate_args(args)

    console = Console()
    start_time = time.perf_counter()

    settings = Settings()
    if not PROPERTY_TOKEN_PATTERN.match(settings.node_uuid_property):
        raise ValueError(
            f"Invalid NODE_UUID_PROPERTY value: {settings.node_uuid_property}"
        )

    queries = args.queries or list(DEFAULT_QUERIES)
    now = datetime.now(timezone.utc)

    client = ArxivApiClient(
        user_agent=USER_AGENT,
        cache_dir=args.cache_dir,
        cache_ttl_seconds=args.cache_ttl_hours * 60 * 60,
        throttle_seconds=args.throttle_seconds,
        timeout_seconds=args.timeout,
    )

    papers, stats = await _collect_papers_async(
        client=client,
        queries=queries,
        limit=args.limit,
        page_size=args.page_size,
        max_pages_per_query=args.max_pages_per_query,
        months_back=args.months_back,
        now=now,
        console=console,
    )

    console.print(
        "Fetch stats: "
        f"raw={stats.raw_papers}, unique_recent={stats.unique_papers}, kept={stats.kept_papers}"
    )

    bundle = build_ingest_bundle(papers, node_uuid_property=settings.node_uuid_property)
    console.print(
        "Prepared bundle: "
        f"papers={len(bundle.papers)}, chunks={len(bundle.chunks)}, authors={len(bundle.authors)}, "
        f"universities={len(bundle.universities)}, categories={len(bundle.categories)}, "
        f"venues={len(bundle.venues)}, vectors={len(bundle.vector_points)}"
    )

    if args.dry_run:
        console.print(
            "[yellow]Dry run mode enabled: skipping Neo4j and Qdrant writes.[/yellow]"
        )
        elapsed = time.perf_counter() - start_time
        console.print(f"Completed dry run in {elapsed:.2f}s")
        return 0

    await _write_graph(
        settings,
        bundle=bundle,
        batch_size=args.batch_size,
        console=console,
    )

    success_count, error_count = await _write_vectors(
        settings,
        points=bundle.vector_points,
        concurrency=args.embedding_concurrency,
        console=console,
    )

    elapsed = time.perf_counter() - start_time
    console.print(
        f"Qdrant upsert complete: success={success_count}, errors={error_count}. "
        f"Elapsed={elapsed:.2f}s"
    )

    if error_count > 0:
        return 1
    return 0


def main() -> int:
    """Run the arXiv ingestion and indexing pipeline."""
    args = _parse_args()
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        Console().print("[red]Interrupted.[/red]")
        return 130
    except Exception as exc:  # noqa: BLE001
        Console().print(f"[red]Error:[/red] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
