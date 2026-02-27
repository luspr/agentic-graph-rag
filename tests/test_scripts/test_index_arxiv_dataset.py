"""Unit tests for arXiv indexing helpers."""

from __future__ import annotations

from datetime import datetime, timezone

from agentic_graph_rag.scripts.arxiv_client import ArxivAuthor, ArxivLink, ArxivPaper
from agentic_graph_rag.scripts.index_arxiv_dataset import (
    CollectedPaper,
    _author_uuid,
    _category_uuid,
    _chunk_uuid,
    _is_recent,
    _paper_uuid,
    _subtract_months,
    _university_uuid,
    _venue_uuid,
    build_ingest_bundle,
)


def _paper(
    *,
    arxiv_id: str,
    title: str,
    summary: str,
    published: str,
    updated: str,
    authors: list[ArxivAuthor],
    categories: list[str],
    primary_category: str | None,
    journal_ref: str | None = None,
) -> ArxivPaper:
    return ArxivPaper(
        id_url=f"http://arxiv.org/abs/{arxiv_id}",
        arxiv_id=arxiv_id,
        arxiv_id_base=arxiv_id.split("v")[0],
        title=title,
        summary=summary,
        published=published,
        updated=updated,
        authors=authors,
        primary_category=primary_category,
        categories=categories,
        doi=None,
        journal_ref=journal_ref,
        comment=None,
        links=[
            ArxivLink(
                href=f"http://arxiv.org/pdf/{arxiv_id}",
                rel=None,
                type="application/pdf",
                title="pdf",
            )
        ],
        query="graph rag",
    )


def test_deterministic_entity_ids() -> None:
    """Entity UUID helper functions are deterministic and normalized."""
    assert _paper_uuid("2501.12345") == "paper:2501.12345"
    assert _chunk_uuid("2501.12345") == "chunk:2501.12345:0"
    assert _author_uuid("Alice Example") == "author:alice_example"
    assert _university_uuid("Example University") == "university:example_university"
    assert _category_uuid("cs.AI") == "category:cs_ai"
    assert _venue_uuid("J. Testing 2025") == "venue:j_testing_2025"


def test_subtract_months_handles_calendar_edges() -> None:
    """Month subtraction clamps day to end-of-month when necessary."""
    reference = datetime(2024, 3, 31, 8, 0, tzinfo=timezone.utc)

    value = _subtract_months(reference, 1)

    assert value.year == 2024
    assert value.month == 2
    assert value.day == 29


def test_is_recent_uses_published_or_updated() -> None:
    """Recency check accepts records with either recent published or updated timestamps."""
    cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)

    published_recent = _paper(
        arxiv_id="2501.00001v1",
        title="Recent paper",
        summary="summary",
        published="2025-01-03T00:00:00Z",
        updated="2025-01-03T00:00:00Z",
        authors=[ArxivAuthor(name="A", affiliations=[])],
        categories=["cs.AI"],
        primary_category="cs.AI",
    )
    updated_recent = _paper(
        arxiv_id="2401.00001v2",
        title="Updated paper",
        summary="summary",
        published="2024-01-03T00:00:00Z",
        updated="2025-01-10T00:00:00Z",
        authors=[ArxivAuthor(name="B", affiliations=[])],
        categories=["cs.IR"],
        primary_category="cs.IR",
    )
    old_paper = _paper(
        arxiv_id="2301.00001v1",
        title="Old paper",
        summary="summary",
        published="2023-01-03T00:00:00Z",
        updated="2023-01-10T00:00:00Z",
        authors=[ArxivAuthor(name="C", affiliations=[])],
        categories=["cs.CL"],
        primary_category="cs.CL",
    )

    assert _is_recent(published_recent, cutoff) is True
    assert _is_recent(updated_recent, cutoff) is True
    assert _is_recent(old_paper, cutoff) is False


def test_build_ingest_bundle_builds_entities_and_payloads() -> None:
    """Bundle builder emits all requested entity rows, edges, and vector payload keys."""
    paper_one = _paper(
        arxiv_id="2501.12345v1",
        title="Graph RAG Paper",
        summary="Uses hybrid retrieval.",
        published="2025-01-20T00:00:00Z",
        updated="2025-01-21T00:00:00Z",
        authors=[
            ArxivAuthor(name="Alice Example", affiliations=["Example University"]),
            ArxivAuthor(name="Bob Example", affiliations=[]),
        ],
        categories=["cs.IR", "cs.AI"],
        primary_category="cs.IR",
        journal_ref="J. Test Systems 2025",
    )
    paper_two = _paper(
        arxiv_id="2501.22222v1",
        title="Knowledge Graph Planning",
        summary="Planning with graph traversal.",
        published="2025-01-18T00:00:00Z",
        updated="2025-01-19T00:00:00Z",
        authors=[
            ArxivAuthor(name="Alice Example", affiliations=["Example University"])
        ],
        categories=["cs.AI"],
        primary_category="cs.AI",
    )

    bundle = build_ingest_bundle(
        [
            CollectedPaper(paper=paper_one, matched_queries=["graph rag"]),
            CollectedPaper(paper=paper_two, matched_queries=["knowledge graph"]),
        ],
        node_uuid_property="uuid",
    )

    assert len(bundle.papers) == 2
    assert len(bundle.chunks) == 2
    assert len(bundle.authors) == 2
    assert len(bundle.universities) == 1
    assert len(bundle.categories) == 2
    assert len(bundle.venues) == 1

    assert len(bundle.authored_edges) == 3
    assert len(bundle.affiliated_edges) == 1
    assert len(bundle.has_chunk_edges) == 2
    assert len(bundle.in_category_edges) == 3
    assert len(bundle.published_in_edges) == 1

    payload_by_uuid = {point.uuid: point.payload for point in bundle.vector_points}
    for payload in payload_by_uuid.values():
        assert payload["uuid"]
        assert payload["dataset"] == "arxiv"
        assert payload["entity_type"]
        assert payload["label"]
        assert payload["labels"]
        assert payload["display_name"]

    assert "paper:2501.12345" in payload_by_uuid
    assert "chunk:2501.12345:0" in payload_by_uuid
    assert "author:alice_example" in payload_by_uuid
    assert "university:example_university" in payload_by_uuid
    assert "category:cs_ir" in payload_by_uuid
    assert "venue:j_test_systems_2025" in payload_by_uuid


def test_build_ingest_bundle_is_deterministic() -> None:
    """Repeated bundle builds over identical input produce identical UUID sets."""
    paper = _paper(
        arxiv_id="2501.12345v1",
        title="Graph RAG Paper",
        summary="Uses hybrid retrieval.",
        published="2025-01-20T00:00:00Z",
        updated="2025-01-21T00:00:00Z",
        authors=[
            ArxivAuthor(name="Alice Example", affiliations=["Example University"])
        ],
        categories=["cs.IR"],
        primary_category="cs.IR",
    )
    collected = [CollectedPaper(paper=paper, matched_queries=["graph rag"])]

    first = build_ingest_bundle(collected, node_uuid_property="uuid")
    second = build_ingest_bundle(collected, node_uuid_property="uuid")

    first_ids = sorted(point.uuid for point in first.vector_points)
    second_ids = sorted(point.uuid for point in second.vector_points)
    assert first_ids == second_ids
