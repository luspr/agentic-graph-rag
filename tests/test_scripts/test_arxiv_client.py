"""Unit tests for the arXiv API client helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agentic_graph_rag.scripts.arxiv_client import (
    ArxivApiClient,
    base_arxiv_id,
    extract_arxiv_id,
    parse_arxiv_feed,
)

SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>42</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2501.12345v2</id>
    <updated>2025-01-25T00:00:00Z</updated>
    <published>2025-01-20T00:00:00Z</published>
    <title> Graph Retrieval with Hybrid Expansion </title>
    <summary> Test abstract text. </summary>
    <author>
      <name> Alice Example </name>
      <arxiv:affiliation>Example University</arxiv:affiliation>
    </author>
    <author>
      <name>Bob Example</name>
    </author>
    <arxiv:doi>10.1234/example</arxiv:doi>
    <arxiv:journal_ref>J. Test Systems 2025</arxiv:journal_ref>
    <arxiv:comment>Accepted at TestConf</arxiv:comment>
    <link rel="alternate" type="text/html" href="http://arxiv.org/abs/2501.12345v2" />
    <link title="pdf" rel="related" type="application/pdf" href="http://arxiv.org/pdf/2501.12345v2" />
    <arxiv:primary_category term="cs.IR" />
    <category term="cs.IR" scheme="http://arxiv.org/schemas/atom" />
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom" />
  </entry>
</feed>
"""


def test_parse_arxiv_feed_maps_fields() -> None:
    """parse_arxiv_feed maps atom/arxiv fields into ArxivPaper dataclass objects."""
    page = parse_arxiv_feed(
        SAMPLE_FEED,
        query="graph rag",
        start=0,
        max_results=10,
    )

    assert page.total_results == 42
    assert len(page.papers) == 1

    paper = page.papers[0]
    assert paper.arxiv_id == "2501.12345v2"
    assert paper.arxiv_id_base == "2501.12345"
    assert paper.title == "Graph Retrieval with Hybrid Expansion"
    assert paper.summary == "Test abstract text."
    assert paper.primary_category == "cs.IR"
    assert paper.categories == ["cs.IR", "cs.AI"]
    assert paper.doi == "10.1234/example"
    assert paper.journal_ref == "J. Test Systems 2025"
    assert paper.comment == "Accepted at TestConf"
    assert paper.pdf_url == "http://arxiv.org/pdf/2501.12345v2"
    assert [author.name for author in paper.authors] == ["Alice Example", "Bob Example"]
    assert paper.authors[0].affiliations == ["Example University"]
    assert paper.authors[1].affiliations == []


def test_extract_arxiv_id_and_base() -> None:
    """Identifier helpers parse and normalize versioned arXiv IDs."""
    assert extract_arxiv_id("https://arxiv.org/abs/2501.12345v3") == "2501.12345v3"
    assert extract_arxiv_id("arxiv:cs/0112017v2") == "cs/0112017v2"
    assert base_arxiv_id("2501.12345v3") == "2501.12345"
    assert base_arxiv_id("cs/0112017v2") == "cs/0112017"


@pytest.mark.anyio
async def test_search_uses_cache(tmp_path: Path) -> None:
    """search() reuses cached feed responses within the configured TTL."""
    client = ArxivApiClient(
        user_agent="test-agent",
        cache_dir=tmp_path,
        cache_ttl_seconds=3600,
        throttle_seconds=0.0,
    )
    client._fetch_feed_with_retry = AsyncMock(return_value=SAMPLE_FEED)  # type: ignore[method-assign]

    first = await client.search("graph rag", start=0, max_results=5)
    second = await client.search("graph rag", start=0, max_results=5)

    assert len(first.papers) == 1
    assert len(second.papers) == 1
    assert client._fetch_feed_with_retry.await_count == 1  # type: ignore[attr-defined]


def test_seconds_until_next_request_respects_throttle(tmp_path: Path) -> None:
    """Internal throttle calculation returns positive wait when requests are too close."""
    client = ArxivApiClient(
        user_agent="test-agent",
        cache_dir=tmp_path,
        throttle_seconds=3.5,
    )
    client._last_request_time = 10.0

    wait = client._seconds_until_next_request(12.0)

    assert wait == pytest.approx(1.5)


def test_retry_delay_adds_jitter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Retry delay uses exponential backoff plus jitter."""
    monkeypatch.setattr(
        "agentic_graph_rag.scripts.arxiv_client.random.uniform", lambda a, b: 0.2
    )

    client = ArxivApiClient(
        user_agent="test-agent",
        cache_dir=tmp_path,
        base_backoff_seconds=1.0,
        jitter_seconds=0.5,
    )

    delay = client._retry_delay(2)

    assert delay == pytest.approx(4.2)
