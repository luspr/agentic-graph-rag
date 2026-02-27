"""ArXiv API client utilities with polite rate limiting and on-disk response cache."""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
import time
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"
OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"
ARXIV_NS = "http://arxiv.org/schemas/atom"
NS = {"atom": ATOM_NS, "opensearch": OPENSEARCH_NS, "arxiv": ARXIV_NS}

_TRANSIENT_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class ArxivLink:
    """Representation of a link attached to an arXiv paper entry."""

    href: str
    rel: str | None
    type: str | None
    title: str | None


@dataclass(frozen=True)
class ArxivAuthor:
    """Representation of a single arXiv author."""

    name: str
    affiliations: list[str]


@dataclass(frozen=True)
class ArxivPaper:
    """Representation of a single arXiv paper entry."""

    id_url: str
    arxiv_id: str
    arxiv_id_base: str
    title: str
    summary: str
    published: str
    updated: str
    authors: list[ArxivAuthor]
    primary_category: str | None
    categories: list[str]
    doi: str | None
    journal_ref: str | None
    comment: str | None
    links: list[ArxivLink]
    query: str

    @property
    def pdf_url(self) -> str | None:
        """Return the PDF URL if present in entry links."""
        for link in self.links:
            if link.type == "application/pdf" and link.href:
                return link.href
            if link.title == "pdf" and link.href:
                return link.href
        return None


@dataclass(frozen=True)
class ArxivSearchPage:
    """A single page of arXiv search results."""

    query: str
    start: int
    max_results: int
    total_results: int | None
    papers: list[ArxivPaper]


class ArxivApiClient:
    """Thin asynchronous arXiv API client with cache and polite throttling."""

    def __init__(
        self,
        *,
        user_agent: str,
        cache_dir: Path,
        cache_ttl_seconds: int = 24 * 60 * 60,
        throttle_seconds: float = 3.5,
        timeout_seconds: float = 30.0,
        max_retries: int = 4,
        base_backoff_seconds: float = 1.0,
        jitter_seconds: float = 0.3,
    ) -> None:
        """Initialize the client.

        Args:
            user_agent: User agent sent with arXiv requests.
            cache_dir: Directory for cached feed responses.
            cache_ttl_seconds: Cache TTL in seconds.
            throttle_seconds: Minimum seconds between upstream requests.
            timeout_seconds: Request timeout.
            max_retries: Max retries for transient failures.
            base_backoff_seconds: Base delay used for retry backoff.
            jitter_seconds: Random jitter added to retry backoff.
        """
        self._user_agent = user_agent
        self._cache_dir = cache_dir
        self._cache_ttl_seconds = cache_ttl_seconds
        self._throttle_seconds = throttle_seconds
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._base_backoff_seconds = base_backoff_seconds
        self._jitter_seconds = jitter_seconds
        self._last_request_time: float | None = None

    async def search(
        self,
        query: str,
        *,
        start: int,
        max_results: int,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
    ) -> ArxivSearchPage:
        """Search arXiv and return a parsed result page."""
        if start < 0:
            raise ValueError("start must be >= 0")
        if max_results <= 0:
            raise ValueError("max_results must be > 0")

        url = self._build_query_url(
            query=query,
            start=start,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        xml_text = await self._get_feed(url)
        return parse_arxiv_feed(
            xml_text,
            query=query,
            start=start,
            max_results=max_results,
        )

    def _build_query_url(
        self,
        *,
        query: str,
        start: int,
        max_results: int,
        sort_by: str,
        sort_order: str,
    ) -> str:
        params = {
            "search_query": f'all:"{query}"',
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        return f"{ARXIV_API_URL}?{urlencode(params)}"

    async def _get_feed(self, url: str) -> str:
        cached = self._read_cache(url)
        if cached is not None:
            return cached

        xml_text = await self._fetch_feed_with_retry(url)
        self._write_cache(url, xml_text)
        return xml_text

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self._cache_dir / f"{digest}.xml"

    def _read_cache(self, url: str) -> str | None:
        cache_path = self._cache_path(url)
        if not cache_path.exists():
            return None
        age_seconds = time.time() - cache_path.stat().st_mtime
        if age_seconds > self._cache_ttl_seconds:
            return None
        return cache_path.read_text(encoding="utf-8")

    def _write_cache(self, url: str, xml_text: str) -> None:
        cache_path = self._cache_path(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(xml_text, encoding="utf-8")

    async def _fetch_feed_with_retry(self, url: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            wait_seconds = self._seconds_until_next_request(time.monotonic())
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            try:
                xml_text = await asyncio.to_thread(self._fetch_feed_blocking, url)
                self._last_request_time = time.monotonic()
                return xml_text
            except (HTTPError, URLError, TimeoutError) as exc:
                self._last_request_time = time.monotonic()
                last_error = exc
                is_transient = self._is_transient_error(exc)
                if not is_transient or attempt >= self._max_retries:
                    break
                await asyncio.sleep(self._retry_delay(attempt))

        raise RuntimeError(f"Failed to fetch arXiv feed: {last_error}") from last_error

    def _fetch_feed_blocking(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": self._user_agent})
        with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310
            return response.read().decode("utf-8")

    def _seconds_until_next_request(self, now_monotonic: float) -> float:
        if self._last_request_time is None:
            return 0.0
        elapsed = now_monotonic - self._last_request_time
        return max(0.0, self._throttle_seconds - elapsed)

    def _retry_delay(self, attempt: int) -> float:
        exponential = self._base_backoff_seconds * (2**attempt)
        jitter = random.uniform(0.0, self._jitter_seconds)
        return exponential + jitter

    def _is_transient_error(self, exc: Exception) -> bool:
        if isinstance(exc, HTTPError):
            return exc.code in _TRANSIENT_HTTP_CODES
        if isinstance(exc, URLError):
            return True
        return isinstance(exc, TimeoutError)


def parse_arxiv_feed(
    xml_text: str,
    *,
    query: str,
    start: int,
    max_results: int,
) -> ArxivSearchPage:
    """Parse an arXiv Atom feed into typed search results."""
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as exc:
        raise RuntimeError(f"Failed to parse arXiv feed: {exc}") from exc

    papers = [
        _entry_to_paper(entry, query=query) for entry in root.findall("atom:entry", NS)
    ]

    total_results = _parse_total_results(root)
    return ArxivSearchPage(
        query=query,
        start=start,
        max_results=max_results,
        total_results=total_results,
        papers=papers,
    )


def extract_arxiv_id(value: str) -> str | None:
    """Extract the arXiv identifier from URL/text."""
    match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", value)
    if match:
        return match.group(0)
    legacy_match = re.search(r"([a-z-]+/\d{7})(v\d+)?", value, re.IGNORECASE)
    if legacy_match:
        return legacy_match.group(0)
    return None


def base_arxiv_id(arxiv_id: str) -> str:
    """Return the versionless arXiv identifier."""
    return re.sub(r"v\d+$", "", arxiv_id)


def _parse_total_results(root: ElementTree.Element) -> int | None:
    total_node = root.find("opensearch:totalResults", NS)
    if total_node is None or not total_node.text:
        return None
    total_text = total_node.text.strip()
    return int(total_text) if total_text.isdigit() else None


def _entry_to_paper(entry: ElementTree.Element, *, query: str) -> ArxivPaper:
    id_url = _text(entry, "atom:id")
    arxiv_id = extract_arxiv_id(id_url) or id_url
    categories = _categories(entry)
    primary_category = _primary_category(entry)
    if primary_category and primary_category not in categories:
        categories.insert(0, primary_category)

    return ArxivPaper(
        id_url=id_url,
        arxiv_id=arxiv_id,
        arxiv_id_base=base_arxiv_id(arxiv_id),
        title=_text(entry, "atom:title"),
        summary=_text(entry, "atom:summary"),
        published=_text(entry, "atom:published"),
        updated=_text(entry, "atom:updated"),
        authors=_authors(entry),
        primary_category=primary_category,
        categories=categories,
        doi=_text_or_none(entry, "arxiv:doi"),
        journal_ref=_text_or_none(entry, "arxiv:journal_ref"),
        comment=_text_or_none(entry, "arxiv:comment"),
        links=_links(entry),
        query=query,
    )


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def _text(entry: ElementTree.Element, tag: str) -> str:
    node = entry.find(tag, NS)
    return _normalize_text(node.text if node is not None else None)


def _text_or_none(entry: ElementTree.Element, tag: str) -> str | None:
    text = _text(entry, tag)
    return text or None


def _authors(entry: ElementTree.Element) -> list[ArxivAuthor]:
    authors: list[ArxivAuthor] = []
    for author_node in entry.findall("atom:author", NS):
        name_node = author_node.find("atom:name", NS)
        name = _normalize_text(name_node.text if name_node is not None else None)
        if not name:
            continue
        affiliations: list[str] = []
        for aff_node in author_node.findall("arxiv:affiliation", NS):
            affiliation = _normalize_text(aff_node.text)
            if affiliation:
                affiliations.append(affiliation)
        authors.append(ArxivAuthor(name=name, affiliations=affiliations))
    return authors


def _links(entry: ElementTree.Element) -> list[ArxivLink]:
    links: list[ArxivLink] = []
    for link_node in entry.findall("atom:link", NS):
        href = _normalize_text(link_node.attrib.get("href"))
        if not href:
            continue
        links.append(
            ArxivLink(
                href=href,
                rel=link_node.attrib.get("rel"),
                type=link_node.attrib.get("type"),
                title=link_node.attrib.get("title"),
            )
        )
    return links


def _categories(entry: ElementTree.Element) -> list[str]:
    categories: list[str] = []
    for category_node in entry.findall("atom:category", NS):
        term = _normalize_text(category_node.attrib.get("term"))
        if term and term not in categories:
            categories.append(term)
    return categories


def _primary_category(entry: ElementTree.Element) -> str | None:
    node = entry.find("arxiv:primary_category", NS)
    if node is None:
        return None
    term = _normalize_text(node.attrib.get("term"))
    return term or None


def paper_to_dict(paper: ArxivPaper) -> dict[str, Any]:
    """Convert an ArxivPaper to a JSON-serializable dictionary."""
    return {
        "id": paper.id_url,
        "arxiv_id": paper.arxiv_id,
        "arxiv_id_base": paper.arxiv_id_base,
        "title": paper.title,
        "authors": [
            {"name": author.name, "affiliations": author.affiliations}
            for author in paper.authors
        ],
        "published": paper.published,
        "updated": paper.updated,
        "summary": paper.summary,
        "url": paper.id_url,
        "pdf_url": paper.pdf_url,
        "primary_category": paper.primary_category,
        "categories": paper.categories,
        "doi": paper.doi,
        "journal_ref": paper.journal_ref,
        "comment": paper.comment,
        "query": paper.query,
        "source": "arxiv",
    }
