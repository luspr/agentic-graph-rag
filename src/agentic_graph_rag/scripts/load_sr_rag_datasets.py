#!/usr/bin/env python3
"""Download and load the SR-RAG knowledge graph and benchmark datasets."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from neo4j import AsyncDriver, NotificationDisabledClassification
from neo4j import AsyncGraphDatabase as Neo4jAsyncGraphDatabase
from neo4j.exceptions import Neo4jError
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


load_dotenv()


HF_BASE_URL = "https://huggingface.co/datasets"
DEFAULT_DATA_DIR = Path("data/sr_rag")
TEMP_LOAD_LABEL = "SRRAG_TMP"
DEFAULT_BATCH_SIZE = 5000

KG_SPEC = {
    "repo": "Ning311/sr-rag-knowledge-graph",
    "files": ("nodes.jsonl.gz", "edges.jsonl.gz", "meta.json"),
}

BENCH_SPEC = {
    "repo": "Ning311/sr-rag-benchmark",
    "files": ("benchmark_1637.jsonl", "summary_benchmark_full.json", "meta.json"),
}

NODE_CONSTRAINTS = [
    "CREATE CONSTRAINT sr_rag_tmp_id IF NOT EXISTS "
    f"FOR (n:{TEMP_LOAD_LABEL}) REQUIRE n.id IS UNIQUE"
]

LABEL_TOKEN_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PROPERTY_TOKEN_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

NODE_UUID_PROPERTY = os.getenv("NODE_UUID_PROPERTY", "uuid")
if not PROPERTY_TOKEN_PATTERN.match(NODE_UUID_PROPERTY):
    raise RuntimeError("NODE_UUID_PROPERTY must be a valid Neo4j property token.")


def _node_merge_query(label: str) -> str:
    return (
        "UNWIND $rows AS row "
        f"MERGE (n:{TEMP_LOAD_LABEL} {{id: row.id}}) "
        "SET n += row.properties "
        f"SET n.{NODE_UUID_PROPERTY} = coalesce(n.{NODE_UUID_PROPERTY}, row.uuid) "
        f"SET n:{label}"
    )


def _edge_merge_query(relation_type: str) -> str:
    return (
        "UNWIND $rows AS row "
        f"MATCH (source:{TEMP_LOAD_LABEL} {{id: row.source}}) "
        f"MATCH (target:{TEMP_LOAD_LABEL} {{id: row.target}}) "
        f"MERGE (source)-[r:{relation_type} {{key: row.key}}]->(target) "
        "SET r.relation = row.relation"
    )


@dataclass(frozen=True)
class Neo4jConfig:
    """Connection settings for Neo4j."""

    uri: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """Load Neo4j connection settings from environment variables."""
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        missing = [
            name
            for name, value in (
                ("NEO4J_URI", uri),
                ("NEO4J_USER", user),
                ("NEO4J_PASSWORD", password),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Missing required environment variables: " + ", ".join(missing)
            )
        return cls(uri=uri or "", user=user or "", password=password or "")


class Neo4jLoader:
    """Minimal async Neo4j wrapper for bulk loading."""

    def __init__(self, config: Neo4jConfig) -> None:
        """Initialize the loader with Neo4j connection settings."""
        self._config = config
        self._driver: AsyncDriver | None = None

    async def __aenter__(self) -> "Neo4jLoader":
        """Open the Neo4j driver."""
        self._driver = Neo4jAsyncGraphDatabase.driver(
            self._config.uri,
            auth=(self._config.user, self._config.password),
            notifications_disabled_classifications=[
                NotificationDisabledClassification.DEPRECATION,
                NotificationDisabledClassification.GENERIC,
            ],
        )
        return self

    async def __aexit__(self, exc_type, exc, exc_tb) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    @property
    def _active_driver(self) -> AsyncDriver:
        """Return the active driver, raising if not connected."""
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized.")
        return self._driver

    async def execute(self, cypher: str, params: dict[str, Any] | None = None) -> None:
        """Execute a Cypher query and raise on failure."""
        try:
            async with self._active_driver.session() as session:
                result = await session.run(cypher, params or {})
                await result.consume()
        except Neo4jError as exc:
            raise RuntimeError(f"Neo4j error: {exc}") from exc


def _build_hf_url(repo: str, filename: str, revision: str) -> str:
    return f"{HF_BASE_URL}/{repo}/resolve/{revision}/{filename}"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _download_file(
    url: str, destination: Path, force: bool, progress: Progress
) -> None:
    if destination.exists() and not force:
        return
    _ensure_dir(destination.parent)

    request = Request(url, headers={"User-Agent": "agentic-graph-rag/0.1"})
    try:
        with urlopen(request, timeout=60) as response, destination.open("wb") as handle:
            total = response.headers.get("Content-Length")
            total_size = int(total) if total and total.isdigit() else None
            task = progress.add_task(
                f"Downloading {destination.name}", total=total_size
            )
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                progress.update(task, advance=len(chunk))
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def _download_dataset(
    spec: dict[str, Any],
    target_dir: Path,
    revision: str,
    force: bool,
    console: Console,
) -> None:
    console.print(f"Downloading [bold]{spec['repo']}[/bold] -> {target_dir}")
    _ensure_dir(target_dir)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        for filename in spec["files"]:
            url = _build_hf_url(spec["repo"], filename, revision)
            destination = target_dir / filename
            _download_file(url, destination, force, progress)


def _iter_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _batch(
    iterator: Iterable[dict[str, Any]], batch_size: int
) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _group_rows(
    rows: Iterable[dict[str, Any]], key: str
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group_key = row[key]
        grouped.setdefault(group_key, []).append(row)
    return grouped


def _node_row(raw: dict[str, Any]) -> dict[str, Any]:
    node_id = str(raw.get("id", "")).strip()
    if not node_id:
        raise ValueError("Node entry missing id.")
    properties: dict[str, Any] = {}
    _add_property(properties, "node_label", raw.get("label"))
    _add_property(properties, "level", raw.get("level"))
    _add_property(properties, "chunk_id", raw.get("chunk_id"))
    _add_property(properties, "schema_type", raw.get("schema_type"))

    name_value = raw.get("name")
    name_text = _format_name_value(name_value)
    if name_text:
        properties["name"] = name_text
    elif name_value is not None:
        _add_property(properties, "name", name_value)

    schema_type = raw.get("schema_type")
    fallback_label = raw.get("label")
    label_source = schema_type or fallback_label or "unclassified"
    label_token = _to_neo4j_token(str(label_source), prefix="Type")
    properties["schema_type_label"] = label_token
    return {
        "id": node_id,
        "label": label_token,
        "properties": properties,
        "uuid": str(uuid.uuid4()),
    }


def _edge_row(raw: dict[str, Any]) -> dict[str, Any]:
    source = str(raw.get("source", "")).strip()
    target = str(raw.get("target", "")).strip()
    relation = str(raw.get("relation", "")).strip()
    if not source or not target or not relation:
        raise ValueError("Edge entry missing source, target, or relation.")
    key_value = raw.get("key")
    if isinstance(key_value, int):
        key = key_value
    elif isinstance(key_value, str) and key_value.isdigit():
        key = int(key_value)
    else:
        key = 0
    relation_type = _to_neo4j_token(relation, prefix="Rel")
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "relation_type": relation_type,
        "key": key,
    }


def _add_property(properties: dict[str, Any], key: str, value: Any) -> None:
    normalized = _normalize_property_value(value)
    if normalized is None:
        return
    properties[key] = normalized


def _normalize_property_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        normalized_items = [_normalize_property_value(item) for item in value]
        if any(item is None for item in normalized_items):
            return json.dumps(value, ensure_ascii=True, sort_keys=True)
        if all(isinstance(item, bool | int | float | str) for item in normalized_items):
            return list(normalized_items)
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _format_name_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return None
    parts: list[str] = []
    for key in ("key", "value", "unit", "cohort"):
        entry = value.get(key)
        if entry is not None:
            parts.append(str(entry))
    return " ".join(parts) if parts else None


def _to_neo4j_token(value: str, prefix: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = f"{prefix}Unknown"
    if cleaned[0].isdigit():
        cleaned = f"{prefix}_{cleaned}"
    if not LABEL_TOKEN_PATTERN.match(cleaned):
        return f"{prefix}Unknown"
    return cleaned


def _load_meta_counts(meta_path: Path) -> tuple[int | None, int | None]:
    if not meta_path.exists():
        return (None, None)
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    node_count = meta.get("node_count")
    edge_count = meta.get("edge_count")
    return (_coerce_int(node_count), _coerce_int(edge_count))


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


async def _ensure_constraints(loader: Neo4jLoader) -> None:
    for query in NODE_CONSTRAINTS:
        await loader.execute(query)


async def _remove_temp_label(loader: Neo4jLoader, console: Console) -> None:
    console.print("Removing temporary load label...")
    await loader.execute(f"MATCH (n:{TEMP_LOAD_LABEL}) REMOVE n:{TEMP_LOAD_LABEL}")


async def _load_nodes(
    loader: Neo4jLoader,
    nodes_path: Path,
    batch_size: int,
    total: int | None,
    progress: Progress,
) -> int:
    loaded = 0
    task = progress.add_task("Loading nodes", total=total)
    for batch in _batch(_iter_jsonl_gz(nodes_path), batch_size):
        rows = [_node_row(node) for node in batch]
        grouped = _group_rows(rows, "label")
        for label, label_rows in grouped.items():
            query = _node_merge_query(label)
            await loader.execute(query, {"rows": label_rows})
        loaded += len(rows)
        progress.update(task, advance=len(rows))
    return loaded


async def _load_edges(
    loader: Neo4jLoader,
    edges_path: Path,
    batch_size: int,
    total: int | None,
    progress: Progress,
) -> int:
    loaded = 0
    task = progress.add_task("Loading edges", total=total)
    for batch in _batch(_iter_jsonl_gz(edges_path), batch_size):
        rows = [_edge_row(edge) for edge in batch]
        grouped = _group_rows(rows, "relation_type")
        for relation_type, rel_rows in grouped.items():
            query = _edge_merge_query(relation_type)
            await loader.execute(query, {"rows": rel_rows})
        loaded += len(rows)
        progress.update(task, advance=len(rows))
    return loaded


async def _load_knowledge_graph(
    config: Neo4jConfig,
    data_dir: Path,
    batch_size: int,
    console: Console,
) -> None:
    nodes_path = data_dir / "nodes.jsonl.gz"
    edges_path = data_dir / "edges.jsonl.gz"
    meta_path = data_dir / "meta.json"

    if not nodes_path.exists() or not edges_path.exists():
        raise RuntimeError(
            "Knowledge graph files not found. Ensure nodes.jsonl.gz and edges.jsonl.gz exist."
        )

    node_total, edge_total = _load_meta_counts(meta_path)

    console.print("Loading knowledge graph into Neo4j...")
    async with Neo4jLoader(config) as loader:
        await _ensure_constraints(loader)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            await _load_nodes(loader, nodes_path, batch_size, node_total, progress)
            await _load_edges(loader, edges_path, batch_size, edge_total, progress)
        await _remove_temp_label(loader, console)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and load SR-RAG datasets into Neo4j."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory to store downloaded datasets.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Hugging Face dataset revision to download.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for Neo4j writes.",
    )
    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Skip loading the knowledge graph into Neo4j (download only).",
    )
    parser.add_argument(
        "--skip-knowledge-graph",
        action="store_true",
        help="Skip the knowledge graph dataset.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the benchmark dataset.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the dataset loader."""
    args = _parse_args()
    console = Console()
    data_dir = args.data_dir
    kg_dir = data_dir / "knowledge_graph"
    bench_dir = data_dir / "benchmark"

    try:
        if not args.skip_knowledge_graph:
            _download_dataset(
                KG_SPEC, kg_dir, args.revision, args.force_download, console
            )

        if not args.skip_benchmark:
            _download_dataset(
                BENCH_SPEC, bench_dir, args.revision, args.force_download, console
            )

        if not args.skip_neo4j and not args.skip_knowledge_graph:
            config = Neo4jConfig.from_env()
            asyncio.run(
                _load_knowledge_graph(
                    config=config,
                    data_dir=kg_dir,
                    batch_size=args.batch_size,
                    console=console,
                )
            )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    console.print("[green]Done.[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
