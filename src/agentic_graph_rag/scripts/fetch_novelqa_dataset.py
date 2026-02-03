#!/usr/bin/env python3
"""Download the NovelQA dataset from Hugging Face."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset
from rich.console import Console


DEFAULT_DATASET = "NovelQA/NovelQA"
DEFAULT_OUTPUT_DIR = Path("data/novelqa")


@dataclass(frozen=True)
class DownloadConfig:
    """Configuration for downloading NovelQA."""

    dataset: str
    revision: str | None
    output_dir: Path
    cache_dir: Path | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the NovelQA dataset from Hugging Face."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset identifier on Hugging Face.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional dataset revision (branch, tag, or commit).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the dataset.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> DownloadConfig:
    return DownloadConfig(
        dataset=args.dataset,
        revision=args.revision,
        output_dir=args.out_dir,
        cache_dir=args.cache_dir,
    )


def _load_dataset(config: DownloadConfig) -> DatasetDict:
    dataset_kwargs: dict[str, Any] = {"path": config.dataset}
    if config.revision:
        dataset_kwargs["revision"] = config.revision
    if config.cache_dir:
        dataset_kwargs["cache_dir"] = str(config.cache_dir)
    return load_dataset(**dataset_kwargs)


def _write_dataset(dataset: DatasetDict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))


def main() -> int:
    """Run the NovelQA download script."""
    args = _parse_args()
    config = _config_from_args(args)
    console = Console()
    console.print(f"Downloading [bold]{config.dataset}[/bold]...")

    try:
        dataset = _load_dataset(config)
        _write_dataset(dataset, config.output_dir)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    console.print(f"[green]Done.[/green] Saved to {config.output_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
