#!/usr/bin/env python3
"""CLI wrapper for downloading the NovelQA dataset."""

from __future__ import annotations

import sys

from agentic_graph_rag.scripts.fetch_novelqa_dataset import main


if __name__ == "__main__":
    sys.exit(main())
