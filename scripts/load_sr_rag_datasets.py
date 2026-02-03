#!/usr/bin/env python3
"""CLI wrapper for the SR-RAG dataset loader."""

from __future__ import annotations

import sys

from agentic_graph_rag.scripts.load_sr_rag_datasets import main


if __name__ == "__main__":
    sys.exit(main())
