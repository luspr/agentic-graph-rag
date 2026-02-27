#!/usr/bin/env python3
"""CLI wrapper for indexing arXiv papers into Neo4j and Qdrant."""

from __future__ import annotations

import sys

from agentic_graph_rag.scripts.index_arxiv_dataset import main


if __name__ == "__main__":
    sys.exit(main())
