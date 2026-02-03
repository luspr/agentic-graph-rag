#!/usr/bin/env python3
"""CLI wrapper for fetching recent arXiv papers."""

from __future__ import annotations

import sys

from agentic_graph_rag.scripts.fetch_recent_papers import main


if __name__ == "__main__":
    sys.exit(main())
