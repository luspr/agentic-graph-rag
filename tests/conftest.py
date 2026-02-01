from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in environments without dev deps
    load_dotenv = None


def pytest_configure() -> None:
    """Load .env for integration tests without overriding existing env vars."""
    if load_dotenv is None:
        return

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", override=False)
