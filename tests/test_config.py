import pytest
from pydantic import ValidationError

from agentic_graph_rag.config import Settings


def test_settings_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings loads all required fields from environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key == "sk-test-key"
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.neo4j_password == "secret"


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default values are applied when optional vars are not set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    for var in (
        "OPENAI_MODEL",
        "QDRANT_HOST",
        "QDRANT_PORT",
        "QDRANT_COLLECTION",
        "EMBEDDING_DIM",
        "NODE_UUID_PROPERTY",
        "MAX_ITERATIONS",
        "MAX_HISTORY_MESSAGES",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = Settings(_env_file=None)

    assert settings.openai_model == "gpt-5.2"
    assert settings.qdrant_host == "localhost"
    assert settings.qdrant_port == 6333
    assert settings.qdrant_collection == "graph_nodes"
    assert settings.embedding_dim == 1536
    assert settings.node_uuid_property == "uuid"
    assert settings.max_iterations == 10
    assert settings.max_history_messages == 10


def test_settings_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables override default values."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("QDRANT_HOST", "qdrant.example.com")
    monkeypatch.setenv("QDRANT_PORT", "6334")
    monkeypatch.setenv("QDRANT_COLLECTION", "custom_collection")
    monkeypatch.setenv("EMBEDDING_DIM", "128")
    monkeypatch.setenv("NODE_UUID_PROPERTY", "node_uuid")
    monkeypatch.setenv("MAX_ITERATIONS", "5")
    monkeypatch.setenv("MAX_HISTORY_MESSAGES", "3")

    settings = Settings(_env_file=None)

    assert settings.openai_model == "gpt-4o"
    assert settings.qdrant_host == "qdrant.example.com"
    assert settings.qdrant_port == 6334
    assert settings.qdrant_collection == "custom_collection"
    assert settings.embedding_dim == 128
    assert settings.node_uuid_property == "node_uuid"
    assert settings.max_iterations == 5
    assert settings.max_history_messages == 3


def test_settings_missing_required_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing required fields raise a ValidationError."""
    # Only set one required field — the rest are missing
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # Clear any env vars that might leak in
    for var in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(ValidationError):
        # Use _env_file=None to prevent loading from .env file during test
        Settings(_env_file=None)
