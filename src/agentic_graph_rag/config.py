from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-5.2"

    # Neo4j
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Agent
    max_iterations: int = Field(default=10, ge=1)
    max_history_messages: int = Field(default=10, ge=0)

    # Trace Logging
    trace_log_dir: str = "logs"
    trace_logging_enabled: bool = True
