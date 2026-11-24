"""Configuration via environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings — all overridable via env vars or .env file."""

    # Core
    environment: str = Field(default="development", env="ENV")
    log_dir: Path = Field(default=Path("./logs"), env="LOG_DIR")
    results_dir: Path = Field(default=Path("./results"), env="RESULTS_DIR")

    # Embeddings / Vector search
    embedder: str = Field(default="ollama:nomic-embed-text", env="EMBEDDER")
    use_ollama: bool = Field(default=True, env="USE_OLLAMA")
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")

    # Vector DB
    vector_db: str = Field(default="qdrant", env="VECTOR_DB")
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")

    # LLM via PydanticAI
    llm_model: str = Field(
        default="anthropic:claude-3-5-sonnet-20241022",
        env="LLM_MODEL",
    )

    # Evaluation
    golden_set_path: Path = Field(
        default=Path("./tests/golden_set.jsonl"),
        env="GOLDEN_SET_PATH",
    )
    eval_metrics: str = Field(
        default="groundedness, faithfulness, answer_relevance",
        env="EVAL_METRICS",
    )

    # Observability
    otlp_endpoint: Optional[str] = Field(default=None, env="OTEL_EXPORTER_OTLP_ENDPOINT")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings