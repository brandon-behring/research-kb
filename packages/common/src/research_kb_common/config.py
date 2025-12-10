"""Configuration management using Pydantic BaseSettings.

Loads configuration from environment variables with sensible defaults
for local development. In production, override via environment variables
or a .env file.

Usage:
    from research_kb_common.config import get_settings

    settings = get_settings()
    print(settings.database_url)
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment.

    All settings have development defaults. In production, override via:
    - Environment variables (e.g., DATABASE_URL=...)
    - .env file in working directory

    Attributes:
        database_url: PostgreSQL connection string
        grobid_url: GROBID service URL for PDF extraction
        embedding_model: Sentence transformer model for embeddings
        embedding_cache_dir: Directory for embedding model cache
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Log format (json or console)
        api_host: FastAPI server host
        api_port: FastAPI server port
        s2_api_key: Semantic Scholar API key (optional, higher rate limits)
        daemon_socket_path: Unix socket path for research-kb daemon
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/research_kb",
        description="PostgreSQL connection string",
    )

    # GROBID (PDF extraction)
    grobid_url: str = Field(
        default="http://localhost:8070",
        description="GROBID service URL",
    )

    # Embeddings
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Sentence transformer model name",
    )
    embedding_cache_dir: str = Field(
        default="~/.cache/sentence_transformers",
        description="Cache directory for embedding models",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    log_format: str = Field(
        default="console",
        description="Log format: json or console",
    )

    # API
    api_host: str = Field(
        default="0.0.0.0",
        description="FastAPI server host",
    )
    api_port: int = Field(
        default=8000,
        description="FastAPI server port",
    )

    # External APIs (optional)
    s2_api_key: Optional[str] = Field(
        default=None,
        description="Semantic Scholar API key (optional)",
    )

    # Daemon
    daemon_socket_path: str = Field(
        default="/tmp/research_kb_daemon.sock",
        description="Unix socket path for daemon",
    )

    # Telemetry (optional)
    otel_exporter_otlp_endpoint: Optional[str] = Field(
        default=None,
        description="OpenTelemetry OTLP endpoint",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return upper

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format."""
        valid_formats = {"json", "console"}
        lower = v.lower()
        if lower not in valid_formats:
            raise ValueError(f"log_format must be one of {valid_formats}")
        return lower


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.

    Settings are loaded once and cached for the lifetime of the process.
    Call get_settings.cache_clear() to reload.
    """
    return Settings()
