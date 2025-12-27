"""Research KB Common - Shared utilities.

Version: 1.0.0

This package provides:
- Structured logging (structlog)
- Retry/backoff patterns (tenacity)
- OpenTelemetry instrumentation helpers
- Custom error types
"""

from research_kb_common.config import Settings, get_settings
from research_kb_common.errors import (
    ChunkExtractionError,
    EmbeddingError,
    ExtractionError,
    ExtractionValidationError,
    IngestionError,
    SearchError,
    StorageError,
)
from research_kb_common.instrumentation import (
    get_tracer,
    init_telemetry,
    instrument_function,
)
from research_kb_common.logging_config import configure_logging, get_logger
from research_kb_common.retry import retry_on_exception, with_exponential_backoff

__version__ = "1.0.0"

__all__ = [
    # Config
    "Settings",
    "get_settings",
    # Logging
    "configure_logging",
    "get_logger",
    # Retry
    "retry_on_exception",
    "with_exponential_backoff",
    # Instrumentation
    "init_telemetry",
    "get_tracer",
    "instrument_function",
    # Errors
    "IngestionError",
    "ChunkExtractionError",
    "EmbeddingError",
    "ExtractionError",
    "ExtractionValidationError",
    "SearchError",
    "StorageError",
]
