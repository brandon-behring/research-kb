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
from research_kb_common.gpu_guard import (
    VRAMMonitor,
    abort_if_embed_server_running,
    check_embed_server_running,
    check_vram_available,
    clear_gpu_cache,
    get_safe_batch_size,
    get_vram_stats,
    set_vram_ceiling,
)
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
    # GPU
    "set_vram_ceiling",
    "get_safe_batch_size",
    "get_vram_stats",
    "check_vram_available",
    "check_embed_server_running",
    "abort_if_embed_server_running",
    "clear_gpu_cache",
    "VRAMMonitor",
    # Errors
    "IngestionError",
    "ChunkExtractionError",
    "EmbeddingError",
    "ExtractionError",
    "ExtractionValidationError",
    "SearchError",
    "StorageError",
]
