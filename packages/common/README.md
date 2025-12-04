# research-kb-common

Common utilities for the research-kb system.

**Version**: 1.0.0

## Purpose

This package provides shared utilities used across all research-kb packages:

- **Structured logging** (structlog) - JSON output for production, human-readable for development
- **Retry/backoff patterns** (tenacity) - Resilient network and DB operations
- **OpenTelemetry instrumentation** - Tracing and observability
- **Custom error types** - Explicit error hierarchy

## Dependencies

- `research-kb-contracts` (for type hints only)
- `opentelemetry-api`, `opentelemetry-sdk`
- `tenacity` (retry patterns)
- `structlog` (structured logging)

## Usage

### Logging

```python
from research_kb_common import configure_logging, get_logger

# Configure once at app startup
configure_logging(level="INFO", json_output=False)

# Get logger in each module
logger = get_logger(__name__)

# Log with structured context
logger.info("chunk_created", chunk_id="xyz789", content_length=1024, source_id="abc123")
```

### Retry Patterns

```python
from research_kb_common import retry_on_exception

@retry_on_exception((ConnectionError, TimeoutError), max_attempts=5, min_wait_seconds=2.0)
async def call_grobid_api(pdf_bytes: bytes) -> dict:
    response = await client.post("/api/processFulltextDocument", data=pdf_bytes)
    return response.json()
```

### Instrumentation

```python
from research_kb_common import init_telemetry, instrument_function

# Initialize once at app startup
init_telemetry(service_name="research-kb-ingestion")

# Decorate functions for automatic tracing
@instrument_function("ingest_pdf")
async def ingest_source(file_path: str) -> Source:
    # Function automatically wrapped in OpenTelemetry span
    source = await process_pdf(file_path)
    return source
```

### Error Handling

```python
from research_kb_common import IngestionError, ChunkExtractionError

# Raise specific errors
raise ChunkExtractionError(f"Failed to extract chunks from {file_path}: {error}")

# Catch error hierarchy
try:
    await ingest_document(path)
except IngestionError as e:
    # Catches ChunkExtractionError, EmbeddingError, etc.
    logger.error("ingestion_failed", error=str(e))
```

## Testing

```bash
cd packages/common
poetry install
poetry run pytest
```

## Version Policy

This package follows semantic versioning. Breaking changes increment the major version.

See: `docs/plans/active/dazzling-soaring-origami.md` for architecture details.
