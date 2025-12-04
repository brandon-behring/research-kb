## research-kb-storage

PostgreSQL storage layer for the research-kb system.

**Version**: 1.0.0

## Purpose

This package provides **exclusive database access** for all research-kb operations:

- **Connection management** - asyncpg connection pooling
- **SourceStore** - CRUD operations for sources table
- **ChunkStore** - CRUD operations for chunks table with pgvector support
- **Hybrid search** - FTS + vector similarity search

**Exclusive DB ownership** - No other packages access PostgreSQL directly.

## Dependencies

- `research-kb-contracts` (schemas)
- `research-kb-common` (logging, errors)
- `asyncpg` (async PostgreSQL driver)
- `pgvector` (vector support)

## Usage

### Connection Setup

```python
from research_kb_storage import DatabaseConfig, get_connection_pool

# Configure database
config = DatabaseConfig(
    host="localhost",
    port=5432,
    database="research_kb",
    user="postgres",
    password="postgres",
)

# Initialize connection pool (once at startup)
pool = await get_connection_pool(config)
```

### Source Operations

```python
from research_kb_storage import SourceStore
from research_kb_contracts import SourceType

# Create source
source = await SourceStore.create(
    source_type=SourceType.TEXTBOOK,
    title="Causality: Models, Reasoning, and Inference",
    file_hash="sha256:abc123",
    authors=["Judea Pearl"],
    metadata={"isbn": "978-0521895606"},
)

# Retrieve by ID
source = await SourceStore.get_by_id(source_id)

# Check if file already ingested
existing = await SourceStore.get_by_file_hash("sha256:abc123")

# Update metadata
source = await SourceStore.update_metadata(
    source_id=source.id,
    metadata={"citations_count": 1200},
)
```

### Chunk Operations

```python
from research_kb_storage import ChunkStore

# Create chunk with embedding
chunk = await ChunkStore.create(
    source_id=source.id,
    content="The backdoor criterion states...",
    content_hash="sha256:chunk123",
    location="Chapter 3, p. 73",
    embedding=[0.1] * 384,  # BGE-large-en-v1.5
    metadata={"chunk_type": "theorem"},
)

# Batch create (for ingestion pipeline)
chunks = await ChunkStore.batch_create([
    {"source_id": source.id, "content": "...", "content_hash": "..."},
    {"source_id": source.id, "content": "...", "content_hash": "..."},
])

# List chunks for a source
chunks = await ChunkStore.list_by_source(source.id, limit=100)

# Count chunks
count = await ChunkStore.count_by_source(source.id)
```

### Hybrid Search

```python
from research_kb_storage import SearchQuery, search_hybrid

# FTS + vector hybrid search
results = await search_hybrid(SearchQuery(
    text="backdoor criterion",
    embedding=[0.1] * 384,
    fts_weight=0.3,
    vector_weight=0.7,
    limit=10,
))

for result in results:
    print(f"Rank {result.rank}: {result.source.title}")
    print(f"  Location: {result.chunk.location}")
    print(f"  FTS score: {result.fts_score}")
    print(f"  Vector score: {result.vector_score}")
    print(f"  Combined: {result.combined_score}")
```

## Testing

**Requires PostgreSQL container running:**

```bash
# Start PostgreSQL
cd ~/Claude/research-kb
docker compose up -d postgres

# Run tests
cd packages/storage
poetry install
poetry run pytest
```

## Architecture

### Connection Pooling

- Global connection pool (singleton)
- Default pool size: 2-10 connections
- 60-second command timeout
- Auto-reconnect on failure

### Error Handling

All operations raise `StorageError` on failure:
- Explicit error messages
- Wrapped exceptions with context
- Logged via structlog

### CASCADE Delete

Deleting a source **automatically deletes all its chunks** (PostgreSQL CASCADE).

## Performance

- **Connection pooling**: Reuse connections across requests
- **Batch operations**: `batch_create()` uses transactions
- **Vector search**: pgvector IVFFlat index (lists=100)
- **FTS**: GIN index on tsvector with location boosting

## Version Policy

This package follows semantic versioning. Breaking changes increment the major version.

See: `docs/plans/active/dazzling-soaring-origami.md` for architecture details.
