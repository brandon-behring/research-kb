## research-kb-storage

PostgreSQL + pgvector + KuzuDB storage layer for the research-kb system.

## Purpose

This package provides **exclusive database access** for all research-kb operations:

- **Connection management** -- asyncpg connection pooling (2-10 connections)
- **SourceStore / ChunkStore** -- CRUD for sources and chunks tables
- **CitationStore / BiblioStore** -- Citation network and bibliographic coupling
- **ConceptStore / RelationshipStore / ChunkConceptStore** -- Knowledge graph entities
- **MethodStore / AssumptionStore** -- Method-assumption cache for auditing
- **DomainStore** -- Multi-domain corpus management (19 domains)
- **DiscoveryStore / QueueStore** -- Paper discovery and ingestion queue
- **CrossDomainStore** -- Cross-domain concept links
- **4-way hybrid search** -- FTS + vector + graph + citation authority
- **KuzuDB graph engine** -- Sub-100ms concept traversal
- **HyDE query expansion** -- Hypothetical document embeddings
- **Reranking** -- Cross-encoder reranking for precision
- **Assumption auditing** -- North Star feature for method assumptions

**Exclusive DB ownership** -- No other packages access PostgreSQL directly.

## Dependencies

- `research-kb-contracts` (Pydantic schemas)
- `research-kb-common` (logging, errors, retry)
- `asyncpg` (async PostgreSQL driver)
- `pgvector` (vector support, 1024 dimensions)
- `kuzu` (embedded graph database)

## Usage

### Connection Setup

```python
from research_kb_storage import DatabaseConfig, get_connection_pool

config = DatabaseConfig(
    host="localhost",
    port=5432,
    database="research_kb",
    user="postgres",
    password="postgres",
)

pool = await get_connection_pool(config)
```

### Source Operations

```python
from research_kb_storage import SourceStore
from research_kb_contracts import SourceType

source = await SourceStore.create(
    source_type=SourceType.TEXTBOOK,
    title="Causality: Models, Reasoning, and Inference",
    domain_id="causal_inference",
    file_hash="sha256:abc123",
    authors=["Judea Pearl"],
    metadata={"isbn": "978-0521895606"},
)
```

### Chunk Operations

```python
from research_kb_storage import ChunkStore

chunk = await ChunkStore.create(
    source_id=source.id,
    domain_id="causal_inference",
    content="The backdoor criterion states...",
    content_hash="sha256:chunk123",
    location="Chapter 3, p. 73",
    embedding=[0.1] * 1024,  # BGE-large-en-v1.5 (1024 dimensions)
    metadata={"chunk_type": "theorem"},
)
```

### 4-Way Hybrid Search

```python
from research_kb_storage import SearchQuery, search_hybrid_v2

results = await search_hybrid_v2(SearchQuery(
    text="backdoor criterion",
    embedding=[0.1] * 1024,
    fts_weight=0.3,
    vector_weight=0.7,
    use_graph=True,        # Add knowledge graph signal
    use_citations=True,    # Add citation authority signal
    context_type="balanced",
    limit=10,
))

for result in results:
    print(f"Rank {result.rank}: {result.source.title}")
    print(f"  Combined: {result.combined_score:.3f}")
```

### Knowledge Graph

```python
from research_kb_storage import find_shortest_path, get_neighborhood

# Find conceptual path
path = await find_shortest_path("IV", "unconfoundedness")

# Explore neighborhood (KuzuDB-accelerated)
neighbors = await get_neighborhood(concept_id, hops=2)
```

### Assumption Auditing

```python
from research_kb_storage import MethodAssumptionAuditor

auditor = MethodAssumptionAuditor(pool)
result = await auditor.audit_method("instrumental variables")
for assumption in result.assumptions:
    print(f"  {assumption.name}: {assumption.description}")
```

## Testing

```bash
# Unit tests (mocked, no services needed)
pytest packages/storage/tests/ -m unit -v

# Integration tests (requires PostgreSQL)
docker compose up -d postgres
pytest packages/storage/tests/ -m integration -v
```

## Architecture

### Search Pipeline

4-way ranking: `score = fts_w * fts + vector_w * vector + graph_w * graph + citation_w * citation`

- **FTS**: PostgreSQL full-text search with location boosting
- **Vector**: BGE-large-en-v1.5 cosine similarity (1024 dimensions)
- **Graph**: KuzuDB concept co-occurrence (sub-100ms)
- **Citation**: PageRank-style authority over 15K+ citation links

### KuzuDB Graph Engine

- Primary graph traversal engine (~150ms batch scoring)
- PostgreSQL recursive CTEs as fallback (2-second timeout)
- Data synced via `scripts/sync_kuzu.py`
- Path: `~/.research_kb/kuzu/research_kb.kuzu`

### Error Handling

All operations raise `StorageError` on failure with explicit messages, wrapped exceptions, and structlog context.

### CASCADE Delete

Deleting a source **automatically deletes all its chunks** (PostgreSQL CASCADE).

## Performance

- **Connection pooling**: 2-10 connections (asyncpg)
- **Batch operations**: `batch_create()` uses transactions
- **Vector search**: pgvector HNSW index (1024 dimensions)
- **FTS**: GIN index on tsvector with location boosting
- **Graph**: KuzuDB embedded (sub-100ms traversal)

## Version Policy

This package follows semantic versioning. See CLAUDE.md for architecture details.
