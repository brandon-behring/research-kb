# research-kb-contracts

Pure Pydantic schemas for the research-kb system.

**Version**: 1.0.0 (frozen - breaking changes require new package)

## Purpose

This package contains ONLY Pydantic data models with no business logic:
- Source (textbook, paper, code_repo)
- Chunk (extracted content units)
- IngestionStatus (pipeline state tracking)
- SearchResult (hybrid search results)

## Dependencies

- `pydantic ^2.5.0` (ONLY dependency - no logging, no OpenTelemetry, no DB drivers)

## Usage

```python
from research_kb_contracts import Source, Chunk, SourceType

# Create a source
source = Source(
    id=uuid4(),
    source_type=SourceType.TEXTBOOK,
    title="Causality",
    file_hash="sha256:abc123",
    metadata={"isbn": "978-0521895606"},
    created_at=datetime.now(timezone.utc),
    updated_at=datetime.now(timezone.utc),
)

# Create a chunk
chunk = Chunk(
    id=uuid4(),
    source_id=source.id,
    content="The backdoor criterion...",
    content_hash="sha256:chunk123",
    location="Chapter 3, p. 73",
    embedding=[0.1] * 384,  # BGE-large-en-v1.5
    created_at=datetime.now(timezone.utc),
)
```

## Testing

```bash
cd packages/contracts
poetry install
poetry run pytest
```

## Schema Extensibility

Both `Source` and `Chunk` have `metadata: dict[str, Any]` (JSONB in PostgreSQL):

**Textbook metadata example**:
```python
metadata = {
    "isbn": "978-0521895606",
    "publisher": "Cambridge University Press",
    "total_pages": 464
}
```

**Paper metadata example**:
```python
metadata = {
    "doi": "10.1111/ectj.12097",
    "journal": "Econometrics Journal",
    "authority_tier": "canonical"
}
```

**Chunk metadata example**:
```python
metadata = {
    "chunk_type": "theorem",
    "theorem_name": "Backdoor Criterion",
    "chapter_num": 3,
    "has_proof": True
}
```

## Version Policy

This package is **frozen at 1.0.0**. Breaking changes require creating `research-kb-contracts-v2`.

See: `docs/plans/active/dazzling-soaring-origami.md` for architecture details.
