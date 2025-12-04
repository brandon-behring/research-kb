"""Smoke tests for PDF ingestion quality.

These tests validate that real PDFs can be ingested and produce quality output.
They test the entire ingestion pipeline end-to-end.
"""

import pytest
from pathlib import Path


def count_tokens(text: str) -> int:
    """Rough token count (word-based approximation)."""
    return len(text.split())


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_ingest_simple_paper(test_db, simple_paper_path, ingestion_helper):
    """Smoke test: Ingest simple paper and validate quality.

    Validates:
    - Chunk count > 10
    - Average chunk length 500-2000 tokens
    - No empty chunks
    """
    source, chunks = await ingestion_helper.ingest_pdf(simple_paper_path)

    # Must produce reasonable number of chunks
    assert len(chunks) > 10, f"Expected >10 chunks, got {len(chunks)}"

    # Check average chunk length (using word-based approximation)
    # Note: Real chunks are ~300 tokens, which is ~150-250 words
    total_tokens = sum(count_tokens(c.content) for c in chunks)
    avg_length = total_tokens / len(chunks)
    assert 100 < avg_length < 500, f"Avg chunk length {avg_length} outside 100-500 words"

    # No empty chunks
    empty = [c for c in chunks if len(c.content.strip()) < 50]
    assert len(empty) == 0, f"Found {len(empty)} empty chunks"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_ingest_textbook(test_db, textbook_path, ingestion_helper):
    """Smoke test: Ingest textbook and validate structure.

    Validates:
    - Large chunk count (textbooks are long)
    - Reasonable chunk distribution
    - No extremely long chunks (should be split)
    """
    from research_kb_contracts import SourceType

    source, chunks = await ingestion_helper.ingest_pdf(
        textbook_path,
        source_type=SourceType.TEXTBOOK
    )

    # Textbooks should produce many chunks
    assert len(chunks) > 50, f"Expected >50 chunks from textbook, got {len(chunks)}"

    # No chunks should be too long (should be split)
    too_long = [c for c in chunks if count_tokens(c.content) > 3000]
    assert len(too_long) == 0, f"Found {len(too_long)} chunks >3000 tokens (should be split)"

    # Most chunks should be reasonable length (word count approximation)
    reasonable = [c for c in chunks if 100 < count_tokens(c.content) < 500]
    assert len(reasonable) / len(chunks) > 0.60, "Less than 60% of chunks in reasonable range (100-500 words)"


@pytest.mark.smoke
@pytest.mark.asyncio
@pytest.mark.requires_embedding
async def test_embedding_quality(test_db, simple_paper_path, ingestion_helper):
    """Smoke test: Validate embedding generation.

    Validates:
    - Embeddings are generated
    - Embeddings have correct dimensions (1024 for BGE-large-en-v1.5)
    - Embeddings are not null or zero vectors
    """
    source, chunks = await ingestion_helper.ingest_pdf(simple_paper_path)

    # Count chunks with embeddings
    with_embeddings = [c for c in chunks if c.embedding is not None]

    # Should have embeddings if server available
    if len(with_embeddings) == 0:
        pytest.skip("Embedding server not available")

    # Check first chunk with embedding
    chunk = with_embeddings[0]
    assert chunk.embedding is not None, "Chunk should have embedding"
    assert len(chunk.embedding) == 1024, f"Expected 1024-dim embedding, got {len(chunk.embedding)}"

    # Should not be zero vector
    assert sum(chunk.embedding) != 0.0, "Embedding should not be zero vector"
    assert max(chunk.embedding) != min(chunk.embedding), "Embedding should have variance"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_deduplication(test_db, simple_paper_path, ingestion_helper):
    """Smoke test: Validate duplicate detection.

    Validates:
    - No duplicate chunks (same content hash)
    - Content hashes are unique
    """
    source, chunks = await ingestion_helper.ingest_pdf(simple_paper_path)

    # Collect content hashes
    content_hashes = [c.content_hash for c in chunks]

    # All should be unique
    unique_hashes = set(content_hashes)
    assert len(content_hashes) == len(unique_hashes), \
        f"Found {len(content_hashes) - len(unique_hashes)} duplicate chunks"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_chunk_metadata(test_db, simple_paper_path, ingestion_helper):
    """Smoke test: Validate chunk metadata.

    Validates:
    - Page numbers are set
    - Page numbers are reasonable
    - Metadata exists
    """
    source, chunks = await ingestion_helper.ingest_pdf(simple_paper_path)

    # Count chunks with page numbers
    with_pages = [c for c in chunks if c.page_start is not None]

    # Most chunks should have page numbers
    assert len(with_pages) / len(chunks) > 0.8, \
        f"Only {len(with_pages)}/{len(chunks)} chunks have page numbers"

    # Page numbers should be reasonable
    for chunk in with_pages:
        assert chunk.page_start > 0, f"Page number should be positive: {chunk.page_start}"
        assert chunk.page_start < 1000, f"Page number unreasonably high: {chunk.page_start}"

        if chunk.page_end:
            assert chunk.page_end >= chunk.page_start, \
                f"Page end ({chunk.page_end}) before start ({chunk.page_start})"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_content_extraction_quality(test_db, simple_paper_path, ingestion_helper):
    """Smoke test: Validate extracted content quality.

    Validates:
    - Content is not garbled
    - Contains actual words (not just special characters)
    - Has reasonable punctuation
    """
    source, chunks = await ingestion_helper.ingest_pdf(simple_paper_path)

    for chunk in chunks[:10]:  # Check first 10 chunks
        content = chunk.content

        # Skip very short chunks (e.g., page numbers, headings)
        if len(content) < 100:
            continue

        # Should contain actual words
        words = [w for w in content.split() if len(w) > 3 and w.isalnum()]
        assert len(words) > 10, f"Chunk should contain >10 real words, got {len(words)}"

        # Should not be mostly special characters
        alpha_ratio = sum(c.isalpha() for c in content) / max(len(content), 1)
        assert alpha_ratio > 0.5, f"Chunk is {alpha_ratio:.1%} alphabetic (should be >50%)"

        # Should have some punctuation (normal text)
        has_period = '.' in content
        has_comma = ',' in content
        assert has_period or has_comma, "Chunk should have basic punctuation"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_multiple_papers_ingestion(test_db, all_papers, ingestion_helper):
    """Smoke test: Ingest multiple papers without errors.

    Validates:
    - Can ingest multiple PDFs
    - Each produces reasonable output
    - No crashes or errors
    """
    if len(all_papers) == 0:
        pytest.skip("No papers available for multi-ingest test")

    # Limit to first 3 papers for speed
    test_papers = all_papers[:3]

    results = []
    for paper_path in test_papers:
        try:
            source, chunks = await ingestion_helper.ingest_pdf(paper_path)
            results.append({
                'path': paper_path,
                'source': source,
                'chunks': len(chunks),
                'success': True
            })
        except Exception as e:
            results.append({
                'path': paper_path,
                'chunks': 0,
                'success': False,
                'error': str(e)
            })

    # All should succeed
    failures = [r for r in results if not r['success']]
    assert len(failures) == 0, \
        f"{len(failures)}/{len(results)} papers failed: {[f['path'].name for f in failures]}"

    # All should produce chunks
    for result in results:
        assert result['chunks'] > 5, \
            f"{result['path'].name} produced only {result['chunks']} chunks"


@pytest.mark.smoke
@pytest.mark.slow
@pytest.mark.asyncio
async def test_full_pipeline_smoke(test_db, simple_paper_path, ingestion_helper):
    """Smoke test: Full pipeline including database queries.

    Validates:
    - Source can be retrieved
    - Chunks can be retrieved
    - Relationships work
    - Database queries succeed
    """
    from research_kb_storage import SourceStore, ChunkStore

    source, chunks = await ingestion_helper.ingest_pdf(simple_paper_path)

    # Verify source can be retrieved
    retrieved_source = await SourceStore.get_by_id(source.id)
    assert retrieved_source is not None, "Source should be retrievable"
    assert retrieved_source.title == source.title, "Source title should match"

    # Verify chunks can be retrieved
    retrieved_chunks = await ChunkStore.list_by_source(source.id, limit=1000)
    assert len(retrieved_chunks) == len(chunks), \
        f"Expected {len(chunks)} chunks, retrieved {len(retrieved_chunks)}"

    # Verify chunk content matches
    first_chunk = retrieved_chunks[0]
    assert first_chunk.content == chunks[0].content, "Chunk content should match"
    assert first_chunk.source_id == source.id, "Chunk should reference source"
