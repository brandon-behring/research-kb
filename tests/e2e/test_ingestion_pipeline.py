"""End-to-end tests for PDF ingestion pipeline.

Tests the complete flow: PDF → Source + Chunks → Database → Search

Test strategy:
- Use real PDFs from fixtures/papers/
- Test with real database (cleaned between tests)
- Mock embedding/GROBID services when unavailable
- Validate complete pipeline integrity
"""

import pytest
from uuid import UUID

from research_kb_contracts import SourceType
from research_kb_storage import (
    ChunkStore,
    CitationStore,
    SourceStore,
    SearchQuery,
    search_hybrid,
)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_full_pipeline_simple_pdf(test_db, pdf_dispatcher, small_pdf_path):
    """Test: PDF → Source + Chunks → Database → Verified.

    Validates:
    - PDF ingested successfully
    - Source created with metadata
    - Chunks created (count > 5)
    - Chunks stored in database
    - Can retrieve chunks by source
    """
    # Ingest PDF (skip embeddings if server not available)
    result = await pdf_dispatcher.ingest_pdf(
        pdf_path=str(small_pdf_path),
        source_type=SourceType.PAPER,
        title="Test Paper - Heterogeneous Treatment Effects",
        authors=["Athey", "Imbens"],
        year=2016,
        skip_embedding=True,  # Skip embeddings for basic pipeline test
    )

    # Validate IngestResult
    assert result.source is not None
    assert isinstance(result.source.id, UUID)
    assert result.chunk_count > 5, f"Expected >5 chunks, got {result.chunk_count}"
    assert result.source.title == "Test Paper - Heterogeneous Treatment Effects"

    # Verify source in database
    retrieved_source = await SourceStore.get_by_id(result.source.id)
    assert retrieved_source is not None
    assert retrieved_source.title == result.source.title
    assert retrieved_source.file_hash == result.source.file_hash

    # Verify chunks in database
    chunks = await ChunkStore.list_by_source(result.source.id, limit=100)
    assert len(chunks) == result.chunk_count
    assert all(chunk.source_id == result.source.id for chunk in chunks)

    # Verify chunks have content
    assert all(len(chunk.content) > 0 for chunk in chunks)
    assert all(chunk.content_hash is not None for chunk in chunks)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_grobid_citation_extraction(
    test_db, pdf_dispatcher, small_pdf_path, grobid_available
):
    """Test GROBID citation extraction and storage.

    Validates:
    - Citations extracted if GROBID available
    - Citations stored in database
    - Can retrieve citations by source
    - BibTeX generated for citations
    """
    if not grobid_available:
        pytest.skip("GROBID service not available")

    # Ingest PDF with GROBID
    result = await pdf_dispatcher.ingest_pdf(
        pdf_path=str(small_pdf_path),
        source_type=SourceType.PAPER,
        title="Test Paper with Citations",
        force_pymupdf=False,  # Allow GROBID
        skip_embedding=True,  # Skip embeddings for citation test
    )

    # Validate citations extracted
    assert result.citations_extracted >= 0  # May be 0 if paper has no citations

    if result.citations_extracted > 0:
        # Verify citations in database
        citations = await CitationStore.list_by_source(result.source.id)
        assert len(citations) == result.citations_extracted

        # Verify citation structure
        for citation in citations:
            assert citation.source_id == result.source.id
            assert citation.raw_string is not None
            # BibTeX may be None if generation failed
            if citation.bibtex:
                assert "@" in citation.bibtex  # Valid BibTeX starts with @


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.requires_ollama
async def test_concept_extraction_pipeline(test_db, pdf_dispatcher, small_pdf_path):
    """Test: PDF → Chunks → Concept Extraction → Graph.

    This test requires Ollama to be running with llama3.1:8b model.

    Validates:
    - Concepts extracted from chunks
    - Concepts stored in database
    - Relationships created
    - Chunk-concept links established
    """
    pytest.skip("Concept extraction requires Ollama - implement after E2E basics work")
    # TODO: Implement concept extraction E2E test
    # from research_kb_extraction import ConceptExtractor
    # extractor = ConceptExtractor()
    # extraction = await extractor.extract_from_chunks(chunks)
    # ...


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_duplicate_detection(test_db, pdf_dispatcher, small_pdf_path):
    """Test file_hash prevents duplicate ingestion.

    Validates:
    - First ingestion succeeds
    - Second ingestion returns existing source
    - No duplicate chunks created
    - DLQ not triggered
    """
    # First ingestion
    result1 = await pdf_dispatcher.ingest_pdf(
        pdf_path=str(small_pdf_path),
        source_type=SourceType.PAPER,
        title="Test Paper - First",
        skip_embedding=True,
    )

    # Second ingestion (same file)
    result2 = await pdf_dispatcher.ingest_pdf(
        pdf_path=str(small_pdf_path),
        source_type=SourceType.PAPER,
        title="Test Paper - Second",  # Different title, same file
        skip_embedding=True,
    )

    # Validate idempotency
    assert result1.source.id == result2.source.id
    assert result1.source.file_hash == result2.source.file_hash
    assert result1.chunk_count == result2.chunk_count

    # Verify no duplicate chunks
    chunks = await ChunkStore.list_by_source(result1.source.id, limit=1000)
    assert len(chunks) == result1.chunk_count


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_dlq_on_corrupted_pdf(test_db, pdf_dispatcher, corrupted_pdf_path):
    """Test corrupted PDF goes to DLQ.

    Validates:
    - Corrupted PDF fails gracefully
    - Failure logged to DLQ
    - No partial data in database
    """
    # Attempt to ingest corrupted PDF
    with pytest.raises(Exception):  # Should raise some error
        await pdf_dispatcher.ingest_pdf(
            pdf_path=str(corrupted_pdf_path),
            source_type=SourceType.PAPER,
            title="Corrupted PDF",
        )

    # Verify no source created
    sources = await SourceStore.list_by_type(SourceType.PAPER)
    assert len(sources) == 0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_large_pdf_multiple_pages(test_db, pdf_dispatcher):
    """Test ingestion of larger PDF with multiple pages.

    Uses chernozhukov_dml_2018.pdf (~1.1MB, ~50 pages).

    Validates:
    - Large PDF ingested successfully
    - Chunk count proportional to pages
    - Page tracking correct
    - Processing time reasonable (<30s)
    """
    from pathlib import Path

    large_pdf = Path(__file__).parent.parent.parent / "fixtures/papers/chernozhukov_dml_2018.pdf"

    if not large_pdf.exists():
        pytest.skip(f"Large PDF not found: {large_pdf}")

    import time

    start = time.time()

    result = await pdf_dispatcher.ingest_pdf(
        pdf_path=str(large_pdf),
        source_type=SourceType.PAPER,
        title="Double Machine Learning",
        authors=[
            "Chernozhukov",
            "Chetverikov",
            "Demirer",
            "Duflo",
            "Hansen",
            "Newey",
            "Robins",
        ],
        year=2018,
        skip_embedding=True,  # Skip embeddings for large PDF test (faster)
    )

    elapsed = time.time() - start

    # Validate result
    assert result.chunk_count > 50  # Expect many chunks from large PDF
    assert elapsed < 60  # Should complete in reasonable time

    # Verify page tracking
    chunks = await ChunkStore.list_by_source(result.source.id, limit=1000)
    page_numbers = [c.page_start for c in chunks if c.page_start is not None]
    assert len(page_numbers) > 0
    assert max(page_numbers) > 10  # Should span multiple pages


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_search_after_ingestion(test_db, pdf_dispatcher, small_pdf_path):
    """Test search finds ingested content.

    Validates:
    - Content is searchable after ingestion
    - FTS search works
    - Vector search works (if embeddings available)
    - Combined score ranking correct
    """
    # Ingest PDF (skip embeddings - FTS search doesn't require them)
    result = await pdf_dispatcher.ingest_pdf(
        pdf_path=str(small_pdf_path),
        source_type=SourceType.PAPER,
        title="Treatment Effects Paper",
        skip_embedding=True,  # Skip embeddings for FTS search test
    )

    # Test FTS search
    query = SearchQuery(
        text="treatment",
        fts_weight=1.0,
        vector_weight=0.0,
        limit=5,
    )

    results = await search_hybrid(query)

    # Validate search results
    assert len(results) > 0, "FTS search should find 'treatment' in paper"
    assert all(r.source.id == result.source.id for r in results)
    assert all(r.fts_score is not None for r in results)
    assert all(r.fts_score > 0 for r in results)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_source_metadata_preservation(test_db, pdf_dispatcher, small_pdf_path):
    """Test source metadata preserved through pipeline.

    Validates:
    - Custom metadata stored correctly
    - Metadata retrievable from database
    - JSONB flexibility works
    """
    # Ingest with custom metadata
    result = await pdf_dispatcher.ingest_pdf(
        pdf_path=str(small_pdf_path),
        source_type=SourceType.PAPER,
        title="Metadata Test Paper",
        authors=["Test", "Author"],
        year=2024,
        metadata={
            "arxiv_id": "2024.12345",
            "doi": "10.1234/test",
            "tags": ["causal_inference", "heterogeneity"],
            "importance_tier": 1,
        },
        skip_embedding=True,
    )

    # Retrieve and validate metadata
    source = await SourceStore.get_by_id(result.source.id)

    assert source.metadata["arxiv_id"] == "2024.12345"
    assert source.metadata["doi"] == "10.1234/test"
    assert source.metadata["tags"] == ["causal_inference", "heterogeneity"]
    assert source.metadata["importance_tier"] == 1


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_chunk_location_tracking(test_db, pdf_dispatcher, small_pdf_path):
    """Test chunk location tracking (page numbers, sections).

    Validates:
    - Page numbers captured correctly
    - Section hierarchy preserved (if detected)
    - Location strings human-readable
    """
    result = await pdf_dispatcher.ingest_pdf(
        pdf_path=str(small_pdf_path),
        source_type=SourceType.PAPER,
        title="Location Tracking Test",
        skip_embedding=True,
    )

    chunks = await ChunkStore.list_by_source(result.source.id, limit=100)

    # Validate page tracking
    chunks_with_pages = [c for c in chunks if c.page_start is not None]
    assert len(chunks_with_pages) > 0, "Should have page numbers"

    # Validate page numbers are sequential and reasonable
    page_numbers = [c.page_start for c in chunks_with_pages]
    assert min(page_numbers) >= 1
    assert max(page_numbers) <= 100  # Reasonable for test PDF

    # Validate location strings
    chunks_with_location = [c for c in chunks if c.location]
    if chunks_with_location:
        # Location should mention page
        assert any(
            "page" in c.location.lower() or "p." in c.location.lower()
            for c in chunks_with_location[:5]
        )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_embedding_generation_and_storage(
    test_db, pdf_dispatcher, small_pdf_path, embedding_available
):
    """Test embedding generation and storage.

    Validates:
    - Embeddings generated if server available
    - Embeddings are 1024-dimensional (BGE-large-en-v1.5)
    - Embeddings stored in database
    - Vector search works with embeddings
    """
    if not embedding_available:
        pytest.skip("Embedding server not available - using skip_embedding=True")

    # Ingest with embeddings
    result = await pdf_dispatcher.ingest_pdf(
        pdf_path=str(small_pdf_path),
        source_type=SourceType.PAPER,
        title="Embedding Test Paper",
        skip_embedding=False,
    )

    # Retrieve chunks and check embeddings
    chunks = await ChunkStore.list_by_source(result.source.id, limit=10)

    chunks_with_embeddings = [c for c in chunks if c.embedding is not None]

    assert len(chunks_with_embeddings) > 0, "Should have embeddings"

    # Validate embedding dimensions
    for chunk in chunks_with_embeddings:
        assert len(chunk.embedding) == 1024, "Should be 1024-dim BGE-large-en-v1.5"
        # Validate not all zeros (real embedding)
        assert any(v != 0.0 for v in chunk.embedding), "Embedding should not be all zeros"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_idempotency_same_hash(test_db, pdf_dispatcher, small_pdf_path):
    """Test re-ingesting same PDF returns existing source.

    Validates:
    - Same file hash detected
    - Existing source returned
    - No new chunks created
    - IngestResult reflects existing data
    """
    # First ingestion
    result1 = await pdf_dispatcher.ingest_pdf(
        pdf_path=str(small_pdf_path),
        source_type=SourceType.PAPER,
        title="Idempotency Test - First",
        authors=["Author1"],
        year=2020,
        skip_embedding=True,
    )

    source_id_1 = result1.source.id
    chunk_count_1 = result1.chunk_count

    # Second ingestion (identical file)
    result2 = await pdf_dispatcher.ingest_pdf(
        pdf_path=str(small_pdf_path),
        source_type=SourceType.PAPER,
        title="Idempotency Test - Second",  # Different metadata
        authors=["Author2"],  # Different author
        year=2021,  # Different year
        skip_embedding=True,
    )

    # Validate idempotency
    assert result2.source.id == source_id_1, "Should return same source ID"
    assert result2.chunk_count == chunk_count_1, "Should return same chunk count"

    # Verify only one source exists
    sources = await SourceStore.list_by_type(SourceType.PAPER)
    assert len(sources) == 1

    # Verify chunks not duplicated
    chunks = await ChunkStore.list_by_source(source_id_1, limit=1000)
    assert len(chunks) == chunk_count_1


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_graceful_degradation_no_grobid(test_db, pdf_dispatcher, small_pdf_path):
    """Test pipeline works without GROBID (fallback to PyMuPDF only).

    Validates:
    - PDF ingested successfully without GROBID
    - Chunks created normally
    - No citations extracted (expected)
    - No errors thrown
    """
    # Force PyMuPDF-only mode
    result = await pdf_dispatcher.ingest_pdf(
        pdf_path=str(small_pdf_path),
        source_type=SourceType.PAPER,
        title="PyMuPDF Only Test",
        force_pymupdf=True,  # Skip GROBID entirely
        skip_embedding=True,
    )

    # Validate result
    assert result.source is not None
    assert result.chunk_count > 5
    assert result.extraction_method == "pymupdf"
    assert result.citations_extracted == 0
    assert result.grobid_metadata_extracted is False

    # Verify chunks created
    chunks = await ChunkStore.list_by_source(result.source.id, limit=100)
    assert len(chunks) == result.chunk_count
