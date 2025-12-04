#!/usr/bin/env python3
"""Ingest golden PDFs for Week 2 validation.

This script:
1. Ingests 3 golden PDFs through full pipeline
2. Extracts, chunks, and embeds content
3. Creates Source and Chunk records in database
4. Reports metrics for manual inspection
"""

import asyncio
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import get_logger
from research_kb_contracts import SourceType
from research_kb_pdf import (
    EmbeddingClient,
    chunk_with_sections,
    extract_with_headings,
)
from research_kb_storage import ChunkStore, DatabaseConfig, SourceStore, get_connection_pool

logger = get_logger(__name__)


# Golden PDFs metadata
GOLDEN_PDFS = [
    {
        "file": "fixtures/golden/attention_paper.pdf",
        "title": "Attention Is All You Need",
        "authors": [
            "Vaswani",
            "Shazeer",
            "Parmar",
            "Uszkoreit",
            "Jones",
            "Gomez",
            "Kaiser",
            "Polosukhin",
        ],
        "year": 2017,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "1706.03762", "domain": "NLP"},
    },
    {
        "file": "fixtures/golden/bayesian_deep_learning.pdf",
        "title": "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning",
        "authors": ["Gal", "Ghahramani"],
        "year": 2015,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "1506.02142", "domain": "Bayesian Deep Learning"},
    },
    {
        "file": "fixtures/golden/variational_inference.pdf",
        "title": "Variational Inference: A Review for Statisticians",
        "authors": ["Blei", "Kucukelbir", "McAuliffe"],
        "year": 2016,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "1601.00670", "domain": "Statistics"},
    },
]


async def ingest_pdf(
    pdf_path: str,
    title: str,
    authors: list[str],
    year: int,
    source_type: SourceType,
    metadata: dict,
) -> tuple[int, int, int]:
    """Ingest a single PDF through full pipeline.

    Args:
        pdf_path: Path to PDF file
        title: Document title
        authors: List of authors
        year: Publication year
        source_type: Type of source
        metadata: Additional metadata

    Returns:
        Tuple of (source_id, num_chunks, num_headings)
    """
    pdf_path = Path(pdf_path)

    # 1. Extract with heading detection
    logger.info("extracting_pdf", path=str(pdf_path))
    doc, headings = extract_with_headings(pdf_path)

    logger.info(
        "extraction_complete",
        path=str(pdf_path),
        pages=doc.total_pages,
        headings=len(headings),
    )

    # 2. Chunk with section tracking
    logger.info("chunking_document", path=str(pdf_path))
    chunks = chunk_with_sections(doc, headings)

    logger.info("chunking_complete", path=str(pdf_path), chunks=len(chunks))

    # 3. Calculate file hash for idempotency
    import hashlib

    sha256_hash = hashlib.sha256()
    with pdf_path.open("rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    file_hash = sha256_hash.hexdigest()

    # 4. Create Source record
    logger.info("creating_source", title=title)
    source = await SourceStore.create(
        source_type=source_type,
        title=title,
        authors=authors,
        year=year,
        file_path=str(pdf_path),
        file_hash=file_hash,
        metadata={
            **metadata,
            "extraction_method": "pymupdf",
            "total_pages": doc.total_pages,
            "total_chars": doc.total_chars,
            "total_headings": len(headings),
            "total_chunks": len(chunks),
        },
    )

    logger.info("source_created", source_id=str(source.id))

    # 5. Generate embeddings and create Chunk records
    logger.info("generating_embeddings", chunks=len(chunks))
    embedding_client = EmbeddingClient()

    chunks_created = 0
    for chunk in chunks:
        # Sanitize content (remove null bytes and other control characters)
        sanitized_content = chunk.content.replace("\x00", "").replace("\uFFFD", "")

        # Generate embedding
        embedding = embedding_client.embed(sanitized_content)

        # Calculate content hash
        content_hash = hashlib.sha256(sanitized_content.encode("utf-8")).hexdigest()

        # Create chunk record
        await ChunkStore.create(
            source_id=source.id,
            content=sanitized_content,
            content_hash=content_hash,
            page_start=chunk.start_page,
            page_end=chunk.end_page,
            embedding=embedding,
            metadata=chunk.metadata,  # ChunkStore handles dict→ChunkMetadata conversion
        )
        chunks_created += 1

        # Log progress every 50 chunks
        if chunks_created % 50 == 0:
            logger.info("chunks_progress", created=chunks_created, total=len(chunks))

    logger.info(
        "ingestion_complete",
        source_id=str(source.id),
        chunks_created=chunks_created,
        headings_detected=len(headings),
    )

    return source.id, chunks_created, len(headings)


async def main():
    """Ingest all golden PDFs and report results."""
    logger.info("starting_golden_ingestion", pdfs=len(GOLDEN_PDFS))

    # Initialize database connection pool
    config = DatabaseConfig()
    await get_connection_pool(config)

    results = []

    for pdf_data in GOLDEN_PDFS:
        pdf_path = Path(__file__).parent.parent / pdf_data["file"]

        if not pdf_path.exists():
            logger.error("pdf_not_found", path=str(pdf_path))
            print(f"✗ PDF not found: {pdf_path}")
            continue

        try:
            source_id, num_chunks, num_headings = await ingest_pdf(
                pdf_path=str(pdf_path),
                title=pdf_data["title"],
                authors=pdf_data["authors"],
                year=pdf_data["year"],
                source_type=pdf_data["source_type"],
                metadata=pdf_data["metadata"],
            )

            results.append(
                {
                    "title": pdf_data["title"],
                    "source_id": source_id,
                    "chunks": num_chunks,
                    "headings": num_headings,
                    "status": "success",
                }
            )

            print(f"✓ {pdf_data['title']}")
            print(f"  Source ID: {source_id}")
            print(f"  Chunks: {num_chunks}")
            print(f"  Headings: {num_headings}")

        except Exception as e:
            logger.error(
                "ingestion_failed", title=pdf_data["title"], error=str(e), exc_info=True
            )
            results.append(
                {
                    "title": pdf_data["title"],
                    "status": "failed",
                    "error": str(e),
                }
            )
            print(f"✗ {pdf_data['title']}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("GOLDEN PDF INGESTION SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"\nSuccessful: {len(successful)}/{len(GOLDEN_PDFS)}")
    print(f"Failed: {len(failed)}/{len(GOLDEN_PDFS)}")

    if successful:
        total_chunks = sum(r["chunks"] for r in successful)
        total_headings = sum(r["headings"] for r in successful)

        print(f"\nTotal chunks created: {total_chunks}")
        print(f"Total headings detected: {total_headings}")

        print("\nPer-PDF Breakdown:")
        for r in successful:
            print(f"  {r['title'][:50]:50} | {r['chunks']:3} chunks | {r['headings']:2} headings")

    if failed:
        print("\nFailed PDFs:")
        for r in failed:
            print(f"  ✗ {r['title']}: {r['error']}")

    logger.info(
        "golden_ingestion_complete",
        successful=len(successful),
        failed=len(failed),
        total_chunks=sum(r["chunks"] for r in successful) if successful else 0,
    )


if __name__ == "__main__":
    asyncio.run(main())
