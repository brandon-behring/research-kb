#!/usr/bin/env python3
"""Ingest Google MMM papers for research knowledge base.

This script ingests 8 foundational Marketing Mix Modeling papers:
1. Jin et al. (2017) - Adstock & Saturation curves
2. Sun et al. (2017) - Geo-level Hierarchical MMM
3. Zhang et al. (2024) - Calibration with Bayesian Priors
4. Zhang et al. (2023) - Reach & Frequency modeling
5. Chen et al. (2018) - Bias Correction for Paid Search
6. Chan & Perry (2017) - Challenges and Opportunities
7. Wang et al. (2017) - Category Data hierarchical approach
8. Ng et al. (2021) - Time-varying parameters

These papers form the theoretical foundation for Google Meridian and
modern Bayesian MMM approaches.
"""

import asyncio
import hashlib
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
from research_kb_storage import (
    ChunkStore,
    DatabaseConfig,
    SourceStore,
    get_connection_pool,
)

logger = get_logger(__name__)


# MMM Papers metadata - Google's foundational MMM research
MMM_PAPERS = [
    {
        "file": "fixtures/papers/mmm_google/jin_adstock_saturation_2017.pdf",
        "title": "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects",
        "authors": ["Jin", "Wang", "Sun", "Chan", "Koehler"],
        "year": 2017,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "causal_inference",
            "topic": "mmm",
            "subtopics": ["adstock", "saturation", "hill_curve", "geometric_decay"],
            "google_pubid": "3806",
        },
    },
    {
        "file": "fixtures/papers/mmm_google/sun_geo_hierarchical_2017.pdf",
        "title": "Geo-level Bayesian Hierarchical Media Mix Modeling",
        "authors": ["Sun", "Wang", "Jin", "Chan", "Koehler"],
        "year": 2017,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "causal_inference",
            "topic": "mmm",
            "subtopics": ["hierarchical", "geo_level", "partial_pooling", "shrinkage"],
            "google_pubid": "3804",
        },
    },
    {
        "file": "fixtures/papers/mmm_google/zhang_calibration_2024.pdf",
        "title": "Media Mix Model Calibration With Bayesian Priors",
        "authors": ["Zhang", "Wurm", "Li", "Wakim", "Kelly", "Price", "Liu"],
        "year": 2024,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "causal_inference",
            "topic": "mmm",
            "subtopics": [
                "calibration",
                "priors",
                "roas_reparameterization",
                "experiment_integration",
            ],
            "google_pubid": "7494",
        },
    },
    {
        "file": "fixtures/papers/mmm_google/zhang_reach_frequency_2023.pdf",
        "title": "Bayesian Hierarchical Media Mix Model Incorporating Reach and Frequency Data",
        "authors": ["Zhang", "Wurm", "Wakim", "Li", "Liu"],
        "year": 2023,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "causal_inference",
            "topic": "mmm",
            "subtopics": ["reach", "frequency", "mcmc", "optimal_frequency"],
            "google_pubid": "7327",
        },
    },
    {
        "file": "fixtures/papers/mmm_google/chen_bias_correction_2018.pdf",
        "title": "Bias Correction For Paid Search In Media Mix Modeling",
        "authors": ["Chen", "Chan", "Perry", "Jin", "Sun", "Wang", "Koehler"],
        "year": 2018,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "causal_inference",
            "topic": "mmm",
            "subtopics": [
                "bias_correction",
                "paid_search",
                "backdoor_criterion",
                "confounding",
            ],
            "arxiv_id": "1807.03292",
        },
    },
    {
        "file": "fixtures/papers/mmm_google/chan_challenges_2017.pdf",
        "title": "Challenges and Opportunities in Media Mix Modeling",
        "authors": ["Chan", "Perry"],
        "year": 2017,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "causal_inference",
            "topic": "mmm",
            "subtopics": [
                "selection_bias",
                "data_challenges",
                "simulation",
                "overview",
            ],
            "google_pubid": "3803",
        },
    },
    {
        "file": "fixtures/papers/mmm_google/wang_category_data_2017.pdf",
        "title": "A Hierarchical Bayesian Approach to Improve Media Mix Models Using Category Data",
        "authors": ["Wang", "Jin", "Sun", "Chan", "Koehler"],
        "year": 2017,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "causal_inference",
            "topic": "mmm",
            "subtopics": ["category_data", "hierarchical_priors", "partial_pooling"],
            "google_pubid": "3805",
        },
    },
    {
        "file": "fixtures/papers/mmm_google/ng_time_varying_2021.pdf",
        "title": "Bayesian Time Varying Coefficient Model with Applications to Marketing Mix Modeling",
        "authors": ["Ng", "Vaver", "Chakrabarti", "Kong", "Liu"],
        "year": 2021,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "causal_inference",
            "topic": "mmm",
            "subtopics": ["time_varying", "dynamic_coefficients", "bayesian_updating"],
            "arxiv_id": "2106.03322",
        },
    },
]


async def ingest_pdf(
    pdf_path: str,
    title: str,
    authors: list[str],
    year: int,
    source_type: SourceType,
    metadata: dict,
) -> tuple[str, int, int]:
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

    return str(source.id), chunks_created, len(headings)


async def main():
    """Ingest all MMM papers and report results."""
    logger.info("starting_mmm_ingestion", pdfs=len(MMM_PAPERS))

    # Initialize database connection pool
    config = DatabaseConfig()
    await get_connection_pool(config)

    results = []

    for pdf_data in MMM_PAPERS:
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
            logger.error("ingestion_failed", title=pdf_data["title"], error=str(e), exc_info=True)
            results.append(
                {
                    "title": pdf_data["title"],
                    "status": "failed",
                    "error": str(e),
                }
            )
            print(f"✗ {pdf_data['title']}: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("MMM PAPERS INGESTION SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"\nSuccessful: {len(successful)}/{len(MMM_PAPERS)}")
    print(f"Failed: {len(failed)}/{len(MMM_PAPERS)}")

    if successful:
        total_chunks = sum(r["chunks"] for r in successful)
        total_headings = sum(r["headings"] for r in successful)

        print(f"\nTotal chunks created: {total_chunks}")
        print(f"Total headings detected: {total_headings}")

        print("\nPer-PDF Breakdown:")
        for r in successful:
            title_short = r["title"][:55]
            print(f"  {title_short:55} | {r['chunks']:3} chunks | {r['headings']:2} headings")

    if failed:
        print("\nFailed PDFs:")
        for r in failed:
            print(f"  ✗ {r['title']}: {r['error']}")

    logger.info(
        "mmm_ingestion_complete",
        successful=len(successful),
        failed=len(failed),
        total_chunks=sum(r["chunks"] for r in successful) if successful else 0,
    )


if __name__ == "__main__":
    asyncio.run(main())
