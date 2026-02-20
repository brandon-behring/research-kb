#!/usr/bin/env python3
"""Ingest all papers from fixtures/papers/ that aren't already in the database.

This script:
1. Scans fixtures/papers/ for all PDFs
2. Checks which ones are already ingested (by file hash)
3. Ingests the missing ones with auto-extracted metadata
"""

import asyncio
import hashlib
import re
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


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def parse_filename_for_metadata(filename: str) -> dict:
    """Extract metadata from filename pattern like 'author_title_year.pdf'"""
    # Remove .pdf extension
    name = filename.replace(".pdf", "")

    # Try to extract year (4 digits at end or in middle)
    year_match = re.search(r"(\d{4})", name)
    year = int(year_match.group(1)) if year_match else None

    # Split by underscore
    parts = name.split("_")

    # First part is usually author(s)
    authors = []
    if parts:
        # Convert "athey_imbens" -> ["Athey", "Imbens"]
        author_parts = []
        for i, part in enumerate(parts):
            if part.isdigit() or len(part) < 2:
                break
            if any(c.isupper() for c in part) or part.lower() in [
                "athey",
                "imbens",
                "chernozhukov",
                "pearl",
                "rubin",
                "angrist",
            ]:
                author_parts.append(part.title())
            else:
                break
        if author_parts:
            authors = author_parts[:3]  # Max 3 authors from filename

    # Title is everything between authors and year
    title = name.replace("_", " ").title()

    return {
        "title": title,
        "authors": authors,
        "year": year,
    }


def load_s2_sidecar(pdf_path: Path) -> dict | None:
    """Load S2 metadata sidecar if it exists.

    The S2 acquisition pipeline saves rich metadata as a JSON sidecar
    alongside each PDF: {filename}.s2.json

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Sidecar data dict if found, None otherwise
    """
    import json

    sidecar_path = pdf_path.with_suffix(".s2.json")
    if not sidecar_path.exists():
        return None

    try:
        with open(sidecar_path) as f:
            data = json.load(f)
        logger.info("loaded_s2_sidecar", path=sidecar_path.name)
        return data
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("sidecar_load_failed", path=sidecar_path.name, error=str(e))
        return None


def get_metadata_for_pdf(pdf_path: Path) -> tuple[str, list[str], int | None, dict]:
    """Get metadata for a PDF, preferring S2 sidecar over filename parsing.

    Returns:
        Tuple of (title, authors, year, extra_metadata)
    """
    # Try S2 sidecar first (rich metadata from Semantic Scholar)
    sidecar = load_s2_sidecar(pdf_path)

    if sidecar:
        # Extract core fields
        title = sidecar.get("title") or pdf_path.stem.replace("_", " ").title()
        authors = sidecar.get("authors") or []
        year = sidecar.get("year")

        # Build extra metadata for sources.metadata JSONB
        extra_metadata = {
            "s2_paper_id": sidecar.get("s2_paper_id"),
            "s2_corpus_id": sidecar.get("s2_corpus_id"),
            "doi": sidecar.get("doi"),
            "arxiv_id": sidecar.get("arxiv_id"),
            "citation_count": sidecar.get("citation_count"),
            "influential_citation_count": sidecar.get("influential_citation_count"),
            "is_open_access": sidecar.get("is_open_access"),
            "fields_of_study": sidecar.get("fields_of_study"),
            "venue": sidecar.get("venue"),
            "abstract": sidecar.get("abstract"),
            "s2_acquired_at": sidecar.get("acquired_at"),
            "metadata_source": "s2_sidecar",
        }
        # Remove None values
        extra_metadata = {k: v for k, v in extra_metadata.items() if v is not None}

        return title, authors, year, extra_metadata

    # Fall back to filename parsing
    meta = parse_filename_for_metadata(pdf_path.name)
    return meta["title"], meta["authors"], meta["year"], {"metadata_source": "filename"}


async def ingest_pdf(
    pdf_path: str,
    title: str,
    authors: list[str],
    year: int | None,
    metadata: dict | None = None,
) -> tuple[str, int, int]:
    """Ingest a single PDF file.

    Returns: (source_id, chunks_created, headings_found)
    """
    logger.info("extracting_pdf", path=pdf_path)

    # Extract text and headings (returns tuple)
    doc, headings = extract_with_headings(pdf_path)

    metadata = metadata or {}
    metadata["extraction_method"] = "pymupdf"
    metadata["total_pages"] = doc.total_pages
    metadata["total_chars"] = doc.total_chars
    metadata["total_headings"] = len(headings)

    # Chunk the document
    logger.info("chunking_document", path=pdf_path)
    chunks = chunk_with_sections(doc, headings, target_tokens=300)
    metadata["total_chunks"] = len(chunks)

    logger.info("chunking_complete", path=pdf_path, chunks=len(chunks))

    # Compute file hash
    file_hash = compute_file_hash(pdf_path)

    # Create source record
    logger.info("creating_source", title=title)
    source = await SourceStore.create(
        source_type=SourceType.PAPER,
        title=title,
        authors=authors,
        year=year,
        file_path=pdf_path,
        file_hash=file_hash,
        metadata=metadata,
    )

    # Generate embeddings and store chunks
    embedding_client = EmbeddingClient()

    logger.info("generating_embeddings", chunks=len(chunks))
    chunks_created = 0

    for i, chunk in enumerate(chunks):
        # embed() is synchronous
        embedding = embedding_client.embed(chunk.content)

        # Compute content hash for deduplication
        content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()

        # ChunkStore.create is a class method
        await ChunkStore.create(
            source_id=source.id,
            content=chunk.content,
            content_hash=content_hash,
            page_start=chunk.start_page,
            page_end=chunk.end_page,
            embedding=embedding,
            metadata={
                "section_header": chunk.metadata.get("section", ""),
                "chunk_index": i,
            },
        )
        chunks_created += 1

        if (i + 1) % 50 == 0:
            logger.info("chunks_progress", current=i + 1, total=len(chunks))

    logger.info(
        "ingestion_complete",
        source_id=str(source.id),
        chunks=chunks_created,
        headings=len(headings),
    )

    return str(source.id), chunks_created, len(headings)


async def main():
    """Ingest all missing papers from fixtures/papers/."""
    papers_dir = Path(__file__).parent.parent / "fixtures" / "papers"

    if not papers_dir.exists():
        print(f"Error: {papers_dir} does not exist")
        return

    # Get all PDFs
    all_pdfs = list(papers_dir.glob("*.pdf"))
    print(f"Found {len(all_pdfs)} PDFs in {papers_dir}")

    # Initialize database connection pool
    config = DatabaseConfig()
    await get_connection_pool(config)

    # Check which are already ingested
    to_ingest = []
    already_ingested = 0

    for pdf_path in all_pdfs:
        file_hash = compute_file_hash(str(pdf_path))
        existing = await SourceStore.get_by_file_hash(file_hash)

        if existing:
            already_ingested += 1
            logger.info("already_ingested", path=pdf_path.name, title=existing.title)
        else:
            to_ingest.append(pdf_path)

    print(f"Already ingested: {already_ingested}")
    print(f"To ingest: {len(to_ingest)}")

    if not to_ingest:
        print("Nothing to ingest!")
        return

    # Ingest missing papers
    results = {"success": [], "failed": []}

    for i, pdf_path in enumerate(to_ingest):
        print(f"\n[{i+1}/{len(to_ingest)}] Processing: {pdf_path.name}")

        # Get metadata (prefers S2 sidecar, falls back to filename parsing)
        title, authors, year, extra_metadata = get_metadata_for_pdf(pdf_path)
        extra_metadata["auto_ingested"] = True

        # Log metadata source
        source = extra_metadata.get("metadata_source", "unknown")
        if source == "s2_sidecar":
            print(f"  📄 Using S2 metadata: {title[:50]}...")
        else:
            print(f"  📝 Using filename metadata: {title[:50]}...")

        try:
            source_id, num_chunks, num_headings = await ingest_pdf(
                pdf_path=str(pdf_path),
                title=title,
                authors=authors,
                year=year,
                metadata=extra_metadata,
            )

            results["success"].append(
                {
                    "file": pdf_path.name,
                    "title": title,
                    "chunks": num_chunks,
                }
            )

            print(f"  ✓ {num_chunks} chunks created")

        except Exception as e:
            logger.error("ingestion_failed", file=pdf_path.name, error=str(e), exc_info=True)
            results["failed"].append(
                {
                    "file": pdf_path.name,
                    "error": str(e),
                }
            )
            print(f"  ✗ Failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(f"Success: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")

    total_chunks = sum(r["chunks"] for r in results["success"])
    print(f"Total new chunks: {total_chunks}")

    if results["failed"]:
        print("\nFailed files:")
        for r in results["failed"]:
            print(f"  - {r['file']}: {r['error'][:50]}")


if __name__ == "__main__":
    asyncio.run(main())
