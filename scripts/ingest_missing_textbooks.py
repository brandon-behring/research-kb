#!/usr/bin/env python3
"""Ingest missing textbooks from fixtures/textbooks/ not in database.

This script:
1. Scans fixtures/textbooks/ for all PDFs
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
from research_kb_storage import ChunkStore, DatabaseConfig, SourceStore, get_connection_pool

logger = get_logger(__name__)


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_sidecar_metadata(pdf_path: Path) -> dict | None:
    """Load metadata from JSON sidecar if it exists."""
    json_path = pdf_path.with_suffix(".json")
    if json_path.exists():
        import json
        try:
            with open(json_path) as f:
                data = json.load(f)
            # Clean up title from sidecar (remove underscores)
            title = data.get("title", "").replace("_", " ").strip()
            return {
                "title": title,
                "authors": data.get("authors", []),
                "year": data.get("year"),
                "source_db": data.get("source_db"),
                "domain": data.get("domain"),
            }
        except Exception:
            return None
    return None


def parse_filename_for_metadata(filename: str) -> dict:
    """Extract metadata from textbook filename patterns."""
    name = filename.replace(".pdf", "")

    # Handle tier-prefixed files (tier1_01_title_year.pdf)
    tier_match = re.match(r'^tier\d+_\d+_(.+)_(\d{4})$', name)
    if tier_match:
        title = tier_match.group(1).replace("_", " ").title()
        year = int(tier_match.group(2))
        return {"title": title, "authors": [], "year": year}

    # Handle Train Discrete Choice chapters
    if "train_discrete_choice" in name.lower():
        return {
            "title": name.replace("_", " ").title(),
            "authors": ["Train, Kenneth E."],
            "year": 2009,
        }

    # Try standard patterns
    year_match = re.search(r'(\d{4})', name)
    year = int(year_match.group(1)) if year_match else None

    # Handle "Applied Bayesian..." long format
    if "Applied Bayesian" in name or "Rubin" in name:
        return {
            "title": "Applied Bayesian Modeling and Causal Inference from Incomplete-Data Perspectives",
            "authors": ["Gelman, Andrew", "Rubin, Donald B.", "Meng, Xiao-Li"],
            "year": 2004,
        }

    title = name.replace("_", " ").title()
    return {"title": title, "authors": [], "year": year}


async def ingest_textbook(
    pdf_path: str,
    title: str,
    authors: list[str],
    year: int | None,
    metadata: dict | None = None,
) -> tuple[str, int, int]:
    """Ingest a single textbook PDF.

    Returns: (source_id, chunks_created, headings_found)
    """
    logger.info("extracting_pdf", path=pdf_path)

    # Extract text and headings
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

    # Create source record as TEXTBOOK
    logger.info("creating_source", title=title)
    source = await SourceStore.create(
        source_type=SourceType.TEXTBOOK,  # Key difference from papers
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
        embedding = embedding_client.embed(chunk.content)

        # Compute content hash for deduplication
        content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()

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

    logger.info("ingestion_complete",
                source_id=str(source.id),
                chunks=chunks_created,
                headings=len(headings))

    return str(source.id), chunks_created, len(headings)


# Skip list - auxiliary files without content
SKIP_PATTERNS = [
    "Ch0_Front",
    "Index_p",
    "Refs_p",
    "_Glossary",
    "Quicksheet",
]


async def main():
    """Ingest all missing textbooks from fixtures/textbooks/."""
    textbooks_dir = Path(__file__).parent.parent / "fixtures" / "textbooks"

    if not textbooks_dir.exists():
        print(f"Error: {textbooks_dir} does not exist")
        return

    # Get all PDFs recursively
    all_pdfs = list(textbooks_dir.rglob("*.pdf"))
    print(f"Found {len(all_pdfs)} PDFs in {textbooks_dir}")

    # Initialize database connection pool
    config = DatabaseConfig()
    await get_connection_pool(config)

    # Check which are already ingested and filter skipped
    to_ingest = []
    already_ingested = 0
    skipped = 0

    for pdf_path in all_pdfs:
        # Skip auxiliary files
        if any(skip in pdf_path.name for skip in SKIP_PATTERNS):
            skipped += 1
            logger.info("skipping_auxiliary", path=pdf_path.name)
            continue

        file_hash = compute_file_hash(str(pdf_path))
        existing = await SourceStore.get_by_file_hash(file_hash)

        if existing:
            already_ingested += 1
            logger.info("already_ingested", path=pdf_path.name, title=existing.title)
        else:
            to_ingest.append(pdf_path)

    print(f"Already ingested: {already_ingested}")
    print(f"Skipped (auxiliary): {skipped}")
    print(f"To ingest: {len(to_ingest)}")

    if not to_ingest:
        print("Nothing to ingest!")
        return

    # Ingest missing textbooks
    results = {"success": [], "failed": []}

    for i, pdf_path in enumerate(to_ingest):
        print(f"\n[{i+1}/{len(to_ingest)}] Processing: {pdf_path.name}")

        # Try sidecar metadata first, fall back to filename parsing
        meta = load_sidecar_metadata(pdf_path)
        if not meta or not meta.get("title"):
            meta = parse_filename_for_metadata(pdf_path.name)

        try:
            source_id, num_chunks, num_headings = await ingest_textbook(
                pdf_path=str(pdf_path),
                title=meta["title"],
                authors=meta["authors"],
                year=meta["year"],
                metadata={"auto_ingested": True, "source": "missing_textbooks_script"},
            )

            results["success"].append({
                "file": pdf_path.name,
                "title": meta["title"],
                "chunks": num_chunks,
            })

            print(f"  ✓ {num_chunks} chunks created")

        except Exception as e:
            logger.error("ingestion_failed", file=pdf_path.name, error=str(e), exc_info=True)
            results["failed"].append({
                "file": pdf_path.name,
                "error": str(e),
            })
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
            print(f"  - {r['file']}: {r['error'][:80]}")


if __name__ == "__main__":
    asyncio.run(main())
