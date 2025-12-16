#!/usr/bin/env python3
"""Ingest missing textbooks from fixtures/textbooks/ not in database.

This script:
1. Scans fixtures/textbooks/ for all PDFs
2. Checks which ones are already ingested (by file hash)
3. Ingests the missing ones with auto-extracted metadata

Usage:
    python scripts/ingest_missing_textbooks.py          # Normal output
    python scripts/ingest_missing_textbooks.py --quiet  # Errors + summary only
"""

import argparse
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
    quiet: bool = False,
) -> tuple[str, int, int]:
    """Ingest a single textbook PDF.

    Returns: (source_id, chunks_created, headings_found)
    """
    if not quiet:
        logger.info("extracting_pdf", path=pdf_path)

    # Extract text and headings
    doc, headings = extract_with_headings(pdf_path)

    metadata = metadata or {}
    metadata["extraction_method"] = "pymupdf"
    metadata["total_pages"] = doc.total_pages
    metadata["total_chars"] = doc.total_chars
    metadata["total_headings"] = len(headings)

    # Chunk the document
    if not quiet:
        logger.info("chunking_document", path=pdf_path)
    chunks = chunk_with_sections(doc, headings, target_tokens=300)
    metadata["total_chunks"] = len(chunks)

    if not quiet:
        logger.info("chunking_complete", path=pdf_path, chunks=len(chunks))

    # Compute file hash
    file_hash = compute_file_hash(pdf_path)

    # Create source record as TEXTBOOK
    if not quiet:
        logger.info("creating_source", title=title)
    source = await SourceStore.create(
        source_type=SourceType.TEXTBOOK,
        title=title,
        authors=authors,
        year=year,
        file_path=pdf_path,
        file_hash=file_hash,
        metadata=metadata,
    )

    # Batch generate embeddings
    embedding_client = EmbeddingClient()
    texts = [chunk.content for chunk in chunks]

    if not quiet:
        logger.info("generating_embeddings", chunks=len(chunks))

    embeddings = embedding_client.embed_batch(texts, batch_size=32)

    # Prepare batch data for insertion
    chunks_data = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
        chunks_data.append({
            "source_id": source.id,
            "content": chunk.content,
            "content_hash": content_hash,
            "page_start": chunk.start_page,
            "page_end": chunk.end_page,
            "embedding": embedding,
            "metadata": {
                "section_header": chunk.metadata.get("section", ""),
                "chunk_index": i,
            },
        })

    # Batch insert in groups of 100
    BATCH_SIZE = 100
    chunks_created = 0
    for i in range(0, len(chunks_data), BATCH_SIZE):
        batch = chunks_data[i:i + BATCH_SIZE]
        await ChunkStore.batch_create(batch)
        chunks_created += len(batch)

    if not quiet:
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest missing textbooks from fixtures/textbooks/."
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Minimal output (one line per PDF + final summary)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output summary as JSON (for programmatic parsing)"
    )
    return parser.parse_args()


async def main():
    """Ingest all missing textbooks from fixtures/textbooks/."""
    import json as json_module
    import time

    from research_kb_common import EmbeddingError, StorageError, configure_logging

    args = parse_args()
    quiet = args.quiet
    json_output = args.json

    # Configure logging based on quiet mode
    if quiet:
        configure_logging(level="ERROR")

    textbooks_dir = Path(__file__).parent.parent / "fixtures" / "textbooks"

    if not textbooks_dir.exists():
        if json_output:
            print(json_module.dumps({"error": f"{textbooks_dir} does not exist"}))
        else:
            print(f"Error: {textbooks_dir} does not exist")
        return

    # Get all PDFs recursively
    all_pdfs = list(textbooks_dir.rglob("*.pdf"))
    if not quiet:
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
            if not quiet:
                logger.info("skipping_auxiliary", path=pdf_path.name)
            continue

        file_hash = compute_file_hash(str(pdf_path))
        existing = await SourceStore.get_by_file_hash(file_hash)

        if existing:
            already_ingested += 1
            if not quiet:
                logger.info("already_ingested", path=pdf_path.name, title=existing.title)
        else:
            to_ingest.append(pdf_path)

    if not quiet:
        print(f"Already ingested: {already_ingested}")
        print(f"Skipped (auxiliary): {skipped}")
        print(f"To ingest: {len(to_ingest)}")

    if not to_ingest:
        if json_output:
            print(json_module.dumps({
                "success_count": 0,
                "failed_count": 0,
                "total_chunks": 0,
                "already_ingested": already_ingested,
                "skipped": skipped,
                "failed_files": []
            }))
        elif not quiet:
            print("Nothing to ingest!")
        return

    # Ingest missing textbooks
    results = {"success": [], "failed": []}

    for i, pdf_path in enumerate(to_ingest):
        if not quiet and not json_output:
            print(f"\n[{i+1}/{len(to_ingest)}] Processing: {pdf_path.name}")

        # Try sidecar metadata first, fall back to filename parsing
        meta = load_sidecar_metadata(pdf_path)
        if not meta or not meta.get("title"):
            meta = parse_filename_for_metadata(pdf_path.name)

        start_time = time.time()

        try:
            source_id, num_chunks, num_headings = await ingest_textbook(
                pdf_path=str(pdf_path),
                title=meta["title"],
                authors=meta["authors"],
                year=meta["year"],
                metadata={"auto_ingested": True, "source": "missing_textbooks_script"},
                quiet=quiet,
            )

            elapsed = time.time() - start_time

            results["success"].append({
                "file": pdf_path.name,
                "title": meta["title"],
                "chunks": num_chunks,
                "elapsed_seconds": round(elapsed, 1),
            })

            # Single-line progress in quiet mode
            if quiet and not json_output:
                print(f"✓ {pdf_path.name}: {num_chunks} chunks ({elapsed:.0f}s)")
            elif not quiet and not json_output:
                print(f"  ✓ {num_chunks} chunks created")

        except (MemoryError, OSError) as e:
            # System-level errors: not recoverable
            error_type = "memory_exhausted" if isinstance(e, MemoryError) else "file_io_error"
            if not json_output:
                logger.error("system_error", file=pdf_path.name, error_type=error_type, error=str(e))
            results["failed"].append({
                "file": pdf_path.name,
                "error": f"{error_type}: {str(e)[:100]}",
                "recoverable": False,
            })
            if quiet and not json_output:
                print(f"✗ {pdf_path.name}: {error_type}")
            elif not json_output:
                print(f"  ✗ System error: {error_type}")

        except (EmbeddingError, ConnectionError) as e:
            # Embedding service errors: recoverable (retry later)
            if not json_output:
                logger.error("embedding_service_failure", file=pdf_path.name, error=str(e))
            results["failed"].append({
                "file": pdf_path.name,
                "error": "Embedding service failure (retries exhausted)",
                "recoverable": True,
            })
            if quiet and not json_output:
                print(f"✗ {pdf_path.name}: embedding service failure (recoverable)")
            elif not json_output:
                print(f"  ✗ Embedding service failure (retry ingestion later)")

        except StorageError as e:
            # Database errors: recoverable
            if not json_output:
                logger.error("storage_error", file=pdf_path.name, error=str(e))
            results["failed"].append({
                "file": pdf_path.name,
                "error": f"Database error: {str(e)[:100]}",
                "recoverable": True,
            })
            if quiet and not json_output:
                print(f"✗ {pdf_path.name}: database error (recoverable)")
            elif not json_output:
                print(f"  ✗ Database error")

        except Exception as e:
            # Unknown errors: not recoverable
            if not json_output:
                logger.error("ingestion_failed", file=pdf_path.name, error=str(e), exc_info=True)
            results["failed"].append({
                "file": pdf_path.name,
                "error": str(e)[:100],
                "recoverable": False,
            })
            if quiet and not json_output:
                print(f"✗ {pdf_path.name}: {str(e)[:60]}")
            elif not json_output:
                print(f"  ✗ Failed: {e}")

    # Summary
    total_chunks = sum(r["chunks"] for r in results["success"])

    if json_output:
        # JSON summary for programmatic parsing
        summary = {
            "success_count": len(results["success"]),
            "failed_count": len(results["failed"]),
            "total_chunks": total_chunks,
            "already_ingested": already_ingested,
            "skipped": skipped,
            "failed_files": [
                {"file": r["file"], "error": r["error"], "recoverable": r.get("recoverable", False)}
                for r in results["failed"]
            ]
        }
        print(json_module.dumps(summary, indent=2))
    elif quiet:
        # Minimal summary (already printed single-line per PDF)
        print(f"\nIngested: {len(results['success'])} textbooks | {total_chunks} chunks")
        if results["failed"]:
            recoverable = sum(1 for r in results["failed"] if r.get("recoverable", False))
            print(f"Failed: {len(results['failed'])} ({recoverable} recoverable)")
    else:
        # Full summary
        print("\n" + "=" * 70)
        print("INGESTION SUMMARY")
        print("=" * 70)
        print(f"Success: {len(results['success'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"Total new chunks: {total_chunks}")

        if results["failed"]:
            print("\nFailed files:")
            for r in results["failed"]:
                status = "(recoverable)" if r.get("recoverable", False) else "(not recoverable)"
                print(f"  - {r['file']}: {r['error'][:60]} {status}")


if __name__ == "__main__":
    asyncio.run(main())
