#!/usr/bin/env python3
"""Ingest papers from fixtures/papers subdirectories (arxiv, acquired_*).

This script scans subdirectories that ingest_missing_papers.py doesn't cover.
"""

import asyncio
import hashlib
import json
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import get_logger
from research_kb_contracts import SourceType
from research_kb_pdf import PDFDispatcher
from research_kb_storage import DatabaseConfig, get_connection_pool

logger = get_logger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_s2_sidecar(pdf_path: Path) -> dict | None:
    """Load S2 metadata sidecar if it exists."""
    sidecar_path = pdf_path.with_suffix(".s2.json")
    if not sidecar_path.exists():
        return None
    try:
        return json.loads(sidecar_path.read_text())
    except (json.JSONDecodeError, IOError):
        return None


def parse_arxiv_filename(filename: str) -> dict:
    """Parse arXiv ID from filename like '1608.00060.pdf'."""
    name = filename.replace(".pdf", "")
    # arXiv IDs are like YYMM.NNNNN or YYMM.NNNNNvN
    import re

    match = re.match(r"^(\d{4}\.\d{4,5})(v\d+)?$", name)
    if match:
        arxiv_id = match.group(1)
        return {
            "title": f"arXiv:{arxiv_id}",
            "authors": [],
            "year": 2000 + int(arxiv_id[:2]),  # YYMM -> year
            "arxiv_id": arxiv_id,
        }
    return {
        "title": name.replace("_", " ").title(),
        "authors": [],
        "year": None,
    }


async def get_existing_hashes() -> set[str]:
    """Get all existing file hashes from the database."""
    config = DatabaseConfig()
    pool = await get_connection_pool(config)

    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT file_hash FROM sources WHERE file_hash IS NOT NULL")
        return {r["file_hash"] for r in rows}


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ingest papers from subdirectories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested")
    parser.add_argument("--limit", type=int, default=None, help="Max papers to ingest")
    args = parser.parse_args()

    # Find all subdirectories with PDFs
    papers_dir = Path(__file__).parent.parent / "fixtures" / "papers"
    subdirs = [
        papers_dir / "arxiv",
        papers_dir / "acquired_2025_12_26",
        papers_dir / "acquired",
    ]

    # Get existing hashes
    existing = await get_existing_hashes()
    print(f"Existing sources: {len(existing)}")

    # Find missing PDFs
    missing = []
    for subdir in subdirs:
        if not subdir.exists():
            continue
        for pdf in subdir.glob("*.pdf"):
            h = compute_file_hash(pdf)
            if h not in existing:
                missing.append((pdf, h))

    print(f"Missing PDFs in subdirectories: {len(missing)}")

    if args.limit:
        missing = missing[: args.limit]

    if args.dry_run:
        for pdf, _ in missing:
            print(f"  Would ingest: {pdf.name}")
        return

    if not missing:
        print("Nothing to ingest!")
        return

    # Initialize dispatcher
    dispatcher = PDFDispatcher()

    success = 0
    failed = 0

    for pdf, file_hash in missing:
        # Try to get metadata from sidecar
        sidecar = load_s2_sidecar(pdf)

        if sidecar:
            title = sidecar.get("title") or pdf.stem
            authors = sidecar.get("authors") or []
            year = sidecar.get("year")
            metadata = {
                "s2_paper_id": sidecar.get("s2_paper_id"),
                "doi": sidecar.get("doi"),
                "arxiv_id": sidecar.get("arxiv_id"),
                "citation_count": sidecar.get("citation_count"),
                "venue": sidecar.get("venue"),
                "metadata_source": "s2_sidecar",
            }
        else:
            # Parse from filename
            info = parse_arxiv_filename(pdf.name)
            title = info["title"]
            authors = info["authors"]
            year = info["year"]
            metadata = {
                "arxiv_id": info.get("arxiv_id"),
                "metadata_source": "filename",
            }

        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}

        try:
            result = await dispatcher.ingest_pdf(
                pdf_path=pdf,
                source_type=SourceType.PAPER,
                title=title,
                authors=authors,
                year=year,
                metadata=metadata,
            )
            print(f"✓ {pdf.name}: {result.chunk_count} chunks")
            success += 1
        except Exception as e:
            print(f"✗ {pdf.name}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Summary: {success} success, {failed} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
