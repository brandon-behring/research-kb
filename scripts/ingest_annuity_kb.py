#!/usr/bin/env python3
"""Ingest annuity-knowledge-base markdown documents into research-kb.

Ingests validated findings from ~/Claude/annuity-knowledge-base/domain/
into the actuarial_insurance domain of research-kb.

These are high-value curated documents:
- MYGA_REGIME_BEHAVIOR.md — validated regime detection finding
- MYGA_ECONOMICS.md — MYGA market economics
- MYGA_WINK_SCHEMA.md — Wink data schema documentation
- PRODUCT_COMPARISON.md — annuity product comparison
- SIGN_CONVENTIONS.md — sign convention standards
- DATA_INVENTORY.md — data source inventory

Usage:
    cd ~/Claude/research-kb
    .venv/bin/python scripts/ingest_annuity_kb.py
    .venv/bin/python scripts/ingest_annuity_kb.py --quiet
"""

import argparse
import asyncio
import hashlib
import re
import sys
from pathlib import Path
from uuid import UUID

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import configure_logging, get_logger
from research_kb_contracts import SourceType
from research_kb_storage import ChunkStore, DatabaseConfig, SourceStore, get_connection_pool

logger = get_logger(__name__)

DOMAIN = "actuarial_insurance"
SOURCE_DIR = Path.home() / "Claude" / "annuity-knowledge-base" / "domain"
TARGET_CHUNK_CHARS = 1500
OVERLAP_CHARS = 200

# Metadata for each document
DOCUMENTS = {
    "MYGA_REGIME_BEHAVIOR.md": {
        "title": "MYGA Regime Behavior — Validated Findings",
        "year": 2026,
    },
    "MYGA_ECONOMICS.md": {
        "title": "MYGA Market Economics",
        "year": 2026,
    },
    "MYGA_WINK_SCHEMA.md": {
        "title": "Wink Data Schema Documentation",
        "year": 2026,
    },
    "PRODUCT_COMPARISON.md": {
        "title": "Annuity Product Comparison",
        "year": 2026,
    },
    "SIGN_CONVENTIONS.md": {
        "title": "Sign Convention Standards for Annuity Analytics",
        "year": 2026,
    },
    "DATA_INVENTORY.md": {
        "title": "Annuity Data Source Inventory",
        "year": 2026,
    },
}


def compute_hash(text: str) -> str:
    """Compute SHA256 hash for deduplication."""
    return f"sha256:{hashlib.sha256(text.encode()).hexdigest()}"


def chunk_markdown(text: str) -> list[dict]:
    """Split markdown into chunks by heading boundaries.

    Respects heading structure: never splits within a heading section
    if possible. Falls back to paragraph splitting for large sections.

    Returns list of dicts with 'content' and 'location' keys.
    """
    # Split by headings (##, ###, etc.)
    sections = re.split(r'(?=^#{1,4}\s)', text, flags=re.MULTILINE)
    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract heading for location
        heading_match = re.match(r'^(#{1,4})\s+(.+)', section)
        heading = heading_match.group(2).strip() if heading_match else "Preamble"

        if len(section) <= TARGET_CHUNK_CHARS:
            chunks.append({"content": section, "location": heading})
        else:
            # Split large sections by paragraphs
            paragraphs = section.split('\n\n')
            current = ""
            for para in paragraphs:
                if len(current) + len(para) > TARGET_CHUNK_CHARS and current:
                    chunks.append({"content": current.strip(), "location": heading})
                    # Overlap: keep last OVERLAP_CHARS of previous chunk
                    current = current[-OVERLAP_CHARS:] + "\n\n" + para
                else:
                    current = current + "\n\n" + para if current else para

            if current.strip():
                chunks.append({"content": current.strip(), "location": heading})

    return chunks


async def ingest_document(filepath: Path, metadata: dict, quiet: bool = False) -> bool:
    """Ingest a single markdown document into research-kb.

    Returns True if successfully ingested, False otherwise.
    """
    filename = filepath.name
    text = filepath.read_text(encoding="utf-8")
    file_hash = compute_hash(text)

    # Check if already ingested
    try:
        existing = await SourceStore.get_by_hash(file_hash)
        if existing:
            if not quiet:
                print(f"  SKIP: {filename} (already ingested as source {existing.id})")
            return True
    except Exception:
        pass  # get_by_hash may not exist; proceed to create

    # Create source record
    try:
        source = await SourceStore.create(
            source_type=SourceType.TEXTBOOK,
            title=metadata["title"],
            file_hash=file_hash,
            domain_id=DOMAIN,
            authors=["Brandon Behring"],
            year=metadata.get("year", 2026),
            file_path=str(filepath),
            metadata={"origin": "annuity-knowledge-base", "format": "markdown"},
        )
        if not quiet:
            print(f"  SOURCE: {filename} -> id={source.id}")
    except Exception as e:
        if "duplicate" in str(e).lower() or "unique" in str(e).lower():
            if not quiet:
                print(f"  SKIP: {filename} (duplicate hash)")
            return True
        print(f"  ERROR: {filename} source creation failed: {e}", file=sys.stderr)
        return False

    # Chunk the markdown
    chunks = chunk_markdown(text)
    if not quiet:
        print(f"  CHUNKS: {len(chunks)} chunks from {filename}")

    # Store each chunk
    stored = 0
    for chunk_data in chunks:
        try:
            await ChunkStore.create(
                source_id=source.id,
                content=chunk_data["content"],
                content_hash=compute_hash(chunk_data["content"]),
                domain_id=DOMAIN,
                location=chunk_data["location"],
                metadata={"chunk_type": "markdown", "origin": "annuity-knowledge-base"},
            )
            stored += 1
        except Exception as e:
            if "duplicate" not in str(e).lower():
                print(f"  WARN: chunk storage failed: {e}", file=sys.stderr)

    if not quiet:
        print(f"  STORED: {stored}/{len(chunks)} chunks for {filename}")

    return stored > 0


async def main() -> None:
    """Ingest all annuity-knowledge-base documents."""
    parser = argparse.ArgumentParser(description="Ingest annuity-knowledge-base into research-kb")
    parser.add_argument("--quiet", action="store_true", help="Errors + summary only")
    args = parser.parse_args()

    if not SOURCE_DIR.exists():
        print(f"ERROR: Source directory not found: {SOURCE_DIR}", file=sys.stderr)
        sys.exit(1)

    # Initialize database connection
    config = DatabaseConfig()
    await get_connection_pool(config)

    if not args.quiet:
        print(f"Ingesting from: {SOURCE_DIR}")
        print(f"Domain: {DOMAIN}")
        print()

    success = 0
    failed = 0

    for filename, metadata in DOCUMENTS.items():
        filepath = SOURCE_DIR / filename
        if not filepath.exists():
            if not args.quiet:
                print(f"  MISSING: {filename}")
            failed += 1
            continue

        if await ingest_document(filepath, metadata, quiet=args.quiet):
            success += 1
        else:
            failed += 1

    print(f"\nIngestion complete: {success} succeeded, {failed} failed")


if __name__ == "__main__":
    configure_logging(level="WARNING")
    asyncio.run(main())
