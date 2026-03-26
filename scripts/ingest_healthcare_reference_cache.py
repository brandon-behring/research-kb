#!/usr/bin/env python3
"""Ingest healthcare reference cache into research-kb.

Ingests 35 healthcare documents (32 markdown + 3 PDFs) from
data/healthcare/reference_cache/ into the healthcare domain.

Source: precision_aq project reference cache
Content: MCO structures, PBM regulation, PIE best practices,
         RWE standards, value-based contracting, drug pricing policy.

Usage:
    .venv/bin/python scripts/ingest_healthcare_reference_cache.py --dry-run
    .venv/bin/python scripts/ingest_healthcare_reference_cache.py --no-embed --quiet
    .venv/bin/python scripts/ingest_healthcare_reference_cache.py --markdown-only
"""

import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import configure_logging, get_logger
from research_kb_contracts import SourceType
from research_kb_storage import ChunkStore, DatabaseConfig, SourceStore, get_connection_pool

logger = get_logger(__name__)

DOMAIN = "healthcare"
REFERENCE_CACHE_DIR = Path(__file__).parent.parent / "data" / "healthcare" / "reference_cache"
INDEX_JSON = REFERENCE_CACHE_DIR / "index.json"
TARGET_CHUNK_CHARS = 1500
OVERLAP_CHARS = 200


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown text.

    Strips the --- delimited YAML block and returns parsed metadata
    and the remaining body text. Uses regex (no PyYAML dependency).

    Handles values containing colons (e.g., URLs) by splitting on
    first colon only.

    Returns:
        (frontmatter_dict, body_text). Empty dict if no frontmatter found.
    """
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        return {}, text
    yaml_block = match.group(1)
    body = text[match.end() :]
    meta = {}
    for line in yaml_block.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"')
    return meta, body


def compute_text_hash(text: str) -> str:
    """Compute SHA256 hash of text content with sha256: prefix."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of binary file with sha256: prefix."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return f"sha256:{sha256_hash.hexdigest()}"


def chunk_markdown(text: str) -> list[dict]:
    """Split markdown into chunks by heading boundaries.

    Respects heading structure: never splits within a heading section
    if possible. Falls back to paragraph splitting for large sections.

    Returns list of dicts with 'content' and 'location' keys.

    Reused from ingest_annuity_kb.py (lines 80-118).
    """
    # Split by headings (##, ###, etc.)
    sections = re.split(r"(?=^#{1,4}\s)", text, flags=re.MULTILINE)
    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract heading for location
        heading_match = re.match(r"^(#{1,4})\s+(.+)", section)
        heading = heading_match.group(2).strip() if heading_match else "Preamble"

        if len(section) <= TARGET_CHUNK_CHARS:
            chunks.append({"content": section, "location": heading})
        else:
            # Split large sections by paragraphs
            paragraphs = section.split("\n\n")
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


def load_index() -> list[dict]:
    """Load and filter index.json to cached entries only."""
    with open(INDEX_JSON) as f:
        data = json.load(f)

    cached = []
    for entry in data["entries"]:
        if entry.get("status") == "cached" and entry.get("cached_files"):
            cached.append(entry)
    return cached


def build_source_metadata(entry: dict, frontmatter: dict) -> dict:
    """Build metadata JSONB dict from index entry and frontmatter."""
    return {
        "bib_key": entry.get("bib_key", ""),
        "url": entry.get("url", ""),
        "content_type": entry.get("content_type", "unknown"),
        "tags": entry.get("tags", []),
        "acquisition_method": entry.get("method", "unknown"),
        "fetched_date": frontmatter.get("fetched_date", entry.get("fetched_date", "")),
        "origin": "precision_aq_reference_cache",
    }


def classify_entry(entry: dict) -> str:
    """Classify entry as 'markdown' or 'pdf' based on cached file extension."""
    cached_file = entry["cached_files"][0]
    if cached_file.endswith(".pdf"):
        return "pdf"
    return "markdown"


async def ingest_markdown_entry(
    entry: dict,
    quiet: bool = False,
    no_embed: bool = False,
) -> dict:
    """Ingest a single markdown file.

    Returns result dict with keys: bib_key, status, chunks, error.
    """
    bib_key = entry.get("bib_key", "unknown")
    cached_file = entry["cached_files"][0]
    file_path = REFERENCE_CACHE_DIR / cached_file

    # Check file exists
    if not file_path.exists():
        if not quiet:
            logger.warning("file_missing", bib_key=bib_key, path=str(file_path))
        return {"bib_key": bib_key, "status": "missing", "chunks": 0, "error": None}

    # Read and parse
    text = file_path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(text)
    file_hash = compute_text_hash(text)

    # Check dedup
    try:
        existing = await SourceStore.get_by_file_hash(file_hash)
        if existing:
            if not quiet:
                print(f"  SKIP {bib_key} (already ingested: {existing.id})")
            return {"bib_key": bib_key, "status": "skipped", "chunks": 0, "error": None}
    except Exception:
        pass  # get_by_file_hash may raise on new hash format; proceed

    # Create source
    metadata = build_source_metadata(entry, frontmatter)
    try:
        source = await SourceStore.create(
            source_type=SourceType.PAPER,
            title=entry.get("title", bib_key),
            file_hash=file_hash,
            domain_id=DOMAIN,
            authors=[],
            year=None,
            file_path=str(file_path),
            metadata=metadata,
        )
    except Exception as e:
        if "duplicate" in str(e).lower() or "unique" in str(e).lower():
            if not quiet:
                print(f"  SKIP {bib_key} (duplicate hash)")
            return {"bib_key": bib_key, "status": "skipped", "chunks": 0, "error": None}
        logger.error("source_creation_failed", bib_key=bib_key, error=str(e))
        return {"bib_key": bib_key, "status": "failed", "chunks": 0, "error": str(e)}

    # Chunk
    chunks = chunk_markdown(body)
    if not chunks:
        if not quiet:
            print(f"  WARN {bib_key}: no chunks produced from body")
        return {"bib_key": bib_key, "status": "ingested", "chunks": 0, "error": None}

    # Embeddings
    embeddings = [None] * len(chunks)
    if not no_embed:
        try:
            from research_kb_pdf import EmbeddingClient

            client = EmbeddingClient()
            texts = [c["content"] for c in chunks]
            embeddings = client.embed_batch(texts, batch_size=32)
        except Exception as e:
            logger.warning("embedding_failed", bib_key=bib_key, error=str(e))
            embeddings = [None] * len(chunks)

    # Store chunks
    stored = 0
    for i, (chunk_data, embedding) in enumerate(zip(chunks, embeddings)):
        try:
            await ChunkStore.create(
                source_id=source.id,
                content=chunk_data["content"],
                content_hash=compute_text_hash(chunk_data["content"]),
                domain_id=DOMAIN,
                location=chunk_data["location"],
                embedding=embedding,
                metadata={"chunk_index": i, "chunk_type": "markdown"},
            )
            stored += 1
        except Exception as e:
            if "duplicate" not in str(e).lower():
                logger.warning("chunk_failed", bib_key=bib_key, chunk=i, error=str(e))

    if not quiet:
        print(f"  OK   {bib_key} -> {stored} chunks")

    return {"bib_key": bib_key, "status": "ingested", "chunks": stored, "error": None}


async def ingest_pdf_entry(
    entry: dict,
    quiet: bool = False,
    no_embed: bool = False,
) -> dict:
    """Ingest a single PDF file via Docling.

    Returns result dict with keys: bib_key, status, chunks, error.
    """
    bib_key = entry.get("bib_key", "unknown")
    cached_file = entry["cached_files"][0]
    file_path = REFERENCE_CACHE_DIR / cached_file

    # Check file exists
    if not file_path.exists():
        if not quiet:
            logger.warning("file_missing", bib_key=bib_key, path=str(file_path))
        return {"bib_key": bib_key, "status": "missing", "chunks": 0, "error": None}

    # Hash and dedup
    file_hash = compute_file_hash(file_path)
    try:
        existing = await SourceStore.get_by_file_hash(file_hash)
        if existing:
            if not quiet:
                print(f"  SKIP {bib_key} (already ingested: {existing.id})")
            return {"bib_key": bib_key, "status": "skipped", "chunks": 0, "error": None}
    except Exception:
        pass

    # Extract with Docling
    try:
        from research_kb_pdf import extract_and_chunk

        if not quiet:
            print(f"  EXTRACT {bib_key} ...")
        extraction_result, chunks = extract_and_chunk(str(file_path), max_tokens=300)
    except Exception as e:
        logger.error("pdf_extraction_failed", bib_key=bib_key, error=str(e))
        return {"bib_key": bib_key, "status": "failed", "chunks": 0, "error": str(e)}

    # Build metadata
    metadata = build_source_metadata(entry, {})
    metadata["extraction_method"] = "docling"
    metadata["total_pages"] = extraction_result.total_pages
    metadata["total_chars"] = extraction_result.total_chars
    metadata["total_chunks"] = len(chunks)

    # Create source
    try:
        source = await SourceStore.create(
            source_type=SourceType.PAPER,
            title=entry.get("title", bib_key),
            file_hash=file_hash,
            domain_id=DOMAIN,
            authors=[],
            year=None,
            file_path=str(file_path),
            metadata=metadata,
        )
    except Exception as e:
        if "duplicate" in str(e).lower() or "unique" in str(e).lower():
            if not quiet:
                print(f"  SKIP {bib_key} (duplicate hash)")
            return {"bib_key": bib_key, "status": "skipped", "chunks": 0, "error": None}
        logger.error("source_creation_failed", bib_key=bib_key, error=str(e))
        return {"bib_key": bib_key, "status": "failed", "chunks": 0, "error": str(e)}

    # Embeddings
    if no_embed:
        embeddings = [None] * len(chunks)
    else:
        try:
            from research_kb_pdf import EmbeddingClient

            client = EmbeddingClient()
            texts = [chunk.content for chunk in chunks]
            if not quiet:
                print(f"  EMBED {bib_key} ({len(texts)} chunks) ...")
            embeddings = client.embed_batch(texts, batch_size=32)
        except Exception as e:
            logger.warning("embedding_failed", bib_key=bib_key, error=str(e))
            embeddings = [None] * len(chunks)

    # Batch insert chunks
    chunks_data = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
        chunks_data.append(
            {
                "source_id": source.id,
                "content": chunk.content,
                "content_hash": f"sha256:{content_hash}",
                "page_start": chunk.start_page,
                "page_end": chunk.end_page,
                "embedding": embedding,
                "domain_id": DOMAIN,
                "metadata": {
                    "section_header": chunk.metadata.get("section", ""),
                    "chunk_index": i,
                    "chunk_type": "pdf",
                },
            }
        )

    BATCH_SIZE = 100
    chunks_created = 0
    for i in range(0, len(chunks_data), BATCH_SIZE):
        batch = chunks_data[i : i + BATCH_SIZE]
        await ChunkStore.batch_create(batch)
        chunks_created += len(batch)

    if not quiet:
        print(f"  OK   {bib_key} -> {chunks_created} chunks")

    return {"bib_key": bib_key, "status": "ingested", "chunks": chunks_created, "error": None}


async def main() -> None:
    """Ingest healthcare reference cache into research-kb."""
    parser = argparse.ArgumentParser(
        description="Ingest healthcare reference cache into research-kb"
    )
    parser.add_argument("--quiet", action="store_true", help="Errors + summary only")
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip embeddings (two-phase GPU workflow)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested without writing",
    )
    parser.add_argument("--markdown-only", action="store_true", help="Skip PDF entries")
    parser.add_argument("--pdf-only", action="store_true", help="Skip markdown entries")
    args = parser.parse_args()

    if not INDEX_JSON.exists():
        print(f"ERROR: index.json not found: {INDEX_JSON}", file=sys.stderr)
        print("Run: cp -r ~/Claude/precision_aq/reference_cache/ data/healthcare/reference_cache/")
        sys.exit(1)

    # Load and classify entries
    entries = load_index()
    md_entries = [e for e in entries if classify_entry(e) == "markdown"]
    pdf_entries = [e for e in entries if classify_entry(e) == "pdf"]

    if args.markdown_only:
        pdf_entries = []
    if args.pdf_only:
        md_entries = []

    if not args.quiet:
        print(f"Healthcare Reference Cache Ingestion")
        print(f"{'=' * 40}")
        print(f"Source: {REFERENCE_CACHE_DIR}")
        print(f"Domain: {DOMAIN}")
        print(f"Entries: {len(md_entries)} markdown, {len(pdf_entries)} PDF")
        if args.no_embed:
            print("Embeddings: SKIPPED (--no-embed)")
        if args.dry_run:
            print("Mode: DRY RUN")
        print()

    if args.dry_run:
        print("Markdown files:")
        for e in md_entries:
            path = REFERENCE_CACHE_DIR / e["cached_files"][0]
            exists = "OK" if path.exists() else "MISSING"
            size = f"{path.stat().st_size:,} bytes" if path.exists() else "---"
            print(f"  [{exists}] {e['bib_key']}: {e['title']} ({size})")
        print()
        print("PDF files:")
        for e in pdf_entries:
            path = REFERENCE_CACHE_DIR / e["cached_files"][0]
            exists = "OK" if path.exists() else "MISSING"
            size = f"{path.stat().st_size:,} bytes" if path.exists() else "---"
            print(f"  [{exists}] {e['bib_key']}: {e['title']} ({size})")
        print()
        print(f"Total: {len(md_entries) + len(pdf_entries)} entries would be ingested")
        return

    # Initialize database
    config = DatabaseConfig()
    await get_connection_pool(config)

    results = []

    # Ingest markdown
    if md_entries:
        if not args.quiet:
            print(f"Markdown ({len(md_entries)} files):")
        for entry in md_entries:
            result = await ingest_markdown_entry(entry, quiet=args.quiet, no_embed=args.no_embed)
            results.append(result)

    # Ingest PDFs
    if pdf_entries:
        if not args.quiet:
            print(f"\nPDFs ({len(pdf_entries)} files):")
        for entry in pdf_entries:
            result = await ingest_pdf_entry(entry, quiet=args.quiet, no_embed=args.no_embed)
            results.append(result)

    # Summary
    ingested = [r for r in results if r["status"] == "ingested"]
    skipped = [r for r in results if r["status"] == "skipped"]
    missing = [r for r in results if r["status"] == "missing"]
    failed = [r for r in results if r["status"] == "failed"]
    total_chunks = sum(r["chunks"] for r in results)

    print(f"\n{'=' * 40}")
    print(f"SUMMARY")
    print(f"  Ingested: {len(ingested)} sources, {total_chunks} chunks")
    print(f"  Skipped:  {len(skipped)} (already in DB)")
    print(f"  Missing:  {len(missing)} (file not found)")
    print(f"  Failed:   {len(failed)}")

    if missing and not args.quiet:
        print(f"\nMissing files:")
        for r in missing:
            print(f"  - {r['bib_key']}")

    if failed:
        print(f"\nFailed entries:", file=sys.stderr)
        for r in failed:
            print(f"  - {r['bib_key']}: {r['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    configure_logging(level="WARNING")
    asyncio.run(main())
