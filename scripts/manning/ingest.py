"""Multi-domain PDF ingestion from Documents/ directory.

Reads catalog for book list and metadata. For each un-ingested book:
1. Get canonical PDF path from dedup
2. Check file_hash → skip if already in DB
3. Extract text + headings (PyMuPDF)
4. Chunk with section tracking
5. Embed in batches of ≤100 (avoid 60s timeout on CPU)
6. Create source record with domain metadata
7. Batch insert chunks

Usage:
    python -m scripts.manning ingest --tier 1 --quiet
    python -m scripts.manning ingest --domain rag_llm --quiet
    python -m scripts.manning ingest --title "Essential GraphRAG"
    python -m scripts.manning ingest --tier 1 --dry-run
    python -m scripts.manning ingest --upgrade-meaps --dry-run
"""

import hashlib
import json as json_module
import sys
import time
from pathlib import Path

from .catalog import ManningBook, load_catalog
from .dedup import compute_file_hash

_project_root = Path(__file__).parent.parent.parent
for pkg in ("pdf-tools", "storage", "contracts", "common"):
    sys.path.insert(0, str(_project_root / "packages" / pkg / "src"))

from research_kb_common import (
    EmbeddingError,
    StorageError,
    configure_logging,
    get_logger,
)  # noqa: E402
from research_kb_contracts import SourceType  # noqa: E402
from research_kb_pdf import (
    EmbeddingClient,
    chunk_with_sections,
    extract_with_headings,
)  # noqa: E402
from research_kb_storage import (
    ChunkStore,
    DatabaseConfig,
    SourceStore,
    get_connection_pool,
)  # noqa: E402

logger = get_logger(__name__)

# Client-side embedding batch size (avoid 60s socket timeout on CPU)
EMBED_BATCH = 100
# DB insert batch size
INSERT_BATCH = 100


def filter_books(
    books: list[ManningBook],
    tier: int | None = None,
    domain: str | None = None,
    title: str | None = None,
) -> list[ManningBook]:
    """Filter catalog books by tier, domain, or exact title.

    Args:
        books: Full catalog.
        tier: If set, only books with this tier (1, 2, or 3).
        domain: If set, only books with this domain_id.
        title: If set, only the book matching this title (exact or substring).

    Returns:
        Filtered list of ManningBook.
    """
    result = books

    if tier is not None:
        result = [b for b in result if b.tier == tier]
    if domain is not None:
        result = [b for b in result if b.domain_id == domain]
    if title is not None:
        # Try exact match first, then case-insensitive substring
        exact = [b for b in result if b.title == title]
        if exact:
            result = exact
        else:
            title_lower = title.lower()
            result = [b for b in result if title_lower in b.title.lower()]

    return result


async def check_already_ingested(pool, file_hash: str) -> tuple[bool, dict | None]:
    """Check if a file is already ingested (by hash), and if so, whether it has chunks.

    Returns:
        (is_ingested, source_info_or_none)
    """
    existing = await SourceStore.get_by_file_hash(file_hash)
    if not existing:
        return False, None

    async with pool.acquire() as conn:
        chunk_count = await conn.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE source_id = $1", existing.id
        )

    return chunk_count > 0, {
        "id": str(existing.id),
        "title": existing.title,
        "chunk_count": chunk_count,
    }


async def ingest_one_book(
    book: ManningBook,
    pool,
    quiet: bool = False,
) -> tuple[str, int, int]:
    """Ingest a single book PDF.

    Returns:
        (source_id, chunks_created, headings_found)

    Raises:
        Various exceptions on failure (caught by caller).
    """
    pdf_path = str(book.best_pdf)

    if not quiet:
        logger.info("extracting_pdf", title=book.title, path=pdf_path)

    doc, headings = extract_with_headings(pdf_path)

    metadata = {
        "domain": book.domain_id,
        "extraction_method": "pymupdf",
        "source_platform": "manning",
        "meap_version": book.meap_version,
        "tier": book.tier,
        "auto_ingested": True,
        "ingestion_script": "scripts.manning.ingest",
        "total_pages": doc.total_pages,
        "total_chars": doc.total_chars,
    }

    if not quiet:
        logger.info("chunking_document", title=book.title)
    chunks = chunk_with_sections(doc, headings, target_tokens=300)
    metadata["total_chunks"] = len(chunks)

    if not quiet:
        logger.info("chunking_complete", title=book.title, chunks=len(chunks))

    file_hash = compute_file_hash(pdf_path)

    # Create source record
    if not quiet:
        logger.info("creating_source", title=book.title)
    source = await SourceStore.create(
        source_type=SourceType.TEXTBOOK,
        title=book.title,
        authors=book.authors,
        year=book.year,
        file_path=pdf_path,
        file_hash=file_hash,
        metadata=metadata,
    )

    # Embed in batches of ≤100
    embedding_client = EmbeddingClient()
    texts = [chunk.content for chunk in chunks]

    if not quiet:
        logger.info("generating_embeddings", title=book.title, chunks=len(chunks))

    embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch_texts = texts[i : i + EMBED_BATCH]
        if not quiet:
            logger.info(
                "embedding_batch",
                start=i,
                end=i + len(batch_texts),
                total=len(texts),
            )
        batch_embeddings = embedding_client.embed_batch(batch_texts, batch_size=32)
        embeddings.extend(batch_embeddings)

    # Build chunk data
    chunks_data = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
        chunks_data.append(
            {
                "source_id": source.id,
                "content": chunk.content,
                "content_hash": content_hash,
                "page_start": chunk.start_page,
                "page_end": chunk.end_page,
                "embedding": embedding,
                "metadata": {
                    "section_header": chunk.metadata.get("section", ""),
                    "chunk_index": i,
                    "domain": book.domain_id,
                },
            }
        )

    # Batch insert chunks
    chunks_created = 0
    for i in range(0, len(chunks_data), INSERT_BATCH):
        batch = chunks_data[i : i + INSERT_BATCH]
        await ChunkStore.batch_create(batch)
        chunks_created += len(batch)

    if not quiet:
        logger.info(
            "ingestion_complete",
            source_id=str(source.id),
            title=book.title,
            chunks=chunks_created,
            headings=len(headings),
        )

    return str(source.id), chunks_created, len(headings)


async def check_meap_upgrades(pool, books: list[ManningBook]) -> list[dict]:
    """Find books where catalog has a newer MEAP version than DB.

    A MEAP upgrade means: different file_hash (new PDF content), but same book.

    Returns:
        List of upgrade candidates with old/new version info.
    """
    upgrades: list[dict] = []

    async with pool.acquire() as conn:
        for book in books:
            if book.meap_version is None:
                continue

            # Find matching source in DB
            rows = await conn.fetch(
                """
                SELECT id, title, metadata
                FROM sources
                WHERE title ILIKE $1
                   OR title ILIKE $2
                LIMIT 5
                """,
                f"%{book.title}%",
                f"%{book.title.replace(' ', '_')}%",
            )

            for row in rows:
                meta = row["metadata"] or {}
                if isinstance(meta, str):
                    import json as _json

                    meta = _json.loads(meta)
                db_version = meta.get("meap_version")
                if db_version is not None and book.meap_version > int(db_version):
                    chunk_count = await conn.fetchval(
                        "SELECT COUNT(*) FROM chunks WHERE source_id = $1", row["id"]
                    )
                    upgrades.append(
                        {
                            "title": book.title,
                            "db_source_id": str(row["id"]),
                            "db_version": db_version,
                            "catalog_version": book.meap_version,
                            "db_chunks": chunk_count,
                        }
                    )

    return upgrades


async def upgrade_meap(pool, book: ManningBook, old_source_id: str, quiet: bool = False) -> dict:
    """Replace an old MEAP source with the newer version.

    1. Delete old chunks
    2. Delete old source
    3. Ingest new PDF

    Returns:
        Result dict with new source info.
    """
    import uuid as uuid_mod

    old_uuid = uuid_mod.UUID(old_source_id)

    async with pool.acquire() as conn:
        # Delete old chunks (cascading from chunk_concepts if they exist)
        old_chunk_ids = await conn.fetch("SELECT id FROM chunks WHERE source_id = $1", old_uuid)
        if old_chunk_ids:
            chunk_ids = [r["id"] for r in old_chunk_ids]
            # Delete chunk_concepts first (FK)
            await conn.execute(
                "DELETE FROM chunk_concepts WHERE chunk_id = ANY($1::uuid[])",
                chunk_ids,
            )
            await conn.execute("DELETE FROM chunks WHERE source_id = $1", old_uuid)

        # Delete old source
        await conn.execute("DELETE FROM sources WHERE id = $1", old_uuid)

    # Ingest the new version
    source_id, chunks, headings = await ingest_one_book(book, pool, quiet=quiet)

    return {
        "title": book.title,
        "old_source_id": old_source_id,
        "new_source_id": source_id,
        "new_chunks": chunks,
        "new_version": book.meap_version,
    }


async def run_ingest(
    tier: int | None = None,
    domain: str | None = None,
    title: str | None = None,
    dry_run: bool = False,
    upgrade_meaps: bool = False,
    quiet: bool = False,
    json_output: bool = False,
) -> None:
    """Main ingestion entry point.

    Filters catalog, checks what's already ingested, ingests the rest.
    """
    if quiet:
        configure_logging(level="ERROR")

    books = load_catalog()
    selected = filter_books(books, tier=tier, domain=domain, title=title)

    if not selected:
        msg = "No books match the given filters."
        if json_output:
            print(json_module.dumps({"error": msg}))
        else:
            print(msg)
        return

    config = DatabaseConfig()
    pool = await get_connection_pool(config)

    # Handle MEAP upgrades separately
    if upgrade_meaps:
        upgrades = await check_meap_upgrades(pool, selected)
        if not upgrades:
            msg = "No MEAP upgrades available."
            if json_output:
                print(json_module.dumps({"upgrades": []}))
            else:
                print(msg)
            return

        if not json_output:
            print(f"Found {len(upgrades)} MEAP upgrades:")
            for u in upgrades:
                print(
                    f"  {u['title']}: v{u['db_version']} → v{u['catalog_version']} "
                    f"({u['db_chunks']} chunks to replace)"
                )

        if dry_run:
            if json_output:
                print(json_module.dumps({"upgrades": upgrades, "dry_run": True}))
            else:
                print("\nDry run — no changes made.")
            return

        results: list[dict] = []
        for u in upgrades:
            book = next(b for b in selected if b.title == u["title"])
            result = await upgrade_meap(pool, book, u["db_source_id"], quiet=quiet)
            results.append(result)
            if not json_output:
                print(f"  Upgraded: {result['title']} → {result['new_chunks']} chunks")

        if json_output:
            print(json_module.dumps({"upgrades": results}))
        return

    # Standard ingestion flow
    to_ingest: list[ManningBook] = []
    already_ingested = 0
    skipped_no_pdf: list[str] = []

    for book in selected:
        if book.best_pdf is None or not book.best_pdf.exists():
            skipped_no_pdf.append(book.title)
            continue

        file_hash = compute_file_hash(str(book.best_pdf))
        is_ingested, info = await check_already_ingested(pool, file_hash)

        if is_ingested:
            already_ingested += 1
            if not quiet:
                logger.info("already_ingested", title=book.title)
        else:
            # If source exists but no chunks, clean it up
            if info and info["chunk_count"] == 0:
                import uuid as uuid_mod

                async with pool.acquire() as conn:
                    await conn.execute(
                        "DELETE FROM sources WHERE id = $1",
                        uuid_mod.UUID(info["id"]),
                    )
            to_ingest.append(book)

    if not quiet and not json_output:
        print(f"Selected: {len(selected)} books")
        print(f"Already ingested: {already_ingested}")
        print(f"To ingest: {len(to_ingest)}")
        if skipped_no_pdf:
            print(f"Skipped (no PDF): {len(skipped_no_pdf)}")

    if dry_run:
        if json_output:
            print(
                json_module.dumps(
                    {
                        "dry_run": True,
                        "selected": len(selected),
                        "already_ingested": already_ingested,
                        "to_ingest": len(to_ingest),
                        "books": [
                            {
                                "title": b.title,
                                "domain": b.domain_id,
                                "tier": b.tier,
                                "pdf": str(b.best_pdf),
                            }
                            for b in to_ingest
                        ],
                    }
                )
            )
        else:
            print("\nBooks to ingest:")
            for b in to_ingest:
                print(f"  [{b.domain_id}] {b.title}")
            print("\nDry run — no changes made.")
        return

    if not to_ingest:
        if json_output:
            print(
                json_module.dumps(
                    {
                        "success_count": 0,
                        "failed_count": 0,
                        "total_chunks": 0,
                        "already_ingested": already_ingested,
                        "failed_files": [],
                    }
                )
            )
        elif not quiet:
            print("Nothing to ingest!")
        return

    # Ingest each book
    results_success: list[dict] = []
    results_failed: list[dict] = []

    for i, book in enumerate(to_ingest):
        if not quiet and not json_output:
            print(f"\n[{i + 1}/{len(to_ingest)}] {book.title} ({book.domain_id})")

        start_time = time.time()

        try:
            source_id, num_chunks, num_headings = await ingest_one_book(book, pool, quiet=quiet)
            elapsed = time.time() - start_time

            results_success.append(
                {
                    "title": book.title,
                    "domain": book.domain_id,
                    "chunks": num_chunks,
                    "elapsed_seconds": round(elapsed, 1),
                }
            )

            if quiet and not json_output:
                print(f"  {book.title}: {num_chunks} chunks ({elapsed:.0f}s)")
            elif not quiet and not json_output:
                print(f"  {num_chunks} chunks created ({elapsed:.0f}s)")

        except (MemoryError, OSError) as e:
            error_type = "memory_exhausted" if isinstance(e, MemoryError) else "file_io_error"
            results_failed.append(
                {
                    "title": book.title,
                    "error": f"{error_type}: {str(e)[:100]}",
                    "recoverable": False,
                }
            )
            if not json_output:
                print(f"  {book.title}: {error_type}")

        except (EmbeddingError, ConnectionError):
            results_failed.append(
                {
                    "title": book.title,
                    "error": "Embedding service failure (retries exhausted)",
                    "recoverable": True,
                }
            )
            if not json_output:
                print(f"  {book.title}: embedding service failure (recoverable)")

        except StorageError as e:
            results_failed.append(
                {
                    "title": book.title,
                    "error": f"Database error: {str(e)[:100]}",
                    "recoverable": True,
                }
            )
            if not json_output:
                print(f"  {book.title}: database error (recoverable)")

        except Exception as e:
            results_failed.append(
                {
                    "title": book.title,
                    "error": str(e)[:100],
                    "recoverable": False,
                }
            )
            if not json_output:
                print(f"  {book.title}: {str(e)[:60]}")

    total_chunks = sum(r["chunks"] for r in results_success)

    if json_output:
        summary = {
            "success_count": len(results_success),
            "failed_count": len(results_failed),
            "total_chunks": total_chunks,
            "already_ingested": already_ingested,
            "failed_files": [
                {
                    "title": r["title"],
                    "error": r["error"],
                    "recoverable": r.get("recoverable", False),
                }
                for r in results_failed
            ],
        }
        print(json_module.dumps(summary, indent=2))
    elif quiet:
        print(f"\nIngested: {len(results_success)} books | {total_chunks} chunks")
        if results_failed:
            recoverable = sum(1 for r in results_failed if r.get("recoverable", False))
            print(f"Failed: {len(results_failed)} ({recoverable} recoverable)")
    else:
        print("\n" + "=" * 70)
        print("MANNING INGESTION SUMMARY")
        print("=" * 70)
        print(f"Success: {len(results_success)}")
        print(f"Failed: {len(results_failed)}")
        print(f"Total new chunks: {total_chunks}")

        if results_failed:
            print("\nFailed files:")
            for r in results_failed:
                status = "(recoverable)" if r.get("recoverable", False) else "(not recoverable)"
                print(f"  - {r['title']}: {r['error'][:60]} {status}")

        print("\nNext steps:")
        print(
            "  python scripts/extract_concepts.py --backend anthropic --model haiku --concurrency 4"
        )
        print("  python scripts/sync_kuzu.py")
