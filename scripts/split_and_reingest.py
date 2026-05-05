#!/usr/bin/env python3
"""Split oversized PDFs into N parts and re-register them for MinerU.

Addresses issue #9 corpus-wide equation remediation: the first Tier 1
batch showed that multi-thousand-page textbooks (Thomas Calculus 12th,
Stewart Calculus 9e, Kreyszig Advanced Engineering Math) OOM on
whole-book MinerU extraction against the RTX 2070's 7.6 GB VRAM.
Vol23 Phase F' used a 4-part manual split pattern for Korte and
Kochenderfer-DM. This script automates that pattern.

Flow per source:

    1. Look up source in the DB, verify the PDF exists on disk.
    2. Count total pages via ``pdfinfo``.
    3. Compute balanced page ranges (round-robin remainder to keep
       parts within ±1 page of each other).
    4. Use ``pdfseparate`` + ``pdfunite`` to write ``<stem>_partN.pdf``
       alongside the original.
    5. Insert a new source row per part (fresh UUID, new file_hash,
       metadata.parent_source_id + split_part + page_range). New
       sources have **no chunks** — MinerU extraction is the next
       pass; it creates them.
    6. ``DELETE FROM sources WHERE id = <original>`` — cascades to
       chunks.
    7. Emit a JSONL record of the new UUIDs to ``data/split_log.jsonl``
       so operators have a reconstructable audit trail.

CPU-only — does NOT call MinerU. Follow up with::

    python scripts/reextract_with_mineru.py \\
        --work-list <emitted-work-list.json> \\
        --checkpoint data/checkpoints/tier1_splits.jsonl \\
        --no-embed --auto-stop-services

Usage::

    # Split one source into 4 parts
    python scripts/split_and_reingest.py --source-id <uuid> --parts 4

    # Dry-run (no DB changes, no file writes beyond a summary)
    python scripts/split_and_reingest.py --source-id <uuid> --parts 4 --dry-run

    # Process every source from a DLQ JSONL (one per line)
    python scripts/split_and_reingest.py --from-dlq data/dlq/mineru_failed.jsonl --parts 4
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

# Make packages importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import configure_logging, get_logger  # noqa: E402
from research_kb_storage import (  # noqa: E402
    close_connection_pool,
    get_connection_pool,
)

configure_logging(level="INFO")
logger = get_logger(__name__)


SPLIT_LOG_PATH = Path(__file__).parent.parent / "data" / "split_log.jsonl"


# ── PDF splitting via poppler-utils ─────────────────────────────────────────


def count_pdf_pages(pdf_path: Path) -> int:
    """Return the page count of a PDF via ``pdfinfo``.

    Raises RuntimeError if pdfinfo is missing or the PDF is malformed.
    """
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "pdfinfo not found — install poppler-utils (apt install poppler-utils)"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"pdfinfo failed for {pdf_path}: {e.stderr.strip()}") from e

    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"pdfinfo produced no Pages line for {pdf_path}")


def compute_page_ranges(total_pages: int, parts: int) -> list[tuple[int, int]]:
    """Divide ``total_pages`` into ``parts`` contiguous 1-indexed ranges.

    Distributes the remainder so parts differ by at most 1 page. For
    total=10, parts=3 returns ``[(1,4), (5,7), (8,10)]``.

    Raises ValueError for invalid inputs (non-positive pages, parts<=0,
    or more parts than pages).
    """
    if total_pages <= 0:
        raise ValueError(f"total_pages must be positive, got {total_pages}")
    if parts <= 0:
        raise ValueError(f"parts must be positive, got {parts}")
    if parts > total_pages:
        raise ValueError(
            f"cannot split {total_pages} pages into {parts} parts " f"(would produce empty parts)"
        )

    base, remainder = divmod(total_pages, parts)
    ranges: list[tuple[int, int]] = []
    cursor = 1
    for i in range(parts):
        size = base + (1 if i < remainder else 0)
        end = cursor + size - 1
        ranges.append((cursor, end))
        cursor = end + 1
    return ranges


def split_pdf_range(in_pdf: Path, start: int, end: int, out_pdf: Path) -> None:
    """Write pages ``start..end`` (inclusive, 1-indexed) of ``in_pdf`` to ``out_pdf``.

    Uses ``pdfseparate`` to extract to a tempdir, then ``pdfunite`` to
    concatenate. Robust across poppler versions.
    """
    if start < 1 or end < start:
        raise ValueError(f"invalid range start={start} end={end}")
    with tempfile.TemporaryDirectory() as tmp:
        pattern = Path(tmp) / "page_%05d.pdf"
        try:
            subprocess.run(
                [
                    "pdfseparate",
                    "-f",
                    str(start),
                    "-l",
                    str(end),
                    str(in_pdf),
                    str(pattern),
                ],
                capture_output=True,
                text=True,
                timeout=1800,
                check=True,
            )
        except FileNotFoundError as e:
            raise RuntimeError("pdfseparate not found — install poppler-utils") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"pdfseparate failed on {in_pdf} pages {start}-{end}: " f"{e.stderr.strip()}"
            ) from e

        pages = sorted(Path(tmp).glob("page_*.pdf"))
        if not pages:
            raise RuntimeError(f"pdfseparate produced no output for {in_pdf} pages {start}-{end}")

        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["pdfunite", *map(str, pages), str(out_pdf)],
                capture_output=True,
                text=True,
                timeout=1800,
                check=True,
            )
        except FileNotFoundError as e:
            raise RuntimeError("pdfunite not found — install poppler-utils") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"pdfunite failed writing {out_pdf}: {e.stderr.strip()}") from e


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's contents, as a lowercase hex string."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ── DB operations ───────────────────────────────────────────────────────────


async def _fetch_source(pool, source_id: UUID) -> dict:
    """Fetch source metadata required for splitting."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, source_type, title, file_path, file_hash,
                   authors, year, domain_id, metadata
            FROM sources
            WHERE id = $1
            """,
            source_id,
        )
    if row is None:
        raise LookupError(f"source not found: {source_id}")
    return dict(row)


async def _insert_part_source(
    conn,
    *,
    part_id: UUID,
    parent: dict,
    part_path: Path,
    file_hash: str,
    part_number: int,
    total_parts: int,
    page_start: int,
    page_end: int,
) -> None:
    """Insert one part-source row; chunks are added by a later MinerU pass."""
    # Register jsonb codec so we can pass the metadata dict directly.
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )
    part_metadata = {
        "parent_source_id": str(parent["id"]),
        "parent_title": parent["title"],
        "split_part": part_number,
        "split_total_parts": total_parts,
        "page_start": page_start,
        "page_end": page_end,
        "split_method": "pdfseparate+pdfunite",
        "split_at": datetime.now(timezone.utc).isoformat(),
    }
    await conn.execute(
        """
        INSERT INTO sources (
            id, source_type, title, file_path, file_hash,
            authors, year, domain_id, metadata
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
        part_id,
        parent["source_type"],
        f"{parent['title']} (Part {part_number}/{total_parts})",
        str(part_path),
        file_hash,
        parent["authors"],
        parent["year"],
        parent["domain_id"],
        part_metadata,
    )


async def _delete_source(conn, source_id: UUID) -> None:
    """Delete the original source (chunks cascade via FK)."""
    await conn.execute("DELETE FROM sources WHERE id = $1", source_id)


def _write_split_log(record: dict) -> None:
    """Append one JSONL line to ``data/split_log.jsonl`` for audit."""
    SPLIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SPLIT_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ── Main per-source flow ────────────────────────────────────────────────────


async def split_and_reingest_source(
    pool,
    source_id: UUID,
    parts: int,
    dry_run: bool,
) -> dict:
    """Split one source, insert part rows, delete original.

    Returns a result dict suitable for logging.
    """
    t0 = datetime.now(timezone.utc)
    source = await _fetch_source(pool, source_id)
    pdf_path = Path(source["file_path"])
    if not pdf_path.exists():
        return {
            "source_id": str(source_id),
            "title": source["title"],
            "status": "skipped",
            "reason": f"PDF not found on disk: {pdf_path}",
        }

    total_pages = count_pdf_pages(pdf_path)
    if total_pages < parts:
        return {
            "source_id": str(source_id),
            "title": source["title"],
            "status": "skipped",
            "reason": f"only {total_pages} pages — not worth splitting into {parts}",
        }

    ranges = compute_page_ranges(total_pages, parts)

    planned_parts = []
    for i, (start, end) in enumerate(ranges, 1):
        out_path = pdf_path.parent / f"{pdf_path.stem}_part{i}of{parts}.pdf"
        planned_parts.append({"part": i, "start": start, "end": end, "path": out_path})

    if dry_run:
        return {
            "source_id": str(source_id),
            "title": source["title"],
            "status": "dry_run",
            "total_pages": total_pages,
            "parts": [
                {"part": p["part"], "pages": f"{p['start']}-{p['end']}", "path": str(p["path"])}
                for p in planned_parts
            ],
        }

    # Write the part PDFs (outside the DB transaction so we can see partial
    # progress on crash).
    for p in planned_parts:
        if p["path"].exists():
            raise FileExistsError(
                f"refusing to overwrite existing part file: {p['path']} "
                "(delete it manually if you want to re-split)"
            )
    for p in planned_parts:
        logger.info(
            "split_writing_part",
            part=p["part"],
            pages=f"{p['start']}-{p['end']}",
            path=str(p["path"]),
        )
        split_pdf_range(pdf_path, p["start"], p["end"], p["path"])
        p["hash"] = sha256_file(p["path"])
        p["id"] = uuid4()

    async with pool.acquire() as conn:
        async with conn.transaction():
            for p in planned_parts:
                await _insert_part_source(
                    conn,
                    part_id=p["id"],
                    parent=source,
                    part_path=p["path"],
                    file_hash=p["hash"],
                    part_number=p["part"],
                    total_parts=parts,
                    page_start=p["start"],
                    page_end=p["end"],
                )
            await _delete_source(conn, source_id)

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    record = {
        "source_id": str(source_id),
        "title": source["title"],
        "total_pages": total_pages,
        "parts": [
            {
                "new_id": str(p["id"]),
                "part": p["part"],
                "pages": f"{p['start']}-{p['end']}",
                "path": str(p["path"]),
                "hash": p["hash"],
            }
            for p in planned_parts
        ],
        "status": "ok",
        "elapsed_s": round(elapsed, 1),
        "split_at": t0.isoformat(),
    }
    _write_split_log(record)
    return record


# ── CLI ─────────────────────────────────────────────────────────────────────


def _collect_source_ids(args: argparse.Namespace) -> list[str]:
    """Resolve CLI args into a list of source IDs to process."""
    if args.source_id:
        return [args.source_id]
    if args.from_dlq:
        ids: list[str] = []
        seen: set[str] = set()
        for line in Path(args.from_dlq).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            sid = entry.get("source_id")
            # Dedupe: a DLQ may have multiple entries per source from retries.
            if sid and sid not in seen:
                ids.append(sid)
                seen.add(sid)
        return ids
    raise SystemExit("must specify --source-id or --from-dlq")


async def run(args: argparse.Namespace) -> int:
    source_ids = _collect_source_ids(args)
    logger.info("split_and_reingest_start", count=len(source_ids), parts=args.parts)

    pool = await get_connection_pool()
    results: list[dict] = []
    try:
        for sid in source_ids:
            try:
                result = await split_and_reingest_source(pool, UUID(sid), args.parts, args.dry_run)
            except Exception as e:
                logger.error("split_failed", source_id=sid, error=str(e), exc_info=True)
                result = {
                    "source_id": sid,
                    "status": "failed",
                    "reason": str(e),
                }
            results.append(result)
            print(
                f"  {result['status']:8} {result.get('source_id','')[:8]}… "
                f"{result.get('title','')[:60]}"
            )
    finally:
        await close_connection_pool()

    ok = sum(1 for r in results if r["status"] == "ok")
    failed = sum(1 for r in results if r["status"] == "failed")
    skipped = sum(1 for r in results if r["status"] in {"skipped", "dry_run"})
    print(f"\nSUMMARY: {ok} ok, {failed} failed, {skipped} skipped/dry-run")

    if ok > 0 and not args.dry_run:
        # Emit a ready-to-use work list for the MinerU pass.
        new_ids: list[str] = []
        for r in results:
            if r["status"] == "ok":
                for p in r["parts"]:
                    new_ids.append(p["new_id"])
        worklist_path = Path("data/worklists/post_split_parts.json")
        worklist_path.parent.mkdir(parents=True, exist_ok=True)
        worklist_path.write_text(json.dumps(new_ids, indent=2))
        print(
            f"\nNext step: feed the new part UUIDs to MinerU:\n"
            f"  python scripts/reextract_with_mineru.py \\\n"
            f"    --work-list {worklist_path} \\\n"
            f"    --checkpoint data/checkpoints/post_split.jsonl \\\n"
            f"    --no-embed --auto-stop-services"
        )

    return 0 if failed == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split oversized PDFs into N parts and re-register them for MinerU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--source-id", type=str, help="Source UUID to split.")
    target.add_argument(
        "--from-dlq",
        type=str,
        help="Path to a DLQ JSONL (data/dlq/mineru_failed.jsonl); process every source.",
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=4,
        help="Number of parts to split into (default: 4, matching vol23 Phase F' pattern).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the split but don't write files or touch the DB.",
    )
    args = parser.parse_args()
    rc = asyncio.run(run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
