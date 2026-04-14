#!/usr/bin/env python3
"""Fix the SSM subset after B2 discovered two issues:

1. Ten papers were already ingested under placeholder titles like 'arXiv:XXXX.NNNNN'
   (filename-parsed, no real metadata). Update them in place using the .json
   sidecars written during B2 prep.
2. xLSTM (2405.04517) was ingested but Docling extraction was truncated by CUDA
   OOM (3 chunks from a 55-page paper). Delete and reingest in CPU mode.
3. TTR (2501.12352) failed entirely from CUDA OOM. Ingest fresh in CPU mode.

Written 2026-04-10.
"""

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

# Force Docling to CPU before any torch imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from ingest_missing_papers import ingest_pdf  # noqa: E402
from research_kb_storage import (  # noqa: E402
    ChunkStore,
    DatabaseConfig,
    SourceStore,
    close_connection_pool,
    get_connection_pool,
)

# Sources to reingest (known to be degraded or missing).
REINGEST_IDS = [
    "2405.04517",  # xLSTM — OOM-truncated, delete + reingest
    "2501.12352",  # TTR — never ingested, fresh
]

# Sources with placeholder titles that just need metadata update (keep chunks).
TITLE_FIX_IDS = [
    "2212.14052",
    "2307.08621",
    "2406.00209",
    "2504.19561",
    "2511.17970",
    "2512.16723",
    "2312.04927",
    "2404.06654",
    "2110.13985",
    "2402.18668",
    "2303.06349",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            h.update(block)
    return h.hexdigest()


def load_sidecar(arxiv_id: str, papers_dir: Path) -> dict:
    sidecar_path = papers_dir / f"{arxiv_id}.json"
    with open(sidecar_path) as f:
        return json.load(f)


async def update_title_in_place(arxiv_id: str, papers_dir: Path) -> bool:
    """Update title/authors/year for a source that has placeholder metadata.

    Uses raw SQL since SourceStore only exposes update_metadata (JSONB merge).
    Preserves existing chunks and embeddings.
    """
    pool = await get_connection_pool()
    pdf_path = papers_dir / f"{arxiv_id}.pdf"
    if not pdf_path.exists():
        print(f"  ✗ {arxiv_id}: PDF not found")
        return False

    file_hash = sha256_of(pdf_path)
    source = await SourceStore.get_by_file_hash(file_hash)
    if source is None:
        print(f"  ✗ {arxiv_id}: source not found by hash")
        return False

    sidecar = load_sidecar(arxiv_id, papers_dir)
    new_title = sidecar.get("title")
    raw_authors = sidecar.get("authors", [])
    if isinstance(raw_authors, str):
        new_authors = [a.strip() for a in raw_authors.split(",") if a.strip()]
    else:
        new_authors = list(raw_authors) if raw_authors else []
    new_year = sidecar.get("year")

    if not new_title:
        print(f"  ✗ {arxiv_id}: sidecar has no title")
        return False

    old_title = source.title

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE sources
            SET title = $1,
                authors = $2,
                year = $3,
                metadata = metadata || $4::jsonb,
                updated_at = now()
            WHERE id = $5
            """,
            new_title,
            new_authors,
            new_year,
            json.dumps(
                {
                    "arxiv_id": arxiv_id,
                    "venue": sidecar.get("venue"),
                    "title_fixed_from_sidecar": True,
                    "metadata_source": "sidecar_retrofit",
                }
            ),
            source.id,
        )

    print(f"  ✓ {arxiv_id}: '{old_title}' → '{new_title[:60]}'")
    return True


async def delete_and_reingest(arxiv_id: str, papers_dir: Path) -> bool:
    """Delete existing source (if any) and reingest on CPU."""
    pdf_path = papers_dir / f"{arxiv_id}.pdf"
    if not pdf_path.exists():
        print(f"  ✗ {arxiv_id}: PDF not found")
        return False

    file_hash = sha256_of(pdf_path)
    existing = await SourceStore.get_by_file_hash(file_hash)
    if existing is not None:
        # Delete existing chunks first, then source
        pool = await get_connection_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM chunks WHERE source_id = $1", existing.id)
            await conn.execute("DELETE FROM sources WHERE id = $1", existing.id)
        print(f"  ⊘ {arxiv_id}: deleted existing (source_id={str(existing.id)[:8]})")

    sidecar = load_sidecar(arxiv_id, papers_dir)
    title = sidecar["title"]
    authors = sidecar.get("authors", [])
    year = sidecar.get("year")
    metadata = {
        "arxiv_id": arxiv_id,
        "venue": sidecar.get("venue"),
        "metadata_source": "sidecar_cpu_reingest",
        "auto_ingested": True,
        "ingest_batch": "ssm_subset_cpu_2026_04_10",
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}

    try:
        source_id, num_chunks, num_headings = await ingest_pdf(
            pdf_path=str(pdf_path),
            title=title,
            authors=authors,
            year=year,
            metadata=metadata,
            domain_id="deep_learning",
            no_embed=True,
        )
        print(
            f"  ✓ {arxiv_id}: source_id={source_id[:8]}, {num_chunks} chunks, {num_headings} headings"
        )
        return True
    except Exception as e:
        print(f"  ✗ {arxiv_id}: reingest failed: {e}")
        return False


async def main() -> None:
    papers_dir = Path(__file__).parent.parent / "fixtures" / "papers" / "arxiv"
    await get_connection_pool(DatabaseConfig())

    print("=" * 70)
    print("Phase 1: Update placeholder titles (10 sources)")
    print("=" * 70)
    title_ok = title_fail = 0
    for aid in TITLE_FIX_IDS:
        try:
            ok = await update_title_in_place(aid, papers_dir)
        except Exception as e:
            print(f"  ✗ {aid}: {e}")
            ok = False
        title_ok += int(ok)
        title_fail += int(not ok)

    print(f"\nTitle updates: {title_ok} success, {title_fail} failed")

    print("\n" + "=" * 70)
    print("Phase 2: Delete + reingest degraded papers (CPU mode)")
    print("=" * 70)
    re_ok = re_fail = 0
    for aid in REINGEST_IDS:
        print(f"\n→ {aid}")
        ok = await delete_and_reingest(aid, papers_dir)
        re_ok += int(ok)
        re_fail += int(not ok)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Title updates: {title_ok}/{len(TITLE_FIX_IDS)}")
    print(f"Reingestions:  {re_ok}/{len(REINGEST_IDS)}")

    await close_connection_pool()


if __name__ == "__main__":
    asyncio.run(main())
