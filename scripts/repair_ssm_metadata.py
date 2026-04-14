#!/usr/bin/env python3
"""Repair the metadata column for the SSM subset.

The fix_ssm_subset.py first pass wrote metadata as a list instead of a dict,
which breaks Pydantic Source.metadata validation on read. Also, 9 title fixes
need to be re-applied (the first pass succeeded but the second pass threw).

This script:
1. Resets metadata to a clean dict built from the sidecar (no JSONB merge)
2. Updates title, authors, year from the sidecar
3. Works by file_hash lookup (no Pydantic round-trip)

Idempotent and safe to re-run.
"""

import asyncio
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_storage import (
    DatabaseConfig,
    close_connection_pool,
    get_connection_pool,
)  # noqa: E402

# All 13 SSM-subset arxiv IDs (includes the 2 reingested ones; for those, the
# reingest already wrote good metadata so they're essentially no-ops here).
ALL_IDS = [
    "2212.14052",
    "2307.08621",
    "2405.04517",
    "2501.12352",
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
    with open(papers_dir / f"{arxiv_id}.json") as f:
        return json.load(f)


def normalize_authors(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [a.strip() for a in raw.split(",") if a.strip()]
    return list(raw)


async def main() -> None:
    papers_dir = Path(__file__).parent.parent / "fixtures" / "papers" / "arxiv"
    pool = await get_connection_pool(DatabaseConfig())

    async with pool.acquire() as conn:
        # Install JSONB codec so we can pass python dicts directly
        await conn.set_type_codec(
            "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )

        ok = fail = skipped = 0
        for aid in ALL_IDS:
            pdf_path = papers_dir / f"{aid}.pdf"
            if not pdf_path.exists():
                print(f"  ✗ {aid}: PDF missing")
                fail += 1
                continue

            file_hash = sha256_of(pdf_path)
            row = await conn.fetchrow(
                "SELECT id, title, metadata FROM sources WHERE file_hash = $1", file_hash
            )
            if row is None:
                print(f"  ✗ {aid}: source not in DB")
                fail += 1
                continue

            sidecar = load_sidecar(aid, papers_dir)
            new_title = sidecar.get("title")
            new_authors = normalize_authors(sidecar.get("authors"))
            new_year = sidecar.get("year")
            if not new_title:
                print(f"  ✗ {aid}: sidecar has no title")
                fail += 1
                continue

            # Parse existing metadata (may be broken list) and merge with fresh dict
            existing_raw = row["metadata"]
            try:
                existing = (
                    json.loads(existing_raw) if isinstance(existing_raw, str) else existing_raw
                )
            except (json.JSONDecodeError, TypeError):
                existing = {}
            if not isinstance(existing, dict):
                existing = {}  # discard broken list metadata

            # Only preserve extraction-related fields from existing metadata
            preserved = {
                k: existing.get(k)
                for k in [
                    "extraction_method",
                    "total_pages",
                    "total_chars",
                    "total_headings",
                    "total_chunks",
                    "has_equations",
                ]
                if existing.get(k) is not None
            }

            new_metadata = {
                **preserved,
                "arxiv_id": aid,
                "venue": sidecar.get("venue"),
                "abstract": sidecar.get("abstract"),
                "s2_paper_id": sidecar.get("s2_paper_id"),
                "citation_count": sidecar.get("citation_count"),
                "metadata_source": "sidecar_repair_2026_04_10",
                "ingest_batch": "ssm_subset",
            }
            new_metadata = {k: v for k, v in new_metadata.items() if v is not None}

            old_title = row["title"]
            await conn.execute(
                """
                UPDATE sources
                SET title = $1,
                    authors = $2,
                    year = $3,
                    metadata = $4,
                    updated_at = now()
                WHERE id = $5
                """,
                new_title,
                new_authors,
                new_year,
                new_metadata,
                row["id"],
            )
            marker = "↻" if old_title == new_title else "✓"
            print(f"  {marker} {aid}: '{old_title[:40]}' → '{new_title[:50]}'")
            ok += 1

        print(f"\nRepaired: {ok}, failed: {fail}, skipped: {skipped}")

    await close_connection_pool()


if __name__ == "__main__":
    asyncio.run(main())
