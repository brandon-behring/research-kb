#!/usr/bin/env python3
"""Backfill ``metadata.arxiv_id`` for sources whose PDFs live at
``fixtures/papers/arxiv/<arxiv_id>.pdf`` but whose ``metadata`` JSONB lacks
the ``arxiv_id`` field — which prevents ``SourceStore.get_by_arxiv_id()``
from finding them.

Background: many sources were ingested before ``arxiv_id`` was populated
as a first-class queryable metadata field, so the row exists (matched by
``file_hash``) but downstream consumers using ``get_by_arxiv_id`` miss
them. This bit the rl-and-control citation-graph demo: 83 papers were
"already ingested" per ``ingest_rl_subset.py`` (file-hash match), but only
3 were resolvable via ``get_by_arxiv_id``.

This script is idempotent:
    1. Walk ``fixtures/papers/arxiv/`` for ``*.pdf`` files
    2. Extract arxiv_id from each filename (``<arxiv_id>.pdf``)
    3. Compute file_hash for each PDF; look up the Source
    4. If the source exists and ``metadata.arxiv_id`` is missing or
       different, update via ``SourceStore.update_metadata`` (JSONB merge)
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

# Mirror ingest_*_subset.py path-injection pattern
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from ingest_missing_papers import compute_file_hash  # noqa: E402
from research_kb_common import get_logger  # noqa: E402
from research_kb_storage import (  # noqa: E402
    DatabaseConfig,
    SourceStore,
    close_connection_pool,
    get_connection_pool,
)

logger = get_logger(__name__)

ARXIV_FILENAME_RE = re.compile(r"^(\d{4}\.\d{4,5})\.pdf$")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill metadata.arxiv_id for arxiv-PDF sources lacking it."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N files (debugging).",
    )
    args = parser.parse_args()

    papers_dir = Path(__file__).parent.parent / "fixtures" / "papers" / "arxiv"
    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if args.limit is not None:
        pdf_files = pdf_files[: args.limit]
    print(f"Scanning {len(pdf_files)} PDFs in {papers_dir}")

    config = DatabaseConfig()
    await get_connection_pool(config)

    counts = {
        "scanned": 0,
        "no_arxiv_filename": 0,
        "no_source_for_hash": 0,
        "already_has_arxiv_id": 0,
        "would_update": 0,
        "updated": 0,
        "errors": 0,
    }

    for pdf in pdf_files:
        counts["scanned"] += 1
        m = ARXIV_FILENAME_RE.match(pdf.name)
        if not m:
            counts["no_arxiv_filename"] += 1
            continue
        arxiv_id = m.group(1)

        try:
            file_hash = compute_file_hash(str(pdf))
            source = await SourceStore.get_by_file_hash(file_hash)
        except Exception as e:
            counts["errors"] += 1
            print(f"  ✗ {arxiv_id}: hash/lookup error: {e}")
            continue

        if source is None:
            counts["no_source_for_hash"] += 1
            print(f"  ✗ {arxiv_id}: no source row for file_hash")
            continue

        existing_arxiv = (source.metadata or {}).get("arxiv_id")
        if existing_arxiv == arxiv_id:
            counts["already_has_arxiv_id"] += 1
            continue

        if args.dry_run:
            counts["would_update"] += 1
            print(
                f"  → {arxiv_id}: source_id={str(source.id)[:8]} "
                f"would set metadata.arxiv_id (was: {existing_arxiv!r})"
            )
        else:
            try:
                await SourceStore.update_metadata(source.id, {"arxiv_id": arxiv_id})
                counts["updated"] += 1
                print(f"  ✓ {arxiv_id}: source_id={str(source.id)[:8]} updated")
            except Exception as e:
                counts["errors"] += 1
                print(f"  ✗ {arxiv_id}: update error: {e}")

    print("\n" + "=" * 60)
    print("BACKFILL SUMMARY")
    print("=" * 60)
    for k, v in counts.items():
        print(f"  {k}: {v}")

    await close_connection_pool()


if __name__ == "__main__":
    asyncio.run(main())
