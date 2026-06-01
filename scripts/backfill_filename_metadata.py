#!/usr/bin/env python3
"""Repair filename-ingested junk metadata for a subset of arXiv sources.

Sources auto-ingested *by filename* carry junk core metadata
(``title="<arxiv_id>"``, ``authors=[]``, a guessed ``year``) with
``metadata.metadata_source = "filename"`` (research-kb#20). This backfill
re-fetches real ``title``/``authors``/``year`` from the arXiv API and writes them
to the top-level columns via ``SourceStore.update_core_metadata``.

Scope: the RL+control demo subset by default (``ingest_rl_subset.TARGET_ARXIV_IDS``),
or an explicit ``--arxiv-ids <file>`` (newline-delimited; ``#`` comments allowed) —
e.g. the same list the graph-export pipeline derives from ``paper_index.md``. A
corpus-wide pass over *all* ``metadata_source='filename'`` rows is deferred to a
separate tracked issue.

Idempotent: rows whose ``metadata_source`` is no longer ``"filename"`` are skipped,
so re-running is a no-op. Companion to ``backfill_arxiv_id_metadata.py``, which
must run first so each source carries a queryable ``metadata.arxiv_id``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Mirror ingest_*_subset.py path-injection pattern.
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import fetch_arxiv_metadata, get_logger  # noqa: E402
from research_kb_storage import (  # noqa: E402
    DatabaseConfig,
    SourceStore,
    close_connection_pool,
    get_connection_pool,
)

logger = get_logger(__name__)


def _load_arxiv_ids(path: Path) -> list[str]:
    """Load arXiv IDs from a newline-delimited file (``#`` comments allowed)."""
    ids: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def _default_target_ids() -> list[str]:
    """The RL demo subset, imported lazily to avoid the ingest module's heavy deps."""
    from ingest_rl_subset import TARGET_ARXIV_IDS

    return list(TARGET_ARXIV_IDS)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair filename-junk title/authors/year for a subset of arXiv sources."
    )
    parser.add_argument(
        "--arxiv-ids",
        type=Path,
        default=None,
        help="Newline-delimited file of arXiv IDs to repair (default: the built-in "
        "RL demo subset, ingest_rl_subset.TARGET_ARXIV_IDS).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=3.0,
        help="Seconds to wait between arXiv API calls (default: 3.0, per arXiv guidance).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N ids (debugging).",
    )
    args = parser.parse_args()

    target_ids = _load_arxiv_ids(args.arxiv_ids) if args.arxiv_ids else _default_target_ids()
    if args.limit is not None:
        target_ids = target_ids[: args.limit]
    print(f"Repairing filename-junk metadata for {len(target_ids)} arXiv ids")

    config = DatabaseConfig()
    await get_connection_pool(config)

    counts = {
        "scanned": 0,
        "no_source": 0,
        "already_clean": 0,
        "fetch_failed": 0,
        "would_update": 0,
        "updated": 0,
        "errors": 0,
    }

    for arxiv_id in target_ids:
        counts["scanned"] += 1
        try:
            source = await SourceStore.get_by_arxiv_id(arxiv_id)
        except Exception as exc:
            counts["errors"] += 1
            print(f"  ✗ {arxiv_id}: lookup error: {exc}")
            continue

        if source is None:
            counts["no_source"] += 1
            print(f"  ⊘ {arxiv_id}: no source row (run backfill_arxiv_id_metadata.py first?)")
            continue

        if (source.metadata or {}).get("metadata_source") != "filename":
            counts["already_clean"] += 1
            continue

        fetched = fetch_arxiv_metadata(arxiv_id)
        if args.sleep > 0:
            time.sleep(args.sleep)  # be polite to the arXiv API
        if not fetched:
            counts["fetch_failed"] += 1
            print(f"  ✗ {arxiv_id}: arXiv fetch failed; left as-is")
            continue

        old_title = source.title
        new_title = fetched["title"]
        if args.dry_run:
            counts["would_update"] += 1
            print(f"  → {arxiv_id}: {old_title!r} -> {new_title!r}")
            continue

        try:
            await SourceStore.update_core_metadata(
                source.id,
                title=fetched["title"],
                authors=fetched["authors"],
                year=fetched["year"],
                metadata={"metadata_source": "arxiv_api"},
            )
            counts["updated"] += 1
            print(f"  ✓ {arxiv_id}: {old_title!r} -> {new_title!r}")
        except Exception as exc:
            counts["errors"] += 1
            print(f"  ✗ {arxiv_id}: update error: {exc}")

    print("\n" + "=" * 60)
    print("BACKFILL SUMMARY")
    print("=" * 60)
    for key, value in counts.items():
        print(f"  {key}: {value}")

    await close_connection_pool()


if __name__ == "__main__":
    asyncio.run(main())
