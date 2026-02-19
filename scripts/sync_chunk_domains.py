"""Sync chunks.domain_id from sources.metadata->>'domain'.

The chunks.domain_id column is used for domain-filtered search.
It must match the source's domain tag. This script updates it.

Usage:
    python scripts/sync_chunk_domains.py           # dry run
    python scripts/sync_chunk_domains.py --apply    # apply
"""

import asyncio
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
for pkg in ("storage", "contracts", "common"):
    sys.path.insert(0, str(_root / "packages" / pkg / "src"))

from research_kb_storage import DatabaseConfig, get_connection_pool


async def main(apply: bool = False):
    pool = await get_connection_pool(DatabaseConfig())

    # First, check for sources with NULL domain (would violate NOT NULL)
    null_count = await pool.fetchval(
        "SELECT COUNT(*) FROM sources WHERE metadata->>'domain' IS NULL"
    )
    print(f"Sources with NULL domain: {null_count}")

    if null_count > 0:
        # Set them to 'other' to avoid NOT NULL violation
        if apply:
            await pool.execute(
                "UPDATE sources SET metadata = jsonb_set("
                "  COALESCE(metadata, '{}'::jsonb), '{domain}', '\"other\"'::jsonb"
                ") WHERE metadata->>'domain' IS NULL"
            )
            print(f"  Set {null_count} NULL-domain sources to 'other'")

    # Show current state
    print("\nCurrent chunks.domain_id distribution:")
    rows = await pool.fetch(
        "SELECT domain_id, COUNT(*) as cnt FROM chunks GROUP BY 1 ORDER BY 2 DESC"
    )
    for r in rows:
        d = r["domain_id"] or "(null)"
        print(f"  {d:30s} {r['cnt']}")

    # Count mismatched
    mismatch = await pool.fetchval(
        "SELECT COUNT(*) FROM chunks c "
        "JOIN sources s ON c.source_id = s.id "
        "WHERE c.domain_id IS DISTINCT FROM COALESCE(s.metadata->>'domain', 'other')"
    )
    print(f"\nMismatched chunks: {mismatch}")

    if mismatch == 0:
        print("No sync needed — all chunks match their source domain.")
        await pool.close()
        return

    # Show what would change
    changes = await pool.fetch(
        "SELECT COALESCE(s.metadata->>'domain', 'other') as new_domain, COUNT(*) as cnt "
        "FROM chunks c "
        "JOIN sources s ON c.source_id = s.id "
        "WHERE c.domain_id IS DISTINCT FROM COALESCE(s.metadata->>'domain', 'other') "
        "GROUP BY 1 ORDER BY 2 DESC"
    )
    print("\nChanges by new domain:")
    for r in changes:
        print(f"  → {r['new_domain']:30s} {r['cnt']} chunks")

    if apply:
        result = await pool.execute(
            "UPDATE chunks c "
            "SET domain_id = COALESCE(s.metadata->>'domain', 'other') "
            "FROM sources s "
            "WHERE c.source_id = s.id "
            "  AND c.domain_id IS DISTINCT FROM COALESCE(s.metadata->>'domain', 'other')"
        )
        print(f"\nApplied: {result}")

        # Verify
        print("\nNew chunks.domain_id distribution:")
        rows = await pool.fetch(
            "SELECT domain_id, COUNT(*) as cnt FROM chunks GROUP BY 1 ORDER BY 2 DESC"
        )
        for r in rows:
            print(f"  {r['domain_id']:30s} {r['cnt']}")
    else:
        print(f"\nDRY RUN: Would update {mismatch} chunks. Re-run with --apply.")

    await pool.close()


if __name__ == "__main__":
    do_apply = "--apply" in sys.argv
    asyncio.run(main(apply=do_apply))
