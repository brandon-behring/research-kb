"""Deduplicate chunks: remove same-source, same-content_hash duplicates.

For each (source_id, content_hash) group with count > 1:
1. Keep the chunk with lowest page_start (earliest created_at as tiebreaker)
2. Reassign chunk_concepts from victims to keeper (ON CONFLICT DO NOTHING)
3. Fix concept_relationships.evidence_chunk_ids array references
4. Delete victim chunks (CASCADE handles remaining chunk_concepts)

Usage:
    python scripts/dedup_chunks.py              # Dry run
    python scripts/dedup_chunks.py --apply      # Execute
    python scripts/dedup_chunks.py --verbose     # Show every group
"""

import argparse
import asyncio
import sys

import asyncpg


DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "research_kb"
DB_USER = "postgres"
DB_PASS = "postgres"


async def find_duplicate_groups(conn: asyncpg.Connection) -> list[dict]:
    """Find all (source_id, content_hash) groups with count > 1.

    Returns list of dicts with keys: source_id, content_hash, chunk_ids, keep_id, victim_ids.
    """
    rows = await conn.fetch("""
        SELECT source_id, content_hash,
               array_agg(id ORDER BY page_start NULLS LAST, created_at ASC) as chunk_ids
        FROM chunks
        GROUP BY source_id, content_hash
        HAVING COUNT(*) > 1
        ORDER BY COUNT(*) DESC
    """)

    groups = []
    for row in rows:
        ids = list(row["chunk_ids"])
        groups.append({
            "source_id": row["source_id"],
            "content_hash": row["content_hash"],
            "chunk_ids": ids,
            "keep_id": ids[0],
            "victim_ids": ids[1:],
        })
    return groups


async def dedup_chunks(apply: bool = False, verbose: bool = False) -> dict:
    """Deduplicate chunks across the entire database.

    Args:
        apply: If True, execute deletions. If False, dry run.
        verbose: If True, print every group.

    Returns:
        Summary dict with counts.
    """
    conn = await asyncpg.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASS,
    )

    try:
        groups = await find_duplicate_groups(conn)
        total_groups = len(groups)
        total_victims = sum(len(g["victim_ids"]) for g in groups)

        mode = "APPLY" if apply else "DRY RUN"
        print(f"\nChunk Deduplication ({mode})")
        print("=" * 60)
        print(f"Duplicate groups: {total_groups}")
        print(f"Chunks to remove: {total_victims}")

        if total_groups == 0:
            print("No duplicates found.")
            return {"groups": 0, "removed": 0, "concepts_reassigned": 0, "evidence_fixed": 0}

        concepts_reassigned = 0
        evidence_fixed = 0
        chunks_removed = 0

        if verbose:
            # Show top groups
            for g in groups[:20]:
                print(f"\n  source={g['source_id']}")
                print(f"    hash={g['content_hash'][:30]}...")
                print(f"    keep={g['keep_id']}, victims={len(g['victim_ids'])}")

        if apply:
            # Process in transaction
            async with conn.transaction():
                for i, group in enumerate(groups):
                    keeper = group["keep_id"]
                    victims = group["victim_ids"]

                    for victim in victims:
                        # 1. Reassign chunk_concepts from victim to keeper
                        result = await conn.execute("""
                            INSERT INTO chunk_concepts (chunk_id, concept_id, mention_type, relevance_score, created_at)
                            SELECT $1, concept_id, mention_type, relevance_score, created_at
                            FROM chunk_concepts WHERE chunk_id = $2
                            ON CONFLICT (chunk_id, concept_id, mention_type) DO NOTHING
                        """, keeper, victim)
                        moved = int(result.split()[-1]) if result.startswith("INSERT") else 0
                        concepts_reassigned += moved

                        # 2. Fix concept_relationships.evidence_chunk_ids
                        result = await conn.execute("""
                            UPDATE concept_relationships
                            SET evidence_chunk_ids = array_replace(evidence_chunk_ids, $1, $2)
                            WHERE $1 = ANY(evidence_chunk_ids)
                        """, victim, keeper)
                        fixed = int(result.split()[-1])
                        evidence_fixed += fixed

                        # 3. Delete victim chunk (CASCADE handles chunk_concepts)
                        await conn.execute("DELETE FROM chunks WHERE id = $1", victim)
                        chunks_removed += 1

                    if (i + 1) % 1000 == 0:
                        print(f"  Processed {i + 1}/{total_groups} groups...")

            print(f"\nResults:")
            print(f"  Chunks removed: {chunks_removed}")
            print(f"  Concept links reassigned: {concepts_reassigned}")
            print(f"  Evidence arrays fixed: {evidence_fixed}")
        else:
            print(f"\nDry run complete. Use --apply to execute.")

        return {
            "groups": total_groups,
            "removed": chunks_removed if apply else total_victims,
            "concepts_reassigned": concepts_reassigned,
            "evidence_fixed": evidence_fixed,
        }

    finally:
        await conn.close()


async def verify(conn: asyncpg.Connection) -> bool:
    """Verify no duplicates remain."""
    count = await conn.fetchval("""
        SELECT COUNT(*) FROM (
            SELECT source_id, content_hash FROM chunks
            GROUP BY source_id, content_hash HAVING COUNT(*) > 1
        ) d
    """)
    return count == 0


def main():
    parser = argparse.ArgumentParser(description="Deduplicate chunks (same source + content_hash)")
    parser.add_argument("--apply", action="store_true", help="Execute deletions (default: dry run)")
    parser.add_argument("--verbose", action="store_true", help="Show every duplicate group")
    args = parser.parse_args()

    result = asyncio.run(dedup_chunks(apply=args.apply, verbose=args.verbose))

    if args.apply:
        # Verify
        async def _verify():
            conn = await asyncpg.connect(
                host=DB_HOST, port=DB_PORT, database=DB_NAME,
                user=DB_USER, password=DB_PASS,
            )
            try:
                clean = await verify(conn)
                if clean:
                    print("\nVerification: No duplicates remain.")
                else:
                    print("\nWARNING: Duplicates still exist!")
                    sys.exit(1)
            finally:
                await conn.close()

        asyncio.run(_verify())


if __name__ == "__main__":
    main()
