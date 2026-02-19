"""Diagnose data quality issues — read-only diagnostic script.

Three sections:
  0a. Chunk duplicates (same source, same content_hash)
  0b. Source near-duplicates (pg_trgm similarity)
  0c. time_series golden chunk domain verification

Usage:
    python scripts/diagnose_data_quality.py
"""

import asyncio
import json
import sys
from pathlib import Path

import asyncpg

# Database config
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "research_kb"
DB_USER = "postgres"
DB_PASS = "postgres"

GOLDEN_DATASET = Path(__file__).parent.parent / "fixtures" / "eval" / "golden_dataset.json"


async def diagnose_chunk_duplicates(conn: asyncpg.Connection) -> dict:
    """0a. Find duplicate chunks (same source_id + content_hash)."""
    print("\n" + "=" * 70)
    print("0a. CHUNK DUPLICATES (same source, same content_hash)")
    print("=" * 70)

    # Count duplicate groups
    group_count = await conn.fetchval("""
        SELECT COUNT(*) FROM (
            SELECT source_id, content_hash FROM chunks
            GROUP BY source_id, content_hash HAVING COUNT(*) > 1
        ) d
    """)
    print(f"\nDuplicate groups: {group_count}")

    # Count deletable rows
    deletable = await conn.fetchval("""
        SELECT COALESCE(SUM(cnt - 1), 0) FROM (
            SELECT source_id, content_hash, COUNT(*) as cnt FROM chunks
            GROUP BY source_id, content_hash HAVING COUNT(*) > 1
        ) d
    """)
    print(f"Deletable rows (keeping 1 per group): {deletable}")

    # Top offenders
    top = await conn.fetch("""
        SELECT s.title, c.content_hash, COUNT(*) as n
        FROM chunks c JOIN sources s ON c.source_id = s.id
        GROUP BY s.title, c.content_hash HAVING COUNT(*) > 1
        ORDER BY n DESC LIMIT 15
    """)

    if top:
        print(f"\nTop {len(top)} offenders:")
        for row in top:
            print(f"  [{row['n']}x] {row['title'][:60]} | hash={row['content_hash'][:20]}...")
    else:
        print("\nNo chunk duplicates found.")

    return {"groups": group_count, "deletable": deletable, "top": len(top)}


async def diagnose_source_near_dupes(conn: asyncpg.Connection) -> dict:
    """0b. Find near-duplicate sources via pg_trgm similarity."""
    print("\n" + "=" * 70)
    print("0b. SOURCE NEAR-DUPLICATES (pg_trgm similarity > 0.7)")
    print("=" * 70)

    # Check if pg_trgm is available
    has_trgm = await conn.fetchval(
        "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')"
    )

    if not has_trgm:
        print("\n  pg_trgm extension not available. Trying CREATE EXTENSION...")
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            has_trgm = True
        except Exception as e:
            print(f"  Could not create pg_trgm: {e}")
            print("  Falling back to exact-match check only.")

    if has_trgm:
        pairs = await conn.fetch("""
            SELECT a.id as id_a, b.id as id_b,
                   a.title as title_a, b.title as title_b,
                   similarity(lower(a.title), lower(b.title)) as sim
            FROM sources a JOIN sources b ON a.id < b.id
            WHERE similarity(lower(a.title), lower(b.title)) > 0.7
            ORDER BY sim DESC LIMIT 20
        """)

        if pairs:
            print(f"\n{len(pairs)} near-duplicate pairs found:")
            for row in pairs:
                print(f"\n  sim={row['sim']:.3f}")
                print(f"    A: {row['title_a'][:80]}")
                print(f"    B: {row['title_b'][:80]}")
                print(f"    IDs: {row['id_a']} / {row['id_b']}")
        else:
            print("\nNo near-duplicate sources found (threshold > 0.7).")

        return {"pairs": len(pairs)}
    else:
        # Fallback: check for exact normalized title matches
        print("\nFallback: checking for exact title duplicates (case-insensitive)...")
        dupes = await conn.fetch("""
            SELECT lower(title) as ltitle, COUNT(*) as cnt, array_agg(id) as ids
            FROM sources
            GROUP BY lower(title) HAVING COUNT(*) > 1
            ORDER BY cnt DESC LIMIT 20
        """)
        if dupes:
            for row in dupes:
                print(f"  [{row['cnt']}x] {row['ltitle'][:80]}")
        else:
            print("  No exact duplicates.")
        return {"pairs": len(dupes)}


async def diagnose_time_series_golden(conn: asyncpg.Connection) -> dict:
    """0c. Verify domain_id of all time_series golden chunk targets."""
    print("\n" + "=" * 70)
    print("0c. TIME_SERIES GOLDEN CHUNK DOMAIN VERIFICATION")
    print("=" * 70)

    # Load golden dataset
    if not GOLDEN_DATASET.exists():
        print(f"\n  Golden dataset not found: {GOLDEN_DATASET}")
        return {"error": "file_not_found"}

    with open(GOLDEN_DATASET) as f:
        golden = json.load(f)

    ts_entries = [e for e in golden if e["domain"] == "time_series"]
    print(f"\ntime_series entries in golden dataset: {len(ts_entries)}")

    # Collect all target chunk IDs
    all_chunk_ids = []
    for entry in ts_entries:
        all_chunk_ids.extend(entry["target_chunk_ids"])

    print(f"Total target chunks: {len(all_chunk_ids)}")

    # Check domain_id for each chunk
    rows = await conn.fetch("""
        SELECT c.id, c.domain_id, s.title,
               s.metadata->>'domain' as src_domain
        FROM chunks c JOIN sources s ON c.source_id = s.id
        WHERE c.id = ANY($1::uuid[])
    """, all_chunk_ids)

    found_ids = {str(r["id"]) for r in rows}
    missing = [cid for cid in all_chunk_ids if cid not in found_ids]

    domain_counts = {}
    wrong_domain = []
    for row in rows:
        d = row["domain_id"] or "NULL"
        domain_counts[d] = domain_counts.get(d, 0) + 1
        if d != "time_series":
            wrong_domain.append(row)

    print(f"\nChunks found: {len(rows)} / {len(all_chunk_ids)}")
    print(f"Missing chunks: {len(missing)}")
    if missing:
        for cid in missing[:5]:
            print(f"  MISSING: {cid}")

    print(f"\nDomain distribution of target chunks:")
    for domain, cnt in sorted(domain_counts.items(), key=lambda x: -x[1]):
        marker = " <-- WRONG" if domain != "time_series" else ""
        print(f"  {domain}: {cnt}{marker}")

    if wrong_domain:
        print(f"\nChunks with wrong domain ({len(wrong_domain)}):")
        for row in wrong_domain[:10]:
            print(f"  chunk={row['id']}")
            print(f"    domain_id={row['domain_id']}, source_domain={row['src_domain']}")
            print(f"    source: {row['title'][:60]}")

    # Also check the source itself
    print("\ntime_series sources in DB:")
    ts_sources = await conn.fetch("""
        SELECT id, title, metadata->>'domain' as domain
        FROM sources
        WHERE title ILIKE '%time series%'
           OR metadata->>'domain' = 'time_series'
        ORDER BY title
    """)
    for s in ts_sources:
        print(f"  [{s['domain'] or 'NULL'}] {s['title'][:70]}")

    return {
        "total_entries": len(ts_entries),
        "total_chunks": len(all_chunk_ids),
        "found": len(rows),
        "missing": len(missing),
        "wrong_domain": len(wrong_domain),
        "domain_counts": domain_counts,
    }


async def main():
    """Run all diagnostics."""
    print("Research-KB Data Quality Diagnostic")
    print("=" * 70)
    print("READ-ONLY — no modifications to database")

    conn = await asyncpg.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASS
    )

    try:
        results = {}
        results["chunk_dupes"] = await diagnose_chunk_duplicates(conn)
        results["source_near_dupes"] = await diagnose_source_near_dupes(conn)
        results["time_series_golden"] = await diagnose_time_series_golden(conn)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  Chunk duplicate groups: {results['chunk_dupes']['groups']}")
        print(f"  Deletable duplicate chunks: {results['chunk_dupes']['deletable']}")
        print(f"  Source near-dupe pairs: {results['source_near_dupes'].get('pairs', '?')}")
        ts = results["time_series_golden"]
        print(f"  time_series golden chunks found: {ts.get('found', '?')}/{ts.get('total_chunks', '?')}")
        print(f"  time_series wrong domain: {ts.get('wrong_domain', '?')}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
