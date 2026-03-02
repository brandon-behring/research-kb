#!/usr/bin/env python3
"""Regenerate golden candidate chunk IDs after corpus rechunking.

After rechunking, all chunk UUIDs change. This script maps old golden
candidates to new chunk IDs by:
1. Loading the golden dataset JSON
2. For each entry, searching for top-matching chunks from the same source
3. Selecting the best-matching chunks as new golden candidates
4. Writing updated golden candidates with new UUIDs

Usage:
    python scripts/regenerate_golden_candidates.py                    # Regenerate
    python scripts/regenerate_golden_candidates.py --verify-only      # Check without writing
    python scripts/regenerate_golden_candidates.py --candidates 3     # Top-3 per query

Output:
    fixtures/eval/golden_candidates_YYYY-MM-DD.json
"""

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path
from uuid import UUID

import numpy as np

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import get_logger
from research_kb_pdf import EmbeddingClient
from research_kb_storage import DatabaseConfig, get_connection_pool

logger = get_logger(__name__)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "eval"


def find_latest_golden_file() -> Path:
    """Find the most recent golden_candidates JSON file.

    Returns:
        Path to the latest golden candidates file

    Raises:
        FileNotFoundError: If no golden candidates file exists
    """
    candidates = sorted(FIXTURES_DIR.glob("golden_candidates_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No golden_candidates_*.json files found in {FIXTURES_DIR}"
        )
    return candidates[-1]


async def find_best_chunks_for_query(
    pool,
    embed_client: EmbeddingClient,
    query: str,
    source_title_prefix: str,
    domain: str,
    top_k: int = 2,
) -> list[dict]:
    """Find the best-matching chunks for a query within a specific source.

    Uses vector similarity to find chunks from the expected source that
    best match the query. This is the core mapping logic.

    Args:
        pool: asyncpg connection pool
        embed_client: Embedding client for query embedding
        query: The search query text
        source_title_prefix: First 30 chars of expected source title
        domain: Knowledge domain for the query
        top_k: Number of candidate chunks to return

    Returns:
        List of dicts with chunk_id, similarity, content_preview
    """
    from pgvector.asyncpg import register_vector

    # Generate query embedding
    query_embedding = embed_client.embed_query(query)
    embedding_vec = np.array(query_embedding, dtype=np.float32)

    async with pool.acquire() as conn:
        await register_vector(conn)

        # Find chunks from matching source, ranked by vector similarity
        rows = await conn.fetch(
            """
            SELECT c.id, c.content,
                   1 - (c.embedding <=> $1) as similarity,
                   s.title as source_title
            FROM chunks c
            JOIN sources s ON c.source_id = s.id
            WHERE LOWER(s.title) LIKE $2
              AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> $1
            LIMIT $3
            """,
            embedding_vec,
            f"{source_title_prefix[:30].lower()}%",
            top_k,
        )

    return [
        {
            "chunk_id": str(row["id"]),
            "similarity": float(row["similarity"]),
            "content_preview": row["content"][:100],
            "source_title": row["source_title"],
        }
        for row in rows
    ]


async def regenerate(
    input_path: Path,
    top_k: int = 2,
    verify_only: bool = False,
    verbose: bool = False,
) -> dict:
    """Regenerate golden candidates from old dataset.

    Args:
        input_path: Path to existing golden_candidates JSON
        top_k: Number of candidate chunks per query
        verify_only: If True, only check reachability without writing
        verbose: Print detailed output

    Returns:
        Summary dict with counts and quality metrics
    """
    with open(input_path) as f:
        entries = json.load(f)

    config = DatabaseConfig()
    pool = await get_connection_pool(config)
    embed_client = EmbeddingClient()

    new_entries = []
    matched = 0
    failed = 0
    low_similarity = 0

    print(f"Regenerating {len(entries)} golden candidates (top_k={top_k})...")

    for i, entry in enumerate(entries):
        query = entry["query"]
        source_title = entry.get("source_title", "")
        domain = entry.get("domain", "")
        difficulty = entry.get("difficulty", "medium")

        try:
            candidates = await find_best_chunks_for_query(
                pool, embed_client, query, source_title, domain, top_k=top_k
            )

            if not candidates:
                failed += 1
                if verbose:
                    print(f"  [{i + 1}] MISS: {query} (no chunks from '{source_title[:40]}')")
                # Preserve entry with empty targets so we know it needs attention
                new_entries.append(
                    {
                        "query": query,
                        "target_chunk_ids": [],
                        "domain": domain,
                        "source_title": source_title,
                        "difficulty": difficulty,
                        "_status": "no_match",
                    }
                )
                continue

            # Check similarity quality
            best_sim = candidates[0]["similarity"]
            if best_sim < 0.5:
                low_similarity += 1
                if verbose:
                    print(
                        f"  [{i + 1}] LOW: {query} "
                        f"(best sim={best_sim:.3f}, source='{source_title[:40]}')"
                    )

            new_chunk_ids = [c["chunk_id"] for c in candidates]
            matched += 1

            new_entry = {
                "query": query,
                "target_chunk_ids": new_chunk_ids,
                "domain": domain,
                "source_title": source_title,
                "difficulty": difficulty,
            }
            new_entries.append(new_entry)

            if verbose:
                print(
                    f"  [{i + 1}] OK: {query} "
                    f"(sim={best_sim:.3f}, {len(new_chunk_ids)} candidates)"
                )

        except Exception as e:
            failed += 1
            logger.error(
                "regeneration_failed",
                query=query,
                error=str(e),
            )
            if verbose:
                print(f"  [{i + 1}] ERROR: {query}: {e}")
            new_entries.append(
                {
                    "query": query,
                    "target_chunk_ids": [],
                    "domain": domain,
                    "source_title": source_title,
                    "difficulty": difficulty,
                    "_status": f"error: {str(e)[:80]}",
                }
            )

    summary = {
        "total": len(entries),
        "matched": matched,
        "failed": failed,
        "low_similarity": low_similarity,
        "match_rate": matched / len(entries) if entries else 0,
    }

    if verify_only:
        print(f"\nVERIFICATION ONLY (no file written)")
    else:
        # Write new golden candidates
        output_path = FIXTURES_DIR / f"golden_candidates_{date.today().isoformat()}.json"

        # Remove internal _status fields before writing
        clean_entries = []
        for entry in new_entries:
            clean = {k: v for k, v in entry.items() if not k.startswith("_")}
            clean_entries.append(clean)

        with open(output_path, "w") as f:
            json.dump(clean_entries, f, indent=2)
            f.write("\n")

        summary["output_path"] = str(output_path)
        print(f"\nWritten: {output_path}")

    print(f"\nSummary:")
    print(f"  Matched: {matched}/{len(entries)} ({summary['match_rate']:.0%})")
    print(f"  Failed: {failed}")
    print(f"  Low similarity (<0.5): {low_similarity}")

    if failed > 0:
        print(f"\n  WARNING: {failed} queries could not find matching chunks.")
        print("  These entries have empty target_chunk_ids and need manual review.")

    if low_similarity > 0:
        print(f"\n  NOTE: {low_similarity} queries matched with low similarity.")
        print("  Consider reviewing these for correctness after rechunking.")

    return summary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Regenerate golden candidate chunk IDs after rechunking."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to existing golden_candidates JSON (default: latest)",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=2,
        help="Number of candidate chunks per query (default: 2)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check reachability without writing new file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed per-query output",
    )
    return parser.parse_args()


async def main():
    """Regenerate golden candidates."""
    args = parse_args()

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = find_latest_golden_file()

    print(f"Input: {input_path}")

    summary = await regenerate(
        input_path=input_path,
        top_k=args.candidates,
        verify_only=args.verify_only,
        verbose=args.verbose,
    )

    # Exit non-zero if too many failures
    if summary["match_rate"] < 0.9:
        print(
            f"\nERROR: Match rate {summary['match_rate']:.0%} below 90% threshold. "
            "Check source titles and rechunking status."
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
