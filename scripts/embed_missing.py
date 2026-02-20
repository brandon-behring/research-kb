#!/usr/bin/env python3
"""Generate embeddings for chunks that are missing them.

This is used after loading demo data from fixtures, where chunk text is
loaded but embeddings are not (too large for JSON fixtures).

Requires the embedding server to be running:
    python -m research_kb_pdf.embed_server &

Usage:
    python scripts/embed_missing.py              # Embed all missing chunks
    python scripts/embed_missing.py --batch 50   # Custom batch size
    python scripts/embed_missing.py --limit 100  # Only embed first 100
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "storage" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "common" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "contracts" / "src"))


async def embed_missing(batch_size: int = 50, limit: int | None = None) -> int:
    """Generate embeddings for chunks without them."""
    import asyncpg
    from research_kb_pdf.embed_client import EmbeddingClient

    client = EmbeddingClient()
    try:
        client.ping()
    except Exception as e:
        print(f"Embedding server not available: {e}")
        print("Start with: python -m research_kb_pdf.embed_server &")
        return 0

    pool = await asyncpg.create_pool(
        "postgresql://postgres:postgres@localhost:5432/research_kb",
        min_size=1,
        max_size=3,
    )

    total_embedded = 0
    start = time.time()

    try:
        async with pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")
            if count == 0:
                print("All chunks already have embeddings.")
                return 0

            target = min(count, limit) if limit else count
            print(f"Found {count} chunks without embeddings. Processing {target}...")

        while True:
            async with pool.acquire() as conn:
                query = """
                    SELECT id, content FROM chunks
                    WHERE embedding IS NULL
                    ORDER BY created_at
                    LIMIT $1
                """
                rows = await conn.fetch(query, batch_size)

                if not rows:
                    break

                texts = [r["content"] for r in rows]
                ids = [r["id"] for r in rows]

                # Generate embeddings
                embeddings = client.embed_batch(texts)

                # Store embeddings
                for chunk_id, embedding in zip(ids, embeddings):
                    await conn.execute(
                        "UPDATE chunks SET embedding = $1 WHERE id = $2",
                        str(embedding),
                        chunk_id,
                    )

                total_embedded += len(rows)
                elapsed = time.time() - start
                rate = total_embedded / elapsed if elapsed > 0 else 0
                print(f"  Embedded {total_embedded}/{target} " f"({rate:.1f} chunks/sec)")

                if limit and total_embedded >= limit:
                    break

    finally:
        await pool.close()

    elapsed = time.time() - start
    print(f"\nDone: {total_embedded} chunks embedded in {elapsed:.1f}s")
    return total_embedded


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for missing chunks")
    parser.add_argument(
        "--batch",
        type=int,
        default=50,
        help="Batch size for embedding (default: 50)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of chunks to embed",
    )
    args = parser.parse_args()

    asyncio.run(embed_missing(batch_size=args.batch, limit=args.limit))


if __name__ == "__main__":
    main()
