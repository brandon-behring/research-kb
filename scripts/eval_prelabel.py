#!/usr/bin/env python3
"""Pre-label pooled candidates with suggested relevance grades.

Computes query-chunk cosine similarity using stored embeddings from the database,
then assigns percentile-based grades with absolute floor safety net.

Grading algorithm (per query):
  - Top 10% by similarity -> grade 3 (IF similarity >= 0.45)
  - Next 20% -> grade 2 (IF similarity >= 0.35)
  - Next 30% -> grade 1
  - Bottom 40% -> grade 0
  - Chunks below absolute floor are capped at grade 1

Auto-extracts evidence_text (first 150 chars) for grade >= 2 candidates.

Usage:
    python scripts/eval_prelabel.py --input fixtures/eval/v2_pool_candidates.yaml
    python scripts/eval_prelabel.py --input pool.yaml --output pool_prelabeled.yaml
    python scripts/eval_prelabel.py --input pool.yaml --verbose
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import numpy as np
import yaml

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from pgvector.asyncpg import register_vector
from research_kb_pdf import EmbeddingClient
from research_kb_storage import DatabaseConfig, get_connection_pool

# Absolute floor thresholds — prevent false positives on corpus-gap queries
FLOOR_GRADE_3 = 0.45
FLOOR_GRADE_2 = 0.35


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Parameters
    ----------
    a, b : np.ndarray
        1-D vectors.

    Returns
    -------
    float
        Cosine similarity in [-1, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


async def fetch_chunk_embeddings(chunk_ids: list[str]) -> dict[str, Optional[list[float]]]:
    """Fetch embeddings from the chunks table.

    Parameters
    ----------
    chunk_ids : list[str]
        UUIDs as strings.

    Returns
    -------
    dict[str, list[float] | None]
        Mapping of chunk_id -> embedding (or None if missing).
    """
    if not chunk_ids:
        return {}

    pool = await get_connection_pool()
    async with pool.acquire() as conn:
        await register_vector(conn)
        rows = await conn.fetch(
            "SELECT id, embedding FROM chunks WHERE id = ANY($1::uuid[])",
            [UUID(cid) for cid in chunk_ids],
        )

    result: dict[str, Optional[list[float]]] = {}
    for row in rows:
        embedding = row["embedding"]
        if embedding is not None:
            result[str(row["id"])] = list(embedding)
        else:
            result[str(row["id"])] = None

    # Mark missing chunks
    for cid in chunk_ids:
        if cid not in result:
            result[cid] = None

    return result


def assign_grades(
    candidates: list[dict[str, Any]],
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Assign percentile-based grades with absolute floor.

    Parameters
    ----------
    candidates : list[dict]
        Candidates with similarity_score already computed.
    verbose : bool
        Print grade distribution.

    Returns
    -------
    list[dict]
        Same candidates with suggested_grade and evidence_text filled.
    """
    # Filter to candidates with valid similarity scores
    scored = [c for c in candidates if c.get("similarity_score") is not None]
    unscored = [c for c in candidates if c.get("similarity_score") is None]

    if not scored:
        for c in candidates:
            c["suggested_grade"] = 0
        return candidates

    sims = [c["similarity_score"] for c in scored]

    # Compute percentile thresholds
    p90 = float(np.percentile(sims, 90))  # Top 10%
    p70 = float(np.percentile(sims, 70))  # Next 20%
    p40 = float(np.percentile(sims, 40))  # Next 30%

    grade_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for c in scored:
        sim = c["similarity_score"]

        # Percentile-based assignment
        if sim >= p90:
            grade = 3
        elif sim >= p70:
            grade = 2
        elif sim >= p40:
            grade = 1
        else:
            grade = 0

        # Apply absolute floor safety net
        if grade == 3 and sim < FLOOR_GRADE_3:
            grade = min(grade, 2) if sim >= FLOOR_GRADE_2 else min(grade, 1)
        if grade == 2 and sim < FLOOR_GRADE_2:
            grade = 1

        c["suggested_grade"] = grade
        grade_counts[grade] += 1

        # Auto-extract evidence text for grade >= 2
        if grade >= 2 and c.get("text_snippet"):
            c["evidence_text"] = c["text_snippet"][:150]

    # Unscored chunks get grade 0
    for c in unscored:
        c["suggested_grade"] = 0
        grade_counts[0] += 1

    if verbose:
        total = len(candidates)
        print(f"    Grades: 3={grade_counts[3]} 2={grade_counts[2]} "
              f"1={grade_counts[1]} 0={grade_counts[0]} "
              f"(thresholds: p90={p90:.3f} p70={p70:.3f} p40={p40:.3f})")

    return candidates


async def prelabel_query(
    query_data: dict[str, Any],
    embed_client: EmbeddingClient,
    chunk_embeddings_cache: dict[str, Optional[list[float]]],
    verbose: bool = False,
) -> dict[str, Any]:
    """Compute similarities and assign suggested grades for one query.

    Parameters
    ----------
    query_data : dict
        Query with candidates list.
    embed_client : EmbeddingClient
        For computing query embedding.
    chunk_embeddings_cache : dict
        Pre-fetched chunk embeddings.
    verbose : bool
        Print details.

    Returns
    -------
    dict
        Updated query_data with similarity scores and suggested grades.
    """
    query_text = query_data["query_text"]
    candidates = query_data.get("candidates", [])

    if not candidates:
        return query_data

    # Compute query embedding
    query_embedding = embed_client.embed_query(query_text)
    query_vec = np.array(query_embedding)

    # Compute cosine similarity for each candidate
    missing_count = 0
    for c in candidates:
        chunk_id = c["chunk_id"]
        chunk_emb = chunk_embeddings_cache.get(chunk_id)

        if chunk_emb is not None:
            chunk_vec = np.array(chunk_emb)
            c["similarity_score"] = round(cosine_similarity(query_vec, chunk_vec), 4)
        else:
            c["similarity_score"] = None
            missing_count += 1

    if missing_count > 0 and verbose:
        print(f"    WARNING: {missing_count}/{len(candidates)} chunks missing embeddings")

    # Assign grades
    candidates = assign_grades(candidates, verbose=verbose)

    return query_data


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-label pooled candidates with suggested relevance grades"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Pooled candidates YAML (from eval_pool_candidates.py)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output YAML path (default: overwrite input file)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-query grade distribution",
    )
    args = parser.parse_args()

    output_path = args.output or args.input

    # Load pooled candidates
    with open(args.input) as f:
        data = yaml.safe_load(f)

    queries = data.get("queries", [])
    if not queries:
        print("ERROR: No queries found in input file", file=sys.stderr)
        sys.exit(1)

    # Collect all unique chunk IDs across all queries
    all_chunk_ids: list[str] = []
    for q in queries:
        for c in q.get("candidates", []):
            cid = c.get("chunk_id")
            if cid:
                all_chunk_ids.append(cid)

    unique_chunk_ids = list(set(all_chunk_ids))
    print(f"Pre-labeling {len(queries)} queries, {len(unique_chunk_ids)} unique chunks")

    # Initialize DB + embedding client
    config = DatabaseConfig()
    await get_connection_pool(config)
    embed_client = EmbeddingClient()

    # Batch-fetch all chunk embeddings (single DB round-trip)
    print(f"Fetching embeddings for {len(unique_chunk_ids)} chunks...")
    chunk_embeddings = await fetch_chunk_embeddings(unique_chunk_ids)

    # Count missing
    missing = sum(1 for v in chunk_embeddings.values() if v is None)
    if missing > 0:
        print(f"WARNING: {missing}/{len(unique_chunk_ids)} chunks have no embedding")

    # Pre-label each query
    print("Computing similarities and assigning grades...")
    for i, q in enumerate(queries):
        query_text = q["query_text"]
        candidate_count = len(q.get("candidates", []))
        print(f"  [{i + 1}/{len(queries)}] {query_text[:50]}... ({candidate_count} candidates)")

        await prelabel_query(
            query_data=q,
            embed_client=embed_client,
            chunk_embeddings_cache=chunk_embeddings,
            verbose=args.verbose,
        )

    # Update metadata
    data["metadata"]["prelabeled"] = True
    data["metadata"]["prelabel_floors"] = {"grade_3": FLOOR_GRADE_3, "grade_2": FLOOR_GRADE_2}

    # Compute summary stats
    total_by_grade = {0: 0, 1: 0, 2: 0, 3: 0}
    for q in queries:
        for c in q.get("candidates", []):
            grade = c.get("suggested_grade", 0)
            total_by_grade[grade] = total_by_grade.get(grade, 0) + 1

    data["metadata"]["grade_distribution"] = total_by_grade

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print()
    print(f"Pre-labeling complete:")
    print(f"  Grade 3 (highly relevant): {total_by_grade.get(3, 0)}")
    print(f"  Grade 2 (relevant):        {total_by_grade.get(2, 0)}")
    print(f"  Grade 1 (marginal):        {total_by_grade.get(1, 0)}")
    print(f"  Grade 0 (irrelevant):      {total_by_grade.get(0, 0)}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
