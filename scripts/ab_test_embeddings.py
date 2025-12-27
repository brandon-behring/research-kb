#!/usr/bin/env python3
"""A/B test BGE-large vs BGE-M3 embeddings.

Compares retrieval quality on a 10K chunk sample using the eval test suite.

Usage:
    python scripts/ab_test_embeddings.py
    python scripts/ab_test_embeddings.py --sample-size 5000
    python scripts/ab_test_embeddings.py --verbose
"""

import argparse
import asyncio
import gc
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

import asyncpg

# Configuration
DB_URL = "postgresql://postgres:postgres@localhost:5432/research_kb"
BGE_LARGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class TestCase:
    """A single retrieval test case."""
    query: str
    expected_source_pattern: str
    expected_in_top_k: int


@dataclass
class ModelResult:
    """Results for a single model."""
    name: str
    hit_rate_5: float
    hit_rate_10: float
    mrr: float
    per_query: list[dict]


def load_test_cases(yaml_path: Path) -> list[TestCase]:
    """Load test cases from YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    cases = []
    for tc in data.get("test_cases", []):
        cases.append(TestCase(
            query=tc["query"],
            expected_source_pattern=tc["expected_source_pattern"],
            expected_in_top_k=tc.get("expected_in_top_k", 5),
        ))
    return cases


async def sample_chunks(pool: asyncpg.Pool, sample_size: int, test_cases: list[TestCase]) -> list[dict]:
    """Sample chunks from database, ensuring eval sources are included.

    Strategy:
    1. Get all chunks from sources matching eval test patterns (~2-3K)
    2. Fill remaining quota with random stratified sample
    """
    print(f"Sampling {sample_size} chunks from database...")

    # Build pattern for eval sources
    patterns = [tc.expected_source_pattern for tc in test_cases]
    combined_pattern = "|".join(patterns)

    async with pool.acquire() as conn:
        # First: Get chunks from eval-relevant sources
        eval_chunks = await conn.fetch("""
            SELECT c.id, c.content, s.title as source_title
            FROM chunks c
            JOIN sources s ON c.source_id = s.id
            WHERE s.title ~* $1
            LIMIT 5000
        """, combined_pattern)

        eval_chunk_ids = {r['id'] for r in eval_chunks}
        print(f"  Found {len(eval_chunks)} chunks from eval-relevant sources")

        # Second: Fill with random sample (stratified by source)
        remaining = sample_size - len(eval_chunks)
        if remaining > 0:
            random_chunks = await conn.fetch("""
                SELECT c.id, c.content, s.title as source_title
                FROM chunks c
                JOIN sources s ON c.source_id = s.id
                WHERE c.id != ALL($1::uuid[])
                ORDER BY RANDOM()
                LIMIT $2
            """, list(eval_chunk_ids), remaining)
            print(f"  Added {len(random_chunks)} random chunks")
            all_chunks = list(eval_chunks) + list(random_chunks)
        else:
            all_chunks = list(eval_chunks)[:sample_size]

    print(f"  Total sample: {len(all_chunks)} chunks")
    return [dict(r) for r in all_chunks]


def embed_with_model(
    model_name: str,
    texts: list[str],
    query_prefix: str = "",
    batch_size: int = 32,
) -> tuple[np.ndarray, SentenceTransformer]:
    """Load model and embed texts.

    Returns embeddings and keeps model in memory for query embedding.
    """
    print(f"\nLoading {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"Embedding {len(texts)} texts (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return embeddings, model


def embed_query(model: SentenceTransformer, query: str, prefix: str = "") -> np.ndarray:
    """Embed a single query with optional prefix."""
    text = f"{prefix}{query}" if prefix else query
    return model.encode([text], convert_to_numpy=True)[0]


def cosine_similarity(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and all documents."""
    # Normalize
    query_norm = query_emb / np.linalg.norm(query_emb)
    doc_norms = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    # Dot product
    return np.dot(doc_norms, query_norm)


def evaluate_model(
    model: SentenceTransformer,
    model_name: str,
    chunk_embeddings: np.ndarray,
    chunks: list[dict],
    test_cases: list[TestCase],
    query_prefix: str = "",
    verbose: bool = False,
) -> ModelResult:
    """Run evaluation for a single model."""
    print(f"\nEvaluating {model_name}...")

    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks = []
    per_query = []

    for tc in test_cases:
        # Embed query
        query_emb = embed_query(model, tc.query, query_prefix)

        # Compute similarities
        similarities = cosine_similarity(query_emb, chunk_embeddings)

        # Get top-10 indices
        top_indices = np.argsort(similarities)[::-1][:10]

        # Check if expected source is in results
        pattern = re.compile(tc.expected_source_pattern, re.IGNORECASE)
        found_rank = None
        found_source = None

        for rank, idx in enumerate(top_indices, 1):
            source_title = chunks[idx]['source_title']
            if pattern.search(source_title):
                found_rank = rank
                found_source = source_title
                break

        # Update metrics
        if found_rank and found_rank <= 5:
            hits_at_5 += 1
        if found_rank and found_rank <= 10:
            hits_at_10 += 1
        if found_rank:
            reciprocal_ranks.append(1.0 / found_rank)

        result = {
            "query": tc.query,
            "found": found_rank is not None,
            "rank": found_rank,
            "source": found_source,
        }
        per_query.append(result)

        if verbose:
            status = "✓" if found_rank and found_rank <= 5 else "✗"
            if found_rank:
                print(f"  {status} [{found_rank}] {tc.query[:40]}... → {found_source[:40]}...")
            else:
                top_source = chunks[top_indices[0]]['source_title']
                print(f"  {status} [--] {tc.query[:40]}... → (top: {top_source[:40]}...)")

    total = len(test_cases)
    return ModelResult(
        name=model_name,
        hit_rate_5=hits_at_5 / total,
        hit_rate_10=hits_at_10 / total,
        mrr=sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0,
        per_query=per_query,
    )


def print_comparison(large_result: ModelResult, m3_result: ModelResult):
    """Print side-by-side comparison."""
    print("\n" + "=" * 70)
    print("A/B TEST RESULTS: BGE-large-en-v1.5 vs BGE-M3")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'BGE-large':>15} {'BGE-M3':>15} {'Δ':>10} {'Winner':>10}")
    print("-" * 70)

    # Hit Rate@5
    delta_5 = m3_result.hit_rate_5 - large_result.hit_rate_5
    winner_5 = "M3" if delta_5 > 0.01 else ("large" if delta_5 < -0.01 else "tie")
    print(f"{'Hit Rate@5':<20} {large_result.hit_rate_5:>14.1%} {m3_result.hit_rate_5:>14.1%} {delta_5:>+9.1%} {winner_5:>10}")

    # Hit Rate@10
    delta_10 = m3_result.hit_rate_10 - large_result.hit_rate_10
    winner_10 = "M3" if delta_10 > 0.01 else ("large" if delta_10 < -0.01 else "tie")
    print(f"{'Hit Rate@10':<20} {large_result.hit_rate_10:>14.1%} {m3_result.hit_rate_10:>14.1%} {delta_10:>+9.1%} {winner_10:>10}")

    # MRR
    delta_mrr = m3_result.mrr - large_result.mrr
    winner_mrr = "M3" if delta_mrr > 0.02 else ("large" if delta_mrr < -0.02 else "tie")
    print(f"{'MRR':<20} {large_result.mrr:>15.3f} {m3_result.mrr:>15.3f} {delta_mrr:>+10.3f} {winner_mrr:>10}")

    print("-" * 70)

    # Decision
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    threshold = 0.02  # 2% improvement threshold

    if delta_5 >= threshold:
        print(f"\n✅ BGE-M3 WINS: Hit Rate improved by {delta_5:.1%} (≥{threshold:.0%} threshold)")
        print("\n   Recommended action: MIGRATE to BGE-M3")
        print("   - Re-embed entire corpus (~45 min)")
        print("   - Remove query instruction prefix")
        print("   - Expected Hit Rate: {:.1%}".format(m3_result.hit_rate_5))
    elif delta_5 <= -threshold:
        print(f"\n❌ BGE-large WINS: BGE-M3 regressed by {abs(delta_5):.1%}")
        print("\n   Recommended action: STAY with BGE-large")
        print("   - No changes needed")
        print("   - Document findings for future reference")
    else:
        print(f"\n🟰 TIE: Difference ({delta_5:+.1%}) within noise threshold (±{threshold:.0%})")
        print("\n   Recommended action: STAY with BGE-large")
        print("   - No migration benefit justifies the effort")
        print("   - Revisit when corpus grows or new models emerge")

    print()


async def main():
    parser = argparse.ArgumentParser(description="A/B test BGE embedding models")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of chunks to sample")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-query results")
    args = parser.parse_args()

    # Load test cases
    yaml_path = Path(__file__).parent.parent / "fixtures" / "eval" / "retrieval_test_cases.yaml"
    test_cases = load_test_cases(yaml_path)
    print(f"Loaded {len(test_cases)} test cases")

    # Connect to database
    pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=2)

    try:
        # Sample chunks
        chunks = await sample_chunks(pool, args.sample_size, test_cases)
        texts = [c['content'] for c in chunks]

        # === BGE-large ===
        large_embeddings, large_model = embed_with_model(
            "BAAI/bge-large-en-v1.5",
            texts,
            batch_size=16,  # Reduced for GPU memory
        )

        large_result = evaluate_model(
            large_model,
            "BGE-large-en-v1.5",
            large_embeddings,
            chunks,
            test_cases,
            query_prefix=BGE_LARGE_QUERY_PREFIX,
            verbose=args.verbose,
        )

        # Free memory
        del large_model
        gc.collect()
        torch.cuda.empty_cache()

        # === BGE-M3 ===
        m3_embeddings, m3_model = embed_with_model(
            "BAAI/bge-m3",
            texts,
            batch_size=16,  # Reduced for GPU memory
        )

        m3_result = evaluate_model(
            m3_model,
            "BGE-M3",
            m3_embeddings,
            chunks,
            test_cases,
            query_prefix="",  # M3 doesn't need prefix
            verbose=args.verbose,
        )

        # Free memory
        del m3_model
        gc.collect()
        torch.cuda.empty_cache()

        # Print comparison
        print_comparison(large_result, m3_result)

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
