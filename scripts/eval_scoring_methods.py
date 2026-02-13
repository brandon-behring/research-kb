#!/usr/bin/env python3
"""Evaluate and compare scoring methods (weighted sum vs RRF) using golden dataset.

Uses chunk-ID-targeted test cases to compute precise ranking metrics.
Runs each query with search_hybrid_v2 (4-way: FTS + vector + graph + citation)
using both scoring methods, then computes:

- Hit Rate@5: % of queries where a target chunk appears in top 5
- Hit Rate@10: % of queries where a target chunk appears in top 10
- MRR: Mean Reciprocal Rank (1/rank of first target chunk found)
- NDCG@5: Normalized Discounted Cumulative Gain at 5
- NDCG@10: Normalized Discounted Cumulative Gain at 10

Usage:
    python scripts/eval_scoring_methods.py
    python scripts/eval_scoring_methods.py --verbose
    python scripts/eval_scoring_methods.py --output fixtures/benchmarks/rrf_vs_weighted.json
"""

import asyncio
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_pdf import EmbeddingClient
from research_kb_storage import DatabaseConfig, SearchQuery, get_connection_pool
from research_kb_storage.search import (
    search_hybrid,
    search_hybrid_v2,
    search_with_expansion,
    search_with_rerank,
)


@dataclass
class GoldenEntry:
    """A single golden dataset entry."""

    query: str
    target_chunk_ids: list[str]
    domain: str
    source_title: str
    difficulty: str


@dataclass
class QueryResult:
    """Result of evaluating a single query with a scoring method."""

    entry: GoldenEntry
    scoring_method: str
    hit_at_5: bool = False
    hit_at_10: bool = False
    first_hit_rank: Optional[int] = None
    latency_ms: float = 0.0
    error: Optional[str] = None


def load_golden_dataset(path: Path) -> list[GoldenEntry]:
    """Load golden dataset from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [GoldenEntry(**entry) for entry in data]


def compute_ndcg_at_k(rank: Optional[int], k: int) -> float:
    """Compute NDCG@K for binary relevance with a single relevant item.

    Args:
        rank: 1-based rank of the relevant item (None if not found)
        k: Cutoff position

    Returns:
        NDCG score (0.0 to 1.0). 1.0 = found at rank 1, 0.0 = not found in top K.
    """
    if rank is None or rank > k:
        return 0.0
    # Binary relevance: DCG = 1/log2(rank+1), ideal DCG = 1/log2(2) = 1.0
    return 1.0 / math.log2(rank + 1)


async def evaluate_query(
    entry: GoldenEntry,
    embed_client: EmbeddingClient,
    scoring_method: str,
    use_v2: bool = True,
    domain_filter: bool = True,
    use_rerank: bool = False,
    use_expand: bool = False,
) -> QueryResult:
    """Run a single query and check if target chunks appear in results.

    Args:
        entry: Golden dataset entry
        embed_client: Embedding client for query embedding
        scoring_method: "weighted" or "rrf"
        use_v2: Use search_hybrid_v2 (4-way) or search_hybrid (2-way)
        domain_filter: Filter by entry's domain (default True)
        use_rerank: Enable cross-encoder reranking
        use_expand: Enable query expansion (synonyms + graph)
    """
    result = QueryResult(entry=entry, scoring_method=scoring_method)

    try:
        query_embedding = embed_client.embed_query(entry.query)

        query = SearchQuery(
            text=entry.query,
            embedding=query_embedding,
            fts_weight=0.2,
            vector_weight=0.4,
            graph_weight=0.2,
            citation_weight=0.2,
            use_graph=True,
            use_citations=True,
            limit=10,
            scoring_method=scoring_method,
            domain_id=entry.domain if domain_filter else None,
        )

        start = time.monotonic()
        if use_expand:
            results, _expansion = await search_with_expansion(
                query,
                use_synonyms=True,
                use_graph_expansion=True,
                use_llm_expansion=False,
                use_rerank=use_rerank,
                rerank_top_k=10,
            )
        elif use_rerank:
            results = await search_with_rerank(
                query,
                rerank_top_k=10,
                fetch_multiplier=5,
            )
        elif use_v2:
            results = await search_hybrid_v2(query)
        else:
            results = await search_hybrid(query)
        result.latency_ms = (time.monotonic() - start) * 1000

        target_set = set(entry.target_chunk_ids)

        for i, r in enumerate(results):
            chunk_id = str(r.chunk.id)
            if chunk_id in target_set:
                rank = i + 1
                if result.first_hit_rank is None:
                    result.first_hit_rank = rank
                if rank <= 5:
                    result.hit_at_5 = True
                if rank <= 10:
                    result.hit_at_10 = True
                break  # Only need first hit for MRR

    except Exception as e:
        result.error = str(e)

    return result


def compute_metrics(results: list[QueryResult]) -> dict:
    """Compute aggregate metrics from query results."""
    n = len(results)
    if n == 0:
        return {}

    valid = [r for r in results if r.error is None]
    n_valid = len(valid)

    hit_5 = sum(1 for r in valid if r.hit_at_5)
    hit_10 = sum(1 for r in valid if r.hit_at_10)

    # MRR: average of 1/rank for queries with hits
    reciprocal_ranks = [
        1.0 / r.first_hit_rank for r in valid if r.first_hit_rank is not None
    ]
    mrr = sum(reciprocal_ranks) / n_valid if n_valid > 0 else 0.0

    # NDCG@5 and NDCG@10
    ndcg_5_scores = [compute_ndcg_at_k(r.first_hit_rank, 5) for r in valid]
    ndcg_10_scores = [compute_ndcg_at_k(r.first_hit_rank, 10) for r in valid]
    ndcg_5 = sum(ndcg_5_scores) / n_valid if n_valid > 0 else 0.0
    ndcg_10 = sum(ndcg_10_scores) / n_valid if n_valid > 0 else 0.0

    # Latency
    latencies = [r.latency_ms for r in valid]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "total_queries": n,
        "valid_queries": n_valid,
        "errors": n - n_valid,
        "hit_rate_at_5": hit_5 / n_valid if n_valid > 0 else 0.0,
        "hit_rate_at_10": hit_10 / n_valid if n_valid > 0 else 0.0,
        "mrr": mrr,
        "ndcg_at_5": ndcg_5,
        "ndcg_at_10": ndcg_10,
        "avg_latency_ms": avg_latency,
    }


def compute_domain_metrics(results: list[QueryResult]) -> dict[str, dict]:
    """Compute metrics broken down by domain."""
    domains: dict[str, list[QueryResult]] = {}
    for r in results:
        d = r.entry.domain
        if d not in domains:
            domains[d] = []
        domains[d].append(r)

    return {domain: compute_metrics(rs) for domain, rs in sorted(domains.items())}


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare scoring methods using golden dataset")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose per-query output")
    parser.add_argument("--output", "-o", help="Output JSON file for comparison results")
    parser.add_argument(
        "--golden",
        default=str(Path(__file__).parent.parent / "fixtures" / "eval" / "golden_dataset.json"),
        help="Path to golden dataset JSON",
    )
    parser.add_argument(
        "--use-v1",
        action="store_true",
        help="Use search_hybrid instead of search_hybrid_v2 (for testing D1 fix)",
    )
    parser.add_argument(
        "--no-domain-filter",
        action="store_true",
        help="Search entire corpus (don't filter by domain)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking (requires rerank server running)",
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Enable query expansion (synonyms + graph expansion)",
    )
    args = parser.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"Error: Golden dataset not found at {golden_path}")
        sys.exit(1)

    entries = load_golden_dataset(golden_path)
    print(f"Loaded {len(entries)} golden entries")

    # Initialize
    config = DatabaseConfig()
    await get_connection_pool(config)
    embed_client = EmbeddingClient()

    use_v2 = not args.use_v1
    domain_filter = not args.no_domain_filter
    use_rerank = args.rerank
    use_expand = args.expand

    # Determine search mode label
    if use_expand:
        search_fn_name = "search_with_expansion" + (" + rerank" if use_rerank else "")
    elif use_rerank:
        search_fn_name = "search_with_rerank"
    elif use_v2:
        search_fn_name = "search_hybrid_v2"
    else:
        search_fn_name = "search_hybrid"

    mode_flags = []
    if domain_filter:
        mode_flags.append("domain_filter=ON")
    else:
        mode_flags.append("domain_filter=OFF")
    if use_rerank:
        mode_flags.append("rerank=ON")
    if use_expand:
        mode_flags.append("expand=ON")

    print(f"Search function: {search_fn_name}")
    print(f"Mode: {', '.join(mode_flags)}")

    all_results = {}
    for method in ["weighted", "rrf"]:
        print(f"\n{'=' * 60}")
        print(f"  Scoring method: {method.upper()}")
        print(f"{'=' * 60}")

        results = []
        for entry in entries:
            result = await evaluate_query(
                entry,
                embed_client,
                method,
                use_v2=use_v2,
                domain_filter=domain_filter,
                use_rerank=use_rerank,
                use_expand=use_expand,
            )
            results.append(result)

            if args.verbose:
                status = "HIT@5" if result.hit_at_5 else ("HIT@10" if result.hit_at_10 else "MISS")
                rank_str = f"rank {result.first_hit_rank}" if result.first_hit_rank else "not found"
                print(f"  [{status:6s}] {entry.query:45s} | {rank_str:12s} | {result.latency_ms:.0f}ms")

        all_results[method] = results

    # Compute and display comparison
    print(f"\n{'=' * 60}")
    print("  COMPARISON: Weighted vs RRF")
    print(f"{'=' * 60}\n")

    metrics = {}
    for method in ["weighted", "rrf"]:
        metrics[method] = compute_metrics(all_results[method])

    # Print comparison table
    print(f"{'Metric':<20s} {'Weighted':>12s} {'RRF':>12s} {'Winner':>10s}")
    print("-" * 56)

    comparison_metrics = [
        ("Hit Rate@5", "hit_rate_at_5"),
        ("Hit Rate@10", "hit_rate_at_10"),
        ("MRR", "mrr"),
        ("NDCG@5", "ndcg_at_5"),
        ("NDCG@10", "ndcg_at_10"),
        ("Avg Latency (ms)", "avg_latency_ms"),
    ]

    rrf_wins = 0
    weighted_wins = 0

    for label, key in comparison_metrics:
        w = metrics["weighted"].get(key, 0.0)
        r = metrics["rrf"].get(key, 0.0)

        if key == "avg_latency_ms":
            # Lower is better for latency
            winner = "RRF" if r < w else ("Weighted" if w < r else "Tie")
        else:
            # Higher is better for all other metrics
            winner = "RRF" if r > w + 0.001 else ("Weighted" if w > r + 0.001 else "Tie")

        if winner == "RRF":
            rrf_wins += 1
        elif winner == "Weighted":
            weighted_wins += 1

        if key == "avg_latency_ms":
            print(f"{label:<20s} {w:>10.0f}ms {r:>10.0f}ms {winner:>10s}")
        else:
            print(f"{label:<20s} {w:>11.3f} {r:>12.3f} {winner:>10s}")

    # Overall verdict
    print(f"\nMetrics won: Weighted={weighted_wins}, RRF={rrf_wins}")
    if rrf_wins >= 3:
        verdict = "RRF wins — consider changing default"
    elif weighted_wins >= 3:
        verdict = "Weighted wins — keep current default"
    else:
        verdict = "Inconclusive — keep weighted (status quo)"
    print(f"Verdict: {verdict}")

    # Per-domain breakdown
    print(f"\n{'=' * 60}")
    print("  PER-DOMAIN BREAKDOWN")
    print(f"{'=' * 60}\n")

    for method in ["weighted", "rrf"]:
        domain_metrics = compute_domain_metrics(all_results[method])
        print(f"\n  {method.upper()}:")
        print(f"  {'Domain':<20s} {'Hit@5':>8s} {'Hit@10':>8s} {'MRR':>8s} {'NDCG@5':>8s}")
        print(f"  {'-' * 54}")
        for domain, dm in domain_metrics.items():
            print(
                f"  {domain:<20s} "
                f"{dm['hit_rate_at_5']:>7.1%} "
                f"{dm['hit_rate_at_10']:>7.1%} "
                f"{dm['mrr']:>7.3f} "
                f"{dm['ndcg_at_5']:>7.3f}"
            )

    # Output JSON
    if args.output:
        output = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "search_function": search_fn_name,
            "mode": {
                "domain_filter": domain_filter,
                "rerank": use_rerank,
                "expand": use_expand,
            },
            "golden_entries": len(entries),
            "metrics": metrics,
            "domain_metrics": {
                method: compute_domain_metrics(all_results[method])
                for method in ["weighted", "rrf"]
            },
            "verdict": verdict,
            "rrf_wins": rrf_wins,
            "weighted_wins": weighted_wins,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
