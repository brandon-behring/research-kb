#!/usr/bin/env python3
"""Optimize search weights using the golden evaluation dataset.

Tunes the 4 search weights (FTS/vector/graph/citation) to maximize MRR
using scipy.optimize with sum-to-1 constraints.

Supports:
- Global optimization across all eval cases
- Per-domain optimization (only for domains with >= min_cases)
- K-fold cross-validation to detect overfitting
- Comparison report: current vs optimized weights

Usage:
    python scripts/optimize_weights.py                      # Global optimization
    python scripts/optimize_weights.py --per-domain         # Per-domain tuning
    python scripts/optimize_weights.py --cross-validate     # With 5-fold CV
    python scripts/optimize_weights.py --dry-run            # Show current, don't optimize
    python scripts/optimize_weights.py --method nelder-mead # Optimization method
"""

import nest_asyncio

nest_asyncio.apply()  # Allow nested event loops (scipy calls run_until_complete inside asyncio.run)

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_pdf import EmbeddingClient
from research_kb_storage import DatabaseConfig, SearchQuery, get_connection_pool, search_hybrid

try:
    from research_kb_storage.search import search_hybrid_v2
except ImportError:
    search_hybrid_v2 = None  # Fallback to search_hybrid if v2 not available

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "eval"


@dataclass
class EvalCase:
    """A golden evaluation case with precomputed embedding."""

    query: str
    target_chunk_ids: set[UUID]
    domain: str
    source_title: str
    difficulty: str
    embedding: list[float]


@dataclass
class ChunkScores:
    """Raw per-signal scores for a single chunk result."""

    chunk_id: UUID
    fts_score: float
    vector_score: float
    graph_score: float
    citation_score: float


@dataclass
class CachedCase:
    """Precomputed scores for an eval case (all top-N results with raw signal scores)."""

    case: EvalCase
    results: list[ChunkScores]  # Top-N results with raw scores


@dataclass
class WeightConfig:
    """A set of search weights (always normalized to sum to 1)."""

    fts: float
    vector: float
    graph: float
    citation: float

    def __post_init__(self):
        """Normalize weights to sum to 1."""
        total = self.fts + self.vector + self.graph + self.citation
        if total > 0:
            self.fts /= total
            self.vector /= total
            self.graph /= total
            self.citation /= total

    def to_dict(self) -> dict:
        """Convert to dict with rounded values."""
        return {
            "fts_weight": round(self.fts, 4),
            "vector_weight": round(self.vector, 4),
            "graph_weight": round(self.graph, 4),
            "citation_weight": round(self.citation, 4),
        }

    def __repr__(self) -> str:
        return (
            f"W(fts={self.fts:.3f}, vec={self.vector:.3f}, "
            f"graph={self.graph:.3f}, cite={self.citation:.3f})"
        )


# Current default weights (as set in CLI _shared.py)
CURRENT_WEIGHTS = WeightConfig(fts=0.3, vector=0.7, graph=0.2, citation=0.15)


def find_latest_golden_file() -> Path:
    """Find the most recent golden_candidates JSON file."""
    candidates = sorted(FIXTURES_DIR.glob("golden_candidates_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No golden_candidates_*.json files found in {FIXTURES_DIR}"
        )
    return candidates[-1]


async def load_eval_cases(dataset_path: Path, embed_client: EmbeddingClient) -> list[EvalCase]:
    """Load golden dataset and precompute embeddings.

    Args:
        dataset_path: Path to golden_candidates JSON
        embed_client: Embedding client for query embedding

    Returns:
        List of EvalCase with precomputed embeddings
    """
    with open(dataset_path) as f:
        entries = json.load(f)

    print(f"Precomputing embeddings for {len(entries)} queries...")
    cases = []
    for entry in entries:
        target_ids = {UUID(cid) for cid in entry.get("target_chunk_ids", []) if cid}
        if not target_ids:
            continue  # Skip entries with no targets

        embedding = embed_client.embed_query(entry["query"])
        cases.append(
            EvalCase(
                query=entry["query"],
                target_chunk_ids=target_ids,
                domain=entry.get("domain", "unknown"),
                source_title=entry.get("source_title", ""),
                difficulty=entry.get("difficulty", "medium"),
                embedding=embedding,
            )
        )

    print(f"  Loaded {len(cases)} eval cases with embeddings")
    return cases


async def precompute_scores(
    cases: list[EvalCase],
    use_graph: bool = False,
    use_citations: bool = True,
    fetch_limit: int = 50,
) -> list[CachedCase]:
    """Precompute raw signal scores for all eval cases (run search ONCE per case).

    Fetches a wide result set (top-50) so different weight combinations can
    re-rank without missing candidates. This is the expensive step (~18s/case).

    Args:
        cases: Eval cases with precomputed embeddings
        use_graph: Include graph signal in precomputation
        use_citations: Include citation signal
        fetch_limit: Number of results to cache per query (wider = better coverage)

    Returns:
        List of CachedCase with raw per-signal scores
    """
    cached = []
    total = len(cases)

    for i, case in enumerate(cases, 1):
        query = SearchQuery(
            text=case.query,
            embedding=case.embedding,
            fts_weight=0.3,
            vector_weight=0.7,
            graph_weight=0.2 if use_graph else 0.0,
            use_graph=use_graph,
            citation_weight=0.15 if use_citations else 0.0,
            use_citations=use_citations,
            limit=fetch_limit,
        )

        try:
            if use_citations and search_hybrid_v2 is not None:
                results = await search_hybrid_v2(query)
            else:
                results = await search_hybrid(query)

            chunk_scores = []
            for r in results:
                chunk_scores.append(
                    ChunkScores(
                        chunk_id=r.chunk.id,
                        fts_score=r.fts_score or 0.0,
                        vector_score=r.vector_score or 0.0,
                        graph_score=r.graph_score or 0.0,
                        citation_score=r.citation_score or 0.0,
                    )
                )
            cached.append(CachedCase(case=case, results=chunk_scores))
        except Exception as e:
            print(f"  WARNING: Failed to precompute case {i}: {e}")
            cached.append(CachedCase(case=case, results=[]))

        if i % 20 == 0 or i == total:
            print(f"  Precomputed {i}/{total} cases")

    return cached


def evaluate_weights_cached(
    cached_cases: list[CachedCase],
    weights: WeightConfig,
    top_k: int = 10,
) -> dict:
    """Evaluate weights using precomputed scores (pure arithmetic, no DB queries).

    Re-ranks cached results with new weights and computes MRR/Hit@K.
    Runs in <1ms per evaluation.

    Args:
        cached_cases: Cases with precomputed raw scores
        weights: Weight configuration to evaluate
        top_k: Number of results to consider for ranking

    Returns:
        Dict with mrr, hit_rate_10, per_domain metrics
    """
    hits = 0
    reciprocal_ranks = []
    domain_results: dict[str, list[float]] = defaultdict(list)

    for cc in cached_cases:
        # Re-score all cached results with current weights
        scored = []
        for cs in cc.results:
            combined = (
                weights.fts * cs.fts_score
                + weights.vector * cs.vector_score
                + weights.graph * cs.graph_score
                + weights.citation * cs.citation_score
            )
            scored.append((cs.chunk_id, combined))

        # Sort by combined score descending
        scored.sort(key=lambda x: -x[1])

        # Find rank of first matching chunk in top-K
        matched_rank = None
        for rank, (chunk_id, _) in enumerate(scored[:top_k], 1):
            if chunk_id in cc.case.target_chunk_ids:
                matched_rank = rank
                break

        if matched_rank is not None:
            hits += 1
            rr = 1.0 / matched_rank
        else:
            rr = 0.0

        reciprocal_ranks.append(rr)
        domain_results[cc.case.domain].append(rr)

    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    hit_rate = hits / len(cached_cases) if cached_cases else 0.0

    per_domain = {}
    for domain, rrs in domain_results.items():
        per_domain[domain] = {
            "mrr": float(np.mean(rrs)),
            "hit_rate": sum(1 for r in rrs if r > 0) / len(rrs),
            "count": len(rrs),
        }

    return {
        "mrr": float(mrr),
        "hit_rate_10": float(hit_rate),
        "total_cases": len(cached_cases),
        "per_domain": per_domain,
    }


async def evaluate_weights(
    cases: list[EvalCase],
    weights: WeightConfig,
    use_graph: bool = True,
    use_citations: bool = True,
) -> dict:
    """Legacy evaluate_weights — calls search per case. Use evaluate_weights_cached instead.

    Kept for cross-validation on unseen data where precomputed cache doesn't apply.
    """
    hits = 0
    reciprocal_ranks = []
    domain_results: dict[str, list[float]] = defaultdict(list)

    for case in cases:
        query = SearchQuery(
            text=case.query,
            embedding=case.embedding,
            fts_weight=weights.fts,
            vector_weight=weights.vector,
            graph_weight=weights.graph if use_graph else 0.0,
            use_graph=use_graph,
            citation_weight=weights.citation if use_citations else 0.0,
            use_citations=use_citations,
            limit=10,
        )

        try:
            results = await search_hybrid(query)

            matched_rank = None
            for r in results:
                if r.chunk.id in case.target_chunk_ids:
                    matched_rank = r.rank
                    break

            if matched_rank is not None:
                hits += 1
                rr = 1.0 / matched_rank
            else:
                rr = 0.0

            reciprocal_ranks.append(rr)
            domain_results[case.domain].append(rr)

        except Exception:
            reciprocal_ranks.append(0.0)
            domain_results[case.domain].append(0.0)

    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    hit_rate = hits / len(cases) if cases else 0.0

    per_domain = {}
    for domain, rrs in domain_results.items():
        per_domain[domain] = {
            "mrr": float(np.mean(rrs)),
            "hit_rate": sum(1 for r in rrs if r > 0) / len(rrs),
            "count": len(rrs),
        }

    return {
        "mrr": float(mrr),
        "hit_rate_10": float(hit_rate),
        "total_cases": len(cases),
        "per_domain": per_domain,
    }


def optimize_global_cached(
    cached_cases: list[CachedCase],
    method: str = "nelder-mead",
    use_graph: bool = True,
    use_citations: bool = True,
) -> tuple[WeightConfig, dict]:
    """Find optimal global weights using precomputed scores.

    Uses Nelder-Mead simplex (gradient-free) to minimize negative MRR.
    Each evaluation is pure arithmetic (~1ms), making optimization fast.

    Args:
        cached_cases: Cases with precomputed raw scores
        method: Optimization method (nelder-mead or differential_evolution)
        use_graph: Include graph signal
        use_citations: Include citation signal

    Returns:
        Tuple of (optimal WeightConfig, metrics dict)
    """
    from scipy.optimize import minimize, differential_evolution

    eval_count = [0]

    def objective(params):
        """Objective function: negative MRR (pure arithmetic, no DB queries)."""
        exp_params = np.exp(params - np.max(params))
        w = exp_params / exp_params.sum()

        weights = WeightConfig(
            fts=float(w[0]),
            vector=float(w[1]),
            graph=float(w[2]) if use_graph else 0.0,
            citation=float(w[3]) if use_citations else 0.0,
        )

        metrics = evaluate_weights_cached(cached_cases, weights)
        eval_count[0] += 1

        if eval_count[0] % 50 == 0:
            print(
                f"  Iteration {eval_count[0]}: MRR={metrics['mrr']:.4f} "
                f"{weights}"
            )

        return -metrics["mrr"]

    # Initial point: current weights (log-space for softmax parameterization)
    n_params = 4
    if use_graph and use_citations:
        x0 = np.log([CURRENT_WEIGHTS.fts, CURRENT_WEIGHTS.vector,
                      CURRENT_WEIGHTS.graph, CURRENT_WEIGHTS.citation])
    elif use_graph:
        x0 = np.log([CURRENT_WEIGHTS.fts, CURRENT_WEIGHTS.vector,
                      CURRENT_WEIGHTS.graph, 0.01])
    elif use_citations:
        x0 = np.log([CURRENT_WEIGHTS.fts, CURRENT_WEIGHTS.vector,
                      0.01, CURRENT_WEIGHTS.citation])
    else:
        x0 = np.log([CURRENT_WEIGHTS.fts, CURRENT_WEIGHTS.vector, 0.01, 0.01])

    print(f"\nOptimizing with {method} ({len(cached_cases)} cases, precomputed scores)...")
    t0 = time.time()

    if method == "differential_evolution":
        bounds = [(-4.0, 2.0)] * n_params
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=50,
            tol=0.001,
            seed=42,
            popsize=10,
        )
    else:
        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 200, "xatol": 0.005, "fatol": 0.0005},
        )

    elapsed = time.time() - t0

    # Decode optimal weights
    exp_params = np.exp(result.x - np.max(result.x))
    w = exp_params / exp_params.sum()
    optimal = WeightConfig(
        fts=float(w[0]),
        vector=float(w[1]),
        graph=float(w[2]) if use_graph else 0.0,
        citation=float(w[3]) if use_citations else 0.0,
    )

    final_metrics = evaluate_weights_cached(cached_cases, optimal)

    print(f"\nOptimization complete ({elapsed:.1f}s, {eval_count[0]} evaluations)")
    print(f"  Optimal: {optimal}")
    print(f"  MRR: {final_metrics['mrr']:.4f}")
    print(f"  Hit@10: {final_metrics['hit_rate_10']:.1%}")

    return optimal, final_metrics


async def optimize_global(
    cases: list[EvalCase],
    method: str = "nelder-mead",
    use_graph: bool = True,
    use_citations: bool = True,
) -> tuple[WeightConfig, dict]:
    """Legacy optimizer — runs search per case per iteration. Slow.

    Use optimize_global_cached() instead for ~60x speedup.
    """
    from scipy.optimize import minimize, differential_evolution

    eval_count = [0]

    async def objective_async(params):
        exp_params = np.exp(params - np.max(params))
        w = exp_params / exp_params.sum()
        weights = WeightConfig(
            fts=float(w[0]), vector=float(w[1]),
            graph=float(w[2]) if use_graph else 0.0,
            citation=float(w[3]) if use_citations else 0.0,
        )
        metrics = await evaluate_weights(cases, weights, use_graph, use_citations)
        eval_count[0] += 1
        if eval_count[0] % 10 == 0:
            print(f"  Iteration {eval_count[0]}: MRR={metrics['mrr']:.4f} {weights}")
        return -metrics["mrr"]

    def objective(params):
        return asyncio.get_event_loop().run_until_complete(objective_async(params))

    n_params = 4
    x0 = np.log([CURRENT_WEIGHTS.fts, CURRENT_WEIGHTS.vector, 0.01, CURRENT_WEIGHTS.citation])

    print(f"\nOptimizing with {method} ({len(cases)} cases, LIVE QUERIES - slow)...")
    t0 = time.time()
    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 100, "xatol": 0.01, "fatol": 0.001})
    elapsed = time.time() - t0

    exp_params = np.exp(result.x - np.max(result.x))
    w = exp_params / exp_params.sum()
    optimal = WeightConfig(
        fts=float(w[0]), vector=float(w[1]),
        graph=float(w[2]) if use_graph else 0.0,
        citation=float(w[3]) if use_citations else 0.0,
    )
    final_metrics = await evaluate_weights(cases, optimal, use_graph, use_citations)
    print(f"\nOptimization complete ({elapsed:.1f}s, {eval_count[0]} evaluations)")
    print(f"  Optimal: {optimal}")
    print(f"  MRR: {final_metrics['mrr']:.4f}")
    return optimal, final_metrics


async def cross_validate(
    cases: list[EvalCase],
    weights: WeightConfig,
    k_folds: int = 5,
    use_graph: bool = True,
    use_citations: bool = True,
) -> dict:
    """K-fold cross-validation for weight robustness.

    Args:
        cases: All eval cases
        weights: Weights to validate
        k_folds: Number of folds
        use_graph: Enable graph signal
        use_citations: Enable citation signal

    Returns:
        Dict with mean_mrr, std_mrr, fold_mrrs
    """
    indices = np.arange(len(cases))
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_folds)

    fold_mrrs = []
    for i, test_indices in enumerate(folds):
        test_cases = [cases[j] for j in test_indices]
        metrics = await evaluate_weights(test_cases, weights, use_graph, use_citations)
        fold_mrrs.append(metrics["mrr"])
        print(f"  Fold {i + 1}/{k_folds}: MRR={metrics['mrr']:.4f} ({len(test_cases)} cases)")

    return {
        "mean_mrr": float(np.mean(fold_mrrs)),
        "std_mrr": float(np.std(fold_mrrs)),
        "fold_mrrs": fold_mrrs,
        "k_folds": k_folds,
    }


async def optimize_per_domain(
    cases: list[EvalCase],
    min_cases: int = 5,
    use_graph: bool = True,
    use_citations: bool = True,
) -> dict[str, tuple[WeightConfig, dict]]:
    """Optimize weights per domain (only domains with >= min_cases).

    Args:
        cases: All eval cases
        min_cases: Minimum cases required for per-domain optimization
        use_graph: Enable graph signal
        use_citations: Enable citation signal

    Returns:
        Dict mapping domain -> (optimal weights, metrics)
    """
    domain_cases: dict[str, list[EvalCase]] = defaultdict(list)
    for case in cases:
        domain_cases[case.domain].append(case)

    results = {}
    for domain, dcases in sorted(domain_cases.items()):
        if len(dcases) < min_cases:
            print(f"\n  {domain}: SKIP ({len(dcases)} < {min_cases} min cases)")
            continue

        print(f"\n  {domain} ({len(dcases)} cases):")
        optimal, metrics = await optimize_global(
            dcases, method="nelder-mead", use_graph=use_graph, use_citations=use_citations
        )
        results[domain] = (optimal, metrics)

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize search weights using golden evaluation dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to golden_candidates JSON (default: latest)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate current weights without optimizing",
    )
    parser.add_argument(
        "--per-domain",
        action="store_true",
        help="Optimize per-domain weights (min 5 cases per domain)",
    )
    parser.add_argument(
        "--min-cases",
        type=int,
        default=5,
        help="Minimum cases for per-domain optimization (default: 5)",
    )
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Run 5-fold cross-validation on results",
    )
    parser.add_argument(
        "--method",
        choices=["nelder-mead", "differential_evolution"],
        default="nelder-mead",
        help="Optimization method (default: nelder-mead)",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable graph signal (optimize FTS/vector/citation only)",
    )
    parser.add_argument(
        "--no-citations",
        action="store_true",
        help="Disable citation signal (optimize FTS/vector/graph only)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    return parser.parse_args()


async def main():
    """Optimize search weights."""
    args = parse_args()

    use_graph = not args.no_graph
    use_citations = not args.no_citations

    signals = []
    if use_graph:
        signals.append("graph")
    if use_citations:
        signals.append("citation")
    signal_label = f"FTS+vector+{'+'.join(signals)}" if signals else "FTS+vector"
    print(f"Signals: {signal_label}")

    # Load dataset
    if args.dataset:
        dataset_path = Path(args.dataset)
    else:
        dataset_path = find_latest_golden_file()
    print(f"Dataset: {dataset_path}")

    # Initialize
    config = DatabaseConfig()
    await get_connection_pool(config)
    embed_client = EmbeddingClient()

    cases = await load_eval_cases(dataset_path, embed_client)
    if not cases:
        print("ERROR: No valid eval cases found")
        sys.exit(1)

    # Precompute scores (the expensive step — ~18s/case, done once)
    print(f"\n--- Precomputing raw scores (this takes ~{len(cases) * 18 // 60} min) ---")
    cached = await precompute_scores(cases, use_graph=use_graph, use_citations=use_citations)

    # Evaluate current weights using cached scores (instant)
    print(f"\n--- Current weights: {CURRENT_WEIGHTS} ---")
    current_metrics = evaluate_weights_cached(cached, CURRENT_WEIGHTS)
    print(f"  MRR: {current_metrics['mrr']:.4f}")
    print(f"  Hit@10: {current_metrics['hit_rate_10']:.1%}")

    if current_metrics.get("per_domain"):
        print(f"\n  Per-domain MRR:")
        for domain, dm in sorted(
            current_metrics["per_domain"].items(),
            key=lambda x: -x[1]["mrr"],
        ):
            print(f"    {domain:25s}: MRR={dm['mrr']:.3f} Hit={dm['hit_rate']:.0%} (n={dm['count']})")

    if args.dry_run:
        print("\n[dry-run] Stopping before optimization.")
        return

    # Global optimization (fast — uses cached scores)
    print("\n--- Global optimization ---")
    optimal, optimal_metrics = optimize_global_cached(
        cached, method=args.method, use_graph=use_graph, use_citations=use_citations
    )

    # Improvement
    delta_mrr = optimal_metrics["mrr"] - current_metrics["mrr"]
    print(f"\n  Improvement: MRR {current_metrics['mrr']:.4f} -> {optimal_metrics['mrr']:.4f} (delta {delta_mrr:+.4f})")

    # Cross-validation (using cached scores — fast)
    cv_results = None
    if args.cross_validate:
        print("\n--- Cross-validation (optimal weights, cached) ---")
        indices = np.arange(len(cached))
        np.random.seed(42)
        np.random.shuffle(indices)
        folds = np.array_split(indices, 5)
        fold_mrrs = []
        for i, test_indices in enumerate(folds):
            test_cached = [cached[j] for j in test_indices]
            fold_metrics = evaluate_weights_cached(test_cached, optimal)
            fold_mrrs.append(fold_metrics["mrr"])
            print(f"  Fold {i+1}/5: MRR={fold_metrics['mrr']:.4f} ({len(test_cached)} cases)")
        cv_results = {
            "mean_mrr": float(np.mean(fold_mrrs)),
            "std_mrr": float(np.std(fold_mrrs)),
            "fold_mrrs": fold_mrrs,
            "k_folds": 5,
        }
        print(f"\n  CV MRR: {cv_results['mean_mrr']:.4f} +/- {cv_results['std_mrr']:.4f}")
        if cv_results["std_mrr"] > 0.05:
            print("  WARNING: High variance suggests overfitting risk")

    # Per-domain optimization (using cached scores — fast)
    domain_results = None
    if args.per_domain:
        print("\n--- Per-domain optimization (cached) ---")
        domain_cached: dict[str, list[CachedCase]] = defaultdict(list)
        for cc in cached:
            domain_cached[cc.case.domain].append(cc)
        domain_results = {}
        for domain, dcached in sorted(domain_cached.items()):
            if len(dcached) < args.min_cases:
                print(f"\n  {domain}: SKIP ({len(dcached)} < {args.min_cases} min cases)")
                continue
            print(f"\n  {domain} ({len(dcached)} cases):")
            dopt, dmetrics = optimize_global_cached(
                dcached, method="nelder-mead", use_graph=use_graph, use_citations=use_citations
            )
            domain_results[domain] = (dopt, dmetrics)

    # Output summary
    output = {
        "current_weights": CURRENT_WEIGHTS.to_dict(),
        "current_metrics": {
            "mrr": current_metrics["mrr"],
            "hit_rate_10": current_metrics["hit_rate_10"],
        },
        "optimal_weights": optimal.to_dict(),
        "optimal_metrics": {
            "mrr": optimal_metrics["mrr"],
            "hit_rate_10": optimal_metrics["hit_rate_10"],
        },
        "delta_mrr": delta_mrr,
        "method": args.method,
        "signals": signal_label,
        "total_cases": len(cases),
    }

    if cv_results:
        output["cross_validation"] = {
            "mean_mrr": cv_results["mean_mrr"],
            "std_mrr": cv_results["std_mrr"],
            "fold_mrrs": cv_results["fold_mrrs"],
        }

    if domain_results:
        output["per_domain"] = {
            domain: {
                "weights": w.to_dict(),
                "mrr": m["mrr"],
                "hit_rate_10": m["hit_rate_10"],
            }
            for domain, (w, m) in domain_results.items()
        }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
            f.write("\n")
        print(f"\nResults saved: {args.output}")

    # Deployment guidance
    print("\n--- Deployment ---")
    print(f"Optimal weights: {optimal}")
    print(f"\nTo deploy globally, update these locations:")
    print(f"  1. packages/cli/src/research_kb_cli/_shared.py:88-89 (graph_weight, citation_weight)")
    print(f"  2. packages/cli/src/research_kb_cli/_shared.py:74-79 (get_context_weights FTS/vector)")
    print(f"  3. packages/mcp-server/src/research_kb_mcp/tools/search.py:20-34 (MCP tool defaults)")
    print(f"  4. packages/storage/src/research_kb_storage/domain_store.py:105-108 (DB defaults)")

    if domain_results:
        print(f"\nPer-domain weights (deploy via domain_store UPDATE):")
        for domain, (w, _) in sorted(domain_results.items()):
            print(f"  UPDATE domains SET {w.to_dict()} WHERE domain_id = '{domain}';")


if __name__ == "__main__":
    asyncio.run(main())
