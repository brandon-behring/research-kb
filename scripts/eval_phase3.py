#!/usr/bin/env python3
"""Phase 3 Evaluation - Benchmark enhanced retrieval against baseline.

Compares search quality across configurations:
- Baseline: FTS + vector only
- Graph-boosted: FTS + vector + graph signals
- With reranking: + cross-encoder reranking
- With expansion: + query expansion (synonyms + graph)
- Full pipeline: All enhancements combined

Metrics:
- P@K (Precision at K): Fraction of top-K results that are relevant
- MRR (Mean Reciprocal Rank): 1/rank of first relevant result
- Latency: P50 and P99 query times

Usage:
    python scripts/eval_phase3.py                    # Run evaluation
    python scripts/eval_phase3.py --generate-queries # Generate candidate queries
    python scripts/eval_phase3.py --quick            # Quick smoke test (5 queries)

Master Plan Reference: Phase 3D Evaluation
"""

import argparse
import asyncio
import json
import statistics
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
from research_kb_storage import (
    ConceptStore,
    DatabaseConfig,
    SearchQuery,
    get_connection_pool,
    search_hybrid,
    search_hybrid_v2,
    search_with_rerank,
    search_with_expansion,
)


@dataclass
class QueryWithRelevance:
    """A test query with known relevant concepts/keywords.

    Attributes:
        query: The query text
        relevant_concepts: Concept names that should appear in results
        relevant_keywords: Keywords that indicate relevance
        expected_sources: Source titles that should be highly ranked
    """

    query: str
    relevant_concepts: list[str] = field(default_factory=list)
    relevant_keywords: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Evaluation results for a single configuration."""

    config_name: str
    precision_at_5: float
    precision_at_10: float
    mrr: float
    latency_p50_ms: float
    latency_p99_ms: float
    query_count: int


# Test queries covering major causal inference topics
# These are manually curated with known relevant concepts
TEST_QUERIES = [
    QueryWithRelevance(
        query="instrumental variables for causal identification",
        relevant_concepts=["instrumental variables", "iv", "2sls", "endogeneity"],
        relevant_keywords=["instrument", "endogenous", "exclusion restriction", "relevance"],
    ),
    QueryWithRelevance(
        query="difference in differences parallel trends",
        relevant_concepts=["difference-in-differences", "did", "parallel trends"],
        relevant_keywords=["treatment", "control", "pre-trend", "common trends"],
    ),
    QueryWithRelevance(
        query="regression discontinuity design",
        relevant_concepts=["rdd", "regression discontinuity", "sharp", "fuzzy"],
        relevant_keywords=["cutoff", "threshold", "bandwidth", "running variable"],
    ),
    QueryWithRelevance(
        query="propensity score matching",
        relevant_concepts=["propensity score", "matching", "psm"],
        relevant_keywords=["treatment", "control", "balance", "covariate"],
    ),
    QueryWithRelevance(
        query="double machine learning causal inference",
        relevant_concepts=["dml", "double machine learning", "cross-fitting"],
        relevant_keywords=["nuisance", "orthogonal", "debiased", "sample splitting"],
    ),
    QueryWithRelevance(
        query="synthetic control method",
        relevant_concepts=["synthetic control", "scm", "abadie"],
        relevant_keywords=["donor", "weights", "pre-treatment", "counterfactual"],
    ),
    QueryWithRelevance(
        query="average treatment effect estimation",
        relevant_concepts=["ate", "average treatment effect", "causal effect"],
        relevant_keywords=["treatment", "outcome", "potential outcomes"],
    ),
    QueryWithRelevance(
        query="conditional average treatment effect heterogeneity",
        relevant_concepts=["cate", "heterogeneous treatment", "hte"],
        relevant_keywords=["subgroup", "heterogeneity", "conditional"],
    ),
    QueryWithRelevance(
        query="backdoor criterion causal graphs",
        relevant_concepts=["backdoor", "dag", "d-separation"],
        relevant_keywords=["confounding", "adjustment", "path", "blocking"],
    ),
    QueryWithRelevance(
        query="selection bias and confounding",
        relevant_concepts=["selection bias", "confounding", "confounder"],
        relevant_keywords=["bias", "unobserved", "omitted variable"],
    ),
    QueryWithRelevance(
        query="local average treatment effect compliers",
        relevant_concepts=["late", "local average treatment effect", "complier"],
        relevant_keywords=["instrument", "compliance", "monotonicity"],
    ),
    QueryWithRelevance(
        query="inverse probability weighting",
        relevant_concepts=["iptw", "inverse probability", "weighting"],
        relevant_keywords=["propensity", "weight", "reweight"],
    ),
    QueryWithRelevance(
        query="doubly robust estimation",
        relevant_concepts=["doubly robust", "aipw", "augmented ipw"],
        relevant_keywords=["robust", "consistent", "model misspecification"],
    ),
    QueryWithRelevance(
        query="fixed effects panel data",
        relevant_concepts=["fixed effects", "panel", "twfe"],
        relevant_keywords=["individual", "time", "within", "demeaning"],
    ),
    QueryWithRelevance(
        query="event study staggered adoption",
        relevant_concepts=["event study", "staggered", "did"],
        relevant_keywords=["pre-trend", "dynamic", "treatment timing"],
    ),
    QueryWithRelevance(
        query="causal forest heterogeneous effects",
        relevant_concepts=["causal forest", "grf", "random forest"],
        relevant_keywords=["tree", "splitting", "honest", "heterogeneity"],
    ),
    QueryWithRelevance(
        query="sensitivity analysis unmeasured confounding",
        relevant_concepts=["sensitivity analysis", "rosenbaum bounds"],
        relevant_keywords=["robustness", "unmeasured", "hidden bias"],
    ),
    QueryWithRelevance(
        query="external validity generalization",
        relevant_concepts=["external validity", "generalizability", "transportability"],
        relevant_keywords=["population", "target", "sample"],
    ),
    QueryWithRelevance(
        query="mediation analysis causal mechanisms",
        relevant_concepts=["mediation", "mediator", "mechanism"],
        relevant_keywords=["direct effect", "indirect effect", "pathway"],
    ),
    QueryWithRelevance(
        query="potential outcomes framework rubin",
        relevant_concepts=["potential outcomes", "rubin", "counterfactual"],
        relevant_keywords=["treatment", "control", "neyman"],
    ),
]


def is_relevant(result, query: QueryWithRelevance) -> bool:
    """Check if a search result is relevant to the query.

    Uses concept and keyword matching as proxy for relevance.
    """
    content_lower = result.chunk.content.lower()
    title_lower = (result.source.title or "").lower()

    # Check for relevant concepts
    for concept in query.relevant_concepts:
        if concept.lower() in content_lower or concept.lower() in title_lower:
            return True

    # Check for relevant keywords (require at least 2)
    keyword_matches = sum(
        1 for kw in query.relevant_keywords if kw.lower() in content_lower
    )
    if keyword_matches >= 2:
        return True

    # Check for expected sources
    for source in query.expected_sources:
        if source.lower() in title_lower:
            return True

    return False


def compute_precision_at_k(results: list, query: QueryWithRelevance, k: int) -> float:
    """Compute precision@K for results."""
    if not results or k <= 0:
        return 0.0

    top_k = results[:k]
    relevant_count = sum(1 for r in top_k if is_relevant(r, query))
    return relevant_count / k


def compute_mrr(results: list, query: QueryWithRelevance) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, result in enumerate(results):
        if is_relevant(result, query):
            return 1.0 / (i + 1)
    return 0.0


async def run_search_config(
    query_text: str,
    embedding: list[float],
    config: str,
    limit: int = 10,
) -> tuple[list, float]:
    """Run search with specified configuration.

    Returns (results, latency_ms)
    """
    start = time.perf_counter()

    search_query = SearchQuery(
        text=query_text,
        embedding=embedding,
        fts_weight=0.3,
        vector_weight=0.7,
        limit=limit,
    )

    if config == "baseline":
        # FTS + vector only
        results = await search_hybrid(search_query)

    elif config == "graph":
        # FTS + vector + graph
        search_query.use_graph = True
        search_query.graph_weight = 0.2
        # Renormalize weights
        total = search_query.fts_weight + search_query.vector_weight + search_query.graph_weight
        search_query.fts_weight /= total
        search_query.vector_weight /= total
        search_query.graph_weight /= total
        results = await search_hybrid_v2(search_query)

    elif config == "rerank":
        # FTS + vector + reranking
        results = await search_with_rerank(search_query, rerank_top_k=limit)

    elif config == "graph_rerank":
        # FTS + vector + graph + reranking
        search_query.use_graph = True
        search_query.graph_weight = 0.2
        total = search_query.fts_weight + search_query.vector_weight + search_query.graph_weight
        search_query.fts_weight /= total
        search_query.vector_weight /= total
        search_query.graph_weight /= total
        results = await search_with_rerank(search_query, rerank_top_k=limit)

    elif config == "expand":
        # FTS + vector + expansion (no rerank)
        results, _ = await search_with_expansion(
            search_query,
            use_synonyms=True,
            use_graph_expansion=False,
            use_llm_expansion=False,
            use_rerank=False,
        )

    elif config == "full":
        # All enhancements: graph + expansion + reranking
        search_query.use_graph = True
        search_query.graph_weight = 0.2
        total = search_query.fts_weight + search_query.vector_weight + search_query.graph_weight
        search_query.fts_weight /= total
        search_query.vector_weight /= total
        search_query.graph_weight /= total

        results, _ = await search_with_expansion(
            search_query,
            use_synonyms=True,
            use_graph_expansion=True,
            use_llm_expansion=False,
            use_rerank=True,
            rerank_top_k=limit,
        )

    else:
        raise ValueError(f"Unknown config: {config}")

    latency_ms = (time.perf_counter() - start) * 1000
    return results, latency_ms


async def evaluate_config(
    queries: list[QueryWithRelevance],
    embed_client: EmbeddingClient,
    config: str,
    limit: int = 10,
) -> EvalResult:
    """Evaluate a search configuration across all queries."""
    p5_scores = []
    p10_scores = []
    mrr_scores = []
    latencies = []

    for query in queries:
        # Generate embedding (uses BGE query instruction prefix)
        embedding = embed_client.embed_query(query.query)

        # Run search
        results, latency_ms = await run_search_config(
            query.query, embedding, config, limit=limit
        )
        latencies.append(latency_ms)

        # Compute metrics
        p5 = compute_precision_at_k(results, query, 5)
        p10 = compute_precision_at_k(results, query, 10)
        mrr = compute_mrr(results, query)

        p5_scores.append(p5)
        p10_scores.append(p10)
        mrr_scores.append(mrr)

    return EvalResult(
        config_name=config,
        precision_at_5=statistics.mean(p5_scores) if p5_scores else 0.0,
        precision_at_10=statistics.mean(p10_scores) if p10_scores else 0.0,
        mrr=statistics.mean(mrr_scores) if mrr_scores else 0.0,
        latency_p50_ms=statistics.median(latencies) if latencies else 0.0,
        latency_p99_ms=(
            sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[0] if latencies else 0.0
        ),
        query_count=len(queries),
    )


def print_results_table(results: list[EvalResult]) -> None:
    """Print results as formatted table."""
    print("\n" + "=" * 80)
    print("PHASE 3 EVALUATION RESULTS")
    print("=" * 80)

    # Header
    print(f"{'Configuration':<20} {'P@5':<8} {'P@10':<8} {'MRR':<8} {'P50(ms)':<10} {'P99(ms)':<10}")
    print("-" * 80)

    # Results
    for r in results:
        print(
            f"{r.config_name:<20} {r.precision_at_5:<8.3f} {r.precision_at_10:<8.3f} "
            f"{r.mrr:<8.3f} {r.latency_p50_ms:<10.1f} {r.latency_p99_ms:<10.1f}"
        )

    print("-" * 80)
    print(f"Queries evaluated: {results[0].query_count if results else 0}")
    print("=" * 80)

    # Compute improvements
    if len(results) >= 2:
        baseline = results[0]
        best = max(results, key=lambda r: r.precision_at_5)

        if baseline.precision_at_5 > 0:
            p5_improvement = (best.precision_at_5 - baseline.precision_at_5) / baseline.precision_at_5 * 100
            print(f"\nBest P@5 improvement over baseline: +{p5_improvement:.1f}% ({best.config_name})")

        if baseline.mrr > 0:
            mrr_improvement = (best.mrr - baseline.mrr) / baseline.mrr * 100
            print(f"Best MRR improvement over baseline: +{mrr_improvement:.1f}% ({best.config_name})")


async def generate_candidate_queries() -> None:
    """Generate candidate queries from concept names.

    Outputs to fixtures/eval/queries_candidates.yaml for manual labeling.
    """
    config = DatabaseConfig()
    await get_connection_pool(config)

    concepts = await ConceptStore.list_all(limit=200)

    candidates = []
    for concept in concepts:
        if concept.confidence_score and concept.confidence_score >= 0.7:
            candidates.append({
                "query": concept.canonical_name or concept.name,
                "concept_type": concept.concept_type.value,
                "confidence": concept.confidence_score,
            })

    # Sort by confidence
    candidates.sort(key=lambda x: x["confidence"], reverse=True)

    # Output
    output_dir = Path(__file__).parent.parent / "fixtures" / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "queries_candidates.json"

    with open(output_path, "w") as f:
        json.dump(candidates[:50], f, indent=2)

    print(f"Generated {len(candidates[:50])} candidate queries to: {output_path}")
    print("Review and add relevance labels for manual evaluation.")


async def main() -> None:
    """Run Phase 3 evaluation."""
    parser = argparse.ArgumentParser(description="Phase 3 Evaluation")
    parser.add_argument(
        "--generate-queries",
        action="store_true",
        help="Generate candidate queries from concepts",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test with 5 queries",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["baseline", "graph", "rerank", "graph_rerank", "expand", "full"],
        help="Configurations to evaluate",
    )
    args = parser.parse_args()

    # Initialize database
    config = DatabaseConfig()
    await get_connection_pool(config)

    if args.generate_queries:
        await generate_candidate_queries()
        return

    # Select queries
    queries = TEST_QUERIES[:5] if args.quick else TEST_QUERIES

    print(f"Evaluating {len(queries)} queries across {len(args.configs)} configurations...")

    # Initialize embedding client
    embed_client = EmbeddingClient()

    # Check if server is available
    try:
        _ = embed_client.embed("test")
    except Exception as e:
        print(f"Error: Embedding server not available: {e}")
        print("Start with: python -m research_kb_pdf.embed_server")
        sys.exit(1)

    # Run evaluations
    results = []
    for cfg in args.configs:
        print(f"  Evaluating: {cfg}...")
        try:
            result = await evaluate_config(queries, embed_client, cfg)
            results.append(result)
        except Exception as e:
            print(f"    Warning: {cfg} failed: {e}")

    # Print results
    print_results_table(results)

    # Save detailed results
    output_dir = Path(__file__).parent.parent / "fixtures" / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase3_results.json"

    with open(output_path, "w") as f:
        json.dump(
            [
                {
                    "config": r.config_name,
                    "p5": r.precision_at_5,
                    "p10": r.precision_at_10,
                    "mrr": r.mrr,
                    "latency_p50": r.latency_p50_ms,
                    "latency_p99": r.latency_p99_ms,
                }
                for r in results
            ],
            f,
            indent=2,
        )

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
