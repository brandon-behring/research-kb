#!/usr/bin/env python3
"""Compare search configurations on golden dataset.

Phase 3.4: Systematic comparison of search configurations.

Configurations compared:
1. baseline: FTS + vector, weighted sum
2. graph: + use_graph=True
3. citations: + use_citations=True
4. full: graph + citations + rerank
5. rrf: Same as full but with RRF scoring

Usage:
    python scripts/compare_search_configs.py
    python scripts/compare_search_configs.py --configs baseline,graph,rrf
    python scripts/compare_search_configs.py --output comparison.json
    python scripts/compare_search_configs.py --format table
"""

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np
import yaml
from sklearn.metrics import ndcg_score

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_pdf import EmbeddingClient
from research_kb_storage import (
    DatabaseConfig,
    SearchQuery,
    get_connection_pool,
    search_hybrid,
    search_hybrid_v2,
)


@dataclass
class SearchConfig:
    """Configuration for a search variant."""

    name: str
    use_graph: bool = False
    use_citations: bool = False
    use_rerank: bool = False
    use_expand: bool = False
    scoring_method: str = "weighted"
    graph_weight: float = 0.2
    citation_weight: float = 0.15
    fts_weight: float = 0.3
    vector_weight: float = 0.7

    def as_query_kwargs(self) -> dict:
        """Convert config to SearchQuery kwargs."""
        return {
            "use_graph": self.use_graph,
            "graph_weight": self.graph_weight if self.use_graph else 0.0,
            "use_citations": self.use_citations,
            "citation_weight": self.citation_weight if self.use_citations else 0.0,
            "fts_weight": self.fts_weight,
            "vector_weight": self.vector_weight,
            "scoring_method": self.scoring_method,
        }


# Predefined configurations
CONFIGS = {
    "baseline": SearchConfig(
        name="baseline",
        use_graph=False,
        use_citations=False,
        use_rerank=False,
        use_expand=False,
        scoring_method="weighted",
    ),
    "graph": SearchConfig(
        name="graph",
        use_graph=True,
        use_citations=False,
        use_rerank=False,
        use_expand=False,
        scoring_method="weighted",
    ),
    "citations": SearchConfig(
        name="citations",
        use_graph=False,
        use_citations=True,
        use_rerank=False,
        use_expand=False,
        scoring_method="weighted",
    ),
    "full": SearchConfig(
        name="full",
        use_graph=True,
        use_citations=True,
        use_rerank=False,  # Rerank requires running service
        use_expand=False,  # Expansion requires running service
        scoring_method="weighted",
    ),
    "full-rrf": SearchConfig(
        name="full-rrf",
        use_graph=True,
        use_citations=True,
        use_rerank=False,
        use_expand=False,
        scoring_method="rrf",
    ),
    # Fair comparison: same features (graph+citations), different scoring
    "rrf": SearchConfig(
        name="rrf",
        use_graph=True,
        use_citations=True,
        use_rerank=False,
        use_expand=False,
        scoring_method="rrf",
    ),
}


@dataclass
class TestCase:
    """A single retrieval test case."""

    query: str
    expected_source_pattern: str
    expected_in_top_k: int
    expected_page_range: Optional[tuple[int, int]] = None
    expected_concepts: list[str] = field(default_factory=list)


@dataclass
class ConfigResult:
    """Results for a single configuration."""

    config_name: str
    hit_rate: float
    mrr: float
    ndcg_5: float
    concept_recall: Optional[float]
    avg_latency_ms: float
    p99_latency_ms: float
    passed: int
    total: int


def load_test_cases(yaml_path: Path) -> list[TestCase]:
    """Load test cases from YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    cases = []
    for tc in data.get("test_cases", []):
        cases.append(
            TestCase(
                query=tc["query"],
                expected_source_pattern=tc["expected_source_pattern"],
                expected_in_top_k=tc.get("expected_in_top_k", 5),
                expected_page_range=(
                    tuple(tc["expected_page_range"]) if tc.get("expected_page_range") else None
                ),
                expected_concepts=tc.get("expected_concepts", []),
            )
        )

    return cases


async def get_concepts_for_chunks(pool, chunk_ids: list[UUID]) -> set[str]:
    """Get all concept names linked to a list of chunks."""
    if not chunk_ids:
        return set()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT LOWER(c.name) as name
            FROM chunk_concepts cc
            JOIN concepts c ON cc.concept_id = c.id
            WHERE cc.chunk_id = ANY($1::uuid[])
            """,
            chunk_ids,
        )
        return {row["name"] for row in rows}


async def run_search(
    embed_client: EmbeddingClient,
    pool,
    query_text: str,
    config: SearchConfig,
    limit: int = 5,
):
    """Execute search with given configuration."""
    query_embedding = embed_client.embed_query(query_text)

    query = SearchQuery(
        text=query_text,
        embedding=query_embedding,
        limit=limit,
        **config.as_query_kwargs(),
    )

    # Use hybrid_v2 if graph or citations enabled, else basic hybrid
    if config.use_graph or config.use_citations:
        return await search_hybrid_v2(query)
    else:
        return await search_hybrid(query)


async def evaluate_config(
    config: SearchConfig,
    test_cases: list[TestCase],
    embed_client: EmbeddingClient,
    pool,
) -> ConfigResult:
    """Evaluate a single configuration against all test cases."""
    passed = 0
    reciprocal_ranks = []
    ndcg_scores = []
    concept_recalls = []
    latencies = []

    for tc in test_cases:
        pattern = re.compile(tc.expected_source_pattern, re.IGNORECASE)

        start_time = time.perf_counter()
        results = await run_search(
            embed_client, pool, tc.query, config, limit=tc.expected_in_top_k
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(latency_ms)

        # Check for match
        matched_rank = None
        for result in results:
            if pattern.search(result.source.title.lower()):
                matched_rank = result.rank
                break

        if matched_rank is not None:
            passed += 1
            reciprocal_ranks.append(1.0 / matched_rank)

            # NDCG with binary relevance
            y_true = np.zeros(tc.expected_in_top_k)
            y_true[matched_rank - 1] = 1.0
            y_score = np.arange(tc.expected_in_top_k, 0, -1, dtype=float)
            ndcg_scores.append(ndcg_score([y_true], [y_score], k=tc.expected_in_top_k))
        else:
            ndcg_scores.append(0.0)

        # Concept recall
        if tc.expected_concepts:
            chunk_ids = [r.chunk.id for r in results]
            all_concepts = await get_concepts_for_chunks(pool, chunk_ids)
            expected_lower = {c.lower() for c in tc.expected_concepts}
            matched = sum(
                1
                for exp in expected_lower
                if any(exp in found or found in exp for found in all_concepts)
            )
            concept_recalls.append(matched / len(expected_lower))

    total = len(test_cases)
    hit_rate = passed / total if total > 0 else 0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    ndcg_5 = float(np.mean(ndcg_scores)) if ndcg_scores else 0
    concept_recall = (
        sum(concept_recalls) / len(concept_recalls) if concept_recalls else None
    )

    return ConfigResult(
        config_name=config.name,
        hit_rate=hit_rate,
        mrr=mrr,
        ndcg_5=ndcg_5,
        concept_recall=concept_recall,
        avg_latency_ms=np.mean(latencies),
        p99_latency_ms=np.percentile(latencies, 99) if len(latencies) > 1 else latencies[0],
        passed=passed,
        total=total,
    )


def print_table(results: list[ConfigResult]):
    """Print results in table format."""
    print(f"\nConfiguration Comparison ({results[0].total} test cases)")
    print("=" * 90)
    print(
        f"{'Config':<12} | {'Hit@5':>7} | {'MRR':>7} | {'NDCG@5':>7} | {'Concept':>8} | {'p50(ms)':>8} | {'p99(ms)':>8}"
    )
    print("-" * 90)

    for r in results:
        concept_str = f"{r.concept_recall:.1%}" if r.concept_recall is not None else "N/A"
        print(
            f"{r.config_name:<12} | {r.hit_rate:>6.1%} | {r.mrr:>7.3f} | {r.ndcg_5:>7.3f} | {concept_str:>8} | {r.avg_latency_ms:>8.1f} | {r.p99_latency_ms:>8.1f}"
        )

    print("=" * 90)

    # Find winner
    best = max(results, key=lambda r: (r.hit_rate, r.mrr))
    baseline = next((r for r in results if r.config_name == "baseline"), results[0])

    if best.config_name != "baseline":
        hit_delta = (best.hit_rate - baseline.hit_rate) * 100
        mrr_delta = (best.mrr - baseline.mrr) * 100
        print(f"Winner: {best.config_name} (+{hit_delta:.1f}% Hit@5, +{mrr_delta:.1f}% MRR vs baseline)")
    else:
        print("Winner: baseline (no improvement from enhancements)")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare search configurations")
    parser.add_argument(
        "--configs",
        "-c",
        default="baseline,graph,citations,full,rrf",
        help="Comma-separated list of configs to compare",
    )
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="Output format",
    )
    args = parser.parse_args()

    yaml_path = Path(__file__).parent.parent / "fixtures" / "eval" / "retrieval_test_cases.yaml"

    if not yaml_path.exists():
        print(f"Error: Test cases not found at {yaml_path}")
        sys.exit(1)

    # Parse configs
    config_names = [c.strip() for c in args.configs.split(",")]
    configs = []
    for name in config_names:
        if name not in CONFIGS:
            print(f"Error: Unknown config '{name}'. Available: {list(CONFIGS.keys())}")
            sys.exit(1)
        configs.append(CONFIGS[name])

    # Initialize
    test_cases = load_test_cases(yaml_path)
    print(f"Loaded {len(test_cases)} test cases")
    print(f"Comparing configs: {config_names}")

    config = DatabaseConfig()
    pool = await get_connection_pool(config)
    embed_client = EmbeddingClient()

    # Evaluate each config
    results = []
    for cfg in configs:
        print(f"  Evaluating {cfg.name}...")
        result = await evaluate_config(cfg, test_cases, embed_client, pool)
        results.append(result)

    # Output
    if args.format == "table" or not args.output:
        print_table(results)

    if args.output:
        output_data = {
            "test_count": results[0].total if results else 0,
            "configs": {
                r.config_name: {
                    "hit_rate": r.hit_rate,
                    "mrr": r.mrr,
                    "ndcg_5": r.ndcg_5,
                    "concept_recall": r.concept_recall,
                    "avg_latency_ms": r.avg_latency_ms,
                    "p99_latency_ms": r.p99_latency_ms,
                    "passed": r.passed,
                    "total": r.total,
                }
                for r in results
            },
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
