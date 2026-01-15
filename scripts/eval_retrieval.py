#!/usr/bin/env python3
"""Evaluate retrieval quality for research-kb.

Master Plan Reference: Lines 596-601, 1382-1383

Metrics:
- Hit Rate@K: % of queries where expected result appears in top K (target: 90%)
- MRR (Mean Reciprocal Rank): average of 1/rank for successful queries
  - MRR=1.0 means always rank 1, MRR=0.5 means average rank 2
- NDCG@K (Normalized Discounted Cumulative Gain): standard ranking metric
  - Accounts for position of relevant results (earlier = better)
  - Range 0-1, where 1.0 is perfect ranking
- Concept Recall: % of expected concepts found in search results (target: 70%)
  - Measures whether relevant concepts are surfaced alongside sources

Note: True Precision@K requires graded relevance labels for ALL retrieved
results. Since we only label expected sources, we use Hit Rate, MRR, and NDCG.

Usage:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --verbose
    python scripts/eval_retrieval.py --tag core
    python scripts/eval_retrieval.py --output metrics.json  # For CI
"""

import asyncio
import json
import re
import sys
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
from research_kb_storage import DatabaseConfig, SearchQuery, get_connection_pool, search_hybrid


@dataclass
class TestCase:
    """A single retrieval test case."""

    query: str
    expected_source_pattern: str
    expected_in_top_k: int
    expected_page_range: Optional[tuple[int, int]] = None
    expect_mixed_sources: bool = False
    expected_concepts: list[str] = field(default_factory=list)  # Phase 2: concept recall
    relevance_grade: int = 3  # 0=irrelevant, 1=related, 2=relevant, 3=exact
    tags: list[str] = None
    notes: Optional[str] = None


@dataclass
class TestResult:
    """Result of running a test case."""

    test_case: TestCase
    passed: bool
    matched_rank: Optional[int] = None
    matched_source: Optional[str] = None
    matched_page: Optional[int] = None
    concept_recall: Optional[float] = None  # Phase 2: concept recall score
    found_concepts: list[str] = field(default_factory=list)  # Concepts found in results
    error: Optional[str] = None


async def get_concepts_for_chunks(chunk_ids: list[UUID]) -> set[str]:
    """Get all concept names linked to a list of chunks.

    Args:
        chunk_ids: List of chunk UUIDs

    Returns:
        Set of concept names (lowercase for matching)
    """
    if not chunk_ids:
        return set()

    pool = await get_connection_pool()
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


def load_test_cases(yaml_path: Path, tag_filter: Optional[str] = None) -> list[TestCase]:
    """Load test cases from YAML file.

    Args:
        yaml_path: Path to test cases YAML
        tag_filter: Optional tag to filter by

    Returns:
        List of TestCase objects
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    cases = []
    for tc in data.get("test_cases", []):
        tags = tc.get("tags", [])

        # Filter by tag if specified
        if tag_filter and tag_filter not in tags:
            continue

        cases.append(TestCase(
            query=tc["query"],
            expected_source_pattern=tc["expected_source_pattern"],
            expected_in_top_k=tc.get("expected_in_top_k", 5),
            expected_page_range=tuple(tc["expected_page_range"]) if tc.get("expected_page_range") else None,
            expect_mixed_sources=tc.get("expect_mixed_sources", False),
            expected_concepts=tc.get("expected_concepts", []),  # Phase 2
            relevance_grade=tc.get("relevance_grade", 3),  # Phase 2
            tags=tags,
            notes=tc.get("notes"),
        ))

    return cases


async def run_test_case(
    test_case: TestCase,
    embed_client: EmbeddingClient,
    scoring_method: str = "weighted",
) -> TestResult:
    """Run a single test case.

    Args:
        test_case: The test case to run
        embed_client: Embedding client for query embedding
        scoring_method: Score combination method - "weighted" or "rrf"

    Returns:
        TestResult with pass/fail and details
    """
    try:
        # Generate query embedding (uses BGE query instruction prefix)
        query_embedding = embed_client.embed_query(test_case.query)

        # Execute search
        query = SearchQuery(
            text=test_case.query,
            embedding=query_embedding,
            fts_weight=0.3,
            vector_weight=0.7,
            limit=test_case.expected_in_top_k,
            scoring_method=scoring_method,
        )

        results = await search_hybrid(query)

        if not results:
            return TestResult(
                test_case=test_case,
                passed=False,
                error="No results returned",
            )

        # Phase 2: Compute concept recall if expected_concepts specified
        concept_recall = None
        found_concepts = []
        if test_case.expected_concepts:
            chunk_ids = [r.chunk.id for r in results]
            all_concepts = await get_concepts_for_chunks(chunk_ids)
            found_concepts = list(all_concepts)

            # Match expected concepts (case-insensitive, partial match)
            expected_lower = {c.lower() for c in test_case.expected_concepts}
            matched = 0
            for expected in expected_lower:
                # Check for exact or partial match
                if any(expected in found or found in expected for found in all_concepts):
                    matched += 1

            concept_recall = matched / len(expected_lower) if expected_lower else 1.0

        # Check if expected source appears in top-K
        pattern = re.compile(test_case.expected_source_pattern, re.IGNORECASE)

        for result in results:
            source_title = result.source.title.lower()

            if pattern.search(source_title):
                # Check page range if specified
                page_valid = True
                if test_case.expected_page_range:
                    page = result.chunk.page_start or 0
                    min_page, max_page = test_case.expected_page_range
                    page_valid = min_page <= page <= max_page

                return TestResult(
                    test_case=test_case,
                    passed=page_valid,
                    matched_rank=result.rank,
                    matched_source=result.source.title,
                    matched_page=result.chunk.page_start,
                    concept_recall=concept_recall,
                    found_concepts=found_concepts[:10],  # Limit for display
                    error=None if page_valid else f"Page {result.chunk.page_start} outside expected range {test_case.expected_page_range}",
                )

        # No match found
        top_sources = [r.source.title for r in results[:3]]
        return TestResult(
            test_case=test_case,
            passed=False,
            concept_recall=concept_recall,
            found_concepts=found_concepts[:10],
            error=f"Pattern '{test_case.expected_source_pattern}' not found. Top sources: {top_sources}",
        )

    except Exception as e:
        return TestResult(
            test_case=test_case,
            passed=False,
            error=str(e),
        )


async def run_eval(
    yaml_path: Path,
    tag_filter: Optional[str] = None,
    verbose: bool = False,
    scoring_method: str = "weighted",
) -> tuple[list[TestResult], dict]:
    """Run full evaluation suite.

    Args:
        yaml_path: Path to test cases YAML
        tag_filter: Optional tag to filter by
        verbose: Print detailed output
        scoring_method: Score combination method - "weighted" or "rrf"

    Returns:
        Tuple of (results list, metrics dict)
    """
    # Load test cases
    test_cases = load_test_cases(yaml_path, tag_filter)

    if not test_cases:
        print("No test cases found!")
        return [], {}

    print(f"Running {len(test_cases)} test cases...")

    # Initialize
    config = DatabaseConfig()
    await get_connection_pool(config)
    embed_client = EmbeddingClient()

    # Run tests
    results = []
    for tc in test_cases:
        if verbose:
            print(f"  Testing: {tc.query}")

        result = await run_test_case(tc, embed_client, scoring_method=scoring_method)
        results.append(result)

        if verbose:
            status = "✓" if result.passed else "✗"
            if result.passed:
                print(f"    {status} Found at rank {result.matched_rank}: {result.matched_source}")
            else:
                print(f"    {status} {result.error}")

    # Calculate metrics
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    # Mean Reciprocal Rank (MRR): average of 1/rank for successful queries
    # Measures ranking quality - higher is better, 1.0 means always rank 1
    reciprocal_ranks = [
        1.0 / r.matched_rank
        for r in results
        if r.passed and r.matched_rank
    ]
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    # NDCG@K (Normalized Discounted Cumulative Gain)
    # For binary relevance with 1 relevant doc: NDCG = 1/log2(rank+1) if found, 0 otherwise
    # Uses sklearn for standard computation
    def compute_ndcg_at_k(results: list[TestResult], k: int) -> float:
        """Compute NDCG@K for binary relevance with single relevant item per query."""
        ndcg_scores = []
        for r in results:
            if r.passed and r.matched_rank and r.matched_rank <= k:
                # Binary relevance: 1 at matched position, 0 elsewhere
                y_true = np.zeros(k)
                y_true[r.matched_rank - 1] = 1.0
                # Predicted scores: decreasing by rank (position 1 = highest score)
                y_score = np.arange(k, 0, -1, dtype=float)
                ndcg_scores.append(ndcg_score([y_true], [y_score], k=k))
            else:
                # Not found in top K → NDCG = 0
                ndcg_scores.append(0.0)
        return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    ndcg_5 = compute_ndcg_at_k(results, 5)
    ndcg_10 = compute_ndcg_at_k(results, 10)

    # Phase 2: Compute average concept recall
    concept_recalls = [
        r.concept_recall for r in results
        if r.concept_recall is not None
    ]
    avg_concept_recall = (
        sum(concept_recalls) / len(concept_recalls)
        if concept_recalls else None
    )

    metrics = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        # Hit Rate@K: % of queries where expected result appears in top K
        # (Previously mislabeled as "precision_at_k")
        "hit_rate_at_k": passed / total if total > 0 else 0,
        # Mean Reciprocal Rank: average of 1/rank for successful queries
        # MRR=1.0 means always rank 1, MRR=0.5 means avg rank 2
        "mrr": mrr,
        # NDCG: accounts for position (earlier = better), range 0-1
        "ndcg_5": ndcg_5,
        "ndcg_10": ndcg_10,
        # Phase 2: Concept Recall - % of expected concepts found
        "concept_recall": avg_concept_recall,
        "concept_recall_tests": len(concept_recalls),
    }

    return results, metrics


def print_summary(results: list[TestResult], metrics: dict):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal tests: {metrics['total']}")
    print(f"Passed: {metrics['passed']}")
    print(f"Failed: {metrics['failed']}")

    # Metrics
    print("\nMetrics:")
    target_hit_rate = 0.90
    actual_hit_rate = metrics['hit_rate_at_k']
    status = "✓" if actual_hit_rate >= target_hit_rate else "✗"
    print(f"  {status} Hit Rate@K: {actual_hit_rate:.1%} (target: ≥{target_hit_rate:.0%})")
    print(f"      (% of queries where expected result appears in top K)")

    mrr = metrics['mrr']
    mrr_status = "✓" if mrr >= 0.5 else "✗"  # MRR >= 0.5 means avg rank ≤ 2
    print(f"  {mrr_status} MRR: {mrr:.3f}")
    print(f"      (Mean Reciprocal Rank: 1.0=always rank 1, 0.5=avg rank 2)")

    # NDCG metrics
    ndcg_5 = metrics.get('ndcg_5', 0.0)
    ndcg_10 = metrics.get('ndcg_10', 0.0)
    ndcg_status = "✓" if ndcg_5 >= 0.7 else "✗"  # NDCG ≥ 0.7 is good ranking
    print(f"  {ndcg_status} NDCG@5: {ndcg_5:.3f}")
    print(f"    NDCG@10: {ndcg_10:.3f}")
    print(f"      (Position-weighted: 1.0=perfect, 0.5=avg rank 3)")

    # Phase 2: Concept Recall metrics
    concept_recall = metrics.get('concept_recall')
    concept_tests = metrics.get('concept_recall_tests', 0)
    if concept_recall is not None:
        target_concept_recall = 0.70
        cr_status = "✓" if concept_recall >= target_concept_recall else "✗"
        print(f"  {cr_status} Concept Recall: {concept_recall:.1%} (target: ≥{target_concept_recall:.0%})")
        print(f"      ({concept_tests} tests with expected_concepts)")
    else:
        print(f"  ⊘ Concept Recall: N/A (no tests have expected_concepts defined)")

    # List failures
    failures = [r for r in results if not r.passed]
    if failures:
        print("\nFailed Tests:")
        for r in failures:
            print(f"  ✗ {r.test_case.query}")
            print(f"    {r.error}")

    print()


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--tag", "-t", help="Filter by tag (e.g., 'core')")
    parser.add_argument("--output", "-o", help="Output JSON file for CI (e.g., metrics.json)")
    parser.add_argument(
        "--scoring", "-s",
        choices=["weighted", "rrf"],
        default="weighted",
        help="Scoring method: 'weighted' (default) or 'rrf' (Reciprocal Rank Fusion)"
    )
    args = parser.parse_args()

    yaml_path = Path(__file__).parent.parent / "fixtures" / "eval" / "retrieval_test_cases.yaml"

    if not yaml_path.exists():
        print(f"Error: Test cases not found at {yaml_path}")
        sys.exit(1)

    print(f"Scoring method: {args.scoring}")
    results, metrics = await run_eval(
        yaml_path,
        tag_filter=args.tag,
        verbose=args.verbose,
        scoring_method=args.scoring,
    )

    print_summary(results, metrics)

    # Phase 2: Output JSON for CI quality gate
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics written to: {output_path}")

    # Exit with error code if tests failed
    if metrics.get("failed", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
