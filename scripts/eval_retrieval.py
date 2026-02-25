#!/usr/bin/env python3
"""Evaluate retrieval quality for research-kb.

See docs/INDEX.md for architecture overview.

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
from research_kb_storage import (
    DatabaseConfig,
    SearchQuery,
    get_connection_pool,
    search_hybrid,
)


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
    domain: Optional[str] = None  # Explicit domain for per-domain breakdown
    difficulty: str = "medium"  # easy|medium|hard for per-difficulty breakdown


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
    source_matched_rank: Optional[int] = None  # Source-level match (lenient metric)


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

        cases.append(
            TestCase(
                query=tc["query"],
                expected_source_pattern=tc["expected_source_pattern"],
                expected_in_top_k=tc.get("expected_in_top_k", 5),
                expected_page_range=(
                    tuple(tc["expected_page_range"]) if tc.get("expected_page_range") else None
                ),
                expect_mixed_sources=tc.get("expect_mixed_sources", False),
                expected_concepts=tc.get("expected_concepts", []),  # Phase 2
                relevance_grade=tc.get("relevance_grade", 3),  # Phase 2
                tags=tags,
                notes=tc.get("notes"),
                domain=tc.get("domain"),
                difficulty=tc.get("difficulty", "medium"),
            )
        )

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
                    error=(
                        None
                        if page_valid
                        else f"Page {result.chunk.page_start} outside expected range {test_case.expected_page_range}"
                    ),
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

    # Calculate metrics using shared function
    metrics = compute_metrics_for_results(results)

    return results, metrics


def compute_metrics_for_results(results: list[TestResult]) -> dict:
    """Compute metrics for a set of test results.

    Reports both chunk-level (strict) and source-level (lenient) hit rates.

    Args:
        results: List of TestResult objects

    Returns:
        Dict of computed metrics
    """
    if not results:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "hit_rate_at_k": 0.0,
            "source_hit_rate_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_5": 0.0,
            "ndcg_10": 0.0,
            "concept_recall": None,
            "concept_recall_tests": 0,
        }

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    # Source-level hits (lenient): chunk match OR source match
    source_hits = sum(1 for r in results if r.passed or r.source_matched_rank is not None)

    reciprocal_ranks = [1.0 / r.matched_rank for r in results if r.passed and r.matched_rank]
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    def compute_ndcg_at_k(results: list[TestResult], k: int) -> float:
        ndcg_scores = []
        for r in results:
            if r.passed and r.matched_rank and r.matched_rank <= k:
                y_true = np.zeros(k)
                y_true[r.matched_rank - 1] = 1.0
                y_score = np.arange(k, 0, -1, dtype=float)
                ndcg_scores.append(ndcg_score([y_true], [y_score], k=k))
            else:
                ndcg_scores.append(0.0)
        return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    concept_recalls = [r.concept_recall for r in results if r.concept_recall is not None]
    avg_concept_recall = sum(concept_recalls) / len(concept_recalls) if concept_recalls else None

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "hit_rate_at_k": passed / total if total > 0 else 0,
        "source_hit_rate_at_k": source_hits / total if total > 0 else 0,
        "mrr": mrr,
        "ndcg_5": compute_ndcg_at_k(results, 5),
        "ndcg_10": compute_ndcg_at_k(results, 10),
        "concept_recall": avg_concept_recall,
        "concept_recall_tests": len(concept_recalls),
    }


def _print_metrics_block(metrics: dict, indent: str = "  "):
    """Print a single metrics block (used by both summary and per-domain).

    Args:
        metrics: Dict of computed metrics
        indent: Indentation prefix
    """
    target_hit_rate = 0.90
    actual_hit_rate = metrics["hit_rate_at_k"]
    status = "+" if actual_hit_rate >= target_hit_rate else "-"
    print(
        f"{indent}{status} Chunk Hit Rate@K: {actual_hit_rate:.1%} (target: >={target_hit_rate:.0%})"
    )

    source_hit_rate = metrics.get("source_hit_rate_at_k")
    if source_hit_rate is not None:
        src_status = "+" if source_hit_rate >= target_hit_rate else "-"
        print(f"{indent}{src_status} Source Hit Rate@K: {source_hit_rate:.1%} (lenient)")

    mrr = metrics["mrr"]
    mrr_status = "+" if mrr >= 0.5 else "-"
    print(f"{indent}{mrr_status} MRR: {mrr:.3f}")

    ndcg_5 = metrics.get("ndcg_5", 0.0)
    ndcg_10 = metrics.get("ndcg_10", 0.0)
    ndcg_status = "+" if ndcg_5 >= 0.7 else "-"
    print(f"{indent}{ndcg_status} NDCG@5: {ndcg_5:.3f}  NDCG@10: {ndcg_10:.3f}")

    concept_recall = metrics.get("concept_recall")
    concept_tests = metrics.get("concept_recall_tests", 0)
    if concept_recall is not None:
        target_cr = 0.70
        cr_status = "+" if concept_recall >= target_cr else "-"
        print(f"{indent}{cr_status} Concept Recall: {concept_recall:.1%} ({concept_tests} tests)")


def print_per_domain(results: list[TestResult]):
    """Print per-domain evaluation breakdown.

    Groups results by explicit domain field, falling back to tag scan
    for backward compatibility with golden dataset entries.

    Args:
        results: List of TestResult objects
    """
    # Group by explicit domain field (primary), then tag scan (fallback)
    domain_results = {}
    for r in results:
        domain = r.test_case.domain
        if domain:
            domain_results.setdefault(domain, []).append(r)
        else:
            # Backward compat: scan tags for domain-like entries
            tags = r.test_case.tags or []
            for tag in tags:
                if tag in (
                    "causal_inference",
                    "econometrics",
                    "time_series",
                    "rag_llm",
                    "interview_prep",
                    "deep_learning",
                    "software_engineering",
                    "machine_learning",
                    "mathematics",
                    "statistics",
                    "finance",
                    "ml_engineering",
                    "data_science",
                    "algorithms",
                    "functional_programming",
                    "fitness",
                    "forecasting",
                    "sql",
                    "recommender_systems",
                    "adtech",
                ):
                    domain_results.setdefault(tag, []).append(r)
                    break  # Only assign to first matching domain tag

    print("\n" + "=" * 60)
    print("PER-DOMAIN BREAKDOWN")
    print("=" * 60)

    # For each domain, compute metrics
    domain_metrics = {}
    for domain in sorted(domain_results.keys()):
        dr = domain_results[domain]
        if dr:
            metrics = compute_metrics_for_results(dr)
            domain_metrics[domain] = metrics

            print(f"\n  {domain} ({metrics['total']} tests):")
            _print_metrics_block(metrics, indent="    ")

    # Summary table
    print("\n" + "-" * 70)
    print(f"  {'Domain':<25} {'Tests':>5} {'Chunk%':>7} {'Source%':>8} {'MRR':>6} {'NDCG@5':>7}")
    print("  " + "-" * 65)
    for domain in sorted(domain_metrics.keys()):
        m = domain_metrics[domain]
        src_hit = m.get("source_hit_rate_at_k", m["hit_rate_at_k"])
        print(
            f"  {domain:<25} {m['total']:>5} {m['hit_rate_at_k']:>6.0%} {src_hit:>7.0%} {m['mrr']:>6.3f} {m['ndcg_5']:>7.3f}"
        )
    print()

    return domain_metrics


def print_per_difficulty(results: list[TestResult]):
    """Print per-difficulty evaluation breakdown.

    Groups results by explicit difficulty field (easy/medium/hard).

    Args:
        results: List of TestResult objects
    """
    difficulty_levels = ["easy", "medium", "hard"]

    print("\n" + "=" * 60)
    print("PER-DIFFICULTY BREAKDOWN")
    print("=" * 60)

    difficulty_metrics = {}
    for level in difficulty_levels:
        level_results = [r for r in results if r.test_case.difficulty == level]
        if level_results:
            metrics = compute_metrics_for_results(level_results)
            difficulty_metrics[level] = metrics
            print(f"\n  {level} ({metrics['total']} tests):")
            _print_metrics_block(metrics, indent="    ")

    # Summary table
    print("\n" + "-" * 70)
    print(
        f"  {'Difficulty':<12} {'Tests':>5} {'Chunk%':>7} {'Source%':>8} {'MRR':>6} {'NDCG@5':>7}"
    )
    print("  " + "-" * 50)
    for level in difficulty_levels:
        if level in difficulty_metrics:
            m = difficulty_metrics[level]
            src_hit = m.get("source_hit_rate_at_k", m["hit_rate_at_k"])
            print(
                f"  {level:<12} {m['total']:>5} {m['hit_rate_at_k']:>6.0%} {src_hit:>7.0%} {m['mrr']:>6.3f} {m['ndcg_5']:>7.3f}"
            )
    print()

    return difficulty_metrics


def print_summary(results: list[TestResult], metrics: dict):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal tests: {metrics['total']}")
    print(f"Passed (chunk match): {metrics['passed']}")
    print(f"Failed (chunk match): {metrics['failed']}")

    # Source-level stats
    source_hit_rate = metrics.get("source_hit_rate_at_k", 0.0)
    source_hits = round(source_hit_rate * metrics["total"])
    source_only = source_hits - metrics["passed"]
    if source_only > 0:
        print(f"Source-only hits (near-misses): {source_only}")

    # Metrics
    print("\nMetrics:")
    target_hit_rate = 0.90
    actual_hit_rate = metrics["hit_rate_at_k"]
    status = "+" if actual_hit_rate >= target_hit_rate else "-"
    print(f"  {status} Chunk Hit Rate@K: {actual_hit_rate:.1%} (target: >={target_hit_rate:.0%})")
    print("      (% of queries where exact target chunk appears in top K)")

    src_status = "+" if source_hit_rate >= target_hit_rate else "-"
    print(f"  {src_status} Source Hit Rate@K: {source_hit_rate:.1%} (lenient)")
    print("      (% of queries where any chunk from expected source appears in top K)")

    mrr = metrics["mrr"]
    mrr_status = "+" if mrr >= 0.5 else "-"  # MRR >= 0.5 means avg rank <= 2
    print(f"  {mrr_status} MRR: {mrr:.3f}")
    print("      (Mean Reciprocal Rank: 1.0=always rank 1, 0.5=avg rank 2)")

    # NDCG metrics
    ndcg_5 = metrics.get("ndcg_5", 0.0)
    ndcg_10 = metrics.get("ndcg_10", 0.0)
    ndcg_status = "+" if ndcg_5 >= 0.7 else "-"  # NDCG >= 0.7 is good ranking
    print(f"  {ndcg_status} NDCG@5: {ndcg_5:.3f}")
    print(f"    NDCG@10: {ndcg_10:.3f}")
    print("      (Position-weighted: 1.0=perfect, 0.5=avg rank 3)")

    # Phase 2: Concept Recall metrics
    concept_recall = metrics.get("concept_recall")
    concept_tests = metrics.get("concept_recall_tests", 0)
    if concept_recall is not None:
        target_concept_recall = 0.70
        cr_status = "+" if concept_recall >= target_concept_recall else "-"
        print(
            f"  {cr_status} Concept Recall: {concept_recall:.1%} (target: >={target_concept_recall:.0%})"
        )
        print(f"      ({concept_tests} tests with expected_concepts)")
    else:
        print("  - Concept Recall: N/A (no tests have expected_concepts defined)")

    # List failures
    failures = [r for r in results if not r.passed]
    if failures:
        print("\nFailed Tests:")
        for r in failures:
            print(f"  - {r.test_case.query}")
            print(f"    {r.error}")

    print()


async def run_golden_dataset_eval(
    dataset_path: Path,
    verbose: bool = False,
    scoring_method: str = "weighted",
    domain_filter: bool = False,
) -> tuple[list[TestResult], dict]:
    """Run evaluation against a golden dataset JSON file.

    Golden dataset entries have: query, target_chunk_ids, domain, source_title, difficulty.
    Evaluates whether target_chunk_ids appear in top-K search results.
    Reports both chunk-level (strict) and source-level (lenient) hit rates.

    Args:
        dataset_path: Path to golden_dataset.json
        verbose: Print detailed output
        scoring_method: Score combination method
        domain_filter: Pass domain_id to SearchQuery (diagnostic mode)

    Returns:
        Tuple of (results list, metrics dict)
    """
    with open(dataset_path) as f:
        entries = json.load(f)

    if not entries:
        print("No entries in golden dataset!")
        return [], {}

    mode_label = "domain-filtered" if domain_filter else "global"
    print(f"Running {len(entries)} golden dataset queries ({mode_label})...")

    config = DatabaseConfig()
    pool = await get_connection_pool(config)
    embed_client = EmbeddingClient()

    results = []
    for entry in entries:
        query_text = entry["query"]
        target_ids = {UUID(cid) for cid in entry.get("target_chunk_ids", [])}
        source_title = entry.get("source_title", "?")
        domain = entry.get("domain", "?")
        difficulty = entry.get("difficulty", "medium")

        if verbose:
            print(f"  [{domain}|{difficulty}] {query_text}")

        try:
            query_embedding = embed_client.embed_query(query_text)
            query = SearchQuery(
                text=query_text,
                embedding=query_embedding,
                fts_weight=0.3,
                vector_weight=0.7,
                limit=10,
                scoring_method=scoring_method,
                domain_id=domain if domain_filter else None,
            )
            search_results = await search_hybrid(query)

            # Strict: exact chunk_id match
            matched_rank = None
            matched_source = None
            for r in search_results:
                if r.chunk.id in target_ids:
                    matched_rank = r.rank
                    matched_source = r.source.title
                    break

            # Lenient: source-level match (any chunk from expected source)
            source_matched_rank = None
            if source_title and source_title != "?":
                source_prefix = source_title[:30].lower()
                for r in search_results:
                    if source_prefix in r.source.title.lower():
                        source_matched_rank = r.rank
                        break

            tc = TestCase(
                query=query_text,
                expected_source_pattern=re.escape(source_title[:30]),
                expected_in_top_k=10,
                tags=[domain, difficulty] if domain else [difficulty],
            )
            result = TestResult(
                test_case=tc,
                passed=matched_rank is not None,
                matched_rank=matched_rank,
                matched_source=matched_source,
                source_matched_rank=source_matched_rank,
                error=(
                    None
                    if matched_rank is not None
                    else f"No target chunk in top-10 for '{source_title[:50]}'"
                ),
            )
            results.append(result)

            if verbose:
                status = "+" if result.passed else "-"
                if result.passed:
                    print(f"    {status} Chunk rank {matched_rank}: {matched_source}")
                elif source_matched_rank:
                    print(
                        f"    {status} Chunk miss, source hit rank {source_matched_rank}: {source_title[:40]}"
                    )
                else:
                    top = [r.source.title[:40] for r in search_results[:3]]
                    print(f"    {status} Not found. Top: {top}")

        except Exception as e:
            tc = TestCase(
                query=query_text,
                expected_source_pattern="",
                expected_in_top_k=10,
                tags=[domain, difficulty] if domain else [difficulty],
            )
            results.append(TestResult(test_case=tc, passed=False, error=f"Error: {e}"))
            if verbose:
                print(f"    - Error: {e}")

    metrics = compute_metrics_for_results(results)
    return results, metrics


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--tag", "-t", help="Filter by tag (e.g., 'core')")
    parser.add_argument("--output", "-o", help="Output JSON file for CI (e.g., metrics.json)")
    parser.add_argument(
        "--scoring",
        "-s",
        choices=["weighted", "rrf"],
        default="weighted",
        help="Scoring method: 'weighted' (default) or 'rrf' (Reciprocal Rank Fusion)",
    )
    parser.add_argument(
        "--per-domain",
        action="store_true",
        help="Show per-domain breakdown of Hit Rate, MRR, NDCG, Concept Recall",
    )
    parser.add_argument(
        "--dataset",
        help="Path to golden dataset JSON (deprecated; YAML is now the default)",
    )
    parser.add_argument(
        "--domain-filter",
        action="store_true",
        help="Pass domain_id to search queries (diagnostic: compare global vs domain-filtered)",
    )
    parser.add_argument(
        "--fail-below",
        type=float,
        default=None,
        help="Exit non-zero if MRR falls below this threshold (e.g., 0.7)",
    )
    args = parser.parse_args()

    print(f"Scoring method: {args.scoring}")
    if args.domain_filter:
        print("Domain filter: ENABLED (passing domain_id to search)")

    # Choose evaluation mode: golden dataset JSON or YAML test cases
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Error: Dataset not found at {dataset_path}")
            sys.exit(1)
        print(f"Using golden dataset: {dataset_path}")
        results, metrics = await run_golden_dataset_eval(
            dataset_path,
            verbose=args.verbose,
            scoring_method=args.scoring,
            domain_filter=args.domain_filter,
        )
    else:
        yaml_path = Path(__file__).parent.parent / "fixtures" / "eval" / "retrieval_test_cases.yaml"
        if not yaml_path.exists():
            print(f"Error: Test cases not found at {yaml_path}")
            sys.exit(1)
        results, metrics = await run_eval(
            yaml_path,
            tag_filter=args.tag,
            verbose=args.verbose,
            scoring_method=args.scoring,
        )

    print_summary(results, metrics)

    # Per-domain breakdown
    domain_metrics = None
    if args.per_domain:
        domain_metrics = print_per_domain(results)

    # Per-difficulty breakdown (show whenever per-domain is requested)
    difficulty_metrics = None
    if args.per_domain:
        difficulty_metrics = print_per_difficulty(results)

    # Output JSON for CI quality gate
    if args.output:
        output_path = Path(args.output)
        output_data = dict(metrics)
        if domain_metrics:
            output_data["per_domain"] = domain_metrics
        if difficulty_metrics:
            output_data["per_difficulty"] = difficulty_metrics
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nMetrics written to: {output_path}")

    # MRR threshold gate (for CI)
    if args.fail_below is not None:
        mrr = metrics.get("mrr", 0.0)
        if mrr < args.fail_below:
            print(f"\nFAIL: MRR {mrr:.3f} below threshold {args.fail_below}")
            sys.exit(1)
        else:
            print(f"\nPASS: MRR {mrr:.3f} >= threshold {args.fail_below}")

    # Exit with error code if tests failed
    if metrics.get("failed", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
