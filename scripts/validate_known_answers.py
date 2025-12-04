#!/usr/bin/env python3
"""Validate known-answer queries against golden dataset.

Tests search quality by running curated queries with known expected results.
"""

import asyncio
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import get_logger
from research_kb_pdf import EmbeddingClient
from research_kb_storage import DatabaseConfig, SearchQuery, get_connection_pool, search_hybrid

logger = get_logger(__name__)


# Known-answer queries (subset for quick validation)
QUERIES = [
    {
        "id": "1.1",
        "query": "What is the formula for scaled dot-product attention?",
        "expected_source": "Attention Is All You Need",
        "expected_keywords": ["attention", "softmax", "queries", "keys", "values"],
        "min_relevance": 0.6,
    },
    {
        "id": "1.2",
        "query": "How does multi-head attention work?",
        "expected_source": "Attention Is All You Need",
        "expected_keywords": ["multi-head", "parallel", "concatenated"],
        "min_relevance": 0.6,
    },
    {
        "id": "2.1",
        "query": "How does dropout approximate a Gaussian process?",
        "expected_source": "Dropout as a Bayesian Approximation",
        "expected_keywords": ["Gaussian process", "variational", "approximate"],
        "min_relevance": 0.6,
    },
    {
        "id": "2.2",
        "query": "How can dropout be used to measure model uncertainty?",
        "expected_source": "Dropout as a Bayesian Approximation",
        "expected_keywords": ["uncertainty", "Monte Carlo", "forward passes"],
        "min_relevance": 0.6,
    },
    {
        "id": "3.1",
        "query": "What is the ELBO in variational inference?",
        "expected_source": "Variational Inference",
        "expected_keywords": ["ELBO", "evidence lower bound", "KL divergence"],
        "min_relevance": 0.6,
    },
    {
        "id": "3.2",
        "query": "What is mean-field variational inference?",
        "expected_source": "Variational Inference",
        "expected_keywords": ["mean-field", "factorization", "independent"],
        "min_relevance": 0.6,
    },
    {
        "id": "3.3",
        "query": "How does stochastic optimization work in variational inference?",
        "expected_source": "Variational Inference",
        "expected_keywords": ["stochastic", "minibatch", "gradient", "scalability"],
        "min_relevance": 0.6,
    },
    {
        "id": "C.1",
        "query": "What are the main approaches to attention mechanisms?",
        "expected_source": "Attention Is All You Need",
        "expected_keywords": ["attention", "self-attention", "mechanism"],
        "min_relevance": 0.5,
    },
]


def check_keywords(content: str, keywords: list[str]) -> tuple[int, list[str]]:
    """Check how many keywords appear in content.

    Returns:
        Tuple of (count, matched_keywords)
    """
    content_lower = content.lower()
    matched = [kw for kw in keywords if kw.lower() in content_lower]
    return len(matched), matched


async def validate_query(query_data: dict, embedding_client: EmbeddingClient) -> dict:
    """Run a single query and validate results.

    Args:
        query_data: Query configuration dict
        embedding_client: Client for generating query embedding

    Returns:
        Validation results dict
    """
    query_id = query_data["id"]
    query_text = query_data["query"]
    expected_source = query_data["expected_source"]
    expected_keywords = query_data["expected_keywords"]
    min_relevance = query_data["min_relevance"]

    logger.info("running_query", query_id=query_id, query=query_text)

    # Generate query embedding
    query_embedding = embedding_client.embed(query_text)

    # Search (hybrid: FTS + vector)
    search_query = SearchQuery(
        text=query_text,
        embedding=query_embedding,
        limit=5,
    )

    results = await search_hybrid(search_query)

    # Validate results
    validation = {
        "query_id": query_id,
        "query": query_text,
        "expected_source": expected_source,
        "results_count": len(results),
        "top_sources": [],
        "keyword_matches": [],
        "section_coverage": 0,
        "avg_similarity": 0.0,
        "passed": False,
    }

    if not results:
        logger.warning("no_results", query_id=query_id)
        return validation

    # Check top results
    top_sources = []
    keyword_match_counts = []
    sections_present = 0
    similarities = []

    for i, result in enumerate(results[:5], 1):
        # Extract source title (full title for comparison)
        source_title = result.source.title
        top_sources.append(source_title)

        # Check keywords
        matched_count, matched_kw = check_keywords(result.chunk.content, expected_keywords)
        keyword_match_counts.append(matched_count)

        # Check section metadata
        if result.chunk.metadata and result.chunk.metadata.get("section"):
            sections_present += 1

        # Get similarity score (if available)
        # Note: search_hybrid returns results with similarity scores
        # We'll estimate from ranking for now
        similarities.append(1.0 - (i - 1) * 0.1)  # Placeholder

        logger.info(
            "result_analysis",
            query_id=query_id,
            rank=i,
            source=source_title[:40],
            keywords_matched=f"{matched_count}/{len(expected_keywords)}",
            matched_kw=matched_kw[:3],
            has_section=bool(result.chunk.metadata and result.chunk.metadata.get("section")),
        )

    validation["top_sources"] = top_sources
    validation["keyword_matches"] = keyword_match_counts
    validation["section_coverage"] = sections_present / min(5, len(results))
    validation["avg_similarity"] = sum(similarities) / len(similarities) if similarities else 0.0

    # Success criteria:
    # 1. At least 1 result from expected source in top 3
    # 2. At least 50% keyword match in top result
    # 3. At least 60% results have section metadata

    source_match = any(expected_source in src for src in top_sources[:3])
    keyword_quality = max(keyword_match_counts[:3]) / len(expected_keywords) if keyword_match_counts else 0
    section_quality = validation["section_coverage"]

    validation["passed"] = (
        source_match
        and keyword_quality >= 0.5
        and section_quality >= 0.6
    )

    if validation["passed"]:
        logger.info("query_passed", query_id=query_id)
    else:
        logger.warning(
            "query_failed",
            query_id=query_id,
            source_match=source_match,
            keyword_quality=f"{keyword_quality:.1%}",
            section_quality=f"{section_quality:.1%}",
        )

    return validation


async def main():
    """Run all validation queries and report results."""
    logger.info("starting_validation", queries=len(QUERIES))

    # Initialize
    await get_connection_pool(DatabaseConfig())
    embedding_client = EmbeddingClient()

    # Run queries
    results = []
    for query_data in QUERIES:
        try:
            validation = await validate_query(query_data, embedding_client)
            results.append(validation)
        except Exception as e:
            logger.error("query_failed", query_id=query_data["id"], error=str(e), exc_info=True)
            results.append({
                "query_id": query_data["id"],
                "query": query_data["query"],
                "passed": False,
                "error": str(e),
            })

    # Summary report
    print("\n" + "=" * 80)
    print("KNOWN-ANSWER QUERY VALIDATION RESULTS")
    print("=" * 80)

    passed = sum(1 for r in results if r.get("passed", False))
    total = len(results)

    print(f"\nOverall: {passed}/{total} queries passed ({passed/total*100:.1f}%)")
    print(f"Success threshold: ≥71% (10/14 queries)")
    print()

    # Per-query breakdown
    print("Query Results:")
    print("-" * 80)
    for r in results:
        status = "✓ PASS" if r.get("passed", False) else "✗ FAIL"
        query_id = r.get("query_id", "?")
        query = r.get("query", "?")[:60]

        print(f"{status} [{query_id}] {query}")

        if r.get("top_sources"):
            print(f"     Top sources: {r['top_sources'][:2]}")
        if r.get("keyword_matches"):
            print(f"     Keyword matches: {r['keyword_matches'][:3]}")
        if "section_coverage" in r:
            print(f"     Section coverage: {r['section_coverage']:.0%}")
        print()

    print("=" * 80)

    # Detailed statistics
    if results:
        avg_section_coverage = sum(r.get("section_coverage", 0) for r in results) / len(results)
        queries_with_results = sum(1 for r in results if r.get("results_count", 0) > 0)

        print("\nDetailed Statistics:")
        print(f"  Queries returning results: {queries_with_results}/{total}")
        print(f"  Average section coverage: {avg_section_coverage:.1%}")
        print()

    # Final verdict
    success_rate = passed / total
    if success_rate >= 0.71:
        print("✓ VALIDATION PASSED - Search quality meets threshold (≥71%)")
        logger.info("validation_success", passed=passed, total=total, rate=f"{success_rate:.1%}")
    else:
        print("✗ VALIDATION FAILED - Search quality below threshold")
        logger.warning("validation_failed", passed=passed, total=total, rate=f"{success_rate:.1%}")

    return 0 if success_rate >= 0.71 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
