#!/usr/bin/env python3
"""
Validate ProactiveContext ↔ research-kb daemon integration.

Measures:
- Latency: P50/P95/P99 via ResearchKBBridge.query()
- Coverage: % of domain-specific prompts receiving enrichment
- Quality: Mean vector_score of returned results
- Fallback: Graceful behavior when daemon stopped

Usage:
    python scripts/validate_proactive_context.py
    python scripts/validate_proactive_context.py --json          # JSON output
    python scripts/validate_proactive_context.py --daemon-only   # Skip CLI fallback tests

Exit criteria (Phase 4.3):
    - Latency P50 < 500ms
    - Coverage >= 80% of domain prompts enriched
    - Quality mean vector_score > 0.6
    - Fallback: empty results (no crash) when daemon down
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Add lever_of_archimedes to path for bridge import
LEVER_PATH = Path.home() / "Claude" / "lever_of_archimedes"
sys.path.insert(0, str(LEVER_PATH))

from services.proactive_context.research_kb_bridge import ResearchKBBridge

# ── Test corpus: 50 prompts across 5 domains ──────────────────────────────────

TEST_PROMPTS = {
    "causal_inference": [
        "What are the key assumptions for instrumental variables estimation?",
        "Explain the difference between ATE and ATT in causal inference",
        "How does double machine learning handle nuisance parameters?",
        "When should I use regression discontinuity design?",
        "What is the unconfoundedness assumption in observational studies?",
        "Explain difference-in-differences with staggered treatment adoption",
        "How does propensity score matching reduce selection bias?",
        "What are the identification conditions for LATE?",
        "How does synthetic control compare to diff-in-diff?",
        "What is the do-calculus in Pearl's framework?",
    ],
    "time_series": [
        "How do I choose between ARIMA and exponential smoothing?",
        "What is the temporal fusion transformer architecture?",
        "Explain GARCH models for volatility forecasting",
        "When should I use state space models over ARIMA?",
        "How does TsMixer handle multivariate time series?",
        "What stationarity tests should I run before fitting VAR?",
        "Explain walk-forward validation for time series models",
        "How does Prophet handle seasonality and holidays?",
        "What is the Informer attention mechanism for long sequences?",
        "How do I implement cross-validation for time series data?",
    ],
    "rag_llm": [
        "What are the best chunking strategies for RAG pipelines?",
        "How does hybrid search combine BM25 and dense retrieval?",
        "Explain the reranking stage in retrieval augmented generation",
        "What embedding models work best for semantic search?",
        "How do I reduce hallucination in RAG systems?",
        "What is the difference between sparse and dense retrieval?",
        "How does LangChain handle document splitting?",
        "What are effective prompt engineering techniques for RAG?",
        "How do sentence transformers generate embeddings?",
        "What vector databases support hybrid search?",
    ],
    "interview_prep": [
        "How should I structure a system design interview answer?",
        "What coding interview patterns should I know for trees and graphs?",
        "Explain the STAR method for behavioral interview questions",
        "What are the most common machine learning interview questions?",
        "How do I prepare for statistics interview questions?",
        "What probability questions come up in data science interviews?",
        "How should I approach algorithm interview problems?",
        "What data structure interview topics are most important?",
        "How do I practice for a technical interview efficiently?",
        "What case study interview frameworks should I know?",
    ],
    "healthcare": [
        "What assumptions does a randomized controlled trial require?",
        "How do I calculate odds ratio vs hazard ratio?",
        "Explain the intention-to-treat principle in clinical trials",
        "When should I use Kaplan-Meier survival analysis?",
        "What is the difference between meta-analysis and systematic review?",
        "How does Cox proportional hazards regression work?",
        "What are the phases of a clinical trial?",
        "How do I handle missing data in RCT analysis?",
        "What biostatistics methods detect treatment interaction?",
        "How does per-protocol analysis differ from ITT?",
    ],
}


@dataclass
class PromptResult:
    """Result from a single prompt test."""

    domain: str
    prompt: str
    latency_ms: float
    num_results: int
    mean_vector_score: float | None
    enriched: bool  # True if at least 1 result returned


@dataclass
class DomainSummary:
    """Summary statistics for a domain."""

    domain: str
    num_prompts: int
    enriched_count: int
    coverage_pct: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    mean_quality: float | None  # Mean vector_score across all results


@dataclass
class ValidationReport:
    """Complete validation report."""

    timestamp: str
    daemon_available: bool
    total_prompts: int
    total_enriched: int
    overall_coverage_pct: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    mean_quality: float | None
    domains: list[DomainSummary] = field(default_factory=list)
    exit_criteria: dict = field(default_factory=dict)
    all_passed: bool = False


def percentile(data: list[float], pct: float) -> float:
    """Calculate percentile from sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * pct / 100.0
    lower = int(idx)
    upper = min(lower + 1, len(sorted_data) - 1)
    frac = idx - lower
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac


def run_prompt(bridge: ResearchKBBridge, domain: str, prompt: str) -> PromptResult:
    """Run a single prompt and measure results."""
    start = time.monotonic()
    results = bridge.query(prompt, limit=3)
    elapsed_ms = (time.monotonic() - start) * 1000

    vector_scores = [r.confidence for r in results if r.confidence is not None]
    mean_vs = statistics.mean(vector_scores) if vector_scores else None

    return PromptResult(
        domain=domain,
        prompt=prompt,
        latency_ms=elapsed_ms,
        num_results=len(results),
        mean_vector_score=mean_vs,
        enriched=len(results) > 0,
    )


def run_validation(daemon_only: bool = False) -> ValidationReport:
    """Run full validation suite."""
    bridge = ResearchKBBridge()
    daemon_available = os.path.exists(bridge.socket_path)

    if daemon_only and not daemon_available:
        print("ERROR: --daemon-only specified but daemon not running", file=sys.stderr)
        sys.exit(1)

    # Run all prompts
    all_results: list[PromptResult] = []
    for domain, prompts in TEST_PROMPTS.items():
        for prompt in prompts:
            result = run_prompt(bridge, domain, prompt)
            all_results.append(result)

    # Overall statistics
    all_latencies = [r.latency_ms for r in all_results]
    all_scores = [r.mean_vector_score for r in all_results if r.mean_vector_score is not None]
    total_enriched = sum(1 for r in all_results if r.enriched)

    # Per-domain summaries
    domain_summaries = []
    for domain in TEST_PROMPTS:
        domain_results = [r for r in all_results if r.domain == domain]
        latencies = [r.latency_ms for r in domain_results]
        scores = [r.mean_vector_score for r in domain_results if r.mean_vector_score is not None]
        enriched = sum(1 for r in domain_results if r.enriched)

        domain_summaries.append(
            DomainSummary(
                domain=domain,
                num_prompts=len(domain_results),
                enriched_count=enriched,
                coverage_pct=(enriched / len(domain_results) * 100 if domain_results else 0),
                latency_p50_ms=percentile(latencies, 50),
                latency_p95_ms=percentile(latencies, 95),
                latency_p99_ms=percentile(latencies, 99),
                mean_quality=statistics.mean(scores) if scores else None,
            )
        )

    overall_coverage = total_enriched / len(all_results) * 100 if all_results else 0
    p50 = percentile(all_latencies, 50)
    p95 = percentile(all_latencies, 95)
    p99 = percentile(all_latencies, 99)
    mean_q = statistics.mean(all_scores) if all_scores else None

    # Check exit criteria
    exit_criteria = {
        "latency_p50_under_500ms": {"target": 500, "actual": p50, "passed": p50 < 500},
        "coverage_above_80pct": {
            "target": 80,
            "actual": overall_coverage,
            "passed": overall_coverage >= 80,
        },
        "quality_above_0.6": {
            "target": 0.6,
            "actual": mean_q,
            "passed": (mean_q or 0) > 0.6,
        },
    }
    all_passed = all(c["passed"] for c in exit_criteria.values())

    # Fallback test: query with nonexistent socket
    fallback_bridge = ResearchKBBridge(
        socket_path="/nonexistent/socket.sock",
        cli_path="/nonexistent/cli",
    )
    try:
        fallback_results = fallback_bridge.query("test fallback")
        fallback_ok = isinstance(fallback_results, list)
    except Exception:
        fallback_ok = False

    exit_criteria["fallback_graceful"] = {
        "target": True,
        "actual": fallback_ok,
        "passed": fallback_ok,
    }
    all_passed = all_passed and fallback_ok

    return ValidationReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        daemon_available=daemon_available,
        total_prompts=len(all_results),
        total_enriched=total_enriched,
        overall_coverage_pct=overall_coverage,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
        mean_quality=mean_q,
        domains=domain_summaries,
        exit_criteria=exit_criteria,
        all_passed=all_passed,
    )


def print_text_report(report: ValidationReport) -> None:
    """Print human-readable validation report."""
    print("=" * 70)
    print("ProactiveContext Integration Validation")
    print("=" * 70)
    print(f"Timestamp:        {report.timestamp}")
    print(f"Daemon available: {report.daemon_available}")
    print(f"Total prompts:    {report.total_prompts}")
    print(f"Total enriched:   {report.total_enriched}")
    print()

    # Overall metrics
    print("── Overall Metrics ──────────────────────────────────────────────")
    print(f"  Coverage:     {report.overall_coverage_pct:.1f}%")
    print(f"  Latency P50:  {report.latency_p50_ms:.0f}ms")
    print(f"  Latency P95:  {report.latency_p95_ms:.0f}ms")
    print(f"  Latency P99:  {report.latency_p99_ms:.0f}ms")
    if report.mean_quality is not None:
        print(f"  Quality:      {report.mean_quality:.3f}")
    print()

    # Per-domain breakdown
    print("── Per-Domain Breakdown ─────────────────────────────────────────")
    print(f"  {'Domain':<20} {'Coverage':>8} {'P50 ms':>8} {'P95 ms':>8} {'Quality':>8}")
    print(f"  {'─' * 20} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
    for d in report.domains:
        quality_str = f"{d.mean_quality:.3f}" if d.mean_quality is not None else "N/A"
        print(
            f"  {d.domain:<20} {d.coverage_pct:>7.0f}% "
            f"{d.latency_p50_ms:>7.0f} {d.latency_p95_ms:>7.0f} "
            f"{quality_str:>8}"
        )
    print()

    # Exit criteria
    print("── Exit Criteria ───────────────────────────────────────────────")
    for name, check in report.exit_criteria.items():
        status = "PASS" if check["passed"] else "FAIL"
        actual = check["actual"]
        if isinstance(actual, float):
            actual_str = f"{actual:.2f}"
        else:
            actual_str = str(actual)
        print(f"  [{status}] {name}: {actual_str} (target: {check['target']})")
    print()

    overall = "ALL PASSED" if report.all_passed else "SOME FAILED"
    print(f"Result: {overall}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Validate ProactiveContext integration")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--daemon-only", action="store_true", help="Fail if daemon not running")
    args = parser.parse_args()

    report = run_validation(daemon_only=args.daemon_only)

    if args.json:
        # Convert to JSON-serializable dict
        report_dict = {
            "timestamp": report.timestamp,
            "daemon_available": report.daemon_available,
            "total_prompts": report.total_prompts,
            "total_enriched": report.total_enriched,
            "overall_coverage_pct": report.overall_coverage_pct,
            "latency_p50_ms": report.latency_p50_ms,
            "latency_p95_ms": report.latency_p95_ms,
            "latency_p99_ms": report.latency_p99_ms,
            "mean_quality": report.mean_quality,
            "domains": [asdict(d) for d in report.domains],
            "exit_criteria": report.exit_criteria,
            "all_passed": report.all_passed,
        }
        print(json.dumps(report_dict, indent=2))
    else:
        print_text_report(report)

    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
