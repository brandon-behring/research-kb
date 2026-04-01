#!/usr/bin/env python3
"""Compare two retrieval evaluation runs and report differences.

Reads JSON output from eval_retrieval.py (--output flag) and computes
per-metric deltas, per-domain improvements/regressions, and highlights
queries that changed ranking.

Usage:
    python scripts/eval_compare.py baseline.json candidate.json
    python scripts/eval_compare.py baseline.json candidate.json --per-domain
    python scripts/eval_compare.py baseline.json candidate.json --threshold 0.05
    python scripts/eval_compare.py baseline.json candidate.json --output comparison.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Metrics we track and their "higher is better" direction
TRACKED_METRICS = {
    "hit_rate_at_k": {"label": "Chunk Hit@K", "format": ".1%", "higher_better": True},
    "source_hit_rate_at_k": {"label": "Source Hit@K", "format": ".1%", "higher_better": True},
    "mrr": {"label": "MRR", "format": ".4f", "higher_better": True},
    "ndcg_5": {"label": "NDCG@5", "format": ".4f", "higher_better": True},
    "ndcg_10": {"label": "NDCG@10", "format": ".4f", "higher_better": True},
}


@dataclass
class MetricDelta:
    """A single metric comparison between baseline and candidate."""

    name: str
    label: str
    baseline: float
    candidate: float
    delta: float
    pct_change: float
    improved: bool
    regressed: bool
    format_spec: str = ".4f"

    def format_value(self, value: float) -> str:
        """Format a value according to the metric's format spec."""
        if self.format_spec == ".1%":
            return f"{value:.1%}"
        return f"{value:{self.format_spec}}"

    def format_delta(self) -> str:
        """Format the delta with direction indicator."""
        if abs(self.delta) < 1e-6:
            return "  (no change)"
        sign = "+" if self.delta > 0 else ""
        if self.format_spec == ".1%":
            return f"{sign}{self.delta:.1%}"
        return f"{sign}{self.delta:{self.format_spec}}"


@dataclass
class DomainComparison:
    """Comparison for a single domain."""

    domain: str
    baseline_tests: int
    candidate_tests: int
    metric_deltas: list[MetricDelta] = field(default_factory=list)
    improved: bool = False
    regressed: bool = False


@dataclass
class ComparisonReport:
    """Full comparison between two eval runs."""

    baseline_path: str
    candidate_path: str
    aggregate_deltas: list[MetricDelta] = field(default_factory=list)
    domain_comparisons: list[DomainComparison] = field(default_factory=list)
    difficulty_comparisons: list[DomainComparison] = field(default_factory=list)
    improved_domains: list[str] = field(default_factory=list)
    regressed_domains: list[str] = field(default_factory=list)
    unchanged_domains: list[str] = field(default_factory=list)


def compute_metric_delta(
    name: str,
    baseline_val: Optional[float],
    candidate_val: Optional[float],
    meta: dict,
) -> Optional[MetricDelta]:
    """Compute delta between baseline and candidate for one metric.

    Parameters
    ----------
    name : str
        Internal metric name.
    baseline_val : float or None
        Baseline value.
    candidate_val : float or None
        Candidate value.
    meta : dict
        Metric metadata (label, format, higher_better).

    Returns
    -------
    MetricDelta or None
        Comparison result, or None if either value is missing.
    """
    if baseline_val is None or candidate_val is None:
        return None

    delta = candidate_val - baseline_val
    pct_change = (delta / baseline_val * 100) if baseline_val != 0 else 0.0

    higher_better = meta["higher_better"]
    improved = (delta > 1e-6) if higher_better else (delta < -1e-6)
    regressed = (delta < -1e-6) if higher_better else (delta > 1e-6)

    return MetricDelta(
        name=name,
        label=meta["label"],
        baseline=baseline_val,
        candidate=candidate_val,
        delta=delta,
        pct_change=pct_change,
        improved=improved,
        regressed=regressed,
        format_spec=meta["format"],
    )


def compare_metric_dicts(
    baseline: dict,
    candidate: dict,
) -> list[MetricDelta]:
    """Compare all tracked metrics between two result dicts.

    Parameters
    ----------
    baseline : dict
        Baseline metrics dict from eval_retrieval.py.
    candidate : dict
        Candidate metrics dict from eval_retrieval.py.

    Returns
    -------
    list[MetricDelta]
        List of computed deltas for all available metrics.
    """
    deltas = []
    for name, meta in TRACKED_METRICS.items():
        d = compute_metric_delta(
            name,
            baseline.get(name),
            candidate.get(name),
            meta,
        )
        if d is not None:
            deltas.append(d)
    return deltas


def compare_evals(
    baseline: dict,
    candidate: dict,
    threshold: float = 0.0,
) -> ComparisonReport:
    """Compare two full evaluation result JSONs.

    Parameters
    ----------
    baseline : dict
        Baseline eval output from eval_retrieval.py --output.
    candidate : dict
        Candidate eval output from eval_retrieval.py --output.
    threshold : float
        Minimum absolute delta to consider as a real change (noise filter).

    Returns
    -------
    ComparisonReport
        Full comparison with aggregate, per-domain, and per-difficulty breakdowns.
    """
    report = ComparisonReport(
        baseline_path="",
        candidate_path="",
    )

    # Aggregate comparison
    report.aggregate_deltas = compare_metric_dicts(baseline, candidate)

    # Per-domain comparison
    baseline_domains = baseline.get("per_domain", {})
    candidate_domains = candidate.get("per_domain", {})
    all_domains = sorted(set(baseline_domains.keys()) | set(candidate_domains.keys()))

    for domain in all_domains:
        b = baseline_domains.get(domain, {})
        c = candidate_domains.get(domain, {})

        dc = DomainComparison(
            domain=domain,
            baseline_tests=b.get("total", 0),
            candidate_tests=c.get("total", 0),
        )

        if b and c:
            dc.metric_deltas = compare_metric_dicts(b, c)

            # Determine domain-level verdict using MRR as primary signal
            mrr_delta = next((d for d in dc.metric_deltas if d.name == "mrr"), None)
            if mrr_delta:
                if mrr_delta.delta > threshold:
                    dc.improved = True
                    report.improved_domains.append(domain)
                elif mrr_delta.delta < -threshold:
                    dc.regressed = True
                    report.regressed_domains.append(domain)
                else:
                    report.unchanged_domains.append(domain)
        elif c and not b:
            report.improved_domains.append(domain)
            dc.improved = True
        elif b and not c:
            report.regressed_domains.append(domain)
            dc.regressed = True

        report.domain_comparisons.append(dc)

    # Per-difficulty comparison
    baseline_diff = baseline.get("per_difficulty", {})
    candidate_diff = candidate.get("per_difficulty", {})
    for level in ["easy", "medium", "hard"]:
        b = baseline_diff.get(level, {})
        c = candidate_diff.get(level, {})
        if b or c:
            dc = DomainComparison(
                domain=level,
                baseline_tests=b.get("total", 0),
                candidate_tests=c.get("total", 0),
            )
            if b and c:
                dc.metric_deltas = compare_metric_dicts(b, c)
            report.difficulty_comparisons.append(dc)

    return report


def print_report(report: ComparisonReport, show_domains: bool = False) -> None:
    """Print comparison report to stdout.

    Parameters
    ----------
    report : ComparisonReport
        The comparison to display.
    show_domains : bool
        Whether to show per-domain breakdown.
    """
    print("=" * 70)
    print("RETRIEVAL EVALUATION COMPARISON")
    print("=" * 70)
    print(f"  Baseline:  {report.baseline_path}")
    print(f"  Candidate: {report.candidate_path}")

    # Aggregate metrics
    print("\n--- Aggregate Metrics ---\n")
    print(f"  {'Metric':<18} {'Baseline':>10} {'Candidate':>10} {'Delta':>14} {'Change':>8}")
    print("  " + "-" * 62)
    for d in report.aggregate_deltas:
        indicator = ""
        if d.improved:
            indicator = " [+]"
        elif d.regressed:
            indicator = " [-]"

        print(
            f"  {d.label:<18} {d.format_value(d.baseline):>10} "
            f"{d.format_value(d.candidate):>10} {d.format_delta():>14} "
            f"{d.pct_change:>+6.1f}%{indicator}"
        )

    # Per-difficulty breakdown
    if report.difficulty_comparisons:
        print("\n--- Per-Difficulty ---\n")
        print(f"  {'Level':<12} {'Tests':>5}  {'MRR Base':>9} {'MRR Cand':>9} {'Delta':>9}")
        print("  " + "-" * 50)
        for dc in report.difficulty_comparisons:
            mrr_d = next((d for d in dc.metric_deltas if d.name == "mrr"), None)
            if mrr_d:
                print(
                    f"  {dc.domain:<12} {dc.baseline_tests:>5}  "
                    f"{mrr_d.format_value(mrr_d.baseline):>9} "
                    f"{mrr_d.format_value(mrr_d.candidate):>9} "
                    f"{mrr_d.format_delta():>9}"
                )

    # Per-domain breakdown
    if show_domains and report.domain_comparisons:
        print("\n--- Per-Domain Breakdown ---\n")
        print(
            f"  {'Domain':<25} {'Tests':>5}  "
            f"{'MRR Base':>9} {'MRR Cand':>9} {'Delta':>9} {'Verdict':>9}"
        )
        print("  " + "-" * 72)
        for dc in report.domain_comparisons:
            mrr_d = next((d for d in dc.metric_deltas if d.name == "mrr"), None)
            if mrr_d:
                verdict = "SAME"
                if dc.improved:
                    verdict = "BETTER"
                elif dc.regressed:
                    verdict = "WORSE"

                print(
                    f"  {dc.domain:<25} {dc.baseline_tests:>5}  "
                    f"{mrr_d.format_value(mrr_d.baseline):>9} "
                    f"{mrr_d.format_value(mrr_d.candidate):>9} "
                    f"{mrr_d.format_delta():>9} {verdict:>9}"
                )

    # Summary
    print("\n--- Summary ---\n")
    n_improved = len(report.improved_domains)
    n_regressed = len(report.regressed_domains)
    n_unchanged = len(report.unchanged_domains)
    total = n_improved + n_regressed + n_unchanged

    print(f"  Domains improved:  {n_improved}/{total}")
    print(f"  Domains regressed: {n_regressed}/{total}")
    print(f"  Domains unchanged: {n_unchanged}/{total}")

    if report.regressed_domains:
        print(f"\n  Regressed: {', '.join(report.regressed_domains)}")

    # Overall verdict
    agg_mrr = next((d for d in report.aggregate_deltas if d.name == "mrr"), None)
    if agg_mrr:
        if agg_mrr.improved:
            print(f"\n  VERDICT: MRR improved by {agg_mrr.format_delta()}")
        elif agg_mrr.regressed:
            print(f"\n  VERDICT: MRR regressed by {agg_mrr.format_delta()}")
        else:
            print("\n  VERDICT: No significant MRR change")

    print()


def report_to_dict(report: ComparisonReport) -> dict:
    """Convert report to JSON-serializable dict.

    Parameters
    ----------
    report : ComparisonReport
        The comparison report.

    Returns
    -------
    dict
        JSON-serializable representation.
    """
    return {
        "baseline_path": report.baseline_path,
        "candidate_path": report.candidate_path,
        "aggregate": {
            d.name: {
                "baseline": d.baseline,
                "candidate": d.candidate,
                "delta": d.delta,
                "pct_change": d.pct_change,
                "improved": d.improved,
                "regressed": d.regressed,
            }
            for d in report.aggregate_deltas
        },
        "per_domain": {
            dc.domain: {
                "baseline_tests": dc.baseline_tests,
                "candidate_tests": dc.candidate_tests,
                "improved": dc.improved,
                "regressed": dc.regressed,
                "metrics": {
                    d.name: {"delta": d.delta, "pct_change": d.pct_change} for d in dc.metric_deltas
                },
            }
            for dc in report.domain_comparisons
        },
        "summary": {
            "improved_domains": report.improved_domains,
            "regressed_domains": report.regressed_domains,
            "unchanged_domains": report.unchanged_domains,
        },
    }


def main() -> None:
    """CLI entry point for eval comparison."""
    parser = argparse.ArgumentParser(
        description="Compare two retrieval evaluation runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/eval_compare.py evaluation_runs/baseline.json evaluation_runs/after_change.json
  python scripts/eval_compare.py baseline.json candidate.json --per-domain
  python scripts/eval_compare.py baseline.json candidate.json --threshold 0.02 --output diff.json
        """,
    )
    parser.add_argument(
        "baseline", help="Path to baseline eval JSON (from eval_retrieval.py --output)"
    )
    parser.add_argument("candidate", help="Path to candidate eval JSON")
    parser.add_argument(
        "--per-domain",
        action="store_true",
        help="Show per-domain breakdown",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum MRR delta to count as improvement/regression (default: 0.0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Write comparison JSON to file",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if aggregate MRR regressed (for CI)",
    )
    args = parser.parse_args()

    # Load files
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)

    if not baseline_path.exists():
        print(f"Error: Baseline file not found: {baseline_path}")
        sys.exit(1)
    if not candidate_path.exists():
        print(f"Error: Candidate file not found: {candidate_path}")
        sys.exit(1)

    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(candidate_path) as f:
        candidate = json.load(f)

    # Compare
    report = compare_evals(baseline, candidate, threshold=args.threshold)
    report.baseline_path = str(baseline_path)
    report.candidate_path = str(candidate_path)

    # Print
    print_report(report, show_domains=args.per_domain)

    # Output JSON
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(report_to_dict(report), f, indent=2)
        print(f"Comparison written to: {output_path}")

    # CI gate
    if args.fail_on_regression:
        agg_mrr = next((d for d in report.aggregate_deltas if d.name == "mrr"), None)
        if agg_mrr and agg_mrr.regressed:
            print(f"FAIL: MRR regressed by {agg_mrr.format_delta()}")
            sys.exit(1)


if __name__ == "__main__":
    main()
