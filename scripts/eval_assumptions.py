#!/usr/bin/env python3
"""Evaluate assumption auditing quality against golden dataset.

Phase 4.1c: Validation for Assumption Auditing (North Star feature)

Metrics:
- Recall: What fraction of expected assumptions were found?
- Precision: What fraction of returned assumptions were expected?
- F1 Score: Harmonic mean of precision and recall
- Importance-weighted scores: Critical assumptions count more

Usage:
    python scripts/eval_assumptions.py
    python scripts/eval_assumptions.py --output metrics.json
    python scripts/eval_assumptions.py --verbose
    python scripts/eval_assumptions.py --method "double machine learning"
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

# Add packages to path for development mode
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))


@dataclass
class AssumptionMatch:
    """Record of whether an expected assumption was found."""

    expected_name: str
    expected_importance: str
    matched: bool
    matched_name: Optional[str] = None
    similarity: float = 0.0


@dataclass
class MethodEvaluation:
    """Evaluation results for a single method."""

    method_name: str
    expected_count: int
    returned_count: int
    matches: list[AssumptionMatch] = field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    weighted_recall: float = 0.0
    source: str = "unknown"
    error: Optional[str] = None


@dataclass
class EvaluationSummary:
    """Summary of evaluation across all methods."""

    total_methods: int = 0
    methods_with_results: int = 0
    methods_not_found: int = 0
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_f1: float = 0.0
    avg_weighted_recall: float = 0.0
    total_expected: int = 0
    total_matched: int = 0
    methods: list[MethodEvaluation] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "total_methods": self.total_methods,
            "methods_with_results": self.methods_with_results,
            "methods_not_found": self.methods_not_found,
            "avg_precision": round(self.avg_precision, 4),
            "avg_recall": round(self.avg_recall, 4),
            "avg_f1": round(self.avg_f1, 4),
            "avg_weighted_recall": round(self.avg_weighted_recall, 4),
            "total_expected": self.total_expected,
            "total_matched": self.total_matched,
            "methods": [
                {
                    "name": m.method_name,
                    "expected": m.expected_count,
                    "returned": m.returned_count,
                    "precision": round(m.precision, 4),
                    "recall": round(m.recall, 4),
                    "f1": round(m.f1, 4),
                    "source": m.source,
                    "error": m.error,
                }
                for m in self.methods
            ],
        }


def fuzzy_match(name1: str, name2: str, threshold: float = 0.7) -> tuple[bool, float]:
    """Check if two assumption names match using fuzzy string matching.

    Args:
        name1: First assumption name
        name2: Second assumption name
        threshold: Minimum similarity score (0-1)

    Returns:
        Tuple of (is_match, similarity_score)
    """
    # Normalize names
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()

    # Exact match
    if n1 == n2:
        return True, 1.0

    # Check if one contains the other
    if n1 in n2 or n2 in n1:
        return True, 0.9

    # Fuzzy match using SequenceMatcher
    similarity = SequenceMatcher(None, n1, n2).ratio()

    return similarity >= threshold, similarity


def find_best_match(
    expected_name: str,
    expected_aliases: list[str],
    returned_names: list[str],
    threshold: float = 0.7,
) -> tuple[Optional[str], float]:
    """Find the best matching returned assumption for an expected one.

    Args:
        expected_name: Expected assumption name
        expected_aliases: Alternative names for the assumption
        returned_names: List of assumption names returned by the system
        threshold: Minimum similarity for a match

    Returns:
        Tuple of (matched_name or None, best_similarity)
    """
    all_expected = [expected_name] + expected_aliases
    best_match = None
    best_similarity = 0.0

    for returned in returned_names:
        for expected in all_expected:
            is_match, similarity = fuzzy_match(expected, returned, threshold)
            if similarity > best_similarity:
                best_similarity = similarity
                if is_match:
                    best_match = returned

    return best_match, best_similarity


async def evaluate_method(
    method_config: dict,
    importance_weights: dict[str, float],
    threshold: float,
    use_ollama: bool = False,
    verbose: bool = False,
) -> MethodEvaluation:
    """Evaluate assumption auditing for a single method.

    Args:
        method_config: Method configuration from golden dataset
        importance_weights: Weights for different importance levels
        threshold: Fuzzy match threshold
        use_ollama: Whether to allow Ollama fallback
        verbose: Print detailed output

    Returns:
        MethodEvaluation with results
    """
    from research_kb_storage import MethodAssumptionAuditor, DatabaseConfig, get_connection_pool

    method_name = method_config["name"]
    expected_assumptions = method_config["expected_assumptions"]

    # Initialize database
    config = DatabaseConfig()
    await get_connection_pool(config)

    eval_result = MethodEvaluation(
        method_name=method_name,
        expected_count=len(expected_assumptions),
        returned_count=0,
    )

    try:
        # Run assumption auditing
        result = await MethodAssumptionAuditor.audit_assumptions(
            method_name,
            use_ollama_fallback=use_ollama,
        )

        eval_result.source = result.source
        eval_result.returned_count = len(result.assumptions)

        if result.source == "not_found":
            eval_result.error = "Method not found in knowledge base"
            return eval_result

        # Get returned assumption names
        returned_names = [a.name for a in result.assumptions]

        # Match each expected assumption
        matched_returned = set()
        weighted_matches = 0.0
        total_weight = 0.0

        for exp in expected_assumptions:
            exp_name = exp["name"]
            exp_importance = exp.get("importance", "standard")
            exp_aliases = exp.get("aliases", [])
            weight = importance_weights.get(exp_importance, 1.0)
            total_weight += weight

            matched_name, similarity = find_best_match(
                exp_name, exp_aliases, returned_names, threshold
            )

            match = AssumptionMatch(
                expected_name=exp_name,
                expected_importance=exp_importance,
                matched=matched_name is not None,
                matched_name=matched_name,
                similarity=similarity,
            )
            eval_result.matches.append(match)

            if matched_name:
                matched_returned.add(matched_name)
                weighted_matches += weight

        # Calculate metrics
        true_positives = len([m for m in eval_result.matches if m.matched])

        if eval_result.expected_count > 0:
            eval_result.recall = true_positives / eval_result.expected_count

        if eval_result.returned_count > 0:
            # Precision: matched / returned (being generous with extra assumptions)
            eval_result.precision = len(matched_returned) / eval_result.returned_count

        if eval_result.precision + eval_result.recall > 0:
            eval_result.f1 = (
                2 * eval_result.precision * eval_result.recall
                / (eval_result.precision + eval_result.recall)
            )

        if total_weight > 0:
            eval_result.weighted_recall = weighted_matches / total_weight

        if verbose:
            print(f"\n  {method_name}:")
            print(f"    Source: {result.source}")
            print(f"    Expected: {eval_result.expected_count}, Returned: {eval_result.returned_count}")
            print(f"    Precision: {eval_result.precision:.2%}, Recall: {eval_result.recall:.2%}")
            for match in eval_result.matches:
                status = "✓" if match.matched else "✗"
                print(f"      {status} {match.expected_name} [{match.expected_importance}]", end="")
                if match.matched:
                    print(f" → {match.matched_name} ({match.similarity:.2f})")
                else:
                    print(f" (best similarity: {match.similarity:.2f})")

    except Exception as e:
        eval_result.error = str(e)
        if verbose:
            print(f"\n  {method_name}: ERROR - {e}")

    return eval_result


async def run_evaluation(
    golden_path: Path,
    output_path: Optional[Path] = None,
    method_filter: Optional[str] = None,
    use_ollama: bool = False,
    verbose: bool = False,
) -> EvaluationSummary:
    """Run full evaluation against golden dataset.

    Args:
        golden_path: Path to golden_assumptions.json
        output_path: Optional path to write JSON results
        method_filter: Optional method name to evaluate (single method)
        use_ollama: Whether to allow Ollama fallback
        verbose: Print detailed output

    Returns:
        EvaluationSummary with all results
    """
    # Load golden dataset
    with open(golden_path) as f:
        golden = json.load(f)

    methods = golden["methods"]
    config = golden.get("evaluation_config", {})

    threshold = config.get("fuzzy_match_threshold", 0.7)
    importance_weights = config.get("importance_weights", {
        "critical": 2.0,
        "standard": 1.0,
        "technical": 0.5,
    })

    # Filter if requested
    if method_filter:
        methods = [m for m in methods if method_filter.lower() in m["name"].lower()]
        if not methods:
            print(f"No methods matching '{method_filter}' in golden dataset")
            return EvaluationSummary()

    print(f"Evaluating {len(methods)} methods...")
    if verbose:
        print(f"  Fuzzy match threshold: {threshold}")
        print(f"  Importance weights: {importance_weights}")
        print(f"  Ollama fallback: {use_ollama}")

    summary = EvaluationSummary(total_methods=len(methods))

    # Evaluate each method
    for method_config in methods:
        eval_result = await evaluate_method(
            method_config,
            importance_weights,
            threshold,
            use_ollama,
            verbose,
        )
        summary.methods.append(eval_result)

        if eval_result.source == "not_found":
            summary.methods_not_found += 1
        elif eval_result.returned_count > 0:
            summary.methods_with_results += 1

        summary.total_expected += eval_result.expected_count
        summary.total_matched += len([m for m in eval_result.matches if m.matched])

    # Calculate averages (only for methods with results)
    valid_methods = [m for m in summary.methods if m.returned_count > 0]
    if valid_methods:
        summary.avg_precision = sum(m.precision for m in valid_methods) / len(valid_methods)
        summary.avg_recall = sum(m.recall for m in valid_methods) / len(valid_methods)
        summary.avg_f1 = sum(m.f1 for m in valid_methods) / len(valid_methods)
        summary.avg_weighted_recall = sum(m.weighted_recall for m in valid_methods) / len(valid_methods)

    # Print summary
    print("\n" + "=" * 60)
    print("ASSUMPTION AUDITING EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Methods evaluated:      {summary.total_methods}")
    print(f"Methods with results:   {summary.methods_with_results}")
    print(f"Methods not found:      {summary.methods_not_found}")
    print()
    print(f"Total expected:         {summary.total_expected}")
    print(f"Total matched:          {summary.total_matched}")
    print()
    print(f"Average Precision:      {summary.avg_precision:.1%}")
    print(f"Average Recall:         {summary.avg_recall:.1%}")
    print(f"Average F1:             {summary.avg_f1:.1%}")
    print(f"Weighted Recall:        {summary.avg_weighted_recall:.1%}")
    print("=" * 60)

    # Check against thresholds
    min_recall = config.get("min_recall_threshold", 0.5)
    min_precision = config.get("min_precision_threshold", 0.3)

    passed = True
    if summary.avg_recall < min_recall:
        print(f"\n⚠ FAIL: Recall {summary.avg_recall:.1%} < {min_recall:.1%} threshold")
        passed = False
    if summary.avg_precision < min_precision:
        print(f"\n⚠ FAIL: Precision {summary.avg_precision:.1%} < {min_precision:.1%} threshold")
        passed = False

    if passed:
        print("\n✓ PASS: All thresholds met")

    # Write output if requested
    if output_path:
        result_dict = summary.to_dict()
        result_dict["passed"] = passed
        result_dict["thresholds"] = {
            "min_recall": min_recall,
            "min_precision": min_precision,
        }

        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"\nResults written to: {output_path}")

    return summary


def main():
    """Entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate assumption auditing against golden dataset"
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path(__file__).parent.parent / "fixtures" / "eval" / "golden_assumptions.json",
        help="Path to golden assumptions JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output JSON file for metrics",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        help="Evaluate single method (filter by name)",
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Enable Ollama LLM fallback for sparse results",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed evaluation output",
    )

    args = parser.parse_args()

    if not args.golden.exists():
        print(f"Error: Golden dataset not found: {args.golden}")
        sys.exit(1)

    summary = asyncio.run(
        run_evaluation(
            args.golden,
            args.output,
            args.method,
            args.ollama,
            args.verbose,
        )
    )

    # Exit with error if thresholds not met
    if summary.avg_recall < 0.5 or summary.avg_precision < 0.3:
        sys.exit(1)


if __name__ == "__main__":
    main()
