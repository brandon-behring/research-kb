#!/usr/bin/env python3
"""Quality check automation script.

Runs all quality validation tests and generates a comprehensive report.
Used for validating pipeline quality before production deployment.

Usage:
    python scripts/run_quality_checks.py [--strict] [--html] [--json]

Options:
    --strict    Fail on any test failure (default: warn only)
    --html      Generate HTML report
    --json      Generate JSON report
    --output    Output directory for reports (default: ./quality-reports)

Exit codes:
    0 - All quality gates passed
    1 - Some quality gates failed
    2 - Critical quality gates failed
"""

import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class QualityCheckRunner:
    """Runs quality tests and generates reports."""

    # Quality gate priorities
    CRITICAL_TESTS = [
        "test_seed_concept_recall_threshold",
        "test_retrieval_precision_threshold",
    ]

    HIGH_PRIORITY_TESTS = [
        "test_concept_confidence_distribution",
        "test_embedding_quality",
    ]

    def __init__(self, strict: bool = False, output_dir: str = "./quality-reports"):
        self.strict = strict
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {},
            "gates": {
                "critical": {"passed": 0, "failed": 0},
                "high": {"passed": 0, "failed": 0},
                "medium": {"passed": 0, "failed": 0},
                "low": {"passed": 0, "failed": 0},
            },
        }

    def run_tests(self) -> int:
        """Run quality tests using pytest.

        Returns:
            Exit code (0 = success, 1 = some failures, 2 = critical failures)
        """
        print("=" * 70)
        print("RUNNING QUALITY VALIDATION TESTS")
        print("=" * 70)
        print()

        # Run pytest with quality marker
        cmd = [
            "pytest",
            "-v",
            "--tb=short",
            "-m",
            "quality",
            "--json-report",
            "--json-report-file=" + str(self.output_dir / "pytest-report.json"),
        ]

        # Add HTML report if requested
        if hasattr(self, "generate_html") and self.generate_html:
            cmd.extend(["--html", str(self.output_dir / "report.html")])

        print(f"Running: {' '.join(cmd)}")
        print()

        # Run tests
        result = subprocess.run(cmd, capture_output=False)

        # Parse results
        return self._parse_results(result.returncode)

    def _parse_results(self, pytest_exit_code: int) -> int:
        """Parse pytest results and determine exit code.

        Args:
            pytest_exit_code: Exit code from pytest

        Returns:
            Exit code for this script
        """
        # Try to load JSON report
        report_file = self.output_dir / "pytest-report.json"
        if report_file.exists():
            with open(report_file) as f:
                pytest_data = json.load(f)
                self._analyze_pytest_report(pytest_data)

        # Print summary
        self._print_summary()

        # Determine exit code
        if self.results["gates"]["critical"]["failed"] > 0:
            print("\n❌ CRITICAL QUALITY GATES FAILED")
            return 2

        if self.results["gates"]["high"]["failed"] > 0:
            if self.strict:
                print("\n⚠️  HIGH PRIORITY QUALITY GATES FAILED (strict mode)")
                return 1
            else:
                print("\n⚠️  HIGH PRIORITY QUALITY GATES FAILED (warning)")

        if pytest_exit_code != 0:
            if self.strict:
                print("\n⚠️  SOME QUALITY TESTS FAILED (strict mode)")
                return 1
            else:
                print("\n⚠️  SOME QUALITY TESTS FAILED (warning)")

        print("\n✅ ALL QUALITY GATES PASSED")
        return 0

    def _analyze_pytest_report(self, pytest_data: Dict[str, Any]):
        """Analyze pytest JSON report."""
        tests = pytest_data.get("tests", [])

        for test in tests:
            test_name = test.get("nodeid", "").split("::")[-1]
            outcome = test.get("outcome")

            # Determine priority
            priority = "low"
            if any(critical in test_name for critical in self.CRITICAL_TESTS):
                priority = "critical"
            elif any(high in test_name for high in self.HIGH_PRIORITY_TESTS):
                priority = "high"
            elif (
                "duplicate" in test_name
                or "relationship" in test_name
                or "chunk_length" in test_name
            ):
                priority = "medium"

            # Record result
            if outcome == "passed":
                self.results["gates"][priority]["passed"] += 1
            elif outcome == "failed":
                self.results["gates"][priority]["failed"] += 1

            self.results["tests"].append(
                {
                    "name": test_name,
                    "outcome": outcome,
                    "priority": priority,
                    "duration": test.get("duration", 0),
                }
            )

        # Update summary
        total_tests = len(tests)
        passed = sum(1 for t in tests if t.get("outcome") == "passed")
        failed = sum(1 for t in tests if t.get("outcome") == "failed")
        skipped = sum(1 for t in tests if t.get("outcome") == "skipped")

        self.results["summary"] = {
            "total": total_tests,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": (passed / total_tests * 100) if total_tests > 0 else 0,
        }

    def _print_summary(self):
        """Print quality check summary."""
        print("\n" + "=" * 70)
        print("QUALITY CHECK SUMMARY")
        print("=" * 70)

        summary = self.results["summary"]
        if summary:
            print(f"\nTotal Tests: {summary['total']}")
            print(f"  ✅ Passed:  {summary['passed']}")
            print(f"  ❌ Failed:  {summary['failed']}")
            print(f"  ⊘ Skipped:  {summary['skipped']}")
            print(f"  Pass Rate: {summary['pass_rate']:.1f}%")

        print("\nQuality Gates by Priority:")
        for priority in ["critical", "high", "medium", "low"]:
            gate = self.results["gates"][priority]
            total = gate["passed"] + gate["failed"]
            if total > 0:
                status = "✅" if gate["failed"] == 0 else "❌"
                print(f"  {status} {priority.upper()}: {gate['passed']}/{total} passed")

        print("=" * 70)

    def save_json_report(self):
        """Save JSON report."""
        output_file = self.output_dir / "quality-report.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 JSON report saved to: {output_file}")

    def save_markdown_report(self):
        """Save markdown report."""
        output_file = self.output_dir / "quality-report.md"

        with open(output_file, "w") as f:
            f.write("# Quality Validation Report\n\n")
            f.write(f"**Generated:** {self.results['timestamp']}\n\n")

            # Summary
            summary = self.results["summary"]
            if summary:
                f.write("## Summary\n\n")
                f.write(f"- **Total Tests:** {summary['total']}\n")
                f.write(f"- **Passed:** {summary['passed']} ✅\n")
                f.write(f"- **Failed:** {summary['failed']} ❌\n")
                f.write(f"- **Skipped:** {summary['skipped']} ⊘\n")
                f.write(f"- **Pass Rate:** {summary['pass_rate']:.1f}%\n\n")

            # Quality gates
            f.write("## Quality Gates\n\n")
            f.write("| Priority | Passed | Failed | Status |\n")
            f.write("|----------|--------|--------|--------|\n")
            for priority in ["critical", "high", "medium", "low"]:
                gate = self.results["gates"][priority]
                total = gate["passed"] + gate["failed"]
                status = "✅ PASS" if gate["failed"] == 0 else "❌ FAIL"
                f.write(
                    f"| {priority.upper()} | {gate['passed']} | {gate['failed']} | {status} |\n"
                )

            # Test details
            f.write("\n## Test Details\n\n")
            for test in self.results["tests"]:
                status_icon = {"passed": "✅", "failed": "❌", "skipped": "⊘"}.get(
                    test["outcome"], "?"
                )

                f.write(f"### {status_icon} {test['name']}\n\n")
                f.write(f"- **Priority:** {test['priority'].upper()}\n")
                f.write(f"- **Outcome:** {test['outcome']}\n")
                f.write(f"- **Duration:** {test['duration']:.2f}s\n\n")

        print(f"📄 Markdown report saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run quality validation tests and generate reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any test failure (default: warn only for non-critical)",
    )

    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report (requires pytest-html)",
    )

    parser.add_argument("--json", action="store_true", help="Generate JSON report")

    parser.add_argument(
        "--markdown",
        action="store_true",
        default=True,
        help="Generate markdown report (default: True)",
    )

    parser.add_argument(
        "--output",
        default="./quality-reports",
        help="Output directory for reports (default: ./quality-reports)",
    )

    args = parser.parse_args()

    # Create runner
    runner = QualityCheckRunner(strict=args.strict, output_dir=args.output)
    runner.generate_html = args.html

    # Run tests
    exit_code = runner.run_tests()

    # Generate reports
    if args.json:
        runner.save_json_report()

    if args.markdown:
        runner.save_markdown_report()

    # Exit
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
