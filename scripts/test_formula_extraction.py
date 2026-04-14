#!/usr/bin/env python3
"""Compare PDF equation extraction quality: Docling default vs formula enrichment.

Runs each extraction in a separate subprocess to guarantee full VRAM cleanup
between runs (PyTorch doesn't reliably free CUDA memory within a process).

Usage:
    # Stop Ollama first to free GPU VRAM!
    sudo systemctl stop ollama
    python scripts/test_formula_extraction.py

    # Test a specific PDF
    python scripts/test_formula_extraction.py --pdf path/to/math_heavy.pdf

    # Skip default run (already have it)
    python scripts/test_formula_extraction.py --skip-default

    # Also test CPU-only formula enrichment (slower but no VRAM limit)
    python scripts/test_formula_extraction.py --test-cpu
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Default test PDF: Murty's Problems in Analytic Number Theory (88.3% formula gap rate)
DEFAULT_PDF = Path(
    "fixtures/library_books/mathematics/Problems in Analytic Number Theory (M. Ram Murty).pdf"
)

OUTPUT_DIR = Path("fixtures/eval")


def run_extraction(pdf_path: str, mode: str, output_path: str) -> dict:
    """Run a single extraction in a subprocess.

    Parameters
    ----------
    pdf_path : str
        Path to PDF file.
    mode : str
        One of: "default", "formula_gpu", "formula_cpu".
    output_path : str
        Where to save the markdown output.

    Returns
    -------
    dict
        Extraction stats (time, formula_gaps, chars, etc.).
    """
    script = f"""
import sys, time, json
from pathlib import Path

pdf_path = "{pdf_path}"
mode = "{mode}"
output_path = "{output_path}"

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

pipeline_options = PdfPipelineOptions()

if mode == "default":
    pipeline_options.do_formula_enrichment = False
elif mode == "formula_gpu":
    pipeline_options.do_formula_enrichment = True
elif mode == "formula_cpu":
    pipeline_options.do_formula_enrichment = True
    from docling.datamodel.pipeline_options import AcceleratorOptions
    acc = AcceleratorOptions(device="cpu")
    pipeline_options.accelerator_options = acc

converter = DocumentConverter(
    format_options={{
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
    }}
)

t0 = time.time()
result = converter.convert(pdf_path)
md = result.document.export_to_markdown()
elapsed = time.time() - t0

Path(output_path).write_text(md)

formula_gaps = md.count("formula-not-decoded")
latex_dollar = md.count("$")
lines_count = md.count("\\n")

stats = {{
    "mode": mode,
    "time_seconds": round(elapsed, 1),
    "total_chars": len(md),
    "formula_gaps": formula_gaps,
    "latex_dollar_signs": latex_dollar,
    "lines": lines_count,
}}
print(json.dumps(stats))
"""

    print(f"  Starting subprocess ({mode})...")
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min max
    )

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        stderr_lines = result.stderr.strip().split("\n")
        # Show last 10 lines of stderr for diagnosis
        for line in stderr_lines[-10:]:
            print(f"    {line}")
        return {"mode": mode, "error": stderr_lines[-1] if stderr_lines else "unknown"}

    # Parse stats from last line of stdout
    stdout_lines = result.stdout.strip().split("\n")
    try:
        stats = json.loads(stdout_lines[-1])
    except (json.JSONDecodeError, IndexError):
        stats = {"mode": mode, "error": "failed to parse stats"}

    wall_time = time.time() - t0
    stats["wall_time"] = round(wall_time, 1)
    return stats


def analyze_output(md_path: Path, label: str) -> None:
    """Print sample formulas from extraction output.

    Parameters
    ----------
    md_path : Path
        Path to markdown output file.
    label : str
        Display label.
    """
    if not md_path.exists():
        print(f"  {label}: output file not found")
        return

    md = md_path.read_text()
    lines = md.split("\n")

    # Show a formula gap example
    for i, line in enumerate(lines):
        if "formula-not-decoded" in line:
            ctx_start = max(0, i - 2)
            ctx_end = min(len(lines), i + 3)
            print(f"\n  Gap example (line {i}):")
            for j in range(ctx_start, ctx_end):
                prefix = ">>>" if "formula-not-decoded" in lines[j] else "   "
                print(f"    {prefix} {lines[j][:120]}")
            break

    # Show a decoded formula example
    for i, line in enumerate(lines):
        if ("$$" in line or "\\(" in line) and "formula-not-decoded" not in line and len(line) > 10:
            print(f"\n  Decoded formula example (line {i}):")
            print(f"    {line[:250]}")
            break

    # Count formulas with actual LaTeX content
    decoded_formulas = sum(1 for l in lines if "$$" in l and "formula-not-decoded" not in l)
    print(f"\n  Lines with decoded LaTeX ($$): {decoded_formulas}")


def main() -> None:
    """Run extraction comparison."""
    parser = argparse.ArgumentParser(description="Test PDF formula extraction quality")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="PDF to test")
    parser.add_argument("--skip-default", action="store_true", help="Skip default extraction")
    parser.add_argument(
        "--test-cpu", action="store_true", help="Also test CPU-only formula enrichment"
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"ERROR: PDF not found: {args.pdf}")
        sys.exit(1)

    print(f"Test PDF: {args.pdf.name}")
    print(f"Size: {args.pdf.stat().st_size / 1024 / 1024:.1f} MB")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # --- Run 1: Default ---
    if not args.skip_default:
        print("\n" + "=" * 60)
        print("RUN 1: Docling default (formula enrichment OFF)")
        print("=" * 60)
        out_default = OUTPUT_DIR / "formula_test_default.md"
        stats = run_extraction(str(args.pdf), "default", str(out_default))
        results.append(stats)

        if "error" not in stats:
            print(f"  Time: {stats['time_seconds']}s (wall: {stats['wall_time']}s)")
            print(f"  Chars: {stats['total_chars']:,}")
            print(f"  Formula gaps: {stats['formula_gaps']}")
            print(f"  LaTeX $ signs: {stats['latex_dollar_signs']}")
            analyze_output(out_default, "Default")
        else:
            print(f"  Error: {stats['error']}")

    # --- Run 2: Formula enrichment ON (GPU) ---
    print("\n" + "=" * 60)
    print("RUN 2: Docling + CodeFormulaV2 (GPU)")
    print("=" * 60)
    out_formula_gpu = OUTPUT_DIR / "formula_test_enriched_gpu.md"
    stats = run_extraction(str(args.pdf), "formula_gpu", str(out_formula_gpu))
    results.append(stats)

    if "error" not in stats:
        print(f"  Time: {stats['time_seconds']}s (wall: {stats['wall_time']}s)")
        print(f"  Chars: {stats['total_chars']:,}")
        print(f"  Formula gaps: {stats['formula_gaps']}")
        print(f"  LaTeX $ signs: {stats['latex_dollar_signs']}")
        analyze_output(out_formula_gpu, "Formula GPU")
    else:
        print(f"  Error: {stats['error']}")
        # If GPU failed, try CPU automatically
        if "OutOfMemoryError" in str(stats.get("error", "")):
            print("\n  GPU OOM detected — retrying with CPU...")
            args.test_cpu = True

    # --- Run 3: Formula enrichment ON (CPU) --- optional
    if args.test_cpu:
        print("\n" + "=" * 60)
        print("RUN 3: Docling + CodeFormulaV2 (CPU fallback)")
        print("=" * 60)
        out_formula_cpu = OUTPUT_DIR / "formula_test_enriched_cpu.md"
        stats = run_extraction(str(args.pdf), "formula_cpu", str(out_formula_cpu))
        results.append(stats)

        if "error" not in stats:
            print(f"  Time: {stats['time_seconds']}s (wall: {stats['wall_time']}s)")
            print(f"  Chars: {stats['total_chars']:,}")
            print(f"  Formula gaps: {stats['formula_gaps']}")
            print(f"  LaTeX $ signs: {stats['latex_dollar_signs']}")
            analyze_output(out_formula_cpu, "Formula CPU")
        else:
            print(f"  Error: {stats['error']}")

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n  {'Mode':<25s} {'Time':>8s} {'Chars':>10s} {'Gaps':>8s} {'LaTeX $':>10s}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 10} {'-' * 8} {'-' * 10}")
    for r in results:
        if "error" in r:
            print(f"  {r['mode']:<25s} {'FAILED':>8s}  {r.get('error', 'unknown')[:40]}")
        else:
            print(
                f"  {r['mode']:<25s} {r['time_seconds']:>7.0f}s"
                f" {r['total_chars']:>10,}"
                f" {r['formula_gaps']:>8,}"
                f" {r['latex_dollar_signs']:>10,}"
            )

    # Check if any successful enriched run
    enriched = [r for r in results if r["mode"] != "default" and "error" not in r]
    default = [r for r in results if r["mode"] == "default" and "error" not in r]

    if enriched and default:
        d = default[0]
        e = enriched[0]
        reduction = (d["formula_gaps"] - e["formula_gaps"]) / d["formula_gaps"] * 100
        print(f"\n  Formula gap reduction: {reduction:.1f}%")
        print(f"  ({d['formula_gaps']} → {e['formula_gaps']})")

    print(f"\n  Output files in {OUTPUT_DIR}/:")
    for f in sorted(OUTPUT_DIR.glob("formula_test_*.md")):
        print(f"    {f.name} ({f.stat().st_size / 1024:.0f} KB)")
    print(
        "\n  Diff them: diff fixtures/eval/formula_test_default.md fixtures/eval/formula_test_enriched_*.md"
    )


if __name__ == "__main__":
    main()
