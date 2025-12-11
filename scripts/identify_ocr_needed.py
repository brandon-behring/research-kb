#!/usr/bin/env python3
"""
Identify PDFs that need OCR (scanned, image-only).

Scans a directory of PDFs and identifies which ones have no extractable text
and would require OCR to process. Outputs a quarantine list for manual handling.

Usage:
    python scripts/identify_ocr_needed.py /path/to/pdfs
    python scripts/identify_ocr_needed.py /path/to/pdfs --output quarantine.txt
    python scripts/identify_ocr_needed.py /path/to/pdfs --threshold 100

Created: 2025-12-10
Part of: U1 Track B (PDF Improvements)
"""

import argparse
import sys
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF


def is_text_extractable(pdf_path: Path, min_chars_per_page: int = 100) -> tuple[bool, dict]:
    """
    Check if a PDF has extractable text.

    Args:
        pdf_path: Path to PDF file
        min_chars_per_page: Minimum average chars per page to consider extractable

    Returns:
        Tuple of (is_extractable, metadata dict with details)
    """
    metadata = {
        "path": str(pdf_path),
        "pages": 0,
        "total_chars": 0,
        "avg_chars_per_page": 0.0,
        "error": None,
        "needs_ocr": False,
    }

    try:
        doc = fitz.open(str(pdf_path))
        metadata["pages"] = len(doc)

        if metadata["pages"] == 0:
            metadata["error"] = "empty_pdf"
            metadata["needs_ocr"] = False  # Can't OCR empty PDF
            doc.close()
            return False, metadata

        total_chars = 0
        for page in doc:
            text = page.get_text()
            total_chars += len(text.strip())

        doc.close()

        metadata["total_chars"] = total_chars
        metadata["avg_chars_per_page"] = total_chars / metadata["pages"]

        # Determine if OCR is needed
        if metadata["avg_chars_per_page"] < min_chars_per_page:
            metadata["needs_ocr"] = True
            return False, metadata
        else:
            metadata["needs_ocr"] = False
            return True, metadata

    except fitz.FileDataError as e:
        metadata["error"] = f"corrupted: {str(e)}"
        metadata["needs_ocr"] = False
        return False, metadata

    except fitz.fitz.FileNotFoundError as e:
        metadata["error"] = f"not_found: {str(e)}"
        metadata["needs_ocr"] = False
        return False, metadata

    except Exception as e:
        error_str = str(e).lower()
        # Encrypted PDFs shouldn't be flagged for OCR
        if "encrypted" in error_str or "password" in error_str:
            metadata["error"] = "encrypted"
        else:
            metadata["error"] = str(e)
        metadata["needs_ocr"] = False
        return False, metadata


def scan_directory(
    directory: Path,
    min_chars_per_page: int = 100,
    recursive: bool = True,
) -> Iterator[dict]:
    """
    Scan a directory for PDFs and check text extractability.

    Args:
        directory: Directory to scan
        min_chars_per_page: Threshold for considering text extractable
        recursive: Whether to scan subdirectories

    Yields:
        Metadata dict for each PDF
    """
    pattern = "**/*.pdf" if recursive else "*.pdf"

    pdf_files = sorted(directory.glob(pattern))
    total = len(pdf_files)

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\r  Scanning [{i}/{total}] {pdf_path.name[:50]:<50}", end="", flush=True)
        _, metadata = is_text_extractable(pdf_path, min_chars_per_page)
        yield metadata

    print()  # Newline after progress


def generate_report(results: list[dict]) -> str:
    """Generate a summary report from scan results."""
    total = len(results)
    needs_ocr = [r for r in results if r["needs_ocr"]]
    errors = [r for r in results if r["error"]]
    extractable = [r for r in results if not r["needs_ocr"] and not r["error"]]

    report = []
    report.append("=" * 60)
    report.append("OCR DETECTION REPORT")
    report.append("=" * 60)
    report.append(f"\nTotal PDFs scanned: {total}")
    report.append(f"  ✅ Text extractable: {len(extractable)}")
    report.append(f"  ⚠️  Needs OCR: {len(needs_ocr)}")
    report.append(f"  ❌ Errors: {len(errors)}")

    if needs_ocr:
        report.append(f"\n--- PDFs Needing OCR ({len(needs_ocr)}) ---")
        for r in needs_ocr:
            report.append(f"  {r['path']}")
            report.append(f"    Pages: {r['pages']}, Avg chars/page: {r['avg_chars_per_page']:.1f}")

    if errors:
        report.append(f"\n--- Errors ({len(errors)}) ---")
        for r in errors:
            report.append(f"  {r['path']}")
            report.append(f"    Error: {r['error']}")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Identify PDFs that need OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan library directory
  python scripts/identify_ocr_needed.py ~/library/pdfs

  # Output quarantine list
  python scripts/identify_ocr_needed.py ~/library --output ocr_needed.txt

  # Use higher threshold (more strict)
  python scripts/identify_ocr_needed.py ~/library --threshold 200
        """,
    )
    parser.add_argument("directory", type=Path, help="Directory to scan for PDFs")
    parser.add_argument(
        "--output", "-o", type=Path, help="Output file for quarantine list"
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=100,
        help="Min chars per page threshold (default: 100)",
    )
    parser.add_argument(
        "--no-recursive", action="store_true", help="Don't scan subdirectories"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not args.directory.exists():
        print(f"Error: Directory not found: {args.directory}")
        return 1

    print(f"Scanning {args.directory} for PDFs needing OCR...")
    print(f"Threshold: {args.threshold} chars/page minimum\n")

    results = list(
        scan_directory(
            args.directory,
            min_chars_per_page=args.threshold,
            recursive=not args.no_recursive,
        )
    )

    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        print(generate_report(results))

    # Write quarantine list if requested
    if args.output:
        needs_ocr = [r for r in results if r["needs_ocr"]]
        with open(args.output, "w") as f:
            f.write(f"# PDFs needing OCR (generated by identify_ocr_needed.py)\n")
            f.write(f"# Threshold: {args.threshold} chars/page\n")
            f.write(f"# Total: {len(needs_ocr)} files\n\n")
            for r in needs_ocr:
                f.write(f"{r['path']}\n")
        print(f"\n✅ Quarantine list written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
