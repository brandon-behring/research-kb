#!/usr/bin/env python3
"""
Docling fallback for scanned/image-only PDFs.

Pre-truncates PDFs to 10 pages via pypdfium2, runs Docling OCR extraction,
and patches the enrichment checkpoint with extracted text. Use after
enrich_catalog.py identifies non-extractable entries.

Requires exclusive GPU access (Docling uses Granite-258M, ~3GB VRAM).

Usage:
    python scripts/catalog_library/enrich_catalog_docling.py \
      --input fixtures/library_catalog/catalog_books_r2.json \
      --checkpoint fixtures/library_catalog/.enrichment_checkpoint.json \
      --limit 50 --quiet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("enrich_docling")


def _setup_logging(quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


# ISBN patterns (same as enrich_catalog.py)
_RE_ISBN13 = re.compile(r"ISBN[\s:\-]*(97[89][\d\-]{10,17})", re.I)
_RE_ISBN10 = re.compile(r"ISBN[\s:\-]*(\d{9}[\dXx])", re.I)
_RE_BARE_ISBN13 = re.compile(r"\b(97[89]\d{10})\b")


def _extract_isbns(text: str) -> list[str]:
    """Extract and normalize ISBNs from text."""
    raw: list[str] = []
    raw.extend(_RE_ISBN13.findall(text))
    raw.extend(_RE_ISBN10.findall(text))
    raw.extend(_RE_BARE_ISBN13.findall(text))
    seen: set[str] = set()
    result: list[str] = []
    for isbn in raw:
        clean = isbn.replace("-", "").replace(" ", "").strip()
        if len(clean) in (10, 13) and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def _truncate_pdf(src: Path, dst: Path, max_pages: int = 10) -> bool:
    """Truncate PDF to first N pages using pypdfium2. Returns True on success."""
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(src))
        n_pages = len(pdf)
        if n_pages <= max_pages:
            # Just copy
            import shutil

            shutil.copy2(src, dst)
            pdf.close()
            return True

        # Create truncated version
        new_pdf = pdfium.PdfDocument.new()
        new_pdf.import_pages(pdf, list(range(max_pages)))
        new_pdf.save(str(dst))
        new_pdf.close()
        pdf.close()
        return True

    except Exception as e:
        log.warning("Failed to truncate %s: %s", src.name, e)
        return False


def _make_docling_converter() -> Any:
    """Create a Docling DocumentConverter singleton for reuse across PDFs."""
    from docling.document_converter import DocumentConverter

    return DocumentConverter()


def _run_docling(pdf_path: Path, converter: Any) -> str:
    """Run Docling extraction on a single PDF. Returns extracted text."""
    try:
        result = converter.convert(str(pdf_path))
        return result.document.export_to_markdown()
    except Exception as e:
        log.warning("Docling failed on %s: %s", pdf_path.name, e)
        return ""


def main() -> None:
    p = argparse.ArgumentParser(
        description="Docling fallback for scanned/image-only PDFs",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("fixtures/library_catalog/catalog_books_r2.json"),
        help="R2 catalog with r2_text_extraction fields",
    )
    p.add_argument(
        "--input-other",
        type=Path,
        default=Path("fixtures/library_catalog/catalog_other_r2.json"),
        help="R2 other catalog",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("fixtures/library_catalog/.enrichment_checkpoint.json"),
        help="Enrichment checkpoint to patch",
    )
    p.add_argument("--limit", type=int, default=0, help="Process only first N scanned PDFs")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    _setup_logging(args.quiet)

    # Load catalog + checkpoint
    if not args.input.exists():
        log.error("Input not found: %s", args.input)
        sys.exit(1)

    books = json.loads(args.input.read_text())
    others = json.loads(args.input_other.read_text()) if args.input_other.exists() else []
    all_entries = books + others

    if not args.checkpoint.exists():
        log.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    checkpoint: dict[str, Any] = json.loads(args.checkpoint.read_text())

    # Find non-extractable entries
    scanned: list[dict] = []
    for entry in all_entries:
        sha = entry["sha256"]
        cp = checkpoint.get(sha, {})
        if not cp.get("extractable", True) and cp.get("text_done"):
            scanned.append(entry)

    if not scanned:
        log.info("No scanned/image-only PDFs found — nothing to do")
        return

    if args.limit > 0:
        scanned = scanned[: args.limit]

    log.info("Processing %d scanned PDFs with Docling", len(scanned))
    t0 = time.time()
    success = 0
    errors = 0

    # Create converter once (Granite-258M model load is expensive)
    converter = _make_docling_converter()

    with tempfile.TemporaryDirectory(prefix="docling_trunc_") as tmpdir:
        tmp = Path(tmpdir)

        for i, entry in enumerate(scanned):
            sha = entry["sha256"]
            src = Path(entry["full_path"])

            if not src.exists():
                log.warning("File not found: %s", src)
                errors += 1
                continue

            # Pre-truncate to 10 pages
            trunc_path = tmp / f"{sha[:12]}.pdf"
            if not _truncate_pdf(src, trunc_path, max_pages=10):
                errors += 1
                continue

            # Run Docling
            text = _run_docling(trunc_path, converter)
            if not text or len(text.strip()) < 50:
                log.debug("Docling returned minimal text for %s", entry.get("filename", ""))
                errors += 1
                continue

            # Patch checkpoint
            text_sample = text[:2000]
            normalized = re.sub(r"\s+", " ", text[:5000].lower().strip())
            fingerprint = (
                hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()[:16]
                if len(normalized) > 50
                else None
            )

            first_3k = text[:3000].lower()
            has_toc = "table of contents" in first_3k or "contents\n" in first_3k

            checkpoint[sha] = {
                "text_done": True,
                "text_sample": text_sample,
                "chars_extracted": len(text),
                "content_fingerprint": fingerprint,
                "has_toc": has_toc,
                "extractable": True,  # Now extractable via Docling
                "isbns": _extract_isbns(text),
                "error": None,
                "docling_extracted": True,
            }
            success += 1

            if not args.quiet and (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                log.info("  %d/%d done (%.1fs)", i + 1, len(scanned), elapsed)

            # Clean up truncated file
            trunc_path.unlink(missing_ok=True)

    # Save checkpoint
    args.checkpoint.write_text(json.dumps(checkpoint, indent=2))

    elapsed = time.time() - t0
    log.info(
        "Docling complete: %d success, %d errors in %.1fs",
        success,
        errors,
        elapsed,
    )

    # Summary
    summary = {
        "scanned_total": len(scanned),
        "docling_success": success,
        "docling_errors": errors,
        "elapsed_seconds": round(elapsed, 1),
    }
    print(json.dumps(summary, indent=2))
    log.info("Re-run enrich_catalog.py Stages 4-5 to reclassify patched entries")


if __name__ == "__main__":
    main()
