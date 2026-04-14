#!/usr/bin/env python3
"""Compare PDF formula extraction quality across multiple extractors.

Runs each extractor on a page range of test PDFs, collects metrics, and generates
a side-by-side COMPARISON.md report. Each extractor runs in a subprocess to guarantee
GPU VRAM cleanup between runs.

Extractors:
  docling_default   — Docling with formula enrichment OFF (current pipeline)
  docling_patched   — Docling default + PyMuPDF bbox text fallback for gaps
  pymupdf_raw       — PyMuPDF text extraction (pymupdf4llm + raw page.get_text)
  mineru            — MinerU pipeline with UniMERNet formula recognition
  paddleocr         — PaddleOCR-VL document understanding

Hardware:
  Stop Ollama before GPU extractors: sudo systemctl stop ollama
  RTX 2070 (8GB) — run extractors sequentially, one GPU model at a time.

Usage:
    # Quick sanity check (2 fast extractors, 1 PDF)
    python scripts/compare_formula_extractors.py --extractors docling_default,pymupdf_raw --pdfs murty

    # Full comparison
    python scripts/compare_formula_extractors.py --extractors all

    # Skip venv install (reuse existing)
    python scripts/compare_formula_extractors.py --extractors mineru,paddleocr --skip-install

    # Custom page range
    python scripts/compare_formula_extractors.py --pages 10-15
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "fixtures" / "eval" / "formula_comparison"

# ── Test PDFs ────────────────────────────────────────────────────────────────

TEST_PDFS = {
    "murty": {
        "path": REPO_ROOT
        / "fixtures/library_books/mathematics/Problems in Analytic Number Theory (M. Ram Murty).pdf",
        "domain": "mathematics",
        "gap_rate": "88.3%",
        "note": "458 pages, heavy formulas",
    },
    "ross": {
        "path": REPO_ROOT / "fixtures/library_books/First Course in Probability 7E - Ross.pdf",
        "domain": "probability",
        "gap_rate": "91.4%",
        "note": "1MB, probability formulas",
    },
    "boyd": {
        "path": REPO_ROOT
        / "fixtures/textbooks/migrated/stephen_boyd_lieven_vandenberghe_-_introduction_to_nd.pdf",
        "domain": "mathematics",
        "gap_rate": "49.8%",
        "note": "7MB, optimization/LA",
    },
    "dml": {
        "path": REPO_ROOT / "fixtures/papers/chernozhukov_dml_2018.pdf",
        "domain": "causal_inference",
        "gap_rate": "<10%",
        "note": "Paper, few formulas",
    },
    "manning": {
        "path": REPO_ROOT / "fixtures/textbooks/manning_causal_ai_2024.pdf",
        "domain": "causal_inference",
        "gap_rate": "1.4%",
        "note": "Control case, minimal math",
    },
}

# ── Metrics ──────────────────────────────────────────────────────────────────


@dataclass
class ExtractionMetrics:
    """Metrics for a single extractor run on a single PDF."""

    extractor: str
    pdf_key: str
    pages: str
    time_seconds: float = 0.0
    wall_time: float = 0.0
    total_chars: int = 0
    formula_gaps: int = 0
    picture_omitted: int = 0
    latex_display: int = 0
    latex_inline: int = 0
    lines: int = 0
    error: Optional[str] = None
    sample_formulas: list[str] = field(default_factory=list)
    sample_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "extractor": self.extractor,
            "pdf_key": self.pdf_key,
            "pages": self.pages,
            "time_seconds": self.time_seconds,
            "wall_time": self.wall_time,
            "total_chars": self.total_chars,
            "formula_gaps": self.formula_gaps,
            "picture_omitted": self.picture_omitted,
            "latex_display": self.latex_display,
            "latex_inline": self.latex_inline,
            "lines": self.lines,
            "error": self.error,
            "sample_formulas": self.sample_formulas[:3],
            "sample_gaps": self.sample_gaps[:3],
        }


def compute_metrics(md_text: str, extractor: str, pdf_key: str, pages: str) -> ExtractionMetrics:
    """Compute extraction quality metrics from markdown output.

    Parameters
    ----------
    md_text : str
        Markdown output from extractor.
    extractor : str
        Extractor name.
    pdf_key : str
        PDF identifier.
    pages : str
        Page range string.

    Returns
    -------
    ExtractionMetrics
        Computed metrics.
    """
    lines = md_text.split("\n")

    # Count formula gaps
    formula_gaps = md_text.count("formula-not-decoded")
    picture_omitted = md_text.count("picture") + md_text.count("intentionally omitted")

    # Count LaTeX blocks
    latex_display = md_text.count("$$") // 2
    # Inline LaTeX: $ signs minus display $$ signs
    dollar_count = md_text.count("$")
    latex_inline = max(0, (dollar_count - latex_display * 4) // 2)

    # Collect sample formulas (lines with $$ or \()
    sample_formulas: list[str] = []
    for line in lines:
        if ("$$" in line or "\\(" in line) and "formula-not-decoded" not in line and len(line) > 10:
            sample_formulas.append(line.strip()[:200])
            if len(sample_formulas) >= 5:
                break

    # Collect sample gaps
    sample_gaps: list[str] = []
    for i, line in enumerate(lines):
        if "formula-not-decoded" in line:
            ctx_start = max(0, i - 1)
            ctx_end = min(len(lines), i + 2)
            context = " | ".join(lines[ctx_start:ctx_end])
            sample_gaps.append(context[:200])
            if len(sample_gaps) >= 3:
                break

    return ExtractionMetrics(
        extractor=extractor,
        pdf_key=pdf_key,
        pages=pages,
        total_chars=len(md_text),
        formula_gaps=formula_gaps,
        picture_omitted=picture_omitted,
        latex_display=latex_display,
        latex_inline=latex_inline,
        lines=len(lines),
        sample_formulas=sample_formulas,
        sample_gaps=sample_gaps,
    )


# ── Extractor Runners ────────────────────────────────────────────────────────


def _run_subprocess(
    python_exe: str,
    script: str,
    timeout: int = 600,
    label: str = "extractor",
) -> tuple[str, str, int]:
    """Run a Python script in subprocess.

    Parameters
    ----------
    python_exe : str
        Path to Python interpreter.
    script : str
        Python code to execute.
    timeout : int
        Max seconds.
    label : str
        Display label for logging.

    Returns
    -------
    tuple[str, str, int]
        (stdout, stderr, returncode)
    """
    print(f"    [{label}] Starting subprocess...")
    result = subprocess.run(
        [python_exe, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )
    return result.stdout, result.stderr, result.returncode


def run_docling_default(pdf_path: str, page_start: int, page_end: int) -> tuple[str, float]:
    """Run Docling with default settings (formula enrichment OFF).

    Parameters
    ----------
    pdf_path : str
        Path to PDF.
    page_start : int
        Start page (1-indexed).
    page_end : int
        End page (1-indexed, inclusive).

    Returns
    -------
    tuple[str, float]
        (markdown_output, elapsed_seconds)
    """
    script = textwrap.dedent(f"""\
        import time, json
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_formula_enrichment = False

        converter = DocumentConverter(
            format_options={{
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }}
        )

        t0 = time.time()
        result = converter.convert("{pdf_path}")
        doc = result.document

        # Filter to target pages via iterate_items
        md_parts = []
        for item, level in doc.iterate_items():
            if hasattr(item, 'prov') and item.prov:
                page = item.prov[0].page_no
                if {page_start} <= page <= {page_end}:
                    text = getattr(item, 'text', '') or getattr(item, 'orig', '') or ''
                    label = getattr(item, 'label', '')
                    if 'formula' in str(label).lower() and not text:
                        md_parts.append('<!-- formula-not-decoded -->')
                    elif text:
                        md_parts.append(text)

        # Also get full markdown for page-range comparison
        full_md = doc.export_to_markdown()
        elapsed = time.time() - t0

        # Output: JSON with stats, then separator, then markdown
        stats = {{"time_seconds": round(elapsed, 1)}}
        print(json.dumps(stats))
        print("---MARKDOWN_SEPARATOR---")
        print(full_md)
    """)

    stdout, stderr, rc = _run_subprocess(
        sys.executable, script, timeout=1800, label="docling_default"
    )
    if rc != 0:
        error_lines = stderr.strip().split("\n")[-5:]
        raise RuntimeError(f"docling_default failed: {' '.join(error_lines)}")

    parts = stdout.split("---MARKDOWN_SEPARATOR---", 1)
    stats = json.loads(parts[0].strip().split("\n")[-1])
    md = parts[1] if len(parts) > 1 else ""
    return md, stats["time_seconds"]


def run_docling_patched(pdf_path: str, page_start: int, page_end: int) -> tuple[str, float]:
    """Run Docling default + PyMuPDF bbox text fallback for formula gaps.

    For each FormulaItem where text is empty (formula-not-decoded):
    1. Get bounding box from item.prov[0].bbox
    2. Extract raw text from PyMuPDF at that bbox region
    3. Replace the empty text with extracted text

    Parameters
    ----------
    pdf_path : str
        Path to PDF.
    page_start : int
        Start page (1-indexed).
    page_end : int
        End page (1-indexed, inclusive).

    Returns
    -------
    tuple[str, float]
        (markdown_output, elapsed_seconds)
    """
    script = textwrap.dedent(f"""\
        import time, json
        import pymupdf
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling_core.types.doc.document import FormulaItem

        pdf_path = "{pdf_path}"

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_formula_enrichment = False

        converter = DocumentConverter(
            format_options={{
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }}
        )

        t0 = time.time()
        result = converter.convert(pdf_path)
        doc = result.document

        # Open PyMuPDF for bbox text extraction
        fitz_doc = pymupdf.open(pdf_path)

        # Get page dimensions from Docling for coordinate scaling
        patched = 0
        failed = 0

        for item, level in doc.iterate_items():
            if not isinstance(item, FormulaItem):
                continue
            if item.text and item.text.strip():
                continue  # Already decoded
            if not item.prov:
                continue

            prov = item.prov[0]
            page_no = prov.page_no  # 1-indexed in Docling
            bbox = prov.bbox

            try:
                # Docling page_no is 1-indexed, PyMuPDF is 0-indexed
                fitz_page = fitz_doc[page_no - 1]

                # Get Docling page size for coordinate conversion
                # Docling uses BOTTOMLEFT origin: t > b (t = distance from bottom)
                # PyMuPDF uses TOPLEFT origin: y0 < y1 (y0 = distance from top)
                # Conversion: pymupdf_y0 = page_height - docling_t
                #             pymupdf_y1 = page_height - docling_b
                docling_page = doc.pages.get(page_no)
                if docling_page:
                    dl_w = docling_page.size.width
                    dl_h = docling_page.size.height
                    fm_w = fitz_page.rect.width
                    fm_h = fitz_page.rect.height
                    x_scale = fm_w / dl_w if dl_w > 0 else 1.0
                    y_scale = fm_h / dl_h if dl_h > 0 else 1.0

                    # BOTTOMLEFT -> TOPLEFT: flip Y axis
                    pymupdf_y0 = (dl_h - bbox.t) * y_scale  # top in TOPLEFT
                    pymupdf_y1 = (dl_h - bbox.b) * y_scale  # bottom in TOPLEFT
                    rect = pymupdf.Rect(
                        bbox.l * x_scale,
                        pymupdf_y0,
                        bbox.r * x_scale,
                        pymupdf_y1,
                    )
                else:
                    # No page info — try TOPLEFT directly as fallback
                    rect = pymupdf.Rect(bbox.l, bbox.t, bbox.r, bbox.b)

                raw_text = fitz_page.get_text("text", clip=rect).strip()

                if raw_text:
                    item.text = raw_text
                    patched += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1

        fitz_doc.close()

        md = doc.export_to_markdown()
        elapsed = time.time() - t0

        stats = {{
            "time_seconds": round(elapsed, 1),
            "patched": patched,
            "patch_failed": failed,
        }}
        print(json.dumps(stats))
        print("---MARKDOWN_SEPARATOR---")
        print(md)
    """)

    stdout, stderr, rc = _run_subprocess(
        sys.executable, script, timeout=1800, label="docling_patched"
    )
    if rc != 0:
        error_lines = stderr.strip().split("\n")[-5:]
        raise RuntimeError(f"docling_patched failed: {' '.join(error_lines)}")

    parts = stdout.split("---MARKDOWN_SEPARATOR---", 1)
    stats_line = parts[0].strip().split("\n")[-1]
    stats = json.loads(stats_line)
    md = parts[1] if len(parts) > 1 else ""
    print(
        f"    [docling_patched] Patched: {stats.get('patched', 0)}, Failed: {stats.get('patch_failed', 0)}"
    )
    return md, stats["time_seconds"]


def run_pymupdf_raw(pdf_path: str, page_start: int, page_end: int) -> tuple[str, float]:
    """Run PyMuPDF text extraction (both pymupdf4llm and raw page.get_text).

    Parameters
    ----------
    pdf_path : str
        Path to PDF.
    page_start : int
        Start page (1-indexed).
    page_end : int
        End page (1-indexed, inclusive).

    Returns
    -------
    tuple[str, float]
        (markdown_output, elapsed_seconds)
    """
    script = textwrap.dedent(f"""\
        import time, json
        import pymupdf

        pdf_path = "{pdf_path}"
        page_start = {page_start}
        page_end = {page_end}

        t0 = time.time()

        # Method 1: pymupdf4llm (if available)
        md_parts = []
        try:
            import pymupdf4llm
            # 0-indexed pages for pymupdf4llm
            pages = list(range(page_start - 1, page_end))
            md_llm = pymupdf4llm.to_markdown(pdf_path, pages=pages)
            md_parts.append("## pymupdf4llm output\\n\\n" + md_llm)
        except ImportError:
            md_parts.append("## pymupdf4llm\\n\\n(not installed)\\n")
        except Exception as e:
            md_parts.append(f"## pymupdf4llm\\n\\nERROR: {{e}}\\n")

        # Method 2: Raw page.get_text()
        doc = pymupdf.open(pdf_path)
        raw_parts = []
        for page_no in range(page_start - 1, min(page_end, len(doc))):
            page = doc[page_no]
            raw_text = page.get_text("text")
            raw_parts.append(f"### Page {{page_no + 1}}\\n\\n{{raw_text}}")
        doc.close()
        md_parts.append("## Raw page.get_text() output\\n\\n" + "\\n\\n".join(raw_parts))

        md = "\\n\\n---\\n\\n".join(md_parts)
        elapsed = time.time() - t0

        stats = {{"time_seconds": round(elapsed, 1)}}
        print(json.dumps(stats))
        print("---MARKDOWN_SEPARATOR---")
        print(md)
    """)

    stdout, stderr, rc = _run_subprocess(sys.executable, script, timeout=300, label="pymupdf_raw")
    if rc != 0:
        error_lines = stderr.strip().split("\n")[-5:]
        raise RuntimeError(f"pymupdf_raw failed: {' '.join(error_lines)}")

    parts = stdout.split("---MARKDOWN_SEPARATOR---", 1)
    stats = json.loads(parts[0].strip().split("\n")[-1])
    md = parts[1] if len(parts) > 1 else ""
    return md, stats["time_seconds"]


def _ensure_venv(venv_path: str, install_cmd: list[str], label: str) -> None:
    """Create isolated venv and install packages if needed.

    Parameters
    ----------
    venv_path : str
        Path to venv directory.
    install_cmd : list[str]
        Packages to pip install.
    label : str
        Display label.
    """
    if os.path.exists(os.path.join(venv_path, "bin", "python")):
        print(f"    [{label}] Reusing existing venv at {venv_path}")
        return

    print(f"    [{label}] Creating isolated venv at {venv_path}...")
    subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)

    # Ensure pip is available (some venvs need ensurepip first)
    subprocess.run(
        [os.path.join(venv_path, "bin", "python"), "-m", "ensurepip", "--upgrade"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    print(f"    [{label}] Installing: {' '.join(install_cmd)}")
    result = subprocess.run(
        [os.path.join(venv_path, "bin", "pip"), "install"] + install_cmd,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min — PyTorch/PaddlePaddle are large downloads
    )
    if result.returncode != 0:
        raise RuntimeError(f"Install failed for {label}: {result.stderr[-500:]}")
    print(f"    [{label}] Install complete.")


def run_mineru(
    pdf_path: str, page_start: int, page_end: int, skip_install: bool = False
) -> tuple[str, float]:
    """Run MinerU (magic-pdf) extraction with UniMERNet formula recognition.

    Parameters
    ----------
    pdf_path : str
        Path to PDF.
    page_start : int
        Start page (1-indexed).
    page_end : int
        End page (1-indexed, inclusive).
    skip_install : bool
        Skip venv creation.

    Returns
    -------
    tuple[str, float]
        (markdown_output, elapsed_seconds)
    """
    venv_path = "/tmp/venv_mineru_312"
    if not skip_install:
        _ensure_venv(venv_path, ["magic-pdf[full]"], "mineru")

    script = textwrap.dedent(f"""\
        import time, json, os, tempfile
        from pathlib import Path

        pdf_path = "{pdf_path}"
        page_start = {page_start}
        page_end = {page_end}

        t0 = time.time()

        try:
            from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
            from magic_pdf.data.dataset import PymuDocDataset
            from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

            # Read PDF bytes
            pdf_bytes = Path(pdf_path).read_bytes()

            # Create temp output dir
            output_dir = tempfile.mkdtemp(prefix="mineru_")
            image_dir = os.path.join(output_dir, "images")
            os.makedirs(image_dir, exist_ok=True)

            image_writer = FileBasedDataWriter(image_dir)

            # Create dataset and analyze
            ds = PymuDocDataset(pdf_bytes)

            # Run inference — auto-detects if PDF has text layer
            if ds.classify() == "ocr":
                infer_result = ds.apply(doc_analyze, ocr=True)
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
            else:
                infer_result = ds.apply(doc_analyze, ocr=False)
                pipe_result = infer_result.pipe_txt_mode(image_writer)

            md = pipe_result.get_markdown(image_dir)

            # Filter to target pages if possible (MinerU outputs full doc)
            # We keep all and note this in the report

            elapsed = time.time() - t0
            stats = {{"time_seconds": round(elapsed, 1)}}
            print(json.dumps(stats))
            print("---MARKDOWN_SEPARATOR---")
            print(md)

        except Exception as e:
            elapsed = time.time() - t0
            stats = {{"time_seconds": round(elapsed, 1), "error": str(e)[:200]}}
            print(json.dumps(stats))
            print("---MARKDOWN_SEPARATOR---")
            print(f"ERROR: {{e}}")
    """)

    python_exe = os.path.join(venv_path, "bin", "python")
    stdout, stderr, rc = _run_subprocess(python_exe, script, timeout=1200, label="mineru")

    parts = stdout.split("---MARKDOWN_SEPARATOR---", 1)
    stats_line = parts[0].strip().split("\n")[-1] if parts[0].strip() else "{}"
    try:
        stats = json.loads(stats_line)
    except json.JSONDecodeError:
        stats = {"time_seconds": 0, "error": f"Parse failed. rc={rc}"}

    if stats.get("error"):
        raise RuntimeError(f"mineru failed: {stats['error']}")
    if rc != 0:
        error_lines = stderr.strip().split("\n")[-5:]
        raise RuntimeError(f"mineru failed (rc={rc}): {' '.join(error_lines)}")

    md = parts[1] if len(parts) > 1 else ""
    return md, stats["time_seconds"]


def run_paddleocr(
    pdf_path: str, page_start: int, page_end: int, skip_install: bool = False
) -> tuple[str, float]:
    """Run PaddleOCR-VL document understanding.

    Parameters
    ----------
    pdf_path : str
        Path to PDF.
    page_start : int
        Start page (1-indexed).
    page_end : int
        End page (1-indexed, inclusive).
    skip_install : bool
        Skip venv creation.

    Returns
    -------
    tuple[str, float]
        (markdown_output, elapsed_seconds)
    """
    venv_path = "/tmp/venv_paddle_312"
    if not skip_install:
        _ensure_venv(venv_path, ["paddleocr", "paddlepaddle"], "paddleocr")

    script = textwrap.dedent(f"""\
        import time, json, os, tempfile
        from pathlib import Path

        # Disable slow model source connectivity check
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

        pdf_path = "{pdf_path}"
        page_start = {page_start}
        page_end = {page_end}

        t0 = time.time()

        try:
            from paddleocr import PPStructureV3

            # Try GPU first, fall back to CPU
            import paddle
            if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                dev = "gpu:0"
            else:
                dev = "cpu"

            # Disable PIR (new IR) which causes CPU crashes in PaddlePaddle 3.x
            paddle.framework.set_flags({{"FLAGS_enable_pir_api": False}})

            pipeline = PPStructureV3(
                use_formula_recognition=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                device=dev,
            )

            output_dir = tempfile.mkdtemp(prefix="paddle_")

            md_parts = []
            for result in pipeline.predict(input=pdf_path):
                # Save to markdown and read back
                result.save_to_markdown(output_dir)

            # Read all generated markdown files
            for md_file in sorted(Path(output_dir).rglob("*.md")):
                md_parts.append(md_file.read_text())

            if not md_parts:
                # Fallback: try to get text from result directly
                md_parts.append(str(result) if 'result' in dir() else "(no output)")

            md = "\\n\\n".join(md_parts)
            elapsed = time.time() - t0

            stats = {{"time_seconds": round(elapsed, 1)}}
            print(json.dumps(stats))
            print("---MARKDOWN_SEPARATOR---")
            print(md)

        except Exception as e:
            import traceback
            elapsed = time.time() - t0
            tb = traceback.format_exc()
            stats = {{"time_seconds": round(elapsed, 1), "error": str(e)[:200]}}
            print(json.dumps(stats))
            print("---MARKDOWN_SEPARATOR---")
            print(f"ERROR: {{e}}\\n{{tb}}")
    """)

    python_exe = os.path.join(venv_path, "bin", "python")
    stdout, stderr, rc = _run_subprocess(python_exe, script, timeout=1200, label="paddleocr")

    parts = stdout.split("---MARKDOWN_SEPARATOR---", 1)
    stats_line = parts[0].strip().split("\n")[-1] if parts[0].strip() else "{}"
    try:
        stats = json.loads(stats_line)
    except json.JSONDecodeError:
        stats = {"time_seconds": 0, "error": f"Parse failed. rc={rc}"}

    if stats.get("error"):
        raise RuntimeError(f"paddleocr failed: {stats['error']}")
    if rc != 0:
        error_lines = stderr.strip().split("\n")[-5:]
        raise RuntimeError(f"paddleocr failed (rc={rc}): {' '.join(error_lines)}")

    md = parts[1] if len(parts) > 1 else ""
    return md, stats["time_seconds"]


# ── Extractor Registry ───────────────────────────────────────────────────────

EXTRACTORS = {
    "docling_default": run_docling_default,
    "docling_patched": run_docling_patched,
    "pymupdf_raw": run_pymupdf_raw,
    "mineru": run_mineru,
    "paddleocr": run_paddleocr,
}

# ── Report Generation ────────────────────────────────────────────────────────


def generate_comparison_report(
    all_metrics: list[ExtractionMetrics],
    output_dir: Path,
) -> str:
    """Generate COMPARISON.md from collected metrics.

    Parameters
    ----------
    all_metrics : list[ExtractionMetrics]
        All extraction metrics.
    output_dir : Path
        Output directory.

    Returns
    -------
    str
        Report content.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Formula Extraction Comparison",
        "",
        f"Generated: {now}",
        "",
        "## Summary",
        "",
    ]

    # Group by PDF
    pdfs_seen: dict[str, list[ExtractionMetrics]] = {}
    for m in all_metrics:
        pdfs_seen.setdefault(m.pdf_key, []).append(m)

    # Overall summary table
    lines.append(f"| PDF | Extractor | Gaps | LaTeX (display) | LaTeX (inline) | Chars | Time |")
    lines.append("|-----|-----------|------|-----------------|----------------|-------|------|")
    for pdf_key, metrics in pdfs_seen.items():
        for m in metrics:
            if m.error:
                lines.append(f"| {pdf_key} | {m.extractor} | ERROR | - | - | - | - |")
            else:
                lines.append(
                    f"| {pdf_key} | {m.extractor} | {m.formula_gaps} | {m.latex_display} "
                    f"| {m.latex_inline} | {m.total_chars:,} | {m.time_seconds:.0f}s |"
                )
    lines.append("")

    # Per-PDF detail sections
    for pdf_key, metrics in pdfs_seen.items():
        pdf_info = TEST_PDFS.get(pdf_key, {})
        pdf_name = Path(str(pdf_info.get("path", pdf_key))).stem
        lines.append(f"## {pdf_key}: {pdf_name}")
        lines.append("")
        lines.append(
            f"Domain: {pdf_info.get('domain', '?')} | Baseline gap rate: {pdf_info.get('gap_rate', '?')}"
        )
        lines.append("")

        # Find baseline (docling_default) for gap reduction calculation
        baseline = next((m for m in metrics if m.extractor == "docling_default"), None)

        for m in metrics:
            lines.append(f"### {m.extractor}")
            if m.error:
                lines.append(f"**ERROR**: {m.error}")
                lines.append("")
                continue

            gap_reduction = ""
            if baseline and baseline.formula_gaps > 0 and m.extractor != "docling_default":
                reduction = (baseline.formula_gaps - m.formula_gaps) / baseline.formula_gaps * 100
                gap_reduction = f" ({reduction:+.1f}% vs baseline)"

            lines.append(f"- Formula gaps: **{m.formula_gaps}**{gap_reduction}")
            lines.append(f"- LaTeX display blocks: {m.latex_display}")
            lines.append(f"- LaTeX inline: {m.latex_inline}")
            lines.append(f"- Total chars: {m.total_chars:,}")
            lines.append(f"- Time: {m.time_seconds:.1f}s (wall: {m.wall_time:.1f}s)")

            if m.sample_formulas:
                lines.append("")
                lines.append("**Sample decoded formulas:**")
                for sf in m.sample_formulas[:3]:
                    lines.append(f"```")
                    lines.append(sf)
                    lines.append(f"```")

            if m.sample_gaps:
                lines.append("")
                lines.append("**Sample gaps (context):**")
                for sg in m.sample_gaps[:2]:
                    lines.append(f"> {sg}")

            lines.append("")

    # Decision section
    lines.append("## Decision")
    lines.append("")
    lines.append("TODO: Fill in after reviewing results.")
    lines.append("")
    lines.append("- [ ] Which extractor wins for formula quality?")
    lines.append("- [ ] Is docling_patched good enough to patch existing chunks in-place?")
    lines.append("- [ ] VRAM feasibility for production re-extraction?")
    lines.append("- [ ] Re-extraction scope: all domains or math-heavy only?")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_page_range(s: str) -> tuple[int, int]:
    """Parse 'start-end' page range string.

    Parameters
    ----------
    s : str
        Range like '50-54'.

    Returns
    -------
    tuple[int, int]
        (start, end) 1-indexed inclusive.
    """
    parts = s.split("-")
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    elif len(parts) == 1:
        p = int(parts[0])
        return p, p
    else:
        raise ValueError(f"Invalid page range: {s}")


def main() -> None:
    """Run formula extractor comparison."""
    parser = argparse.ArgumentParser(
        description="Compare PDF formula extraction quality across extractors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--extractors",
        type=str,
        default="docling_default,docling_patched,pymupdf_raw",
        help="Comma-separated extractor names, or 'all' (default: docling_default,docling_patched,pymupdf_raw)",
    )
    parser.add_argument(
        "--pdfs",
        type=str,
        default="murty",
        help="Comma-separated PDF keys from test set, or 'all' (default: murty)",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default="50-54",
        help="Page range to extract (default: 50-54)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip isolated venv creation (reuse existing)",
    )
    args = parser.parse_args()

    # Parse extractors
    if args.extractors == "all":
        extractor_names = list(EXTRACTORS.keys())
    else:
        extractor_names = [e.strip() for e in args.extractors.split(",")]
        for name in extractor_names:
            if name not in EXTRACTORS:
                print(
                    f"ERROR: Unknown extractor '{name}'. Available: {', '.join(EXTRACTORS.keys())}"
                )
                sys.exit(1)

    # Parse PDFs
    if args.pdfs == "all":
        pdf_keys = list(TEST_PDFS.keys())
    else:
        pdf_keys = [p.strip() for p in args.pdfs.split(",")]
        for key in pdf_keys:
            if key not in TEST_PDFS:
                print(f"ERROR: Unknown PDF key '{key}'. Available: {', '.join(TEST_PDFS.keys())}")
                sys.exit(1)

    page_start, page_end = parse_page_range(args.pages)

    # Verify PDFs exist
    for key in pdf_keys:
        pdf_path = TEST_PDFS[key]["path"]
        if not pdf_path.exists():
            print(f"ERROR: PDF not found: {pdf_path}")
            sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Formula Extractor Comparison")
    print(f"  Extractors: {', '.join(extractor_names)}")
    print(f"  PDFs: {', '.join(pdf_keys)}")
    print(f"  Pages: {page_start}-{page_end}")
    print(f"  Output: {args.output}")
    print()

    all_metrics: list[ExtractionMetrics] = []

    for pdf_key in pdf_keys:
        pdf_info = TEST_PDFS[pdf_key]
        pdf_path = str(pdf_info["path"])
        pdf_output_dir = args.output / pdf_key
        pdf_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"{'=' * 70}")
        print(f"PDF: {pdf_key} — {Path(pdf_path).name}")
        print(f"  Domain: {pdf_info['domain']} | Baseline gap rate: {pdf_info['gap_rate']}")
        print(f"{'=' * 70}")

        for ext_name in extractor_names:
            print(f"\n  --- {ext_name} ---")
            runner = EXTRACTORS[ext_name]
            output_path = pdf_output_dir / f"{ext_name}.md"

            t0 = time.time()
            try:
                # Pass skip_install for isolated venv extractors
                if ext_name in ("mineru", "paddleocr"):
                    md, elapsed = runner(
                        pdf_path, page_start, page_end, skip_install=args.skip_install
                    )
                else:
                    md, elapsed = runner(pdf_path, page_start, page_end)

                wall_time = time.time() - t0

                # Save output
                output_path.write_text(md)

                # Compute metrics
                metrics = compute_metrics(md, ext_name, pdf_key, f"{page_start}-{page_end}")
                metrics.time_seconds = elapsed
                metrics.wall_time = wall_time

                print(f"    Time: {elapsed:.1f}s (wall: {wall_time:.1f}s)")
                print(f"    Chars: {metrics.total_chars:,}")
                print(f"    Formula gaps: {metrics.formula_gaps}")
                print(
                    f"    LaTeX display: {metrics.latex_display} | inline: {metrics.latex_inline}"
                )
                print(f"    Saved: {output_path}")

            except Exception as e:
                wall_time = time.time() - t0
                print(f"    FAILED: {e}")
                metrics = ExtractionMetrics(
                    extractor=ext_name,
                    pdf_key=pdf_key,
                    pages=f"{page_start}-{page_end}",
                    wall_time=wall_time,
                    error=str(e)[:300],
                )

            all_metrics.append(metrics)

    # Generate comparison report
    print(f"\n{'=' * 70}")
    print("Generating COMPARISON.md...")
    report = generate_comparison_report(all_metrics, args.output)
    report_path = args.output / "COMPARISON.md"
    report_path.write_text(report)
    print(f"  Saved: {report_path}")

    # Also save raw metrics as JSON
    metrics_path = args.output / "metrics.json"
    metrics_data = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "pages": f"{page_start}-{page_end}",
        "results": [m.to_dict() for m in all_metrics],
    }
    metrics_path.write_text(json.dumps(metrics_data, indent=2))
    print(f"  Metrics: {metrics_path}")

    # Print quick summary
    print(f"\n{'=' * 70}")
    print("QUICK SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Extractor':<20s} {'PDF':<10s} {'Gaps':>6s} {'LaTeX':>7s} {'Time':>7s}")
    print(f"  {'-'*20} {'-'*10} {'-'*6} {'-'*7} {'-'*7}")
    for m in all_metrics:
        if m.error:
            print(f"  {m.extractor:<20s} {m.pdf_key:<10s} {'FAIL':>6s}")
        else:
            latex = m.latex_display + m.latex_inline
            print(
                f"  {m.extractor:<20s} {m.pdf_key:<10s} {m.formula_gaps:>6d} {latex:>7d} {m.time_seconds:>6.0f}s"
            )

    print(f"\n  Full report: {report_path}")
    print(f"  Output dir:  {args.output}/")


if __name__ == "__main__":
    main()
