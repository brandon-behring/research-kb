"""MinerU-based PDF extraction with LaTeX formula recognition.

Uses MinerU (magic-pdf) for document understanding with UniMERNet for formula
recognition. Runs in an isolated Python 3.12 subprocess to avoid dependency
conflicts (main venv is 3.13).

Produces the same (ExtractionResult, list[TextChunk]) interface as
docling_extractor.py for drop-in compatibility with the storage pipeline.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from research_kb_common import get_logger
from research_kb_pdf.chunker import TextChunk, count_tokens

logger = get_logger(__name__)

DEFAULT_VENV_PATH = os.path.expanduser("~/.local/share/mineru-venv")
VRAM_MIN_MB = int(os.environ.get("MINERU_VRAM_MIN_MB", "4000"))  # ~2.5 GB models + ~1.5 GB working


@dataclass
class MinerUExtractionResult:
    """Result of MinerU PDF extraction.

    Attributes
    ----------
    file_path : str
        Path to the source PDF.
    total_pages : int
        Number of pages in the PDF.
    total_chars : int
        Total characters in the extracted markdown.
    has_equations : bool
        Whether LaTeX equations were detected.
    heading_count : int
        Number of headings detected.
    """

    file_path: str
    total_pages: int
    total_chars: int
    has_equations: bool
    heading_count: int = 0


def check_vram_available(min_mb: int = VRAM_MIN_MB) -> tuple[bool, int]:
    """Check if sufficient GPU VRAM is available.

    Parameters
    ----------
    min_mb : int
        Minimum free VRAM in MB.

    Returns
    -------
    tuple[bool, int]
        (is_sufficient, free_mb)
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            free_mb = int(result.stdout.strip().split("\n")[0])
            return free_mb >= min_mb, free_mb
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return False, 0


def _run_mineru_subprocess(
    pdf_path: str,
    venv_path: str = DEFAULT_VENV_PATH,
    timeout: int = 1800,
) -> tuple[str, dict]:
    """Run MinerU extraction in isolated Python 3.12 subprocess.

    Parameters
    ----------
    pdf_path : str
        Path to PDF file.
    venv_path : str
        Path to MinerU Python 3.12 venv.
    timeout : int
        Max seconds for extraction.

    Returns
    -------
    tuple[str, dict]
        (markdown_output, metadata_dict)

    Raises
    ------
    FileNotFoundError
        If PDF or venv doesn't exist.
    RuntimeError
        If extraction fails.
    """
    python_exe = os.path.join(venv_path, "bin", "python")
    if not os.path.exists(python_exe):
        raise FileNotFoundError(
            f"MinerU venv not found at {venv_path}. "
            f"Create it with: uv venv --python 3.12 {venv_path}"
        )

    script = textwrap.dedent(f"""\
        import json, os, sys, tempfile, time
        from pathlib import Path

        pdf_path = {json.dumps(pdf_path)}
        t0 = time.time()

        try:
            from magic_pdf.data.data_reader_writer import FileBasedDataWriter
            from magic_pdf.data.dataset import PymuDocDataset
            from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

            pdf_bytes = Path(pdf_path).read_bytes()
            output_dir = tempfile.mkdtemp(prefix="mineru_")
            image_dir = os.path.join(output_dir, "images")
            os.makedirs(image_dir, exist_ok=True)
            image_writer = FileBasedDataWriter(image_dir)

            ds = PymuDocDataset(pdf_bytes)
            classify = ds.classify()

            if str(classify) == "SupportedPdfParseMethod.OCR":
                infer_result = ds.apply(doc_analyze, ocr=True)
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
            else:
                infer_result = ds.apply(doc_analyze, ocr=False)
                pipe_result = infer_result.pipe_txt_mode(image_writer)

            md = pipe_result.get_markdown(image_dir)
            elapsed = time.time() - t0

            # Count pages from PyMuPDF (MinerU uses it internally)
            import pymupdf
            doc = pymupdf.open(pdf_path)
            total_pages = len(doc)
            doc.close()

            metadata = {{
                "time_seconds": round(elapsed, 1),
                "total_pages": total_pages,
                "total_chars": len(md),
                "classify": str(classify),
            }}

            # Output protocol: JSON metadata on first line, then separator, then markdown
            print(json.dumps(metadata))
            print("---MINERU_SEPARATOR---")
            print(md)

        except Exception as e:
            import traceback
            elapsed = time.time() - t0
            metadata = {{
                "time_seconds": round(elapsed, 1),
                "error": str(e)[:500],
                "traceback": traceback.format_exc()[-500:],
            }}
            print(json.dumps(metadata))
            print("---MINERU_SEPARATOR---")
            sys.exit(1)
    """)

    # Resolve to absolute path for subprocess (it may have different cwd)
    abs_pdf_path = str(Path(pdf_path).resolve())
    script = script.replace(json.dumps(pdf_path), json.dumps(abs_pdf_path))

    logger.info("mineru_subprocess_start", path=abs_pdf_path)
    result = subprocess.run(
        [python_exe, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    # Parse output
    stdout = result.stdout
    parts = stdout.split("---MINERU_SEPARATOR---", 1)

    if not parts[0].strip():
        stderr_tail = result.stderr.strip().split("\n")[-5:]
        raise RuntimeError(f"MinerU produced no output. stderr: {' '.join(stderr_tail)}")

    try:
        metadata = json.loads(parts[0].strip().split("\n")[-1])
    except json.JSONDecodeError:
        raise RuntimeError(f"MinerU output parse failed. stdout: {stdout[:200]}")

    if metadata.get("error"):
        raise RuntimeError(f"MinerU extraction failed: {metadata['error']}")

    if result.returncode != 0:
        stderr_tail = result.stderr.strip().split("\n")[-5:]
        raise RuntimeError(f"MinerU exit code {result.returncode}: {' '.join(stderr_tail)}")

    md = parts[1] if len(parts) > 1 else ""
    logger.info(
        "mineru_subprocess_done",
        path=pdf_path,
        chars=len(md),
        time=metadata.get("time_seconds"),
    )
    return md, metadata


# ── Section-Aware Chunking ───────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_DISPLAY_LATEX_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)


def _parse_sections(markdown: str) -> list[dict]:
    """Parse markdown into sections with heading hierarchy.

    Parameters
    ----------
    markdown : str
        MinerU markdown output.

    Returns
    -------
    list[dict]
        List of sections, each with:
        - level: int (1-6)
        - heading: str
        - content: str
        - breadcrumb: list[str] (full heading path)
    """
    lines = markdown.split("\n")
    sections: list[dict] = []
    current_content: list[str] = []
    heading_stack: dict[int, str] = {}  # level -> heading text
    current_heading: Optional[str] = None
    current_level: int = 0

    def _flush():
        content = "\n".join(current_content).strip()
        if content or current_heading:
            # Build breadcrumb from heading stack
            sorted_levels = sorted(heading_stack.keys())
            breadcrumb = [heading_stack[k] for k in sorted_levels]
            sections.append(
                {
                    "level": current_level,
                    "heading": current_heading or "",
                    "content": content,
                    "breadcrumb": list(breadcrumb),
                }
            )

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            _flush()
            current_content = []
            level = len(match.group(1))
            heading = match.group(2).strip()
            current_heading = heading
            current_level = level

            # Update heading stack: remove all levels >= current
            heading_stack = {k: v for k, v in heading_stack.items() if k < level}
            heading_stack[level] = heading
        else:
            current_content.append(line)

    _flush()
    return sections


def _protect_latex_blocks(text: str) -> tuple[str, list[str]]:
    """Replace display LaTeX blocks with placeholders to prevent splitting mid-formula.

    Parameters
    ----------
    text : str
        Text potentially containing $$...$$ blocks.

    Returns
    -------
    tuple[str, list[str]]
        (text_with_placeholders, list_of_replaced_blocks)
    """
    blocks: list[str] = []

    def _replace(match):
        blocks.append(match.group(0))
        return f"__LATEX_BLOCK_{len(blocks) - 1}__"

    protected = _DISPLAY_LATEX_RE.sub(_replace, text)
    return protected, blocks


def _restore_latex_blocks(text: str, blocks: list[str]) -> str:
    """Restore LaTeX placeholders back to original blocks.

    Parameters
    ----------
    text : str
        Text with __LATEX_BLOCK_N__ placeholders.
    blocks : list[str]
        Original LaTeX blocks.

    Returns
    -------
    str
        Text with LaTeX restored.
    """
    for i, block in enumerate(blocks):
        text = text.replace(f"__LATEX_BLOCK_{i}__", block)
    return text


def _chunk_section(
    content: str,
    breadcrumb: list[str],
    max_tokens: int,
    start_chunk_index: int,
) -> list[TextChunk]:
    """Chunk a single section's content, respecting token limits and LaTeX blocks.

    Parameters
    ----------
    content : str
        Section content (may include LaTeX).
    breadcrumb : list[str]
        Full heading path for this section.
    max_tokens : int
        Maximum tokens per chunk.
    start_chunk_index : int
        Starting chunk_index for numbering.

    Returns
    -------
    list[TextChunk]
        Chunks with breadcrumb context prepended.
    """
    if not content.strip():
        return []

    # Build breadcrumb prefix
    breadcrumb_prefix = " > ".join(breadcrumb) + "\n\n" if breadcrumb else ""
    prefix_tokens = count_tokens(breadcrumb_prefix) if breadcrumb_prefix else 0
    available_tokens = max_tokens - prefix_tokens

    if available_tokens < 50:
        # Breadcrumb too long — use just the leaf heading
        breadcrumb_prefix = breadcrumb[-1] + "\n\n" if breadcrumb else ""
        prefix_tokens = count_tokens(breadcrumb_prefix) if breadcrumb_prefix else 0
        available_tokens = max_tokens - prefix_tokens

    # Protect LaTeX blocks from splitting
    protected, latex_blocks = _protect_latex_blocks(content)

    # Split by paragraphs (double newline)
    paragraphs = re.split(r"\n\n+", protected)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_tokens = 0

    def _flush_chunk():
        nonlocal current_parts, current_tokens
        if not current_parts:
            return
        raw_content = "\n\n".join(current_parts)
        restored = _restore_latex_blocks(raw_content, latex_blocks)
        full_content = breadcrumb_prefix + restored

        # Sanitize
        full_content = full_content.replace("\x00", "").replace("\ufffd", "")

        token_count = count_tokens(full_content)
        chunks.append(
            TextChunk(
                content=full_content,
                start_page=1,  # Will be refined below
                end_page=1,
                token_count=token_count,
                char_count=len(full_content),
                chunk_index=start_chunk_index + len(chunks),
                metadata={
                    "section": breadcrumb[-1] if breadcrumb else None,
                    "breadcrumb": breadcrumb,
                    "heading_level": len(breadcrumb),
                    "chunking_method": "mineru",
                },
            )
        )
        current_parts = []
        current_tokens = 0

    for para in paragraphs:
        # Restore LaTeX in this paragraph to get accurate token count
        para_restored = _restore_latex_blocks(para, latex_blocks)
        para_tokens = count_tokens(para_restored)

        # If a single paragraph (including LaTeX) exceeds the limit,
        # emit it as its own chunk — don't try to split LaTeX blocks
        if para_tokens > available_tokens:
            _flush_chunk()
            current_parts.append(para)
            current_tokens = para_tokens
            _flush_chunk()  # Emit oversized chunk immediately
            continue

        if current_tokens + para_tokens > available_tokens:
            _flush_chunk()

        current_parts.append(para)
        current_tokens += para_tokens

    _flush_chunk()
    return chunks


def _assign_page_numbers(chunks: list[TextChunk], total_pages: int, total_chars: int) -> None:
    """Estimate page numbers for chunks based on character position.

    MinerU doesn't provide per-chunk page numbers in its markdown output,
    so we estimate based on relative position in the document.

    Parameters
    ----------
    chunks : list[TextChunk]
        Chunks to assign page numbers to (modified in place).
    total_pages : int
        Total pages in the PDF.
    total_chars : int
        Total characters in the markdown.
    """
    if total_pages <= 0 or total_chars <= 0:
        return

    chars_per_page = total_chars / total_pages
    cumulative_chars = 0

    for chunk in chunks:
        # Estimate start page from cumulative character position
        start_page = max(1, int(cumulative_chars / chars_per_page) + 1)
        end_page = max(start_page, int((cumulative_chars + chunk.char_count) / chars_per_page) + 1)
        end_page = min(end_page, total_pages)

        chunk.start_page = start_page
        chunk.end_page = end_page
        cumulative_chars += chunk.char_count


# ── Public API ───────────────────────────────────────────────────────────────


def extract_and_chunk(
    pdf_path: str | Path,
    max_tokens: int = 300,
    venv_path: str = DEFAULT_VENV_PATH,
    check_vram: bool = True,
) -> tuple[MinerUExtractionResult, list[TextChunk]]:
    """Extract PDF with MinerU and chunk with section-aware splitter.

    Pipeline: PDF -> MinerU subprocess -> Section parsing -> Token-based chunking

    Equations are preserved as LaTeX ($$...$$ and $...$) in chunk content.
    Each chunk gets a breadcrumb heading prefix for retrieval context.

    Parameters
    ----------
    pdf_path : str or Path
        Path to PDF file.
    max_tokens : int
        Maximum tokens per chunk (default 300, hard limit 512).
    venv_path : str
        Path to MinerU Python 3.12 venv.
    check_vram : bool
        Check GPU VRAM availability before extraction.

    Returns
    -------
    tuple[MinerUExtractionResult, list[TextChunk]]

    Raises
    ------
    FileNotFoundError
        If PDF doesn't exist.
    ValueError
        If PDF is corrupted or extraction fails.
    RuntimeError
        If insufficient VRAM or MinerU subprocess fails.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # VRAM guard
    if check_vram:
        sufficient, free_mb = check_vram_available()
        if not sufficient:
            raise RuntimeError(
                f"Insufficient VRAM: {free_mb} MB free, need {VRAM_MIN_MB} MB. "
                f"Stop embed_server/rerank_server/Ollama first."
            )
        logger.info("vram_check_passed", free_mb=free_mb)

    # Run MinerU extraction in subprocess
    try:
        md, metadata = _run_mineru_subprocess(str(pdf_path), venv_path)
    except Exception as e:
        raise ValueError(f"MinerU extraction failed: {e}") from e

    total_pages = metadata.get("total_pages", 0)
    total_chars = len(md)

    # Parse sections from markdown
    sections = _parse_sections(md)

    # Chunk each section
    all_chunks: list[TextChunk] = []
    for section in sections:
        section_chunks = _chunk_section(
            content=section["content"],
            breadcrumb=section["breadcrumb"],
            max_tokens=max_tokens,
            start_chunk_index=len(all_chunks),
        )
        all_chunks.extend(section_chunks)

    # Assign estimated page numbers
    _assign_page_numbers(all_chunks, total_pages, total_chars)

    # Count headings and equations
    heading_count = sum(1 for s in sections if s["heading"])
    has_equations = "$$" in md or "\\(" in md

    extraction_result = MinerUExtractionResult(
        file_path=str(pdf_path),
        total_pages=total_pages,
        total_chars=total_chars,
        has_equations=has_equations,
        heading_count=heading_count,
    )

    logger.info(
        "mineru_extraction_complete",
        path=str(pdf_path),
        pages=total_pages,
        chars=total_chars,
        chunks=len(all_chunks),
        has_equations=has_equations,
        headings=heading_count,
    )

    return extraction_result, all_chunks
