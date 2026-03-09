"""Docling-based PDF extraction with LaTeX-preserving chunking.

Uses IBM's Docling (Granite-Docling-258M) for document understanding and
HybridChunker for structure-aware chunking. Equations are preserved as LaTeX
($$...$$ or \\(...\\)) in chunk text.

Replaces PyMuPDF extraction pipeline. GROBID remains for citation metadata only.
"""

import threading
from dataclasses import dataclass, field
from pathlib import Path

from research_kb_common import get_logger
from research_kb_pdf.chunker import BGE_MODEL, BGE_REVISION, TextChunk, count_tokens, get_tokenizer

logger = get_logger(__name__)

# Singleton DocumentConverter — loads Granite-Docling-258M once (~5s, ~2.5 GB VRAM)
_converter = None
_converter_lock = threading.Lock()


def get_converter():
    """Lazy-load DocumentConverter singleton (thread-safe).

    Loads Granite-Docling-258M once (~5s, ~2.5 GB VRAM on GPU).
    Reuse across calls to avoid VRAM leaks from repeated model loads.
    """
    global _converter
    if _converter is not None:
        return _converter
    with _converter_lock:
        if _converter is None:
            from docling.document_converter import DocumentConverter

            _converter = DocumentConverter()
        return _converter


def reset_converter():
    """Clear the singleton DocumentConverter.

    Call before switching GPU/CPU mode (e.g., OOM fallback) to force
    reinitialization on next get_converter() call.
    """
    global _converter
    with _converter_lock:
        _converter = None


@dataclass
class DoclingExtractionResult:
    """Result of Docling PDF extraction.

    Attributes:
        file_path: Path to the source PDF
        total_pages: Number of pages in the PDF
        total_chars: Total characters in the extracted markdown
        has_equations: Whether LaTeX equations were detected
        heading_count: Number of headings detected by Docling
    """

    file_path: str
    total_pages: int
    total_chars: int
    has_equations: bool
    heading_count: int = 0


def _get_page_from_prov(chunk) -> tuple[int, int]:
    """Extract start and end page numbers from chunk provenance.

    Args:
        chunk: A DocChunk from HybridChunker

    Returns:
        Tuple of (start_page, end_page), both 1-indexed. Defaults to (1, 1)
        if provenance is unavailable.
    """
    try:
        if hasattr(chunk, "meta") and hasattr(chunk.meta, "doc_items"):
            pages = []
            for item in chunk.meta.doc_items:
                for prov in getattr(item, "prov", []):
                    page_no = getattr(prov, "page_no", None)
                    if page_no is not None:
                        pages.append(page_no)
            if pages:
                return min(pages), max(pages)
        # Fallback: check chunk.meta.page (simpler API path)
        if hasattr(chunk, "meta") and hasattr(chunk.meta, "page"):
            page = chunk.meta.page
            if page is not None:
                return page, page
    except Exception:
        pass
    return 1, 1


def extract_and_chunk(
    pdf_path: str | Path,
    max_tokens: int = 300,
) -> tuple[DoclingExtractionResult, list[TextChunk]]:
    """Extract PDF with Docling and chunk with HybridChunker.

    Pipeline: PDF -> Docling DocumentConverter -> HybridChunker -> TextChunk adapter

    Equations are preserved as LaTeX in chunk content. Uses BGE-large-en-v1.5
    tokenizer for consistent token counts with the embedding model.

    Args:
        pdf_path: Path to PDF file
        max_tokens: Maximum tokens per chunk (default 300, hard limit 512)

    Returns:
        Tuple of (DoclingExtractionResult, list[TextChunk])

    Raises:
        FileNotFoundError: If PDF doesn't exist
        ValueError: If PDF is corrupted or extraction fails

    Example:
        >>> result, chunks = extract_and_chunk("textbook.pdf")
        >>> latex_chunks = [c for c in chunks if "$$" in c.content]
        >>> print(f"{len(chunks)} chunks, {len(latex_chunks)} with LaTeX")
    """
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("docling_extraction_start", path=str(pdf_path))

    # Convert PDF with Docling (singleton — no VRAM leak)
    try:
        converter = get_converter()
        result = converter.convert(str(pdf_path))
        doc = result.document
    except Exception as e:
        raise ValueError(
            f"Docling extraction failed (corrupted or unsupported PDF?): {e}"
        ) from e

    # Use BGE tokenizer for consistent token counts with embedding model
    hf_tokenizer = get_tokenizer()
    tokenizer = HuggingFaceTokenizer(
        tokenizer=hf_tokenizer,
        max_tokens=max_tokens,
    )
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
    )

    doc_chunks = list(chunker.chunk(dl_doc=doc))

    logger.info(
        "docling_chunking_complete",
        path=str(pdf_path),
        chunks=len(doc_chunks),
    )

    # Map DocChunk -> TextChunk (adapter)
    text_chunks: list[TextChunk] = []
    for i, dc in enumerate(doc_chunks):
        # Get enriched text (includes heading context)
        content = chunker.contextualize(dc)

        # Sanitize null bytes and replacement chars (PostgreSQL rejects \x00)
        content = content.replace("\x00", "").replace("\ufffd", "")

        # Extract headings from chunk metadata
        headings: list[str] = []
        if hasattr(dc, "meta") and hasattr(dc.meta, "headings"):
            headings = dc.meta.headings or []

        # Extract page numbers from provenance
        start_page, end_page = _get_page_from_prov(dc)

        text_chunks.append(
            TextChunk(
                content=content,
                start_page=start_page,
                end_page=end_page,
                token_count=count_tokens(content),
                char_count=len(content),
                chunk_index=i,
                metadata={
                    "section": headings[-1] if headings else None,
                    "heading_level": len(headings),
                    "chunking_method": "docling",
                },
            )
        )

    # Export markdown for metadata extraction
    md = doc.export_to_markdown()

    # Count headings in the document
    heading_count = 0
    try:
        if hasattr(doc, "body") and hasattr(doc.body, "children"):
            # Count heading items in document tree
            from docling_core.types.doc import DocItemLabel

            def _count_headings(items):
                count = 0
                for item in items:
                    label = getattr(item, "label", None)
                    if label in (
                        DocItemLabel.SECTION_HEADER,
                        DocItemLabel.PAGE_HEADER,
                        DocItemLabel.TITLE,
                    ):
                        count += 1
                    children = getattr(item, "children", [])
                    if children:
                        count += _count_headings(children)
                return count

            heading_count = _count_headings(doc.body.children)
    except Exception:
        # Fallback: count markdown headings
        heading_count = sum(1 for line in md.split("\n") if line.startswith("#"))

    # Detect total pages
    total_pages = 0
    try:
        if hasattr(doc, "pages") and doc.pages:
            total_pages = len(doc.pages)
    except Exception:
        # Fallback: estimate from chunk page numbers
        if text_chunks:
            total_pages = max(c.end_page for c in text_chunks)

    extraction_result = DoclingExtractionResult(
        file_path=str(pdf_path),
        total_pages=total_pages,
        total_chars=len(md),
        has_equations="$$" in md or "\\(" in md,
        heading_count=heading_count,
    )

    logger.info(
        "docling_extraction_complete",
        path=str(pdf_path),
        pages=extraction_result.total_pages,
        chars=extraction_result.total_chars,
        chunks=len(text_chunks),
        has_equations=extraction_result.has_equations,
        headings=heading_count,
    )

    return extraction_result, text_chunks
