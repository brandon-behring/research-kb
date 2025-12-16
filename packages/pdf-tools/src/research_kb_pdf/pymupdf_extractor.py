"""PyMuPDF-based PDF text extraction.

Extracts text from PDFs with page number tracking. Designed for textbooks
and simple academic papers. Includes heading detection via font-size analysis.
"""

from dataclasses import dataclass
from pathlib import Path
import statistics

import fitz  # PyMuPDF

from research_kb_common import get_logger
from research_kb_contracts import SourceType

logger = get_logger(__name__)


@dataclass
class Heading:
    """A detected heading from a PDF document."""

    text: str
    level: int  # 1=H1, 2=H2, 3=H3
    page_num: int
    font_size: float
    char_offset: int  # Approximate character offset in full document


@dataclass
class ExtractedPage:
    """Single page of extracted content."""

    page_num: int
    text: str
    char_count: int


@dataclass
class ExtractedDocument:
    """Complete extracted document."""

    file_path: str
    total_pages: int
    pages: list[ExtractedPage]
    total_chars: int
    source_type: SourceType = SourceType.TEXTBOOK


def extract_pdf(pdf_path: str | Path) -> ExtractedDocument:
    """Extract text from PDF using PyMuPDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        ExtractedDocument with text and page numbers

    Raises:
        FileNotFoundError: If PDF doesn't exist
        ValueError: If PDF is corrupted or encrypted

    Example:
        >>> doc = extract_pdf("textbook.pdf")
        >>> print(f"Extracted {doc.total_pages} pages, {doc.total_chars} chars")
        >>> print(doc.pages[0].text[:100])
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("extracting_pdf", path=str(pdf_path))

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to open PDF (corrupted or encrypted?): {e}") from e

    if doc.is_encrypted:
        raise ValueError(f"PDF is encrypted: {pdf_path}")

    pages = []
    total_chars = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # Remove null bytes that cause PostgreSQL UTF-8 encoding errors
        text = text.replace("\x00", "")

        # Strip excessive whitespace but preserve paragraph breaks
        text = "\n".join(line.strip() for line in text.split("\n") if line.strip())

        char_count = len(text)
        total_chars += char_count

        pages.append(
            ExtractedPage(
                page_num=page_num + 1,  # 1-indexed for user display
                text=text,
                char_count=char_count,
            )
        )

    doc.close()

    logger.info(
        "pdf_extracted", path=str(pdf_path), pages=len(pages), chars=total_chars
    )

    return ExtractedDocument(
        file_path=str(pdf_path),
        total_pages=len(pages),
        pages=pages,
        total_chars=total_chars,
    )


def get_text_with_page_numbers(document: ExtractedDocument) -> list[tuple[int, str]]:
    """Get document text as list of (page_num, text) tuples.

    Args:
        document: Extracted document

    Returns:
        List of (page_number, page_text) tuples

    Example:
        >>> doc = extract_pdf("book.pdf")
        >>> for page_num, text in get_text_with_page_numbers(doc):
        ...     print(f"Page {page_num}: {len(text)} chars")
    """
    return [(page.page_num, page.text) for page in document.pages]


def get_full_text(document: ExtractedDocument) -> str:
    """Get complete document text as single string.

    Args:
        document: Extracted document

    Returns:
        All pages concatenated with double newline separator

    Example:
        >>> doc = extract_pdf("paper.pdf")
        >>> full_text = get_full_text(doc)
        >>> print(f"Total text length: {len(full_text)}")
    """
    return "\n\n".join(page.text for page in document.pages)


def detect_headings(pdf_path: str | Path) -> list[Heading]:
    """Detect headings via font-size analysis.

    Algorithm:
    1. Extract all text spans with font metadata using get_text("dict")
    2. Calculate median font size across document
    3. Calculate standard deviation of font sizes
    4. Classify by threshold:
       - font_size > median + 2σ → H1
       - font_size > median + 1σ → H2
       - font_size > median + 0.5σ → H3
    5. Filter: text length 3-100 chars, ≤15 words

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of detected Heading objects

    Example:
        >>> headings = detect_headings("textbook.pdf")
        >>> for h in headings:
        ...     print(f"H{h.level}: {h.text} (page {h.page_num})")
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("detecting_headings", path=str(pdf_path))

    doc = fitz.open(pdf_path)

    # Collect all text spans with font sizes
    font_sizes = []
    text_blocks = []  # (text, font_size, page_num, char_offset)

    char_offset = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_dict = page.get_text("dict")

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # Skip non-text blocks
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    # Remove null bytes that cause PostgreSQL UTF-8 encoding errors
                    text = text.replace("\x00", "")
                    font_size = span.get("size", 0.0)

                    if text and font_size > 0:
                        font_sizes.append(font_size)
                        text_blocks.append((text, font_size, page_num + 1, char_offset))
                        char_offset += len(text)

    doc.close()

    if not font_sizes:
        logger.warning("no_text_with_font_metadata", path=str(pdf_path))
        return []

    # Calculate statistics
    median_size = statistics.median(font_sizes)

    # Handle case where all fonts are same size (stdev = 0)
    if len(set(font_sizes)) == 1:
        logger.info("uniform_font_size", path=str(pdf_path), median=median_size)
        return []  # No headings detectable if all fonts same size

    stdev_size = statistics.stdev(font_sizes)

    # Thresholds
    h1_threshold = median_size + 2 * stdev_size
    h2_threshold = median_size + 1 * stdev_size
    h3_threshold = median_size + 0.5 * stdev_size

    logger.info(
        "heading_thresholds",
        median=median_size,
        stdev=stdev_size,
        h1_threshold=h1_threshold,
        h2_threshold=h2_threshold,
        h3_threshold=h3_threshold,
    )

    # Classify headings
    headings = []

    for text, font_size, page_num, char_offset in text_blocks:
        # Filter: text length 3-100 chars, ≤15 words
        if not (3 <= len(text) <= 100):
            continue

        word_count = len(text.split())
        if word_count > 15:
            continue

        # Classify by font size
        level = None
        if font_size >= h1_threshold:
            level = 1
        elif font_size >= h2_threshold:
            level = 2
        elif font_size >= h3_threshold:
            level = 3

        if level:
            headings.append(
                Heading(
                    text=text,
                    level=level,
                    page_num=page_num,
                    font_size=font_size,
                    char_offset=char_offset,
                )
            )

    logger.info("headings_detected", path=str(pdf_path), count=len(headings))

    return headings


def extract_with_headings(
    pdf_path: str | Path,
) -> tuple[ExtractedDocument, list[Heading]]:
    """Extract text AND detect headings from PDF in a single pass.

    Memory-efficient: Opens PDF once, extracts both text and font metadata
    in a single iteration, then closes immediately.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple of (ExtractedDocument, list of Headings)

    Example:
        >>> doc, headings = extract_with_headings("textbook.pdf")
        >>> print(f"Extracted {doc.total_pages} pages, {len(headings)} headings")
        >>> for h in headings[:5]:
        ...     print(f"H{h.level}: {h.text}")
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("extracting_pdf_with_headings", path=str(pdf_path))

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to open PDF (corrupted or encrypted?): {e}") from e

    if doc.is_encrypted:
        raise ValueError(f"PDF is encrypted: {pdf_path}")

    # Single-pass extraction
    pages = []
    font_sizes = []
    text_blocks = []  # (text, font_size, page_num, char_offset)
    total_chars = 0
    char_offset = 0

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Get plain text for document extraction
        text = page.get_text()
        text = text.replace("\x00", "")
        text = "\n".join(line.strip() for line in text.split("\n") if line.strip())

        char_count = len(text)
        total_chars += char_count

        pages.append(
            ExtractedPage(
                page_num=page_num + 1,
                text=text,
                char_count=char_count,
            )
        )

        # Get font metadata for heading detection (same pass)
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip().replace("\x00", "")
                    font_size = span.get("size", 0.0)
                    if span_text and font_size > 0:
                        font_sizes.append(font_size)
                        text_blocks.append((span_text, font_size, page_num + 1, char_offset))
                        char_offset += len(span_text)

    doc.close()  # Close immediately after extraction

    # Create document
    extracted_doc = ExtractedDocument(
        file_path=str(pdf_path),
        total_pages=len(pages),
        pages=pages,
        total_chars=total_chars,
    )

    # Detect headings from collected font data
    headings = _detect_headings_from_blocks(text_blocks, font_sizes, str(pdf_path))

    logger.info(
        "pdf_extracted_with_headings",
        path=str(pdf_path),
        pages=len(pages),
        chars=total_chars,
        headings=len(headings),
    )

    return extracted_doc, headings


def _detect_headings_from_blocks(
    text_blocks: list[tuple[str, float, int, int]],
    font_sizes: list[float],
    pdf_path: str,
) -> list[Heading]:
    """Internal: Detect headings from pre-collected font metadata.

    Args:
        text_blocks: List of (text, font_size, page_num, char_offset) tuples
        font_sizes: List of all font sizes in document
        pdf_path: Path for logging

    Returns:
        List of detected Heading objects
    """
    if not font_sizes:
        logger.warning("no_text_with_font_metadata", path=pdf_path)
        return []

    median_size = statistics.median(font_sizes)

    if len(set(font_sizes)) == 1:
        logger.debug("uniform_font_size", path=pdf_path, median=median_size)
        return []

    stdev_size = statistics.stdev(font_sizes)

    # Thresholds
    h1_threshold = median_size + 2 * stdev_size
    h2_threshold = median_size + 1 * stdev_size
    h3_threshold = median_size + 0.5 * stdev_size

    logger.debug(
        "heading_thresholds",
        median=median_size,
        stdev=stdev_size,
        h1=h1_threshold,
        h2=h2_threshold,
        h3=h3_threshold,
    )

    # Classify headings
    headings = []
    for text, font_size, page_num, char_offset in text_blocks:
        if not (3 <= len(text) <= 100):
            continue
        if len(text.split()) > 15:
            continue

        level = None
        if font_size >= h1_threshold:
            level = 1
        elif font_size >= h2_threshold:
            level = 2
        elif font_size >= h3_threshold:
            level = 3

        if level:
            headings.append(
                Heading(
                    text=text,
                    level=level,
                    page_num=page_num,
                    font_size=font_size,
                    char_offset=char_offset,
                )
            )

    return headings
