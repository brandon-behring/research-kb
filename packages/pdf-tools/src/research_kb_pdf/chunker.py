"""PDF text chunker with token-accurate segmentation.

Provides two chunking strategies:
- chunk_document(): Fill-the-bucket paragraph accumulation (legacy)
- chunk_by_structure(): Structure-driven chunking that respects heading boundaries

Both produce 300±50 token segments with 50-token overlap.
"""

import re
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoTokenizer

from research_kb_pdf.pymupdf_extractor import ExtractedDocument, Heading, get_full_text
from research_kb_common import get_logger

logger = get_logger(__name__)

# BGE-large-en-v1.5 context limit
MAX_EMBEDDING_TOKENS = 512

# Common abbreviations that should NOT trigger sentence splits
_ABBREVIATIONS = {
    "Mr",
    "Mrs",
    "Ms",
    "Dr",
    "Prof",
    "Jr",
    "Sr",
    "vs",
    "etc",
    "al",
    "e.g",
    "i.e",
    "eg",
    "ie",
    "Fig",
    "Eq",
    "Ch",
    "Vol",
    "No",
    "Ref",
    "cf",
}

# Initialize BGE tokenizer (same model we'll use for embeddings)
_tokenizer = None
_tokenizer_lock = threading.Lock()
# Pin revision for reproducibility (same as embed_server.py)
BGE_MODEL = "BAAI/bge-large-en-v1.5"
BGE_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"


def get_tokenizer() -> AutoTokenizer:  # type: ignore[return-value]  # from_pretrained returns Backend, not AutoTokenizer
    """Lazy-load tokenizer to avoid startup cost (thread-safe)."""
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer  # type: ignore[return-value]
    with _tokenizer_lock:
        # Double-check after acquiring lock
        if _tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL, revision=BGE_REVISION)
        return _tokenizer  # type: ignore[return-value]


@dataclass
class TextChunk:
    """A chunk of text from a PDF document."""

    content: str
    start_page: int
    end_page: int
    token_count: int
    char_count: int
    chunk_index: int  # 0-indexed position in document
    metadata: dict = field(
        default_factory=dict
    )  # Extensible metadata (section, heading_level, etc.)


@dataclass
class Section:
    """A structurally-delimited section of a document.

    Represents text between two consecutive headings (or document start/end).
    Preamble (text before first heading) has heading="" and level=0.

    Attributes:
        heading: Heading text ("" for preamble before first heading)
        content: Full section text between this heading and next
        level: 0=preamble, 1-3=heading level
        start_page: First page of section (1-indexed)
        end_page: Last page of section (1-indexed)
    """

    heading: str
    content: str
    level: int
    start_page: int
    end_page: int


def count_tokens(text: str) -> int:
    """Count tokens using BGE tokenizer.

    Args:
        text: Text to count tokens for

    Returns:
        Number of tokens (excluding special tokens)

    Example:
        >>> count_tokens("Hello world")
        2
    """
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))  # type: ignore[attr-defined]  # AutoTokenizer stubs missing .encode()


def chunk_document(
    document: ExtractedDocument,
    target_tokens: int = 300,
    max_variance: int = 50,
    overlap_tokens: int = 50,
) -> list[TextChunk]:
    """Chunk extracted document into token-sized segments.

    Args:
        document: Extracted PDF document
        target_tokens: Target chunk size in tokens (default 300)
        max_variance: Maximum deviation from target (default ±50)
        overlap_tokens: Token overlap between chunks (default 50)

    Returns:
        List of TextChunk objects with content and metadata

    Example:
        >>> doc = extract_pdf("paper.pdf")
        >>> chunks = chunk_document(doc)
        >>> print(f"Created {len(chunks)} chunks")
        >>> print(f"First chunk: {chunks[0].token_count} tokens")
    """
    logger.debug(
        "chunking_document",
        path=document.file_path,
        pages=document.total_pages,
        target_tokens=target_tokens,
    )

    min_tokens = target_tokens - max_variance
    max_tokens = target_tokens + max_variance

    # Get full text
    full_text = get_full_text(document)

    # Split into paragraphs (preserve structure)
    paragraphs = split_paragraphs(full_text)

    chunks = []
    current_chunk: list[str] = []
    current_tokens = 0
    chunk_index = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # If this single paragraph is too large, split it into sentences
        if para_tokens > max_tokens:
            # First, save any accumulated content (only if meets minimum)
            if current_chunk and current_tokens >= min_tokens:
                chunk_content = "\n\n".join(current_chunk)
                chunk = create_chunk(
                    content=chunk_content, document=document, chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
                current_chunk = []
                current_tokens = 0
            elif current_chunk:
                # Accumulated content too small - merge with large paragraph before splitting
                merged = "\n\n".join(current_chunk) + "\n\n" + para
                para = merged
                para_tokens = count_tokens(para)
                current_chunk = []
                current_tokens = 0

            # Split large paragraph into sentence-based chunks
            sentence_chunks = split_large_paragraph(para, target_tokens, max_variance)
            for sent_chunk in sentence_chunks:
                chunk = create_chunk(content=sent_chunk, document=document, chunk_index=chunk_index)
                chunks.append(chunk)
                chunk_index += 1

            continue

        # If adding this paragraph exceeds max, save current chunk
        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunk_content = "\n\n".join(current_chunk)

            # Only save if chunk meets minimum token requirement
            if current_tokens >= min_tokens:
                chunk = create_chunk(
                    content=chunk_content, document=document, chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1

                # Overlap: keep paragraphs worth ~overlap_tokens
                overlap_paras = get_overlap_paragraphs(current_chunk, overlap_tokens)
                current_chunk = overlap_paras
                current_tokens = count_tokens("\n\n".join(current_chunk)) if current_chunk else 0
            else:
                # Chunk too small, keep accumulating
                # Don't reset - just continue adding paragraphs
                pass

        current_chunk.append(para)
        current_tokens += para_tokens

    # Save final chunk
    if current_chunk:
        chunk_content = "\n\n".join(current_chunk)
        chunk = create_chunk(content=chunk_content, document=document, chunk_index=chunk_index)
        chunks.append(chunk)

    # Validate chunks fit embedding model context
    oversized_count = 0
    for chunk in chunks:
        if chunk.token_count > MAX_EMBEDDING_TOKENS:
            oversized_count += 1
            logger.warning(
                "chunk_exceeds_embedding_limit",
                tokens=chunk.token_count,
                limit=MAX_EMBEDDING_TOKENS,
                chunk_index=chunk.chunk_index,
                path=document.file_path,
            )

    logger.debug(
        "document_chunked",
        path=document.file_path,
        num_chunks=len(chunks),
        avg_tokens=sum(c.token_count for c in chunks) / len(chunks) if chunks else 0,
        oversized_chunks=oversized_count,
    )

    return chunks


def split_paragraphs(text: str) -> list[str]:
    """Split text by paragraph boundaries (double newline).

    Args:
        text: Text to split

    Returns:
        List of non-empty paragraphs

    Example:
        >>> split_paragraphs("Para 1\\n\\nPara 2\\n\\n\\nPara 3")
        ['Para 1', 'Para 2', 'Para 3']
    """
    return [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]


def get_overlap_paragraphs(paragraphs: list[str], target_tokens: int) -> list[str]:
    """Get trailing paragraphs worth approximately target_tokens.

    Args:
        paragraphs: List of paragraphs to select from (end to start)
        target_tokens: Target token count for overlap

    Returns:
        Last few paragraphs totaling ~target_tokens

    Example:
        >>> paras = ["Short para", "Medium length para here", "Another one"]
        >>> overlap = get_overlap_paragraphs(paras, 10)
        >>> len(overlap) <= len(paras)
        True
    """
    if not paragraphs:
        return []

    overlap: list[str] = []
    tokens = 0

    for para in reversed(paragraphs):
        para_tokens = count_tokens(para)
        # Don't exceed 1.5x target (avoid too much overlap)
        if tokens + para_tokens > target_tokens * 1.5 and overlap:
            break
        overlap.insert(0, para)
        tokens += para_tokens
        if tokens >= target_tokens:
            break

    return overlap


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, respecting abbreviations.

    Avoids splitting on:
    - Common abbreviations (Dr., Mr., etc.)
    - Single-letter initials (J. Smith)
    - Latin abbreviations (e.g., i.e.)
    - Decimal numbers (3.14)

    Args:
        text: Text to split into sentences

    Returns:
        List of sentences

    Example:
        >>> split_sentences("Dr. Smith said hello. She was right.")
        ['Dr. Smith said hello.', 'She was right.']
        >>> split_sentences("The value is 3.14. Next sentence.")
        ['The value is 3.14.', 'Next sentence.']
    """
    if not text or not text.strip():
        return []

    # First do a simple split on sentence-ending punctuation
    # Pattern: punctuation followed by whitespace
    raw_parts = re.split(r"([.!?]+)\s+", text)

    # Reconstruct, checking for abbreviations
    sentences = []
    current = ""

    i = 0
    while i < len(raw_parts):
        part = raw_parts[i]

        if i + 1 < len(raw_parts) and raw_parts[i + 1] in {
            ".",
            "!",
            "?",
            "..",
            "...",
            ".!",
            "!?",
        }:
            # This part is followed by punctuation
            punct = raw_parts[i + 1]
            combined = part + punct

            # Check if this ends with an abbreviation
            words = part.split()
            last_word = words[-1] if words else ""

            # Check for abbreviations or single letter (initials)
            is_abbreviation = (
                last_word in _ABBREVIATIONS
                or last_word.rstrip(".") in _ABBREVIATIONS
                or (len(last_word) == 1 and last_word.isupper())  # Single capital letter
                or (len(last_word) == 2 and last_word[0].isupper() and last_word[1] == ".")  # "J."
                or re.match(r"^\d+$", last_word)  # Number before decimal
            )

            if is_abbreviation:
                # Don't split here - keep accumulating
                current += combined + " "
            else:
                # This is a real sentence end
                current += combined
                sentences.append(current.strip())
                current = ""
            i += 2
        else:
            # No punctuation follows, just accumulate
            current += part + " "
            i += 1

    # Add any remaining text
    if current.strip():
        sentences.append(current.strip())

    return [s for s in sentences if s]


def split_large_paragraph(paragraph: str, target_tokens: int, max_variance: int) -> list[str]:
    """Split a large paragraph into sentence-based chunks.

    Args:
        paragraph: Large paragraph to split
        target_tokens: Target chunk size
        max_variance: Allowed variance from target

    Returns:
        List of text chunks, each within target±variance tokens

    Example:
        >>> para = "First sentence. Second sentence. Third sentence." * 100
        >>> chunks = split_large_paragraph(para, 300, 50)
        >>> all(250 <= count_tokens(c) <= 350 for c in chunks[:-1])
        True
    """
    max_tokens = target_tokens + max_variance

    # Split into sentences using improved method
    sentences = split_sentences(paragraph)

    chunks = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        # If adding this sentence exceeds max, save current chunk
        if current_tokens + sentence_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0

        current.append(sentence)
        current_tokens += sentence_tokens

    # Save remaining sentences
    if current:
        chunks.append(" ".join(current))

    return chunks


def create_chunk(content: str, document: ExtractedDocument, chunk_index: int) -> TextChunk:
    """Create chunk with page number tracking.

    Args:
        content: Chunk text content
        document: Source document for page lookup
        chunk_index: 0-indexed position in document

    Returns:
        TextChunk with page range estimated from content

    Example:
        >>> doc = extract_pdf("paper.pdf")
        >>> chunk = create_chunk("Introduction text...", doc, 0)
        >>> chunk.start_page >= 1
        True
    """
    # Estimate page numbers by finding content in document pages
    start_page, end_page = estimate_page_range(content, document)

    return TextChunk(
        content=content,
        start_page=start_page,
        end_page=end_page,
        token_count=count_tokens(content),
        char_count=len(content),
        chunk_index=chunk_index,
    )


def estimate_page_range(content: str, document: ExtractedDocument) -> tuple[int, int]:
    """Estimate which pages a chunk spans by finding content matches.

    Uses first and last 100 chars of chunk to find start/end pages.

    Args:
        content: Chunk content to locate
        document: Source document with pages

    Returns:
        Tuple of (start_page, end_page) - both 1-indexed

    Example:
        >>> doc = extract_pdf("paper.pdf")
        >>> full_text = get_full_text(doc)
        >>> start, end = estimate_page_range(full_text[:500], doc)
        >>> start == 1
        True
    """
    # Get first/last snippets for matching
    snippet_len = min(100, len(content) // 2)
    first_snippet = content[:snippet_len].strip()
    last_snippet = content[-snippet_len:].strip() if len(content) > snippet_len else first_snippet

    start_page = None
    end_page = None

    for page in document.pages:
        if start_page is None and first_snippet in page.text:
            start_page = page.page_num
        if last_snippet in page.text:
            end_page = page.page_num

    # Fallback: if not found, assume first page
    if start_page is None:
        start_page = 1
    if end_page is None:
        end_page = start_page

    return start_page, end_page


def chunk_with_sections(
    document: ExtractedDocument,
    headings: list,  # List of Heading objects
    target_tokens: int = 300,
    max_variance: int = 50,
    overlap_tokens: int = 50,
) -> list[TextChunk]:
    """Chunk document with section context tracking.

    Adds metadata to each chunk indicating which section it belongs to.

    Args:
        document: Extracted PDF document
        headings: List of Heading objects from detect_headings()
        target_tokens: Target chunk size in tokens (default 300)
        max_variance: Maximum deviation from target (default ±50)
        overlap_tokens: Token overlap between chunks (default 50)

    Returns:
        List of TextChunk objects with section metadata populated

    Example:
        >>> from research_kb_pdf import extract_with_headings, chunk_with_sections
        >>> doc, headings = extract_with_headings("textbook.pdf")
        >>> chunks = chunk_with_sections(doc, headings)
        >>> for chunk in chunks:
        ...     if chunk.metadata.get("section"):
        ...         print(f"Section: {chunk.metadata['section']}")
    """
    # First chunk normally
    chunks = chunk_document(document, target_tokens, max_variance, overlap_tokens)

    if not headings:
        logger.debug("no_headings_for_section_tracking", path=document.file_path)
        return chunks

    # Build heading index sorted by character offset
    sorted_headings = sorted(headings, key=lambda h: h.char_offset)

    # Track current section for each chunk
    full_text = get_full_text(document)

    for chunk in chunks:
        # Initialize metadata fields (for chunks we can't locate)
        chunk.metadata["section"] = None
        chunk.metadata["heading_level"] = None

        # Find chunk's approximate position in document
        # Unicode-normalize both sides to handle non-breaking spaces, soft hyphens, etc.
        normalized_text = unicodedata.normalize("NFKC", full_text)
        normalized_prefix = unicodedata.normalize("NFKC", chunk.content[:50])
        chunk_start_pos = normalized_text.find(normalized_prefix)

        if chunk_start_pos == -1:
            # Fallback: couldn't locate chunk - metadata already set to None
            continue

        # Find the most recent heading before this chunk
        current_section = None
        current_level = None

        for heading in sorted_headings:
            if heading.char_offset <= chunk_start_pos:
                current_section = heading.text
                current_level = heading.level
            else:
                break  # Headings are sorted, so we can stop

        # Update metadata if section found
        if current_section:
            chunk.metadata["section"] = current_section
            chunk.metadata["heading_level"] = current_level

    logger.debug(
        "sections_tracked",
        path=document.file_path,
        chunks=len(chunks),
        headings=len(headings),
    )

    return chunks


def split_into_sections(
    document: ExtractedDocument,
    headings: list[Heading],
) -> list[Section]:
    """Split document into structurally-delimited sections using heading boundaries.

    Uses page-based heading lookup: finds each heading's text on its known page,
    then splits the document at those points. This avoids char_offset mismatches
    between get_text() and get_text("dict") in PyMuPDF.

    Args:
        document: Extracted PDF document with page text
        headings: List of Heading objects from detect_headings()

    Returns:
        List of Section objects. Text before first heading becomes a preamble
        section (heading="", level=0). If no headings, returns a single
        preamble section containing the full document.

    Example:
        >>> doc, headings = extract_with_headings("textbook.pdf")
        >>> sections = split_into_sections(doc, headings)
        >>> for s in sections:
        ...     print(f"L{s.level}: '{s.heading}' ({len(s.content)} chars, pp.{s.start_page}-{s.end_page})")
    """
    if not headings:
        full_text = get_full_text(document)
        return [
            Section(
                heading="",
                content=full_text,
                level=0,
                start_page=1,
                end_page=document.total_pages or 1,
            )
        ]

    # Sort headings by (page_num, char_offset) for document order
    sorted_headings = sorted(headings, key=lambda h: (h.page_num, h.char_offset))

    # Locate each heading on its page and record split points
    # Each split point: (page_index, char_pos_in_page, heading)
    split_points: list[tuple[int, int, Heading]] = []

    for heading in sorted_headings:
        page_idx = heading.page_num - 1  # Convert to 0-indexed
        if page_idx < 0 or page_idx >= len(document.pages):
            logger.warning(
                "heading_page_out_of_range",
                heading=heading.text,
                page_num=heading.page_num,
                total_pages=len(document.pages),
            )
            continue

        page_text = document.pages[page_idx].text
        pos = page_text.find(heading.text)
        if pos == -1:
            logger.warning(
                "heading_not_found_on_page",
                heading=heading.text,
                page_num=heading.page_num,
            )
            continue

        split_points.append((page_idx, pos, heading))

    if not split_points:
        # All headings failed to locate — return full document as preamble
        full_text = get_full_text(document)
        return [
            Section(
                heading="",
                content=full_text,
                level=0,
                start_page=1,
                end_page=document.total_pages or 1,
            )
        ]

    sections: list[Section] = []

    # Build sections by iterating through split points
    for i, (page_idx, pos_in_page, heading) in enumerate(split_points):
        # Determine content range: from this heading to the next split point
        # heading line itself: text from pos_in_page to end of that line
        heading_line_end = pos_in_page + len(heading.text)
        # Skip past the heading line (find next newline or use end of heading text)
        page_text = document.pages[page_idx].text
        newline_after = page_text.find("\n", heading_line_end)
        content_start_in_page = newline_after + 1 if newline_after != -1 else heading_line_end

        if i == 0 and (page_idx > 0 or pos_in_page > 0):
            # There's text before the first heading — create preamble section
            preamble_parts = []
            for p_idx in range(page_idx):
                preamble_parts.append(document.pages[p_idx].text)
            # Add text on the heading's page before the heading
            pre_heading_text = page_text[:pos_in_page].strip()
            if pre_heading_text:
                preamble_parts.append(pre_heading_text)

            preamble_content = "\n\n".join(p for p in preamble_parts if p.strip())
            if preamble_content.strip():
                sections.append(
                    Section(
                        heading="",
                        content=preamble_content,
                        level=0,
                        start_page=1,
                        end_page=document.pages[page_idx].page_num,
                    )
                )

        # Collect content from after this heading to the next heading (or end of doc)
        if i + 1 < len(split_points):
            next_page_idx, next_pos_in_page, _ = split_points[i + 1]
        else:
            next_page_idx = len(document.pages) - 1
            next_pos_in_page = len(document.pages[next_page_idx].text)

        content_parts = []

        if page_idx == next_page_idx:
            # Same page: slice between heading end and next heading start
            section_text = page_text[content_start_in_page:next_pos_in_page].strip()
            if section_text:
                content_parts.append(section_text)
        else:
            # Remainder of current page after heading
            remainder = page_text[content_start_in_page:].strip()
            if remainder:
                content_parts.append(remainder)

            # Full intermediate pages
            for p_idx in range(page_idx + 1, next_page_idx):
                page_content = document.pages[p_idx].text.strip()
                if page_content:
                    content_parts.append(page_content)

            # Partial last page (up to next heading)
            if i + 1 < len(split_points):
                last_page_text = document.pages[next_page_idx].text[:next_pos_in_page].strip()
            else:
                last_page_text = document.pages[next_page_idx].text.strip()
            if last_page_text:
                content_parts.append(last_page_text)

        section_content = "\n\n".join(content_parts)

        end_page_idx = next_page_idx if i + 1 < len(split_points) else next_page_idx
        sections.append(
            Section(
                heading=heading.text,
                content=section_content,
                level=heading.level,
                start_page=document.pages[page_idx].page_num,
                end_page=document.pages[end_page_idx].page_num,
            )
        )

    logger.debug(
        "document_split_into_sections",
        path=document.file_path,
        sections=len(sections),
        headings_located=len(split_points),
        headings_total=len(headings),
    )

    return sections


def chunk_by_structure(
    document: ExtractedDocument,
    headings: list[Heading],
    target_tokens: int = 300,
    max_variance: int = 50,
    overlap_tokens: int = 50,
    min_section_chars: int = 100,
) -> list[TextChunk]:
    """Chunk document using heading-driven structure boundaries.

    Critical invariant: **no chunk crosses a heading boundary**. Each section
    (text between two consecutive headings) is chunked independently.

    Algorithm:
    1. Split document into sections via split_into_sections()
    2. Merge tiny sections (<min_section_chars) into adjacent sections
    3. Chunk each section independently using paragraph accumulation
    4. Overlap applies within section only — never across headings
    5. Enforce 512-token hard limit via sentence splitting

    Graceful degradation: No headings → single preamble section → same behavior
    as chunk_document() with paragraph boundaries. Strictly better, never worse.

    Args:
        document: Extracted PDF document
        headings: List of Heading objects from detect_headings()
        target_tokens: Target chunk size in tokens (default 300)
        max_variance: Maximum deviation from target (default ±50)
        overlap_tokens: Token overlap between chunks (default 50)
        min_section_chars: Minimum section size before merging (default 100)

    Returns:
        List of TextChunk objects with section metadata:
        - metadata["section"]: heading text
        - metadata["heading_level"]: heading level (0-3)
        - metadata["chunking_method"]: "structure"

    Example:
        >>> doc, headings = extract_with_headings("textbook.pdf")
        >>> chunks = chunk_by_structure(doc, headings)
        >>> for c in chunks[:3]:
        ...     print(f"[{c.metadata['section']}] {c.token_count}t")
    """
    logger.debug(
        "chunk_by_structure_start",
        path=document.file_path,
        headings=len(headings),
        target_tokens=target_tokens,
    )

    sections = split_into_sections(document, headings)

    # Merge tiny sections into neighbors
    sections = _merge_tiny_sections(sections, min_section_chars)

    min_tokens = target_tokens - max_variance
    max_tokens = target_tokens + max_variance
    all_chunks: list[TextChunk] = []
    chunk_index = 0

    for section in sections:
        if not section.content.strip():
            continue

        section_chunks = _chunk_section(
            section=section,
            document=document,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            target_tokens=target_tokens,
            max_variance=max_variance,
            overlap_tokens=overlap_tokens,
            start_chunk_index=chunk_index,
        )

        all_chunks.extend(section_chunks)
        chunk_index += len(section_chunks)

    # Hard limit enforcement: split any chunk >512 tokens at sentence boundaries
    validated_chunks: list[TextChunk] = []
    final_index = 0
    for chunk in all_chunks:
        if chunk.token_count > MAX_EMBEDDING_TOKENS:
            logger.warning(
                "chunk_exceeds_limit_splitting",
                tokens=chunk.token_count,
                limit=MAX_EMBEDDING_TOKENS,
                section=chunk.metadata.get("section", ""),
            )
            sub_texts = split_large_paragraph(
                chunk.content, target_tokens, max_variance
            )
            for sub_text in sub_texts:
                sub_chunk = TextChunk(
                    content=sub_text,
                    start_page=chunk.start_page,
                    end_page=chunk.end_page,
                    token_count=count_tokens(sub_text),
                    char_count=len(sub_text),
                    chunk_index=final_index,
                    metadata=dict(chunk.metadata),
                )
                validated_chunks.append(sub_chunk)
                final_index += 1
        else:
            chunk.chunk_index = final_index
            validated_chunks.append(chunk)
            final_index += 1

    logger.debug(
        "chunk_by_structure_complete",
        path=document.file_path,
        sections=len(sections),
        chunks=len(validated_chunks),
        avg_tokens=(
            sum(c.token_count for c in validated_chunks) / len(validated_chunks)
            if validated_chunks
            else 0
        ),
    )

    return validated_chunks


def _merge_tiny_sections(
    sections: list[Section], min_chars: int
) -> list[Section]:
    """Merge sections shorter than min_chars into adjacent sections.

    Tiny sections are merged forward (into next section). If a tiny section
    is the last one, it's merged backward (into previous section).

    Args:
        sections: List of sections to potentially merge
        min_chars: Minimum character count to keep a section standalone

    Returns:
        List of sections with tiny ones merged into neighbors
    """
    if len(sections) <= 1:
        return sections

    merged: list[Section] = []

    i = 0
    while i < len(sections):
        section = sections[i]

        if len(section.content.strip()) < min_chars:
            if i + 1 < len(sections):
                # Merge forward: prepend this section's content to the next
                next_sec = sections[i + 1]
                combined_content = section.content.strip()
                if combined_content:
                    combined_content += "\n\n" + next_sec.content
                else:
                    combined_content = next_sec.content

                sections[i + 1] = Section(
                    heading=next_sec.heading,
                    content=combined_content,
                    level=next_sec.level,
                    start_page=section.start_page,
                    end_page=next_sec.end_page,
                )
            elif merged:
                # Last section is tiny: merge backward into previous
                prev = merged[-1]
                combined_content = prev.content
                if section.content.strip():
                    combined_content += "\n\n" + section.content.strip()
                merged[-1] = Section(
                    heading=prev.heading,
                    content=combined_content,
                    level=prev.level,
                    start_page=prev.start_page,
                    end_page=section.end_page,
                )
            else:
                # Only section and it's tiny — keep it
                merged.append(section)
        else:
            merged.append(section)

        i += 1

    return merged


def _chunk_section(
    section: Section,
    document: ExtractedDocument,
    min_tokens: int,
    max_tokens: int,
    target_tokens: int,
    max_variance: int,
    overlap_tokens: int,
    start_chunk_index: int,
) -> list[TextChunk]:
    """Chunk a single section using paragraph accumulation.

    Reuses the same fill-the-bucket logic as chunk_document() but scoped
    to a single section's content. Overlap applies within the section only.

    Args:
        section: Section to chunk
        document: Parent document (for page estimation)
        min_tokens: Minimum acceptable chunk tokens
        max_tokens: Maximum acceptable chunk tokens
        target_tokens: Target chunk tokens
        max_variance: Allowed variance from target
        overlap_tokens: Overlap token count
        start_chunk_index: Starting chunk_index for this section's chunks

    Returns:
        List of TextChunk objects with section metadata populated
    """
    paragraphs = split_paragraphs(section.content)
    if not paragraphs:
        return []

    chunks: list[TextChunk] = []
    current_chunk: list[str] = []
    current_tokens = 0
    chunk_index = start_chunk_index

    section_metadata = {
        "section": section.heading or None,
        "heading_level": section.level,
        "chunking_method": "structure",
    }

    def _emit_chunk(content: str) -> TextChunk:
        """Create a TextChunk with section metadata."""
        nonlocal chunk_index
        start_page, end_page = estimate_page_range(content, document)
        # Clamp to section page range
        start_page = max(start_page, section.start_page)
        end_page = min(end_page, section.end_page)

        chunk = TextChunk(
            content=content,
            start_page=start_page,
            end_page=end_page,
            token_count=count_tokens(content),
            char_count=len(content),
            chunk_index=chunk_index,
            metadata=dict(section_metadata),
        )
        chunk_index += 1
        return chunk

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # Large paragraph: split into sentences
        if para_tokens > max_tokens:
            # Save accumulated content first
            if current_chunk and current_tokens >= min_tokens:
                chunk_content = "\n\n".join(current_chunk)
                chunks.append(_emit_chunk(chunk_content))
                current_chunk = []
                current_tokens = 0
            elif current_chunk:
                # Too small — merge with large paragraph
                merged = "\n\n".join(current_chunk) + "\n\n" + para
                para = merged
                current_chunk = []
                current_tokens = 0

            sentence_chunks = split_large_paragraph(para, target_tokens, max_variance)
            for sent_chunk in sentence_chunks:
                chunks.append(_emit_chunk(sent_chunk))
            continue

        # If adding exceeds max, save current chunk
        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunk_content = "\n\n".join(current_chunk)

            if current_tokens >= min_tokens:
                chunks.append(_emit_chunk(chunk_content))
                # Overlap: keep trailing paragraphs within section
                overlap_paras = get_overlap_paragraphs(current_chunk, overlap_tokens)
                current_chunk = overlap_paras
                current_tokens = (
                    count_tokens("\n\n".join(current_chunk)) if current_chunk else 0
                )
            # else: too small, keep accumulating

        current_chunk.append(para)
        current_tokens += para_tokens

    # Final chunk
    if current_chunk:
        chunk_content = "\n\n".join(current_chunk)
        chunks.append(_emit_chunk(chunk_content))

    return chunks
