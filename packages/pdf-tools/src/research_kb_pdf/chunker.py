"""Shared chunking utilities: TextChunk, token counting, text splitting.

Core types and functions used by the Docling extraction pipeline.
Chunking is done by Docling's HybridChunker; this module provides
the TextChunk data class, BGE tokenizer, and text-splitting helpers.
"""

import re
import threading
from dataclasses import dataclass, field

from transformers import AutoTokenizer

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
