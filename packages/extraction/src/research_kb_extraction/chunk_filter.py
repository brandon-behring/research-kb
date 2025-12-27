"""Heuristic chunk filtering for extraction quality.

Filters out chunks that are unlikely to yield useful concept extractions:
- Too short (< 100 characters)
- Too long (> 15,000 characters)
- High noise content (tables, formulas, references)
- Repeated headers/footers
- OCR artifacts

Chunks are classified as:
- PROCESS: Normal extraction
- SKIP: No extraction (too noisy/short)
- FLAG: Extract but mark for review
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from research_kb_common import get_logger

logger = get_logger(__name__)


class FilterDecision(str, Enum):
    """Filtering decision for a chunk."""

    PROCESS = "process"  # Normal extraction
    SKIP = "skip"  # Skip extraction (too noisy)
    FLAG = "flag"  # Extract but flag for review


@dataclass
class FilterResult:
    """Result of chunk filtering."""

    decision: FilterDecision
    reason: Optional[str] = None
    metrics: Optional[dict] = None


# Regex patterns for filtering
PATTERNS = {
    "reference_block": re.compile(
        r"^\s*\[?\d+\]?\s*[A-Z][a-z]+.*\d{4}",
        re.MULTILINE,
    ),
    "table_row": re.compile(
        r"^[\s\d\.\,\-\|]+$",
        re.MULTILINE,
    ),
    "equation_heavy": re.compile(
        r"[∑∏∫∂∇√∞≈≠≤≥∈∉⊂⊃∪∩]"
    ),
    "repeated_header": re.compile(
        r"^(Chapter|Section|Figure|Table)\s+\d",
        re.MULTILINE | re.IGNORECASE,
    ),
    "page_number": re.compile(
        r"^\s*\d+\s*$",
        re.MULTILINE,
    ),
    "ocr_garbage": re.compile(
        r"[^\x00-\x7F]{5,}|[!@#$%^&*]{3,}",
    ),
}


def calculate_metrics(content: str) -> dict:
    """Calculate filtering metrics for a chunk."""
    length = len(content)
    lines = content.split("\n")

    # Count various patterns
    reference_matches = len(PATTERNS["reference_block"].findall(content))
    table_rows = len(PATTERNS["table_row"].findall(content))
    equation_symbols = len(PATTERNS["equation_heavy"].findall(content))
    repeated_headers = len(PATTERNS["repeated_header"].findall(content))
    page_numbers = len(PATTERNS["page_number"].findall(content))
    ocr_garbage = len(PATTERNS["ocr_garbage"].findall(content))

    # Calculate ratios
    alpha_chars = sum(1 for c in content if c.isalpha())
    alpha_ratio = alpha_chars / max(length, 1)

    # Word count
    words = content.split()
    word_count = len(words)
    avg_word_length = sum(len(w) for w in words) / max(word_count, 1)

    return {
        "length": length,
        "line_count": len(lines),
        "word_count": word_count,
        "alpha_ratio": alpha_ratio,
        "avg_word_length": avg_word_length,
        "reference_matches": reference_matches,
        "table_rows": table_rows,
        "equation_symbols": equation_symbols,
        "repeated_headers": repeated_headers,
        "page_numbers": page_numbers,
        "ocr_garbage": ocr_garbage,
    }


def filter_chunk(content: str) -> FilterResult:
    """Apply heuristic filtering to a chunk.

    Args:
        content: Chunk text content

    Returns:
        FilterResult with decision and reason
    """
    metrics = calculate_metrics(content)
    length = metrics["length"]

    # Too short - skip
    if length < 100:
        return FilterResult(
            decision=FilterDecision.SKIP,
            reason="too_short",
            metrics=metrics,
        )

    # Too long - flag for review (might need splitting)
    if length > 15000:
        return FilterResult(
            decision=FilterDecision.FLAG,
            reason="too_long",
            metrics=metrics,
        )

    # Very low alpha ratio (likely table/formula)
    if metrics["alpha_ratio"] < 0.4:
        return FilterResult(
            decision=FilterDecision.SKIP,
            reason="low_alpha_ratio",
            metrics=metrics,
        )

    # Heavy reference content (bibliography)
    if metrics["reference_matches"] > 5:
        return FilterResult(
            decision=FilterDecision.SKIP,
            reason="reference_block",
            metrics=metrics,
        )

    # Many table rows
    if metrics["table_rows"] > metrics["line_count"] * 0.5:
        return FilterResult(
            decision=FilterDecision.FLAG,
            reason="table_heavy",
            metrics=metrics,
        )

    # OCR garbage
    if metrics["ocr_garbage"] > 3:
        return FilterResult(
            decision=FilterDecision.SKIP,
            reason="ocr_garbage",
            metrics=metrics,
        )

    # Equation-heavy content (flag but still extract)
    if metrics["equation_symbols"] > 10:
        return FilterResult(
            decision=FilterDecision.FLAG,
            reason="equation_heavy",
            metrics=metrics,
        )

    # Normal content - process
    return FilterResult(
        decision=FilterDecision.PROCESS,
        reason=None,
        metrics=metrics,
    )


def filter_chunks(
    chunks: list[tuple[str, str]],
    include_flagged: bool = True,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Filter a list of chunks.

    Args:
        chunks: List of (chunk_id, content) tuples
        include_flagged: Include flagged chunks in process list

    Returns:
        Tuple of (process_list, skip_list, flag_list)
    """
    process_list = []
    skip_list = []
    flag_list = []

    for chunk_id, content in chunks:
        result = filter_chunk(content)

        if result.decision == FilterDecision.PROCESS:
            process_list.append((chunk_id, content))
        elif result.decision == FilterDecision.SKIP:
            skip_list.append((chunk_id, content))
            logger.debug(
                "chunk_skipped",
                chunk_id=chunk_id,
                reason=result.reason,
            )
        else:  # FLAG
            flag_list.append((chunk_id, content))
            if include_flagged:
                process_list.append((chunk_id, content))

    logger.info(
        "chunks_filtered",
        total=len(chunks),
        process=len(process_list),
        skip=len(skip_list),
        flagged=len(flag_list),
    )

    return process_list, skip_list, flag_list


def get_filter_stats(chunks: list[tuple[str, str]]) -> dict:
    """Get filtering statistics for a set of chunks.

    Args:
        chunks: List of (chunk_id, content) tuples

    Returns:
        Statistics dict
    """
    results = [filter_chunk(content) for _, content in chunks]

    by_decision = {
        "process": sum(1 for r in results if r.decision == FilterDecision.PROCESS),
        "skip": sum(1 for r in results if r.decision == FilterDecision.SKIP),
        "flag": sum(1 for r in results if r.decision == FilterDecision.FLAG),
    }

    by_reason = {}
    for r in results:
        if r.reason:
            by_reason[r.reason] = by_reason.get(r.reason, 0) + 1

    return {
        "total": len(chunks),
        "by_decision": by_decision,
        "by_reason": by_reason,
    }
