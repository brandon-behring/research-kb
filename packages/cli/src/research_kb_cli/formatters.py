"""Output formatters for CLI results.

Provides multiple output formats:
- markdown: Human-readable markdown with provenance
- json: Machine-parseable JSON
- agent: Agent-friendly format with authority tiers and structured metadata

Master Plan Reference: Lines 588-590, 1318
"""

import json

from research_kb_contracts import SearchResult


def format_result_markdown(result: SearchResult, show_content: bool = True) -> str:
    """Format a single search result as markdown.

    Args:
        result: Search result to format
        show_content: Whether to include content snippet

    Returns:
        Markdown-formatted string
    """
    source = result.source
    chunk = result.chunk

    # Build location string
    location_parts = []
    if chunk.metadata.get("section"):
        location_parts.append(chunk.metadata["section"])
    if chunk.page_start:
        if chunk.page_end and chunk.page_end != chunk.page_start:
            location_parts.append(f"pp. {chunk.page_start}-{chunk.page_end}")
        else:
            location_parts.append(f"p. {chunk.page_start}")

    location = ", ".join(location_parts) if location_parts else "Unknown location"

    # Format score
    score = f"{result.combined_score:.2f}"

    # Build output
    lines = [
        f"## Result {result.rank} (score: {score})",
        f"**Source**: {source.title} ({source.year}) [{source.source_type.value}]",
        f"**Location**: {location}",
        f"**Authority**: {source.metadata.get('authority', 'standard')}",
    ]

    if show_content:
        # Truncate content for display
        content = chunk.content[:500]
        if len(chunk.content) > 500:
            content += "..."
        lines.append(f"\n> {content.replace(chr(10), chr(10) + '> ')}")

    return "\n".join(lines)


def format_results_markdown(
    results: list[SearchResult],
    query: str,
    show_content: bool = True,
) -> str:
    """Format multiple search results as markdown.

    Args:
        results: List of search results
        query: Original query string
        show_content: Whether to include content snippets

    Returns:
        Markdown-formatted string
    """
    if not results:
        return f"No results found for: **{query}**"

    header = f'# Search Results for: "{query}"\n\nFound {len(results)} results:\n'

    formatted = [format_result_markdown(r, show_content) for r in results]

    return header + "\n\n---\n\n".join(formatted)


def format_result_json(result: SearchResult) -> dict:
    """Format a single search result as JSON-serializable dict.

    Args:
        result: Search result to format

    Returns:
        Dictionary representation
    """
    return {
        "rank": result.rank,
        "score": result.combined_score,
        "fts_score": result.fts_score,
        "vector_score": result.vector_score,
        "source": {
            "id": str(result.source.id),
            "title": result.source.title,
            "authors": result.source.authors,
            "year": result.source.year,
            "type": result.source.source_type.value,
            "authority": result.source.metadata.get("authority", "standard"),
        },
        "chunk": {
            "id": str(result.chunk.id),
            "content": result.chunk.content,
            "page_start": result.chunk.page_start,
            "page_end": result.chunk.page_end,
            "section": result.chunk.metadata.get("section"),
            "heading_level": result.chunk.metadata.get("heading_level"),
        },
    }


def format_results_json(results: list[SearchResult], query: str) -> str:
    """Format multiple search results as JSON string.

    Args:
        results: List of search results
        query: Original query string

    Returns:
        JSON string
    """
    output = {
        "query": query,
        "result_count": len(results),
        "results": [format_result_json(r) for r in results],
    }
    return json.dumps(output, indent=2)


def format_result_agent(result: SearchResult) -> str:
    """Format a single search result for agent consumption.

    Agent-friendly format emphasizes:
    - Provenance (exact source, page, section)
    - Authority tier for trust calibration
    - Structured metadata for downstream processing

    Master Plan Reference: Line 1318

    Args:
        result: Search result to format

    Returns:
        Agent-friendly formatted string
    """
    source = result.source
    chunk = result.chunk

    # Build provenance string
    provenance_parts = []
    if source.authors:
        first_author = source.authors[0].split()[-1]  # Last name
        provenance_parts.append(first_author)
    if source.year:
        provenance_parts.append(str(source.year))
    provenance = " ".join(provenance_parts) if provenance_parts else "Unknown"

    # Build location
    page_info = ""
    if chunk.page_start:
        if chunk.page_end and chunk.page_end != chunk.page_start:
            page_info = f"pp.{chunk.page_start}-{chunk.page_end}"
        else:
            page_info = f"p.{chunk.page_start}"

    section = chunk.metadata.get("section", "")

    # Authority tier
    authority = source.metadata.get("authority", "standard")

    # Format for agent
    lines = [
        f"[{result.rank}] {source.title[:60]}",
        f"    CITE: ({provenance})",
        f"    TYPE: {source.source_type.value} | AUTH: {authority}",
        f"    LOC: {section} {page_info}".strip(),
        f"    SCORE: {result.combined_score:.3f}",
        "",
        f"    {chunk.content[:400]}{'...' if len(chunk.content) > 400 else ''}",
    ]

    return "\n".join(lines)


def format_results_agent(
    results: list[SearchResult],
    query: str,
    context_type: str = "balanced",
) -> str:
    """Format multiple search results for agent consumption.

    Args:
        results: List of search results
        query: Original query string
        context_type: Context mode (building, auditing, balanced)

    Returns:
        Agent-friendly formatted string
    """
    if not results:
        return f"[NO RESULTS] Query: {query}"

    header = [
        "RESEARCH_KB_RESULTS",
        f"QUERY: {query}",
        f"CONTEXT: {context_type}",
        f"COUNT: {len(results)}",
        "---",
    ]

    formatted = [format_result_agent(r) for r in results]

    footer = [
        "---",
        "USAGE: Cite sources with (Author Year) format. Verify page numbers.",
    ]

    return "\n".join(header + formatted + footer)
