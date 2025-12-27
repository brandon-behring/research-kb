"""Source tools for MCP server.

Exposes source listing, detail, and citation functionality.
"""

from __future__ import annotations

from typing import Optional

from fastmcp import FastMCP

from uuid import UUID

from research_kb_api.service import (
    get_sources,
    get_source_by_id,
    get_source_chunks,
    get_citations_for_source,
)
from research_kb_storage import get_citing_sources, get_cited_sources
from research_kb_mcp.formatters import (
    format_source_list,
    format_source_detail,
    format_citations,
    format_citing_sources,
    format_cited_sources,
)


def register_source_tools(mcp: FastMCP) -> None:
    """Register source tools with the MCP server."""

    @mcp.tool()
    async def research_kb_list_sources(
        limit: int = 50,
        offset: int = 0,
        source_type: Optional[str] = None,
    ) -> str:
        """List sources (papers and textbooks) in the knowledge base.

        Browse the corpus of causal inference literature including academic
        papers and textbooks.

        Args:
            limit: Maximum number of sources to return (1-100, default 50)
            offset: Pagination offset (default 0)
            source_type: Filter by type ("paper" or "textbook", default all)

        Returns:
            Markdown-formatted list of sources with:
            - Title
            - Authors (first 2 + et al.)
            - Year
            - Type (paper/textbook)
            - Source ID for detail queries
        """
        limit = max(1, min(100, limit))
        offset = max(0, offset)

        sources = await get_sources(
            limit=limit,
            offset=offset,
            source_type=source_type,
        )
        return format_source_list(sources)

    @mcp.tool()
    async def research_kb_get_source(
        source_id: str,
        include_chunks: bool = False,
        chunk_limit: int = 10,
    ) -> str:
        """Get detailed information about a specific source.

        Retrieve full metadata and optionally content chunks for a paper
        or textbook.

        Args:
            source_id: UUID of the source (from search results or list)
            include_chunks: Include content chunks in response (default False)
            chunk_limit: Maximum chunks to include (1-50, default 10)

        Returns:
            Markdown-formatted source details with:
            - Full title
            - All authors
            - Year and type
            - Metadata (DOI, ISBN, etc.)
            - Content chunks (if requested)
        """
        source = await get_source_by_id(source_id)

        if source is None:
            return f"**Error:** Source `{source_id}` not found"

        chunks = None
        if include_chunks:
            chunk_limit = max(1, min(50, chunk_limit))
            chunks = await get_source_chunks(source_id, limit=chunk_limit)

        return format_source_detail(source, chunks)

    @mcp.tool()
    async def research_kb_get_source_citations(source_id: str) -> str:
        """Get citation relationships for a source.

        Find papers that cite this source and papers cited by this source.
        Useful for understanding influence and context.

        Args:
            source_id: UUID of the source

        Returns:
            Markdown-formatted citation information with:
            - Papers citing this source (downstream influence)
            - Papers cited by this source (foundations/context)
            - Source IDs for each related paper
        """
        source = await get_source_by_id(source_id)

        if source is None:
            return f"**Error:** Source `{source_id}` not found"

        citations = await get_citations_for_source(source_id)
        return format_citations(citations)

    @mcp.tool()
    async def research_kb_get_citing_sources(source_id: str) -> str:
        """Find all sources that cite a given source.

        Returns papers in the knowledge base that cite this source.
        Useful for finding downstream influence and who built on this work.

        Args:
            source_id: UUID of the source to find citations for

        Returns:
            Markdown-formatted list of citing sources with:
            - Title
            - Authors
            - Year
            - Source ID for further queries
        """
        source = await get_source_by_id(source_id)

        if source is None:
            return f"**Error:** Source `{source_id}` not found"

        citing = await get_citing_sources(UUID(source_id))
        return format_citing_sources(citing, source_id)

    @mcp.tool()
    async def research_kb_get_cited_sources(source_id: str) -> str:
        """Find all sources that a given source cites.

        Returns papers in the knowledge base that are cited by this source.
        Useful for finding foundations, context, and related work.

        Args:
            source_id: UUID of the source to find references for

        Returns:
            Markdown-formatted list of cited sources with:
            - Title
            - Authors
            - Year
            - Source ID for further queries
        """
        source = await get_source_by_id(source_id)

        if source is None:
            return f"**Error:** Source `{source_id}` not found"

        cited = await get_cited_sources(UUID(source_id))
        return format_cited_sources(cited, source_id)
