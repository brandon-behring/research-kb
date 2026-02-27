"""Citation tools for MCP server.

Exposes citation network and bibliographic coupling functionality.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from fastmcp import FastMCP

from research_kb_api.service import get_source_by_id
from research_kb_storage import BiblioStore, get_citing_sources, get_cited_sources
from research_kb_mcp.formatters import (
    format_citation_network,
    format_citation_network_json,
    format_biblio_similar,
    format_biblio_similar_json,
)


def register_citation_tools(mcp: FastMCP) -> None:
    """Register citation tools with the MCP server."""

    @mcp.tool()
    async def research_kb_citation_network(
        source_id: str,
        limit: int = 20,
        output_format: Literal["markdown", "json"] = "markdown",
    ) -> str:
        """Get bidirectional citation network for a source.

        Shows both papers that cite this source (downstream influence) and
        papers cited by this source (foundations/context). Useful for
        understanding a paper's position in the literature.

        Args:
            source_id: UUID of the source
            limit: Maximum sources per direction (1-50, default 20)
            output_format: Response format - "markdown" (default) or "json"

        Returns:
            Markdown-formatted or JSON citation network with:
            - Papers citing this source (who built on this work)
            - Papers cited by this source (foundations)
            - Source IDs for follow-up queries
        """
        source = await get_source_by_id(source_id)

        if source is None:
            return f"**Error:** Source `{source_id}` not found"

        limit = max(1, min(50, limit))

        # Get bidirectional citations
        citing = await get_citing_sources(UUID(source_id))
        cited = await get_cited_sources(UUID(source_id))

        # Apply limit
        citing = citing[:limit]
        cited = cited[:limit]

        if output_format == "json":
            return format_citation_network_json(citing, cited, source)
        return format_citation_network(citing, cited, source)

    @mcp.tool()
    async def research_kb_biblio_coupling(
        source_id: str,
        limit: int = 10,
        min_coupling: float = 0.1,
        output_format: Literal["markdown", "json"] = "markdown",
    ) -> str:
        """Find sources similar by bibliographic coupling.

        Bibliographic coupling finds sources that cite many of the same
        references. High coupling indicates topical similarity even if
        the papers don't cite each other directly.

        Coupling strength uses Jaccard similarity:
            coupling = shared_refs / (refs_A + refs_B - shared_refs)

        Args:
            source_id: UUID of the source to find similar sources for
            limit: Maximum similar sources to return (1-50, default 10)
            min_coupling: Minimum coupling strength threshold (0.0-1.0, default 0.1)
            output_format: Response format - "markdown" (default) or "json"

        Returns:
            Markdown-formatted or JSON list of similar sources with:
            - Title, authors, year
            - Coupling strength (percentage)
            - Number of shared references
            - Source IDs for follow-up queries

        Example use cases:
            - Find methodologically similar papers
            - Discover papers from the same research tradition
            - Identify potential related work you may have missed
        """
        source = await get_source_by_id(source_id)

        if source is None:
            return f"**Error:** Source `{source_id}` not found"

        limit = max(1, min(50, limit))
        min_coupling = max(0.0, min(1.0, min_coupling))

        # Get similar sources by bibliographic coupling
        similar = await BiblioStore.get_similar_sources(
            UUID(source_id),
            limit=limit,
        )

        # Filter by min_coupling
        similar = [s for s in similar if s["coupling_strength"] >= min_coupling]

        if output_format == "json":
            return format_biblio_similar_json(similar, source)
        return format_biblio_similar(similar, source)
