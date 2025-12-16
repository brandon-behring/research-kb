"""Health and stats tools for MCP server.

Exposes database statistics and health check functionality.
"""

from __future__ import annotations

from fastmcp import FastMCP

from research_kb_api.service import get_stats
from research_kb_mcp.formatters import format_stats, format_health


def register_health_tools(mcp: FastMCP) -> None:
    """Register health tools with the MCP server."""

    @mcp.tool()
    async def research_kb_stats() -> str:
        """Get statistics about the research knowledge base.

        Returns corpus metrics including counts of sources, chunks,
        concepts, and relationships.

        Returns:
            Markdown-formatted table with:
            - Sources: Number of papers and textbooks
            - Chunks: Total text chunks indexed
            - Concepts: Extracted concepts in knowledge graph
            - Relationships: Concept-to-concept relationships
            - Citations: Paper citation links
            - Chunk Concepts: Chunk-to-concept mappings
        """
        stats = await get_stats()
        return format_stats(stats)

    @mcp.tool()
    async def research_kb_health() -> str:
        """Check the health of the research-kb system.

        Performs basic connectivity and availability checks.

        Returns:
            Health status with:
            - Overall status (Healthy/Unhealthy)
            - Component status details
        """
        try:
            # Basic health check: can we get stats?
            stats = await get_stats()

            details = {
                "database": "connected",
                "sources": f"{stats.get('sources', 0):,} indexed",
                "chunks": f"{stats.get('chunks', 0):,} indexed",
                "concepts": f"{stats.get('concepts', 0):,} extracted",
            }

            return format_health(healthy=True, details=details)

        except Exception as e:
            return format_health(
                healthy=False,
                details={"error": str(e)},
            )
