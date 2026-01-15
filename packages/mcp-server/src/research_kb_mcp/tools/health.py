"""Health and stats tools for MCP server.

Exposes database statistics and health check functionality.
"""

from __future__ import annotations

from fastmcp import FastMCP

from research_kb_api.service import get_stats
from research_kb_storage import DomainStore
from research_kb_mcp.formatters import format_stats, format_health, format_domains
from research_kb_common import get_logger

logger = get_logger(__name__)


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

    @mcp.tool()
    async def research_kb_list_domains() -> str:
        """List available knowledge domains and their statistics.

        Returns information about all configured domains in the knowledge base,
        including content counts for each domain.

        Returns:
            Markdown-formatted table with:
            - Domain ID (used in search queries)
            - Domain name
            - Number of sources (papers, textbooks)
            - Number of chunks (text segments)
            - Number of concepts (knowledge graph nodes)

        Example use cases:
            - Discover what domains are available for searching
            - Check content coverage per domain
            - Verify domain-specific ingestion completed
        """
        try:
            domain_stats = await DomainStore.get_all_stats()
            return format_domains(domain_stats)
        except Exception as e:
            return f"❌ **Error listing domains:** {e}"
