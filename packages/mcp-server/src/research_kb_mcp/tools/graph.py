"""Graph tools for MCP server.

Exposes knowledge graph exploration functionality.
"""

from __future__ import annotations

from fastmcp import FastMCP

from research_kb_api.service import (
    get_graph_neighborhood,
    get_graph_path,
)
from research_kb_mcp.formatters import (
    format_graph_neighborhood,
    format_graph_path,
)


def register_graph_tools(mcp: FastMCP) -> None:
    """Register graph tools with the MCP server."""

    @mcp.tool()
    async def research_kb_graph_neighborhood(
        concept_name: str,
        hops: int = 2,
        limit: int = 50,
    ) -> str:
        """Explore the neighborhood of a concept in the knowledge graph.

        Find concepts related to a given concept within N hops. Useful for
        understanding context, prerequisites, and related methods.

        Args:
            concept_name: Name of the concept to explore (fuzzy matched)
            hops: Number of relationship hops (1-3, default 2)
            limit: Maximum connected concepts to return (1-100, default 50)

        Returns:
            Markdown-formatted neighborhood with:
            - Center concept details
            - Connected concepts (names and types)
            - Relationship type distribution

        Example:
            Exploring "double machine learning" with 2 hops might reveal:
            - Direct: Neyman orthogonality, cross-fitting, nuisance estimation
            - 2-hop: propensity score, sample splitting, regularization
        """
        hops = max(1, min(3, hops))
        limit = max(1, min(100, limit))

        neighborhood = await get_graph_neighborhood(
            concept_name=concept_name,
            hops=hops,
            limit=limit,
        )
        return format_graph_neighborhood(neighborhood)

    @mcp.tool()
    async def research_kb_graph_path(
        concept_a: str,
        concept_b: str,
    ) -> str:
        """Find the shortest path between two concepts.

        Discover how two concepts are related through the knowledge graph.
        Useful for understanding connections between methods, assumptions,
        and problems.

        Args:
            concept_a: Name of the first concept (fuzzy matched)
            concept_b: Name of the second concept (fuzzy matched)

        Returns:
            Markdown-formatted path with:
            - Start and end concepts
            - Path length (number of hops)
            - Intermediate concepts along the path

        Example:
            Path from "regression discontinuity" to "instrumental variables"
            might show: RD → local average treatment effect → IV
        """
        path = await get_graph_path(
            concept_a=concept_a,
            concept_b=concept_b,
        )
        return format_graph_path(path)
