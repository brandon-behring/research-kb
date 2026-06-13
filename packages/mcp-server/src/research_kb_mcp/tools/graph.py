"""Graph tools for MCP server — RETIRED (RS4, 2026-06).

research-kb's chunk-level knowledge graph was retired in slice RS4 (decision R3,
ADR-0001): the concept layer is now synthesis-kb (claim-level, eval-gated). The
graph backend (KuzuDB + the graph-score path) was removed; these tools remain
registered as fail-loud tombstones that redirect callers rather than returning
silently-empty graph results. See docs/decisions/0001.
"""

from __future__ import annotations

from typing import Literal, Optional

from fastmcp import FastMCP

_RETIRED = (
    "⚠ RETIRED (RS4) — research-kb's chunk-level knowledge graph is retired "
    "(decision R3, ADR-0001). The concept layer is now **synthesis-kb** "
    "(claim-level, eval-gated). For primary-literature retrieval and citation "
    "structure use `research_kb_search` / `research_kb_citation_network`; for "
    "concepts and cross-domain bridges use synthesis-kb."
)


def register_graph_tools(mcp: FastMCP) -> None:
    """Register graph tools (retirement stubs) with the MCP server."""

    @mcp.tool()
    async def research_kb_graph_neighborhood(
        concept_name: str,
        hops: int = 2,
        limit: int = 50,
        output_format: Literal["markdown", "json"] = "markdown",
    ) -> str:
        """⚠ RETIRED (RS4). Concept-graph neighborhood is retired — concept layer is synthesis-kb (ADR-0001)."""
        return _RETIRED

    @mcp.tool()
    async def research_kb_graph_path(
        concept_a: str,
        concept_b: str,
        include_definitions: bool = True,
        include_synthesis: bool = True,
        synthesis_style: str = "educational",
    ) -> str:
        """⚠ RETIRED (RS4). Concept-graph path is retired — concept layer is synthesis-kb (ADR-0001)."""
        return _RETIRED

    @mcp.tool()
    async def research_kb_cross_domain_concepts(
        source_domain: str,
        target_domain: str,
        concept_id: Optional[str] = None,
        concept_name: Optional[str] = None,
        similarity_threshold: float = 0.85,
        limit: int = 10,
        output_format: Literal["markdown", "json"] = "markdown",
    ) -> str:
        """⚠ RETIRED (RS4). Cross-domain concept bridging is retired — concept layer is synthesis-kb (ADR-0001)."""
        return _RETIRED

    @mcp.tool()
    async def research_kb_explain_connection(
        concept_a: str,
        concept_b: str,
        style: Literal["educational", "research", "implementation"] = "educational",
        max_evidence_per_step: int = 2,
        use_llm: bool = True,
        output_format: Literal["markdown", "json"] = "markdown",
    ) -> str:
        """⚠ RETIRED (RS4). Concept-connection explanation is retired — concept layer is synthesis-kb (ADR-0001)."""
        return _RETIRED
