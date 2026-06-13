"""Concept tools for MCP server — RETIRED (RS4, 2026-06).

research-kb's chunk-level concept layer was retired in slice RS4 (decision R3,
ADR-0001). The Postgres concept tables + data are PRESERVED (dormant) per the
ADR falsifier, but no longer exposed; the live concept layer is synthesis-kb
(claim-level, eval-gated). These tools remain registered as fail-loud tombstones
that redirect callers. See docs/decisions/0001.
"""

from __future__ import annotations

from typing import Literal, Optional

from fastmcp import FastMCP

_RETIRED = (
    "⚠ RETIRED (RS4) — research-kb's chunk-level concept layer is retired "
    "(decision R3, ADR-0001). The concept layer is now **synthesis-kb** "
    "(claim-level, eval-gated). For primary-literature retrieval use "
    "`research_kb_search`; for concepts use synthesis-kb."
)


def register_concept_tools(mcp: FastMCP) -> None:
    """Register concept tools (retirement stubs) with the MCP server."""

    @mcp.tool()
    async def research_kb_list_concepts(
        query: Optional[str] = None,
        limit: int = 50,
        concept_type: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> str:
        """⚠ RETIRED (RS4). Concept listing is retired — concept layer is synthesis-kb (ADR-0001)."""
        return _RETIRED

    @mcp.tool()
    async def research_kb_get_concept(
        concept_id: str,
        include_relationships: bool = True,
        output_format: Literal["markdown", "json"] = "markdown",
    ) -> str:
        """⚠ RETIRED (RS4). Concept detail is retired — concept layer is synthesis-kb (ADR-0001)."""
        return _RETIRED

    @mcp.tool()
    async def research_kb_chunk_concepts(chunk_id: str) -> str:
        """⚠ RETIRED (RS4). Chunk-concept links are retired — concept layer is synthesis-kb (ADR-0001)."""
        return _RETIRED

    @mcp.tool()
    async def research_kb_find_similar_concepts(
        concept_id: str,
        limit: int = 10,
        threshold: float = 0.8,
        domain: Optional[str] = None,
    ) -> str:
        """⚠ RETIRED (RS4). Concept similarity is retired — concept layer is synthesis-kb (ADR-0001)."""
        return _RETIRED
