"""Literature review tool for MCP server — RETIRED (RS4, 2026-06).

The literature-review generator seeded from the chunk-level concept graph, which
was retired in slice RS4 (decision R3, ADR-0001) — it produced silently-weak
reviews for the 34/38 domains without extracted concepts. Retired as a fail-loud
tombstone; a search-seeded (KG-free) rebuild is a recorded future option. The
concept layer is now synthesis-kb. See docs/decisions/0001.
"""

from __future__ import annotations

from typing import Literal, Optional

from fastmcp import FastMCP

_RETIRED = (
    "⚠ RETIRED (RS4) — research-kb's literature-review generator seeded from the "
    "retired chunk-level concept graph (decision R3, ADR-0001) and produced "
    "silently-weak reviews for domains without extracted concepts. For "
    "primary-literature retrieval use `research_kb_search`; the concept layer is "
    "now **synthesis-kb**. A search-seeded rebuild is a recorded option."
)


def register_review_tools(mcp: FastMCP) -> None:
    """Register literature review tools (retirement stub) with the MCP server."""

    @mcp.tool()
    async def research_kb_literature_review(
        topic: str,
        style: Literal["educational", "research", "implementation"] = "educational",
        use_llm: bool = True,
        max_concepts: int = 30,
        max_evidence_per_section: int = 8,
        domain: Optional[str] = None,
        output_format: Literal["markdown", "json"] = "markdown",
    ) -> str:
        """⚠ RETIRED (RS4). Literature-review generation (seeded from the retired concept graph) is retired — see synthesis-kb (ADR-0001)."""
        return _RETIRED
