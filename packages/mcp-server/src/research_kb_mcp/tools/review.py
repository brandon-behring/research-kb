"""Literature review tool for MCP server.

Exposes automated literature review generation to Claude Code.
"""

from __future__ import annotations

from typing import Literal

from fastmcp import FastMCP

from research_kb_storage import generate_literature_review


def register_review_tools(mcp: FastMCP) -> None:
    """Register literature review tools with the MCP server."""

    @mcp.tool()
    async def research_kb_literature_review(
        topic: str,
        style: Literal["educational", "research", "implementation"] = "educational",
        use_llm: bool = True,
        max_concepts: int = 30,
        max_evidence_per_section: int = 8,
        output_format: Literal["markdown", "json"] = "markdown",
    ) -> str:
        """Generate a structured literature review for a topic from the knowledge base.

        Explores the knowledge graph and corpus to produce a multi-section review with:
        - Overview: Core definitions and foundational ideas
        - Methods & Techniques: Algorithms and analytical approaches
        - Assumptions & Conditions: Required assumptions and validity requirements
        - Applications & Extensions: Practical applications and related work

        Each section is backed by evidence from the corpus with source citations.
        LLM synthesis (Anthropic Haiku) generates coherent prose from the evidence.

        Args:
            topic: Topic to review (e.g., "double machine learning", "instrumental variables")
            style: Writing style for synthesis:
                - "educational": Graduate student level, focus on intuition
                - "research": Methodologist level, focus on rigor
                - "implementation": Practitioner level, focus on practical steps
            use_llm: Enable LLM synthesis (requires ANTHROPIC_API_KEY).
                If False, returns structured evidence without synthesis.
            max_concepts: Maximum concepts to explore from knowledge graph (default 30)
            max_evidence_per_section: Evidence chunks per section (default 8)
            output_format: Response format - "markdown" (default) or "json"

        Returns:
            Multi-section literature review with cited evidence.
            Markdown format includes full prose; JSON includes structured data.

        Example topics:
            - "double machine learning"
            - "instrumental variables"
            - "regression discontinuity"
            - "transformer architecture"
            - "causal forests"
        """
        review = await generate_literature_review(
            topic=topic,
            style=style,
            use_llm=use_llm,
            max_concepts=max_concepts,
            max_evidence_per_section=max_evidence_per_section,
        )

        if output_format == "json":
            import json

            return json.dumps(review.to_dict(), indent=2)
        return review.to_markdown()
