"""Assumption auditing tools for MCP server.

Phase 4.1a: North Star feature - Surface method assumptions for Claude Code.

Exposes:
- research_kb_audit_assumptions: Get required assumptions for a method

This tool enables Claude to understand what assumptions underlie methods
when implementing causal inference algorithms, transforming research-kb
from a "filing cabinet" to a "PhD collaborator."
"""

from __future__ import annotations

from fastmcp import FastMCP

from research_kb_storage import MethodAssumptionAuditor, MethodAssumptions
from research_kb_common import get_logger

logger = get_logger(__name__)


def register_assumption_tools(mcp: FastMCP) -> None:
    """Register assumption auditing tools with the MCP server."""

    @mcp.tool()
    async def research_kb_audit_assumptions(
        method_name: str,
        include_docstring: bool = True,
        use_llm_fallback: bool = True,
    ) -> str:
        """Get required assumptions for a statistical/ML method.

        **Primary use case**: When implementing a method like DML, IV, or DiD,
        call this tool to understand what assumptions must hold for valid inference.

        Returns structured data optimized for Claude Code reasoning:
        - Method name and aliases (for code comments)
        - Required assumptions with:
          - Formal mathematical statement (for documentation)
          - Plain English explanation (for comments)
          - Importance level (critical/standard/technical)
          - Verification approaches (what to check)
          - Source citations (for references)
        - Ready-to-paste docstring snippet

        Data Sources:
        1. **Knowledge Graph**: First queries METHOD → REQUIRES/USES → ASSUMPTION
        2. **Cache**: Checks for previously extracted assumptions
        3. **LLM Fallback**: If <3 assumptions found, uses Ollama to extract more
           (results are cached for future queries)

        Args:
            method_name: Method name, abbreviation, or alias.
                        Examples: "DML", "double machine learning", "IV",
                                  "instrumental variables", "DiD"
            include_docstring: Include code docstring snippet (default True)
            use_llm_fallback: Use Ollama LLM to extract assumptions if graph
                             returns fewer than 3 results (default True)

        Returns:
            Markdown-formatted assumption audit with structured data.

        Example queries:
            - "double machine learning" or "DML"
            - "instrumental variables" or "IV"
            - "difference in differences" or "DiD"
            - "regression discontinuity" or "RDD"
            - "propensity score matching"

        Example output structure:
            ## Assumptions for: double machine learning
            **Aliases**: DML, debiased ML

            ### Required Assumptions

            1. **Unconfoundedness** [CRITICAL]
               - Formal: Y(t) ⊥ T | X for all t
               - Plain English: No unmeasured confounders
               - Verify: DAG review, sensitivity analysis
               - Citation: Chernozhukov et al. (2018), Section 2.1

            ### Code Docstring Snippet
            ```
            Assumptions:
                [CRITICAL] - unconfoundedness: No unmeasured confounders
                - overlap: Treatment probability bounded away from 0,1
            ```
        """
        try:
            result = await MethodAssumptionAuditor.audit_assumptions(
                method_name,
                use_ollama_fallback=use_llm_fallback,
            )
            return _format_assumption_audit(result, include_docstring)

        except Exception as e:
            logger.error(
                "audit_assumptions_failed",
                method_name=method_name,
                error=str(e),
            )
            return f"**Error**: Failed to audit assumptions for '{method_name}': {e}"


def _format_assumption_audit(result: MethodAssumptions, include_docstring: bool) -> str:
    """Format MethodAssumptions as Claude-friendly markdown.

    Designed for MCP tool output - structured for LLM reasoning.
    """
    lines = []

    # Header
    lines.append(f"## Assumptions for: {result.method}")

    if result.method_aliases:
        aliases_str = ", ".join(result.method_aliases)
        lines.append(f"**Aliases**: {aliases_str}")

    if result.method_id:
        lines.append(f"**Method ID**: `{result.method_id}`")

    if result.definition:
        lines.append(f"\n**Definition**: {result.definition}")

    lines.append(f"\n**Source**: {result.source}")
    lines.append("")

    # Handle not found case
    if result.source == "not_found":
        lines.append("**Method not found in knowledge base.**")
        lines.append("")
        lines.append("Try:")
        lines.append("- Different spelling or abbreviation")
        lines.append("- `research_kb_list_concepts` to search for related methods")
        lines.append("- `research_kb_search` for full-text search")
        return "\n".join(lines)

    # Assumptions section
    if not result.assumptions:
        lines.append("### No assumptions found")
        lines.append("")
        lines.append("The knowledge graph doesn't have assumption relationships for this method.")
        lines.append("This may indicate:")
        lines.append("- Concept extraction hasn't covered this method yet")
        lines.append("- Method is a general technique without specific identifying assumptions")
        return "\n".join(lines)

    lines.append(f"### Required Assumptions ({len(result.assumptions)} found)")
    lines.append("")

    # Group by importance
    critical = [a for a in result.assumptions if a.importance == "critical"]
    standard = [a for a in result.assumptions if a.importance == "standard"]
    technical = [a for a in result.assumptions if a.importance == "technical"]

    for group, label in [
        (critical, "Critical (identification fails if violated)"),
        (standard, "Standard"),
        (technical, "Technical"),
    ]:
        if not group:
            continue

        lines.append(f"#### {label}")
        lines.append("")

        for i, a in enumerate(group, 1):
            importance_badge = (
                "[CRITICAL]" if a.importance == "critical"
                else "[technical]" if a.importance == "technical"
                else ""
            )

            lines.append(f"**{i}. {a.name}** {importance_badge}")

            if a.formal_statement:
                lines.append(f"   - **Formal**: `{a.formal_statement}`")

            if a.plain_english:
                lines.append(f"   - **Plain English**: {a.plain_english}")

            if a.violation_consequence:
                lines.append(f"   - **If violated**: {a.violation_consequence}")

            if a.verification_approaches:
                approaches = ", ".join(a.verification_approaches)
                lines.append(f"   - **Verify**: {approaches}")

            if a.source_citation:
                lines.append(f"   - **Citation**: {a.source_citation}")

            if a.concept_id:
                lines.append(f"   - **Concept ID**: `{a.concept_id}`")

            if a.relationship_type:
                lines.append(f"   - **Relationship**: {a.relationship_type}")

            lines.append("")

    # Docstring snippet
    if include_docstring and result.code_docstring_snippet:
        lines.append("### Code Docstring Snippet")
        lines.append("")
        lines.append("```python")
        lines.append(result.code_docstring_snippet)
        lines.append("```")
        lines.append("")
        lines.append("*Paste this into your implementation's docstring.*")

    # Structured data for programmatic access
    lines.append("")
    lines.append("---")
    lines.append("*Use `method_id` with `research_kb_get_concept` for full details.*")

    return "\n".join(lines)
