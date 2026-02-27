"""Assumption auditing tools for MCP server.

Phase 4.1a: North Star feature - Surface method assumptions for Claude Code.

Exposes:
- research_kb_audit_assumptions: Get required assumptions for a method

This tool enables Claude to understand what assumptions underlie methods
when implementing causal inference algorithms, transforming research-kb
from a "filing cabinet" to a "PhD collaborator."
"""

from __future__ import annotations

from typing import Literal, Optional

from fastmcp import FastMCP

from research_kb_storage import MethodAssumptionAuditor
from research_kb_mcp.formatters import format_assumption_audit, format_assumption_audit_json
from research_kb_common import get_logger

logger = get_logger(__name__)


def register_assumption_tools(mcp: FastMCP) -> None:
    """Register assumption auditing tools with the MCP server."""

    @mcp.tool()
    async def research_kb_audit_assumptions(
        method_name: str,
        include_docstring: bool = True,
        use_llm_fallback: bool = True,
        output_format: Literal["markdown", "json"] = "markdown",
        domain: Optional[str] = None,
        scope: Literal["general", "applied"] = "general",
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
        3. **LLM Fallback**: If <3 assumptions found, uses Anthropic (Claude Haiku) to extract more
           (results are cached for future queries)

        Args:
            method_name: Method name, abbreviation, or alias.
                        Examples: "DML", "double machine learning", "IV",
                                  "instrumental variables", "DiD"
            include_docstring: Include code docstring snippet (default True)
            use_llm_fallback: Use Anthropic LLM to extract assumptions if graph
                             returns fewer than 3 results (default True)
            output_format: Response format - "markdown" (default) or "json".
                JSON returns structured data via MethodAssumptions.to_dict().
            domain: Domain context for scoped audit (e.g., "time_series",
                   "econometrics", "causal_inference"). When set, graph queries
                   filter to this domain and LLM prompts include domain context
                   (when scope="applied"). Default None uses method's own domain.
            scope: Audit scope — "general" (default) uses standard assumption
                  extraction prompt. "applied" uses a domain-contextual prompt
                  that focuses on domain-specific assumptions and verification
                  approaches. Requires domain to be set for full effect.

        Returns:
            Markdown-formatted or JSON assumption audit with structured data.

        Example queries:
            - "double machine learning" or "DML"
            - "instrumental variables" or "IV"
            - "difference in differences" or "DiD"
            - "regression discontinuity" or "RDD" (with domain="time_series", scope="applied")
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
                use_llm_fallback=use_llm_fallback,
                llm_backend="anthropic",
                domain=domain,
                scope=scope,
            )
            if output_format == "json":
                return format_assumption_audit_json(result)
            return format_assumption_audit(result, include_docstring)

        except Exception as e:
            logger.error(
                "audit_assumptions_failed",
                method_name=method_name,
                error=str(e),
            )
            return f"**Error**: Failed to audit assumptions for '{method_name}': {e}"
