"""Search commands for research-kb.

Commands:
    query              Hybrid search across the knowledge base
    audit-assumptions  Get required assumptions for a statistical method
"""

import asyncio
from typing import Optional

import typer

from research_kb_cli._shared import (
    ContextType,
    OutputFormat,
    ScoringMethod,
    run_query,
)
from research_kb_cli.formatters import (
    format_results_agent,
    format_results_json,
    format_results_markdown,
)

app = typer.Typer(help="Search the knowledge base")


@app.command()
def query(
    query_text: str = typer.Argument(..., help="The query to search for"),
    limit: int = typer.Option(5, "--limit", "-l", help="Maximum number of results"),
    format: OutputFormat = typer.Option(
        OutputFormat.markdown,
        "--format",
        "-f",
        help="Output format",
    ),
    context_type: ContextType = typer.Option(
        ContextType.balanced,
        "--context-type",
        "-c",
        help="Context mode for search weighting",
    ),
    source_type: Optional[str] = typer.Option(
        None,
        "--source-type",
        "-s",
        help="Filter by source type (paper, textbook)",
    ),
    no_content: bool = typer.Option(
        False,
        "--no-content",
        help="Hide content snippets in markdown output",
    ),
    use_graph: bool = typer.Option(
        True,
        "--graph/--no-graph",
        "-g/-G",
        help="Enable/disable graph-boosted ranking (default: enabled)",
    ),
    graph_weight: float = typer.Option(
        0.2,
        "--graph-weight",
        help="Graph score weight (0.0-1.0)",
    ),
    use_rerank: bool = typer.Option(
        True,
        "--rerank/--no-rerank",
        "-r/-R",
        help="Enable/disable cross-encoder reranking (default: enabled)",
    ),
    use_expand: bool = typer.Option(
        True,
        "--expand/--no-expand",
        "-e/-E",
        help="Enable/disable query expansion with synonyms and graph (default: enabled)",
    ),
    use_llm_expand: bool = typer.Option(
        False,
        "--llm-expand",
        help="Enable LLM-based query expansion via Ollama (slower, optional)",
    ),
    use_citations: bool = typer.Option(
        True,
        "--citations/--no-citations",
        "-C/-X",
        help="Enable/disable citation authority boosting (default: enabled)",
    ),
    citation_weight: float = typer.Option(
        0.15,
        "--citation-weight",
        help="Citation score weight (0.0-1.0)",
    ),
    domain: Optional[str] = typer.Option(
        None,
        "--domain",
        "-d",
        help="Knowledge domain (causal_inference, time_series, or None for all)",
    ),
    scoring: ScoringMethod = typer.Option(
        ScoringMethod.weighted,
        "--scoring",
        "-S",
        help="Score combination method: 'weighted' (default) or 'rrf' (Reciprocal Rank Fusion)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show query expansion details",
    ),
):
    """Search the research knowledge base with graph-boosted search and reranking.

    Graph-boosted search, query expansion, citation authority, and cross-encoder
    reranking are enabled by default:
    - Full-text search (keyword matching)
    - Vector similarity (semantic matching)
    - Knowledge graph signals (concept relationships)
    - Citation authority (boost highly-cited sources)
    - Query expansion (synonyms + graph neighbors, improves recall)
    - Cross-encoder reranking (Phase 3, improves precision)

    Examples:

        research-kb search query "backdoor criterion"

        research-kb search query "instrumental variables" --graph-weight 0.3

        research-kb search query "cross-fitting" --no-graph

        research-kb search query "IV" --no-citations

        research-kb search query "IV" --scoring rrf
    """
    try:
        results, expanded_query = asyncio.run(
            run_query(
                query_text,
                limit,
                context_type,
                source_type,
                use_graph,
                graph_weight,
                use_citations,
                citation_weight,
                use_rerank,
                use_expand,
                use_llm_expand,
                verbose,
                domain,
                scoring.value,
            )
        )

        # Show expansion details if verbose
        if verbose and expanded_query and expanded_query.expanded_terms:
            typer.echo("Query Expansion:")
            typer.echo(f"  Original: {expanded_query.original}")
            typer.echo(f"  Expanded terms: {', '.join(expanded_query.expanded_terms)}")
            if expanded_query.expansion_sources:
                for source, terms in expanded_query.expansion_sources.items():
                    typer.echo(f"    {source}: {', '.join(terms)}")
            typer.echo()

        # Format output
        if format == OutputFormat.markdown:
            output = format_results_markdown(results, query_text, show_content=not no_content)
        elif format == OutputFormat.json:
            output = format_results_json(results, query_text)
        elif format == OutputFormat.agent:
            output = format_results_agent(results, query_text, context_type.value)
        else:
            output = format_results_markdown(results, query_text)

        typer.echo(output)

    except ConnectionError:
        typer.echo(
            "Error: Embedding server not running. Start with: python -m research_kb_pdf.embed_server",
            err=True,
        )
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="audit-assumptions")
def audit_assumptions(
    method_name: str = typer.Argument(..., help="Method name, abbreviation, or alias"),
    format: OutputFormat = typer.Option(
        OutputFormat.markdown,
        "--format",
        "-f",
        help="Output format (markdown or json)",
    ),
    include_docstring: bool = typer.Option(
        True,
        "--docstring/--no-docstring",
        "-d/-D",
        help="Include ready-to-paste code docstring snippet",
    ),
    use_ollama: bool = typer.Option(
        True,
        "--ollama/--no-ollama",
        "-o/-O",
        help="Use Ollama LLM fallback if graph returns <3 assumptions",
    ),
    filter_domain: bool = typer.Option(
        True,
        "--filter-domain/--no-filter-domain",
        help="Filter assumptions to same domain as method (prevents cross-domain contamination)",
    ),
):
    """Get required assumptions for a statistical/ML method.

    Queries the knowledge graph for METHOD -> REQUIRES/USES -> ASSUMPTION
    relationships. If fewer than 3 assumptions are found, optionally
    uses Ollama LLM to extract additional assumptions.

    Examples:

        research-kb search audit-assumptions "double machine learning"

        research-kb search audit-assumptions "DML" --format json

        research-kb search audit-assumptions "IV" --no-ollama
    """
    from research_kb_storage import DatabaseConfig, MethodAssumptionAuditor, get_connection_pool

    async def run_audit():
        config = DatabaseConfig()
        await get_connection_pool(config)
        return await MethodAssumptionAuditor.audit_assumptions(
            method_name,
            use_ollama_fallback=use_ollama,
            filter_by_domain=filter_domain,
        )

    try:
        result = asyncio.run(run_audit())

        if format == OutputFormat.json:
            import json

            output = json.dumps(result.to_dict(), indent=2)
        else:
            # Markdown format
            lines = []
            lines.append(f"## Assumptions for: {result.method}")

            if result.method_aliases:
                lines.append(f"**Aliases**: {', '.join(result.method_aliases)}")

            if result.method_id:
                lines.append(f"**Method ID**: `{result.method_id}`")

            if result.definition:
                lines.append(f"\n**Definition**: {result.definition}")

            lines.append(f"\n**Source**: {result.source}")
            lines.append("")

            if result.source == "not_found":
                lines.append("**Method not found in knowledge base.**")
                lines.append("")
                lines.append("Try:")
                lines.append("- Different spelling or abbreviation")
                lines.append("- `research-kb graph concepts` to search for related methods")
                lines.append("- `research-kb search query` for full-text search")
            elif not result.assumptions:
                lines.append("### No assumptions found")
                lines.append("")
                lines.append(
                    "The knowledge graph doesn't have assumption relationships for this method."
                )
            else:
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
                            "[CRITICAL]"
                            if a.importance == "critical"
                            else "[technical]" if a.importance == "technical" else ""
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

            output = "\n".join(lines)

        typer.echo(output)

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
