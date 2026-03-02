"""Literature review commands for research-kb.

Commands:
    generate    Generate a structured literature review for a topic
"""

import asyncio
from typing import Optional

import typer

from research_kb_cli._shared import OutputFormat

app = typer.Typer(help="Generate literature reviews from the knowledge base")


@app.command()
def generate(
    topic: str = typer.Argument(..., help="Topic to review (e.g., 'double machine learning')"),
    style: str = typer.Option(
        "educational",
        "--style",
        "-s",
        help="Writing style: educational, research, implementation",
    ),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help="Skip LLM synthesis (show evidence only)",
    ),
    max_concepts: int = typer.Option(
        30,
        "--max-concepts",
        help="Maximum concepts to explore from graph",
    ),
    max_evidence: int = typer.Option(
        8,
        "--max-evidence",
        help="Maximum evidence chunks per section",
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.markdown,
        "--format",
        "-f",
        help="Output format: markdown, json",
    ),
):
    """Generate a structured literature review for a topic.

    Explores the knowledge graph and corpus to produce a multi-section
    review covering overview, methods, assumptions, and applications.

    Examples:
        research-kb review generate "double machine learning"
        research-kb review generate "instrumental variables" --style research
        research-kb review generate "causal forests" --no-llm --format json
    """
    asyncio.run(
        _run_review(
            topic=topic,
            style=style,
            use_llm=not no_llm,
            max_concepts=max_concepts,
            max_evidence=max_evidence,
            output_format=format,
        )
    )


async def _run_review(
    topic: str,
    style: str,
    use_llm: bool,
    max_concepts: int,
    max_evidence: int,
    output_format: OutputFormat,
):
    """Execute literature review generation."""
    import json

    from research_kb_storage import (
        DatabaseConfig,
        generate_literature_review,
        get_connection_pool,
    )

    config = DatabaseConfig()
    await get_connection_pool(config)

    review = await generate_literature_review(
        topic=topic,
        style=style,
        use_llm=use_llm,
        max_concepts=max_concepts,
        max_evidence_per_section=max_evidence,
    )

    if output_format == OutputFormat.json:
        print(json.dumps(review.to_dict(), indent=2))
    else:
        print(review.to_markdown())

    # Print quality metrics summary
    qm = review.quality_metrics
    if qm and output_format != OutputFormat.json:
        print()
        print("--- Quality Metrics ---")
        print(f"  Concepts explored: {qm.get('concepts_explored', 0)}")
        print(f"  Concepts covered: {qm.get('concepts_covered', 0)}")
        print(f"  Sources cited: {qm.get('sources_cited', 0)}")
        print(f"  Evidence chunks: {qm.get('evidence_chunks', 0)}")
        print(
            f"  Sections synthesized: {qm.get('sections_synthesized', 0)}"
            f"/{qm.get('sections_total', 0)}"
        )
