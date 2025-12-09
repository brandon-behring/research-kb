"""Discovery command for finding new papers via Semantic Scholar.

Usage:
    research-kb discover "double machine learning" --year-from 2020 --min-citations 50

This command leverages the s2-client package to search for papers by topic,
filter by criteria, and optionally acquire open-access PDFs.
"""

import asyncio
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

# Import will be lazy to avoid import errors when s2-client not installed

app = typer.Typer(help="Discover papers via Semantic Scholar")


class OutputFormat(str, Enum):
    """Output format for discovery results."""

    table = "table"
    json = "json"
    markdown = "markdown"


def format_paper_table(papers, show_abstract: bool = False) -> str:
    """Format papers as a table."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'#':3} | {'Year':4} | {'Citations':9} | {'OA':3} | Title")
    lines.append("-" * 80)

    for i, paper in enumerate(papers, 1):
        year = paper.year or "n.d."
        cites = paper.citation_count or 0
        oa = "Yes" if paper.is_open_access else "No"
        title = (paper.title or "No title")[:55]
        if len(paper.title or "") > 55:
            title += "..."

        lines.append(f"{i:3} | {year:4} | {cites:9,} | {oa:3} | {title}")

        if show_abstract and paper.abstract:
            # Truncate abstract
            abstract = paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract
            lines.append(f"    Abstract: {abstract}")

    lines.append("=" * 80)
    return "\n".join(lines)


def format_paper_markdown(papers) -> str:
    """Format papers as markdown."""
    lines = []
    lines.append("# Discovery Results\n")

    for i, paper in enumerate(papers, 1):
        title = paper.title or "No title"
        year = paper.year or "n.d."
        cites = paper.citation_count or 0
        oa = "Open Access" if paper.is_open_access else "Paywalled"

        # Authors
        if paper.authors:
            author_names = [a.name for a in paper.authors[:3] if a.name]
            authors = ", ".join(author_names)
            if len(paper.authors) > 3:
                authors += " et al."
        else:
            authors = "Unknown"

        lines.append(f"## {i}. {title}")
        lines.append(f"**{authors}** ({year}) | {cites:,} citations | {oa}")

        # Identifiers
        ids = []
        if paper.doi:
            ids.append(f"DOI: {paper.doi}")
        if paper.arxiv_id:
            ids.append(f"arXiv: {paper.arxiv_id}")
        if ids:
            lines.append(f"*{' | '.join(ids)}*")

        # Abstract
        if paper.abstract:
            abstract = paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract
            lines.append(f"\n> {abstract}")

        lines.append("")

    return "\n".join(lines)


def format_paper_json(papers) -> str:
    """Format papers as JSON."""
    import json

    data = []
    for paper in papers:
        data.append({
            "paper_id": paper.paper_id,
            "title": paper.title,
            "year": paper.year,
            "authors": [a.name for a in (paper.authors or [])],
            "citation_count": paper.citation_count,
            "influential_citation_count": paper.influential_citation_count,
            "is_open_access": paper.is_open_access,
            "doi": paper.doi,
            "arxiv_id": paper.arxiv_id,
            "open_access_url": paper.open_access_pdf.url if paper.open_access_pdf else None,
            "fields_of_study": [f.get("category") for f in (paper.s2_fields_of_study or [])],
        })

    return json.dumps(data, indent=2)


@app.command(name="search")
def search(
    query: str = typer.Argument(..., help="Search query (e.g., 'double machine learning')"),
    year_from: Optional[int] = typer.Option(
        None,
        "--year-from",
        "-y",
        help="Minimum publication year",
    ),
    year_to: Optional[int] = typer.Option(
        None,
        "--year-to",
        help="Maximum publication year",
    ),
    min_citations: Optional[int] = typer.Option(
        None,
        "--min-citations",
        "-c",
        help="Minimum citation count",
    ),
    open_access_only: bool = typer.Option(
        False,
        "--open-access-only",
        "-o",
        help="Only show open access papers",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum papers to return",
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.table,
        "--format",
        "-f",
        help="Output format",
    ),
    show_abstract: bool = typer.Option(
        False,
        "--abstract",
        "-a",
        help="Show paper abstracts (table format only)",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--acquire",
        help="Dry run (default) or acquire papers",
    ),
):
    """Search for papers on Semantic Scholar.

    Examples:

        research-kb discover search "double machine learning" --year-from 2020

        research-kb discover search "causal forest" --min-citations 100 --open-access-only

        research-kb discover search "RAG retrieval augmented" --limit 50 --format json
    """
    try:
        from s2_client import S2Client, SearchFilters, PaperAcquisition, load_existing_identifiers
    except ImportError:
        typer.echo("Error: s2-client package not installed.", err=True)
        typer.echo("Run: pip install -e packages/s2-client", err=True)
        raise typer.Exit(1)

    async def do_acquire(papers):
        """Acquire open-access papers with deduplication."""
        # Load existing identifiers for dedup
        s2_ids, dois, arxiv_ids, file_hashes = await load_existing_identifiers()

        async with PaperAcquisition(
            existing_s2_ids=s2_ids,
            existing_dois=dois,
            existing_arxiv_ids=arxiv_ids,
            existing_hashes=file_hashes,
        ) as acq:
            result = await acq.acquire_papers(papers)

        # Report results
        summary = result.to_summary_dict()
        typer.echo(f"\nAcquired: {summary['acquired']}")
        typer.echo(f"Skipped (existing): {summary['skipped_existing']}")
        typer.echo(f"Skipped (paywall): {summary['skipped_paywall']}")
        typer.echo(f"Skipped (no URL): {summary['skipped_no_url']}")
        typer.echo(f"Failed: {summary['failed']}")

        if result.acquired:
            typer.echo("\nDownloaded files:")
            for paper, path in result.acquired:
                typer.echo(f"  - {path.name}")

        if result.skipped_paywall:
            typer.echo(f"\nPaywall papers (for shopping list): {len(result.skipped_paywall)}")

    async def do_search():
        async with S2Client() as client:
            # Build year string for S2 API
            year = None
            if year_from and year_to:
                year = f"{year_from}-{year_to}"
            elif year_from:
                year = f"{year_from}-"
            elif year_to:
                year = f"-{year_to}"

            result = await client.search_papers(
                query=query,
                limit=limit,
                year=year,
                open_access_only=open_access_only,
                min_citation_count=min_citations,
            )

            return result

    try:
        typer.echo(f"Searching Semantic Scholar: '{query}'...")
        result = asyncio.run(do_search())

        if not result.data:
            typer.echo("No papers found matching your criteria.")
            return

        typer.echo(f"Found {result.total:,} total results (showing {len(result.data)})\n")

        # Format output
        if format == OutputFormat.table:
            output = format_paper_table(result.data, show_abstract=show_abstract)
        elif format == OutputFormat.markdown:
            output = format_paper_markdown(result.data)
        else:
            output = format_paper_json(result.data)

        typer.echo(output)

        # Summary stats
        open_access_count = sum(1 for p in result.data if p.is_open_access)
        typer.echo(f"\nOpen access: {open_access_count}/{len(result.data)}")

        if not dry_run:
            # Acquire open-access papers
            typer.echo("\n" + "=" * 60)
            typer.echo("ACQUIRING PAPERS")
            typer.echo("=" * 60)

            asyncio.run(do_acquire(result.data))

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="topics")
def discover_topics(
    year_from: int = typer.Option(
        2020,
        "--year-from",
        "-y",
        help="Minimum publication year",
    ),
    min_citations: int = typer.Option(
        50,
        "--min-citations",
        "-c",
        help="Minimum citation count",
    ),
    limit_per_topic: int = typer.Option(
        10,
        "--limit",
        "-l",
        help="Papers per topic",
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.table,
        "--format",
        "-f",
        help="Output format",
    ),
):
    """Discover papers for all pre-configured research topics.

    Searches across all configured discovery topics (causal ML, RAG, world models, etc.)
    with deduplication across topics.

    Examples:

        research-kb discover topics --year-from 2022 --min-citations 100

        research-kb discover topics --format markdown > new_papers.md
    """
    try:
        from s2_client import S2Client, TopicDiscovery, SearchFilters, DiscoveryTopic
    except ImportError:
        typer.echo("Error: s2-client package not installed.", err=True)
        typer.echo("Run: pip install -e packages/s2-client", err=True)
        raise typer.Exit(1)

    async def do_discovery():
        async with S2Client() as client:
            discovery = TopicDiscovery(client)
            filters = SearchFilters(
                year_from=year_from,
                min_citations=min_citations,
            )

            result = await discovery.discover_all_topics(
                filters=filters,
                limit_per_topic=limit_per_topic,
            )

            return result

    try:
        typer.echo("Discovering papers across all research topics...")
        typer.echo(f"Filters: year >= {year_from}, citations >= {min_citations}")
        typer.echo("")

        result = asyncio.run(do_discovery())

        summary = result.to_summary_dict()
        typer.echo(f"Topics searched: {summary['queries_run']}")
        typer.echo(f"Total results: {summary['total_found']:,}")
        typer.echo(f"After filters: {summary['total_after_filters']:,}")
        typer.echo(f"Duplicates removed: {summary['duplicates_removed']}")
        typer.echo(f"Unique papers: {summary['unique_papers']}")
        typer.echo("")

        if not result.papers:
            typer.echo("No papers found matching criteria.")
            return

        # Format output
        if format == OutputFormat.table:
            output = format_paper_table(result.papers)
        elif format == OutputFormat.markdown:
            output = format_paper_markdown(result.papers)
        else:
            output = format_paper_json(result.papers)

        typer.echo(output)

        # Open access summary
        open_access_count = sum(1 for p in result.papers if p.is_open_access)
        typer.echo(f"\nOpen access available: {open_access_count}/{len(result.papers)}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="author")
def discover_author(
    author_id: str = typer.Argument(..., help="Semantic Scholar author ID"),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum papers to return",
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.table,
        "--format",
        "-f",
        help="Output format",
    ),
):
    """Get recent papers by a specific author.

    Examples:

        research-kb discover author 26331346  # Victor Chernozhukov

        research-kb discover author 1688882 --limit 10  # Yann LeCun
    """
    try:
        from s2_client import S2Client
    except ImportError:
        typer.echo("Error: s2-client package not installed.", err=True)
        typer.echo("Run: pip install -e packages/s2-client", err=True)
        raise typer.Exit(1)

    async def get_author_papers():
        async with S2Client() as client:
            # Get author info
            author = await client.get_author(author_id)

            # Get papers
            result = await client.get_author_papers(author_id, limit=limit)

            return author, result

    try:
        typer.echo(f"Fetching papers for author ID: {author_id}...")

        author, result = asyncio.run(get_author_papers())

        typer.echo(f"\nAuthor: {author.name}")
        if author.affiliations:
            typer.echo(f"Affiliations: {', '.join(author.affiliations[:3])}")
        typer.echo(f"Total papers: {author.paper_count}, h-index: {author.h_index}")
        typer.echo(f"Total citations: {author.citation_count:,}")
        typer.echo("")

        if not result.data:
            typer.echo("No papers found.")
            return

        # Format output
        if format == OutputFormat.table:
            output = format_paper_table(result.data)
        elif format == OutputFormat.markdown:
            output = format_paper_markdown(result.data)
        else:
            output = format_paper_json(result.data)

        typer.echo(output)

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
