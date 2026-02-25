"""Citation network commands for research-kb.

Commands:
    list       List citations extracted from a source
    cited-by   Find sources that cite a given source
    cites      Find sources that a given source cites
    stats      Corpus-wide citation graph statistics
    similar    Find sources with similar research focus (bibliographic coupling)
"""

import asyncio
from typing import Optional

import typer

from research_kb_storage import (
    DatabaseConfig,
    get_citation_stats,
    get_citing_sources,
    get_cited_sources,
    get_connection_pool,
    get_corpus_citation_summary,
    get_most_cited_sources,
)

app = typer.Typer(help="Explore the citation network")


@app.command(name="list")
def list_citations(
    source_query: str = typer.Argument(..., help="Source title or partial match"),
    source_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by source type (paper, textbook)",
    ),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum citations to show"),
):
    """List citations extracted from a source.

    Shows all citations found in the specified source, matching by title.

    Examples:

        research-kb citations list "Pearl 2009"

        research-kb citations list "DML" --type paper --limit 10
    """

    async def get_source_citations():
        from research_kb_storage import SourceStore

        config = DatabaseConfig()
        await get_connection_pool(config)

        all_sources = await SourceStore.list_all(limit=10000)
        query_lower = source_query.lower()

        matches = []
        for s in all_sources:
            if query_lower in s.title.lower():
                if source_type is None or s.source_type.value == source_type:
                    matches.append(s)

        if not matches:
            return None, []

        source = matches[0]

        pool = await get_connection_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, title, authors, year, venue, doi, arxiv_id, raw_string
                FROM citations
                WHERE source_id = $1
                ORDER BY year DESC NULLS LAST, title
                LIMIT $2
                """,
                source.id,
                limit,
            )

        return source, rows

    try:
        source, citations_rows = asyncio.run(get_source_citations())

        if not source:
            typer.echo(f"No source found matching: {source_query}")
            return

        typer.echo(f"Citations in: {source.title}")
        typer.echo(f"Source type: {source.source_type.value}")
        typer.echo("=" * 60)
        typer.echo()

        if not citations_rows:
            typer.echo("No citations extracted for this source.")
            typer.echo("Run: python scripts/extract_citations.py")
            return

        typer.echo(f"Found {len(citations_rows)} citations:\n")

        for i, row in enumerate(citations_rows, 1):
            title = row["title"] or "(No title)"
            year = row["year"] or "n.d."
            authors = row["authors"] or []
            author_str = ", ".join(authors[:2]) + (" et al." if len(authors) > 2 else "")

            typer.echo(f"[{i}] {title[:60]}{'...' if len(title) > 60 else ''}")
            if author_str:
                typer.echo(f"    {author_str} ({year})")
            if row["doi"]:
                typer.echo(f"    DOI: {row['doi']}")
            if row["arxiv_id"]:
                typer.echo(f"    arXiv: {row['arxiv_id']}")
            typer.echo()

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="cited-by")
def cited_by(
    source_query: str = typer.Argument(..., help="Source title or partial match"),
    source_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter citing sources by type (paper, textbook)",
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results"),
):
    """Find sources that cite a given source.

    Shows corpus sources that reference the specified source,
    with breakdown by paper/textbook type.

    Examples:

        research-kb citations cited-by "Pearl 2009"

        research-kb citations cited-by "instrumental variables" --type paper
    """

    async def find_citing():
        from research_kb_storage import SourceStore
        from research_kb_contracts import SourceType

        config = DatabaseConfig()
        await get_connection_pool(config)

        all_sources = await SourceStore.list_all(limit=10000)
        query_lower = source_query.lower()

        matches = [s for s in all_sources if query_lower in s.title.lower()]

        if not matches:
            return None, [], {}

        source = matches[0]

        type_filter = None
        if source_type:
            type_filter = SourceType(source_type)

        citing = await get_citing_sources(source.id, source_type=type_filter, limit=limit)
        stats = await get_citation_stats(source.id)

        return source, citing, stats

    try:
        source, citing_sources, stats = asyncio.run(find_citing())

        if not source:
            typer.echo(f"No source found matching: {source_query}")
            return

        typer.echo(f"Who cites: {source.title}")
        typer.echo(f"Source type: {source.source_type.value}")
        typer.echo("=" * 60)
        typer.echo()

        typer.echo(f"Citation Authority Score: {stats.get('authority_score', 0):.4f}")
        typer.echo(
            f"Cited by {stats.get('cited_by_papers', 0)} papers, {stats.get('cited_by_textbooks', 0)} textbooks"
        )
        typer.echo()

        if not citing_sources:
            typer.echo("No corpus sources cite this work (or citation graph not built).")
            typer.echo("Run: python scripts/build_citation_graph.py")
            return

        typer.echo(f"Citing sources ({len(citing_sources)}):\n")

        for i, s in enumerate(citing_sources, 1):
            type_badge = f"[{s.source_type.value}]"
            typer.echo(f"{i:2}. {type_badge:12} {s.title[:50]}...")
            if s.authors:
                authors = ", ".join(s.authors[:2])
                typer.echo(f"    {authors} ({s.year})")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="cites")
def cites_command(
    source_query: str = typer.Argument(..., help="Source title or partial match"),
    source_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter cited sources by type (paper, textbook)",
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results"),
):
    """Find sources that a given source cites.

    Shows corpus sources referenced by the specified source.

    Examples:

        research-kb citations cites "DML"

        research-kb citations cites "Pearl 2009" --limit 20
    """

    async def find_cited():
        from research_kb_storage import SourceStore
        from research_kb_contracts import SourceType

        config = DatabaseConfig()
        await get_connection_pool(config)

        all_sources = await SourceStore.list_all(limit=10000)
        query_lower = source_query.lower()

        matches = [s for s in all_sources if query_lower in s.title.lower()]

        if not matches:
            return None, [], {}

        source = matches[0]

        type_filter = None
        if source_type:
            type_filter = SourceType(source_type)

        cited = await get_cited_sources(source.id, source_type=type_filter, limit=limit)
        stats = await get_citation_stats(source.id)

        return source, cited, stats

    try:
        source, cited_sources, stats = asyncio.run(find_cited())

        if not source:
            typer.echo(f"No source found matching: {source_query}")
            return

        typer.echo(f"What does it cite: {source.title}")
        typer.echo(f"Source type: {source.source_type.value}")
        typer.echo("=" * 60)
        typer.echo()

        typer.echo(
            f"Cites {stats.get('cites_papers', 0)} papers, {stats.get('cites_textbooks', 0)} textbooks in corpus"
        )
        typer.echo()

        if not cited_sources:
            typer.echo("No corpus sources found in citations (or citation graph not built).")
            typer.echo("Run: python scripts/build_citation_graph.py")
            return

        typer.echo(f"Cited corpus sources ({len(cited_sources)}):\n")

        for i, s in enumerate(cited_sources, 1):
            type_badge = f"[{s.source_type.value}]"
            authority = s.citation_authority or 0.0 if hasattr(s, "citation_authority") else 0.0
            typer.echo(f"{i:2}. {type_badge:12} {s.title[:50]}...")
            if s.authors:
                authors = ", ".join(s.authors[:2])
                typer.echo(f"    {authors} ({s.year}) | Authority: {authority:.4f}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="stats")
def citation_stats_command():
    """Show corpus-wide citation graph statistics.

    Displays:
    - Total citations extracted
    - Internal vs external citation links
    - Citation type breakdown (paper->paper, textbook->paper, etc.)
    - Most cited sources in the corpus

    Examples:

        research-kb citations stats
    """

    async def get_all_stats():
        config = DatabaseConfig()
        await get_connection_pool(config)

        summary = await get_corpus_citation_summary()
        most_cited = await get_most_cited_sources(limit=10)

        return summary, most_cited

    try:
        summary, most_cited = asyncio.run(get_all_stats())

        typer.echo("Citation Graph Statistics")
        typer.echo("=" * 60)
        typer.echo()

        typer.echo(f"Total citations extracted:     {summary.get('total_citations', 0):,}")
        typer.echo(f"Total citation graph edges:    {summary.get('total_edges', 0):,}")
        typer.echo(f"  Internal (corpus->corpus):   {summary.get('internal_edges', 0):,}")
        typer.echo(f"  External (corpus->external):  {summary.get('external_edges', 0):,}")
        typer.echo()

        typer.echo("Citation type breakdown:")
        typer.echo(f"  Paper -> Paper:       {summary.get('paper_to_paper', 0):,}")
        typer.echo(f"  Paper -> Textbook:    {summary.get('paper_to_textbook', 0):,}")
        typer.echo(f"  Textbook -> Paper:    {summary.get('textbook_to_paper', 0):,}")
        typer.echo(f"  Textbook -> Textbook: {summary.get('textbook_to_textbook', 0):,}")
        typer.echo()

        if most_cited:
            typer.echo("Most cited sources (within corpus):")
            typer.echo("-" * 60)
            for i, source in enumerate(most_cited, 1):
                type_badge = f"[{source['source_type']}]"
                typer.echo(f"{i:2}. {type_badge:12} {source['title'][:45]}...")
                typer.echo(
                    f"    Cited by: {source['cited_by_count']} | Authority: {source['citation_authority']:.4f}"
                )
        else:
            typer.echo("No citation graph data. Run: python scripts/build_citation_graph.py")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="similar")
def biblio_similar_command(
    source_query: str = typer.Argument(..., help="Source title or partial match"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results"),
):
    """Find sources with similar research focus via bibliographic coupling.

    Bibliographic coupling measures similarity based on shared references:
    sources that cite the same works are likely topically related.

    Coupling strength uses Jaccard similarity (0.0-1.0).

    Examples:

        research-kb citations similar "DML"

        research-kb citations similar "causal forest" --limit 5
    """
    from research_kb_storage import BiblioStore, SourceStore

    async def find_similar():
        config = DatabaseConfig()
        await get_connection_pool(config)

        all_sources = await SourceStore.list_all(limit=10000)
        query_lower = source_query.lower()

        matches = [s for s in all_sources if query_lower in s.title.lower()]

        if not matches:
            return None, []

        source = matches[0]
        similar = await BiblioStore.get_similar_sources(source.id, limit=limit)

        return source, similar

    try:
        source, similar_sources = asyncio.run(find_similar())

        if not source:
            typer.echo(f"No source found matching: {source_query}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Bibliographic Coupling for: {source.title[:60]}...")
        typer.echo()

        if not similar_sources:
            typer.echo("No similar sources found. Ensure bibliographic coupling is computed:")
            typer.echo("  python scripts/compute_bibliographic_coupling.py")
            return

        typer.echo(f"Similar sources ({len(similar_sources)}):\n")

        for i, s in enumerate(similar_sources, 1):
            type_badge = f"[{s['source_type']}]"
            coupling = s["coupling_strength"]
            shared = s["shared_references"]
            typer.echo(f"{i:2}. {type_badge:12} {s['title'][:50]}...")
            if s["authors"]:
                authors = ", ".join(s["authors"][:2]) if s["authors"] else "Unknown"
                typer.echo(
                    f"    {authors} ({s['year']}) | Coupling: {coupling:.3f} ({shared} shared refs)"
                )

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
