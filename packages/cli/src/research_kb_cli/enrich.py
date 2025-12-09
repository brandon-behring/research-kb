"""Citation enrichment command for enriching citations with Semantic Scholar metadata.

Usage:
    research-kb enrich-citations --dry-run
    research-kb enrich-citations --source "Pearl 2009"
    research-kb enrich-citations --all

This command uses the s2-client package to match extracted citations
to Semantic Scholar papers and enrich them with citation counts,
fields of study, and other metadata.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

import typer

app = typer.Typer(help="Enrich citations with Semantic Scholar metadata")


class OutputFormat(str, Enum):
    """Output format for enrichment results."""

    table = "table"
    json = "json"


def format_enrichment_table(results: dict) -> str:
    """Format enrichment results as a table."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'Status':12} | {'Count':6} | {'Details'}")
    lines.append("-" * 80)

    lines.append(f"{'Matched':12} | {results['matched']:6} | DOI: {results['by_method'].get('doi', 0)}, arXiv: {results['by_method'].get('arxiv', 0)}, Multi-signal: {results['by_method'].get('multi_signal', 0)}")
    lines.append(f"{'Ambiguous':12} | {results['ambiguous']:6} | Below 0.8 threshold, logged for review")
    lines.append(f"{'Unmatched':12} | {results['unmatched']:6} | No DOI/arXiv and title search failed")
    lines.append(f"{'Skipped':12} | {results['skipped']:6} | Already enriched within staleness window")
    lines.append("-" * 80)
    lines.append(f"{'Total':12} | {results['total']:6} |")
    lines.append("=" * 80)

    return "\n".join(lines)


@app.command(name="citations")
def enrich_citations(
    source_query: Optional[str] = typer.Option(
        None,
        "--source",
        "-s",
        help="Enrich citations from specific source (title match)",
    ),
    all_citations: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Enrich all citations in database",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--execute",
        help="Dry run (default) or execute enrichment",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-enrichment (ignore staleness)",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        "-l",
        help="Maximum citations to process",
    ),
    staleness_days: int = typer.Option(
        30,
        "--staleness",
        help="Re-enrich citations older than N days",
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.table,
        "--format",
        help="Output format",
    ),
):
    """Enrich citations with Semantic Scholar metadata.

    Uses multi-signal scoring to match citations to S2 papers:
    - DOI match (confidence 1.0)
    - arXiv ID match (confidence 0.95)
    - Title + Year + Author/Venue scoring (threshold 0.8)

    Examples:

        # Dry run - see what would be enriched
        research-kb enrich citations --dry-run

        # Enrich citations from specific source
        research-kb enrich citations --source "Pearl 2009" --execute

        # Enrich all citations (respects 30-day staleness)
        research-kb enrich citations --all --execute

        # Force re-enrich all (ignore staleness)
        research-kb enrich citations --all --force --execute
    """
    try:
        from s2_client import S2Client, Citation, match_citation, citation_to_enrichment_metadata
    except ImportError:
        typer.echo("Error: s2-client package not installed.", err=True)
        typer.echo("Run: pip install -e packages/s2-client", err=True)
        raise typer.Exit(1)

    if not source_query and not all_citations:
        typer.echo("Error: Specify --source or --all", err=True)
        raise typer.Exit(1)

    async def do_enrichment():
        import sys
        from pathlib import Path

        # Add packages to path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "storage" / "src"))
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "common" / "src"))

        from research_kb_storage import DatabaseConfig, get_connection_pool, SourceStore

        config = DatabaseConfig()
        pool = await get_connection_pool(config)

        # Build query based on filters
        staleness_cutoff = datetime.now(timezone.utc) - timedelta(days=staleness_days)

        query_parts = ["SELECT c.id, c.title, c.authors, c.year, c.venue, c.doi, c.arxiv_id, c.metadata, s.title as source_title FROM citations c JOIN sources s ON c.source_id = s.id WHERE 1=1"]
        params = []
        param_idx = 1

        if source_query:
            query_parts.append(f"AND LOWER(s.title) LIKE ${param_idx}")
            params.append(f"%{source_query.lower()}%")
            param_idx += 1

        if not force:
            # Only citations not recently enriched
            query_parts.append(f"AND (c.metadata->>'s2_enriched_at' IS NULL OR (c.metadata->>'s2_enriched_at')::timestamp < ${param_idx})")
            params.append(staleness_cutoff)
            param_idx += 1

        query_parts.append(f"LIMIT ${param_idx}")
        params.append(limit)

        query = " ".join(query_parts)

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        typer.echo(f"Found {len(rows)} citations to process")

        if not rows:
            return {"matched": 0, "ambiguous": 0, "unmatched": 0, "skipped": 0, "total": 0, "by_method": {}}

        results = {
            "matched": 0,
            "ambiguous": 0,
            "unmatched": 0,
            "skipped": 0,
            "total": len(rows),
            "by_method": {"doi": 0, "arxiv": 0, "title_unique": 0, "multi_signal": 0},
            "enriched_citations": [],
        }

        if dry_run:
            typer.echo("\n[DRY RUN] Would process these citations:")
            for i, row in enumerate(rows[:10], 1):
                title = row["title"] or "(No title)"
                doi_status = "✓" if row["doi"] else "✗"
                arxiv_status = "✓" if row["arxiv_id"] else "✗"
                typer.echo(f"  {i}. {title[:50]}... DOI:{doi_status} arXiv:{arxiv_status}")

            if len(rows) > 10:
                typer.echo(f"  ... and {len(rows) - 10} more")

            # Estimate results based on available IDs
            for row in rows:
                if row["doi"]:
                    results["by_method"]["doi"] += 1
                    results["matched"] += 1
                elif row["arxiv_id"]:
                    results["by_method"]["arxiv"] += 1
                    results["matched"] += 1
                else:
                    # Would need title search
                    results["unmatched"] += 1  # Conservative estimate

            return results

        # Execute enrichment
        typer.echo("\nEnriching citations...")

        async with S2Client() as client:
            for i, row in enumerate(rows):
                citation = Citation(
                    id=str(row["id"]),
                    title=row["title"],
                    authors=row["authors"],
                    year=row["year"],
                    venue=row["venue"],
                    doi=row["doi"],
                    arxiv_id=row["arxiv_id"],
                )

                try:
                    result = await match_citation(citation, client)

                    if result.status == "matched":
                        results["matched"] += 1
                        results["by_method"][result.match_method] = results["by_method"].get(result.match_method, 0) + 1

                        # Update database
                        metadata = citation_to_enrichment_metadata(result)
                        async with pool.acquire() as conn:
                            # Merge with existing metadata
                            await conn.execute(
                                """
                                UPDATE citations
                                SET metadata = metadata || $1::jsonb
                                WHERE id = $2
                                """,
                                metadata,
                                row["id"],
                            )

                        results["enriched_citations"].append({
                            "id": str(row["id"]),
                            "title": row["title"],
                            "method": result.match_method,
                            "confidence": result.confidence,
                        })

                    elif result.status == "ambiguous":
                        results["ambiguous"] += 1
                        # Store ambiguous status for review
                        metadata = citation_to_enrichment_metadata(result)
                        async with pool.acquire() as conn:
                            await conn.execute(
                                """
                                UPDATE citations
                                SET metadata = metadata || $1::jsonb
                                WHERE id = $2
                                """,
                                metadata,
                                row["id"],
                            )

                    else:
                        results["unmatched"] += 1

                except Exception as e:
                    typer.echo(f"  Error enriching citation {row['id']}: {e}", err=True)
                    results["unmatched"] += 1

                # Progress indicator
                if (i + 1) % 10 == 0:
                    typer.echo(f"  Processed {i + 1}/{len(rows)} citations...")

        return results

    try:
        results = asyncio.run(do_enrichment())

        typer.echo()
        if format == OutputFormat.table:
            typer.echo(format_enrichment_table(results))
        else:
            import json
            typer.echo(json.dumps(results, indent=2, default=str))

        # Summary
        if not dry_run and results["matched"] > 0:
            typer.echo(f"\n✓ Enriched {results['matched']} citations with S2 metadata")

        if results["ambiguous"] > 0:
            typer.echo(f"\n⚠ {results['ambiguous']} ambiguous matches logged for manual review")
            typer.echo("  Query: SELECT * FROM citations WHERE metadata->>'s2_match_status' = 'ambiguous'")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="status")
def enrichment_status():
    """Show citation enrichment status.

    Displays:
    - Total citations in database
    - Enriched vs unenriched counts
    - Breakdown by match method
    - Staleness statistics
    """

    async def get_status():
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "storage" / "src"))
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "common" / "src"))

        from research_kb_storage import DatabaseConfig, get_connection_pool

        config = DatabaseConfig()
        pool = await get_connection_pool(config)

        async with pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM citations")

            enriched = await conn.fetchval(
                "SELECT COUNT(*) FROM citations WHERE metadata->>'s2_enriched_at' IS NOT NULL"
            )

            by_method = await conn.fetch(
                """
                SELECT metadata->>'s2_match_method' as method, COUNT(*) as count
                FROM citations
                WHERE metadata->>'s2_match_method' IS NOT NULL
                GROUP BY method
                ORDER BY count DESC
                """
            )

            by_status = await conn.fetch(
                """
                SELECT metadata->>'s2_match_status' as status, COUNT(*) as count
                FROM citations
                WHERE metadata->>'s2_match_status' IS NOT NULL
                GROUP BY status
                ORDER BY count DESC
                """
            )

            # Staleness (>30 days since enrichment)
            stale = await conn.fetchval(
                """
                SELECT COUNT(*) FROM citations
                WHERE metadata->>'s2_enriched_at' IS NOT NULL
                AND (metadata->>'s2_enriched_at')::timestamp < NOW() - INTERVAL '30 days'
                """
            )

        return {
            "total": total,
            "enriched": enriched,
            "unenriched": total - enriched,
            "by_method": {r["method"]: r["count"] for r in by_method},
            "by_status": {r["status"]: r["count"] for r in by_status},
            "stale": stale,
        }

    try:
        status = asyncio.run(get_status())

        typer.echo("Citation Enrichment Status")
        typer.echo("=" * 60)
        typer.echo()

        typer.echo(f"Total citations:   {status['total']:,}")
        typer.echo(f"Enriched:          {status['enriched']:,} ({status['enriched'] / max(status['total'], 1) * 100:.1f}%)")
        typer.echo(f"Unenriched:        {status['unenriched']:,}")
        typer.echo(f"Stale (>30 days):  {status['stale']:,}")
        typer.echo()

        if status["by_method"]:
            typer.echo("By match method:")
            for method, count in status["by_method"].items():
                typer.echo(f"  {method:15} {count:,}")
            typer.echo()

        if status["by_status"]:
            typer.echo("By match status:")
            for stat, count in status["by_status"].items():
                typer.echo(f"  {stat:15} {count:,}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
