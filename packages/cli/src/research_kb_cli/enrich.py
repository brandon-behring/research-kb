"""Citation enrichment command for enriching citations with Semantic Scholar metadata.

Usage:
    research-kb enrich-citations --dry-run
    research-kb enrich-citations --source "Pearl 2009"
    research-kb enrich-citations --all
    research-kb enrich-citations --all --execute --slow --background
    research-kb enrich-citations --resume

This command uses the s2-client package to match extracted citations
to Semantic Scholar papers and enrich them with citation counts,
fields of study, and other metadata.

Checkpoint/Resume:
    Long-running jobs (~10 hours at 0.2 RPS) support checkpointing.
    Use --resume to continue from the last checkpoint.
    Press Ctrl+C to gracefully save progress and exit.
"""

import asyncio
from datetime import datetime, timedelta
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

    lines.append(
        f"{'Matched':12} | {results['matched']:6} | DOI: {results['by_method'].get('doi', 0)}, arXiv: {results['by_method'].get('arxiv', 0)}, Multi-signal: {results['by_method'].get('multi_signal', 0)}"
    )
    lines.append(
        f"{'Ambiguous':12} | {results['ambiguous']:6} | Below 0.8 threshold, logged for review"
    )
    lines.append(
        f"{'Unmatched':12} | {results['unmatched']:6} | No DOI/arXiv and title search failed"
    )
    lines.append(
        f"{'Skipped':12} | {results['skipped']:6} | Already enriched within staleness window"
    )
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
    slow: bool = typer.Option(
        False,
        "--slow",
        help="Use slow rate (0.2 RPS) to avoid rate limits without API key",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from last checkpoint (continue interrupted job)",
    ),
    checkpoint_interval: int = typer.Option(
        100,
        "--checkpoint-interval",
        help="Save checkpoint every N citations (default: 100)",
    ),
    job_id: Optional[str] = typer.Option(
        None,
        "--job-id",
        help="Specific job ID to resume (default: most recent)",
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

        # Slow mode (no API key) - ~10 hours for 7K citations
        research-kb enrich citations --all --execute --slow

        # Resume from checkpoint (after Ctrl+C or crash)
        research-kb enrich citations --resume

        # Resume specific job
        research-kb enrich citations --resume --job-id 20251226_143000
    """
    try:
        from s2_client import (
            S2Client,
            Citation,
            match_citation,
            citation_to_enrichment_metadata,
            EnrichmentCheckpoint,
            GracefulShutdown,
        )
    except ImportError:
        typer.echo("Error: s2-client package not installed.", err=True)
        typer.echo("Run: pip install -e packages/s2-client", err=True)
        raise typer.Exit(1)

    # Handle resume mode
    if resume:
        checkpoint = EnrichmentCheckpoint.load(job_id)
        if checkpoint is None:
            typer.echo("Error: No checkpoint found to resume from.", err=True)
            typer.echo(
                "Start a new enrichment job with: research-kb enrich citations --all --execute",
                err=True,
            )
            raise typer.Exit(1)
        typer.echo(
            f"Resuming job {checkpoint.job_id}: {checkpoint.processed_count} already processed"
        )
        typer.echo(f"Progress: {checkpoint.format_progress()}")
        all_citations = True  # Resume implies --all
        dry_run = False  # Resume implies --execute
    else:
        checkpoint = None

    if not source_query and not all_citations:
        typer.echo("Error: Specify --source or --all", err=True)
        raise typer.Exit(1)

    async def do_enrichment():
        import json as json_module  # Import locally to avoid closure issues
        import sys
        from pathlib import Path

        # Add packages to path
        sys.path.insert(
            0,
            str(Path(__file__).parent.parent.parent.parent.parent / "storage" / "src"),
        )
        sys.path.insert(
            0, str(Path(__file__).parent.parent.parent.parent.parent / "common" / "src")
        )

        from research_kb_storage import DatabaseConfig, get_connection_pool

        config = DatabaseConfig()
        pool = await get_connection_pool(config)

        # Build query based on filters
        # Use naive datetime since database stores naive timestamps
        staleness_cutoff = datetime.utcnow() - timedelta(days=staleness_days)

        query_parts = [
            "SELECT c.id, c.title, c.authors, c.year, c.venue, c.doi, c.arxiv_id, c.metadata, s.title as source_title FROM citations c JOIN sources s ON c.source_id = s.id WHERE 1=1"
        ]
        params = []
        param_idx = 1

        if source_query:
            query_parts.append(f"AND LOWER(s.title) LIKE ${param_idx}")
            params.append(f"%{source_query.lower()}%")
            param_idx += 1

        if not force:
            # Only citations not recently enriched
            query_parts.append(
                f"AND (c.metadata->>'s2_enriched_at' IS NULL OR (c.metadata->>'s2_enriched_at')::timestamp < ${param_idx})"
            )
            params.append(staleness_cutoff)
            param_idx += 1

        # Apply limit: --all means no limit, otherwise use --limit value
        if not all_citations:
            query_parts.append(f"LIMIT ${param_idx}")
            params.append(limit)

        query = " ".join(query_parts)

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        typer.echo(f"Found {len(rows)} citations to process")

        if not rows:
            return {
                "matched": 0,
                "ambiguous": 0,
                "unmatched": 0,
                "skipped": 0,
                "total": 0,
                "by_method": {},
            }

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

        # Execute enrichment with checkpoint support
        rps = 0.2 if slow else 10.0  # Slow mode: 1 request per 5 seconds

        # Initialize or resume checkpoint
        nonlocal checkpoint
        if checkpoint is None:
            checkpoint = EnrichmentCheckpoint(
                job_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
                total_citations=len(rows),
                rps=rps,
                checkpoint_interval=checkpoint_interval,
            )
            typer.echo(f"\nStarting new enrichment job: {checkpoint.job_id}")
        else:
            checkpoint.total_citations = len(rows)

        # Filter out already-processed citations (from checkpoint)
        processed_ids = checkpoint.processed_ids
        rows_to_process = [r for r in rows if r["id"] not in processed_ids]

        if len(rows_to_process) < len(rows):
            typer.echo(f"  Skipping {len(rows) - len(rows_to_process)} already-processed citations")

        if not rows_to_process:
            typer.echo("All citations already processed!")
            return results

        if slow:
            eta_hours = len(rows_to_process) * 5 / 3600
            typer.echo(f"\nEnriching {len(rows_to_process)} citations in SLOW mode (0.2 RPS)...")
            typer.echo(f"  Estimated time: {eta_hours:.1f} hours")
            typer.echo("  Press Ctrl+C to save checkpoint and exit")
        else:
            typer.echo(f"\nEnriching {len(rows_to_process)} citations...")

        async with S2Client(requests_per_second=rps) as client:
            with GracefulShutdown(checkpoint) as shutdown:
                for i, row in enumerate(rows_to_process):
                    # Check for shutdown request
                    if shutdown.shutdown_requested:
                        typer.echo("\nShutdown requested. Exiting gracefully...")
                        break

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
                            checkpoint.matched += 1
                            results["by_method"][result.match_method] = (
                                results["by_method"].get(result.match_method, 0) + 1
                            )

                            # Update database
                            metadata = citation_to_enrichment_metadata(result)
                            async with pool.acquire() as conn:
                                await conn.execute(
                                    """
                                    UPDATE citations
                                    SET metadata = metadata || $1::jsonb
                                    WHERE id = $2
                                    """,
                                    json_module.dumps(metadata),
                                    row["id"],
                                )

                            results["enriched_citations"].append(
                                {
                                    "id": str(row["id"]),
                                    "title": row["title"],
                                    "method": result.match_method,
                                    "confidence": result.confidence,
                                }
                            )

                        elif result.status == "ambiguous":
                            results["ambiguous"] += 1
                            checkpoint.ambiguous += 1
                            metadata = citation_to_enrichment_metadata(result)
                            async with pool.acquire() as conn:
                                await conn.execute(
                                    """
                                    UPDATE citations
                                    SET metadata = metadata || $1::jsonb
                                    WHERE id = $2
                                    """,
                                    json_module.dumps(metadata),
                                    row["id"],
                                )

                        else:
                            results["unmatched"] += 1
                            checkpoint.unmatched += 1

                    except Exception as e:
                        typer.echo(f"  Error enriching citation {row['id']}: {e}", err=True)
                        results["unmatched"] += 1
                        checkpoint.errors += 1

                    # Mark as processed
                    checkpoint.processed_ids.add(row["id"])

                    # Periodic checkpoint save and progress
                    if (i + 1) % checkpoint_interval == 0:
                        checkpoint.save()
                        typer.echo(
                            f"  {checkpoint.format_progress()} | {checkpoint.format_stats()}"
                        )

        # Final checkpoint save
        if shutdown.shutdown_requested:
            # Keep checkpoint for resume
            typer.echo("\n✓ Checkpoint saved. Resume with: research-kb enrich citations --resume")
        else:
            # Job completed - delete checkpoint
            checkpoint.delete()
            typer.echo("\n✓ Enrichment complete. Checkpoint cleaned up.")

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
            typer.echo(
                "  Query: SELECT * FROM citations WHERE metadata->>'s2_match_status' = 'ambiguous'"
            )

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

        sys.path.insert(
            0,
            str(Path(__file__).parent.parent.parent.parent.parent / "storage" / "src"),
        )
        sys.path.insert(
            0, str(Path(__file__).parent.parent.parent.parent.parent / "common" / "src")
        )

        from research_kb_storage import DatabaseConfig, get_connection_pool

        config = DatabaseConfig()
        pool = await get_connection_pool(config)

        async with pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM citations")

            enriched = await conn.fetchval(
                "SELECT COUNT(*) FROM citations WHERE metadata->>'s2_enriched_at' IS NOT NULL"
            )

            by_method = await conn.fetch("""
                SELECT metadata->>'s2_match_method' as method, COUNT(*) as count
                FROM citations
                WHERE metadata->>'s2_match_method' IS NOT NULL
                GROUP BY method
                ORDER BY count DESC
                """)

            by_status = await conn.fetch("""
                SELECT metadata->>'s2_match_status' as status, COUNT(*) as count
                FROM citations
                WHERE metadata->>'s2_match_status' IS NOT NULL
                GROUP BY status
                ORDER BY count DESC
                """)

            # Staleness (>30 days since enrichment)
            stale = await conn.fetchval("""
                SELECT COUNT(*) FROM citations
                WHERE metadata->>'s2_enriched_at' IS NOT NULL
                AND (metadata->>'s2_enriched_at')::timestamp < NOW() - INTERVAL '30 days'
                """)

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
        typer.echo(
            f"Enriched:          {status['enriched']:,} ({status['enriched'] / max(status['total'], 1) * 100:.1f}%)"
        )
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


@app.command(name="job-status")
def job_status_command(
    job_id: Optional[str] = typer.Option(
        None,
        "--job",
        "-j",
        help="Specific job ID to check (default: most recent)",
    ),
    list_all: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List all available checkpoints",
    ),
):
    """Check status of enrichment jobs (running or completed).

    Shows progress, statistics, and ETA for checkpoint-based enrichment jobs.

    Examples:

        # Check most recent job
        research-kb enrich job-status

        # Check specific job
        research-kb enrich job-status --job 20251226_143000

        # List all checkpoints
        research-kb enrich job-status --list
    """
    try:
        from s2_client import EnrichmentCheckpoint, list_checkpoints
    except ImportError:
        typer.echo("Error: s2-client package not installed.", err=True)
        raise typer.Exit(1)

    if list_all:
        checkpoints = list_checkpoints()
        if not checkpoints:
            typer.echo("No enrichment checkpoints found.")
            return

        typer.echo("Available Enrichment Checkpoints")
        typer.echo("=" * 60)
        for cp in checkpoints:
            processed = cp["processed"]
            total = cp["total"]
            pct = (processed / total * 100) if total > 0 else 0
            stats = cp.get("stats", {})
            typer.echo(f"\nJob: {cp['job_id']}")
            typer.echo(f"  Started: {cp['started_at']}")
            typer.echo(f"  Last saved: {cp['last_saved_at']}")
            typer.echo(f"  Progress: {processed}/{total} ({pct:.1f}%)")
            typer.echo(
                f"  Matched: {stats.get('matched', 0)}, Ambiguous: {stats.get('ambiguous', 0)}, Unmatched: {stats.get('unmatched', 0)}"
            )
        return

    checkpoint = EnrichmentCheckpoint.load(job_id)
    if checkpoint is None:
        typer.echo("No enrichment checkpoint found.", err=True)
        if job_id:
            typer.echo(f"Job ID '{job_id}' does not exist.", err=True)
        typer.echo(
            "Start a new job with: research-kb enrich citations --all --execute",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo("Enrichment Job Status")
    typer.echo("=" * 60)
    typer.echo(f"\nJob ID:        {checkpoint.job_id}")
    typer.echo(f"Started:       {checkpoint.started_at.isoformat()}")
    if checkpoint.last_saved_at:
        typer.echo(f"Last saved:    {checkpoint.last_saved_at.isoformat()}")
    typer.echo(f"\nProgress:      {checkpoint.format_progress()}")
    typer.echo(f"Statistics:    {checkpoint.format_stats()}")
    typer.echo(f"\nRate:          {checkpoint.rps} requests/second")
    typer.echo(f"Checkpoint:    Every {checkpoint.checkpoint_interval} citations")

    if checkpoint.remaining > 0:
        typer.echo("\nTo resume:     research-kb enrich citations --resume")
    else:
        typer.echo("\n✓ Job appears complete. Delete checkpoint with new job.")
