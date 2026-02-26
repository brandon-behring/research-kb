"""Source management commands for research-kb.

Commands:
    list               List all ingested sources
    stats              Show knowledge base statistics
    extraction-status  Show extraction pipeline statistics
"""

import asyncio

import typer

from research_kb_storage import DatabaseConfig, get_connection_pool

app = typer.Typer(help="Manage and inspect sources")


@app.command(name="list")
def list_sources():
    """List all ingested sources in the knowledge base.

    Examples:

        research-kb sources list
    """

    async def _list_sources():
        from research_kb_storage import SourceStore

        config = DatabaseConfig()
        await get_connection_pool(config)
        return await SourceStore.list_all(limit=10000)

    try:
        sources = asyncio.run(_list_sources())

        if not sources:
            typer.echo("No sources in knowledge base.")
            return

        typer.echo(f"Found {len(sources)} sources:\n")

        for source in sources:
            type_badge = f"[{source.source_type.value}]"
            authors = ", ".join(source.authors[:2])
            if len(source.authors) > 2:
                authors += " et al."
            typer.echo(f"  {type_badge:12} {source.title[:50]:50} ({authors}, {source.year})")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def stats():
    """Show knowledge base statistics.

    Examples:

        research-kb sources stats
    """

    async def get_stats():
        from research_kb_storage.connection import get_connection_pool

        config = DatabaseConfig()
        pool = await get_connection_pool(config)

        async with pool.acquire() as conn:
            source_count = await conn.fetchval("SELECT COUNT(*) FROM sources")
            chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")

            by_type = await conn.fetch("""
                SELECT source_type, COUNT(*) as count
                FROM sources GROUP BY source_type
            """)

        return source_count, chunk_count, by_type

    try:
        source_count, chunk_count, by_type = asyncio.run(get_stats())

        typer.echo("Research KB Statistics")
        typer.echo("=" * 40)
        typer.echo(f"Total sources: {source_count}")
        typer.echo(f"Total chunks:  {chunk_count}")
        typer.echo()
        typer.echo("By source type:")
        for row in by_type:
            typer.echo(f"  {row['source_type']:12} {row['count']:5}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="extraction-status")
def extraction_status():
    """Show extraction pipeline statistics.

    Displays:
    - Total extracted concepts by type
    - Total relationships by type
    - Concept validation status
    - Extraction quality metrics

    Examples:

        research-kb sources extraction-status
    """

    async def get_extraction_stats():
        config = DatabaseConfig()
        pool = await get_connection_pool(config)

        async with pool.acquire() as conn:
            concept_count = await conn.fetchval("SELECT COUNT(*) FROM concepts")

            concepts_by_type = await conn.fetch("""
                SELECT concept_type, COUNT(*) as count
                FROM concepts
                GROUP BY concept_type
                ORDER BY count DESC
            """)

            relationship_count = await conn.fetchval("SELECT COUNT(*) FROM concept_relationships")

            relationships_by_type = await conn.fetch("""
                SELECT relationship_type, COUNT(*) as count
                FROM concept_relationships
                GROUP BY relationship_type
                ORDER BY count DESC
            """)

            validated_count = await conn.fetchval(
                "SELECT COUNT(*) FROM concepts WHERE validated = TRUE"
            )

            avg_confidence = await conn.fetchval("""
                SELECT AVG(confidence_score)
                FROM concepts
                WHERE confidence_score IS NOT NULL
            """)

            confidence_dist = await conn.fetch("""
                SELECT
                    CASE
                        WHEN confidence_score >= 0.9 THEN 'High (>=0.9)'
                        WHEN confidence_score >= 0.7 THEN 'Medium (0.7-0.9)'
                        WHEN confidence_score >= 0.5 THEN 'Low (0.5-0.7)'
                        ELSE 'Very Low (<0.5)'
                    END AS confidence_range,
                    COUNT(*) as count
                FROM concepts
                WHERE confidence_score IS NOT NULL
                GROUP BY confidence_range
                ORDER BY MIN(confidence_score) DESC
            """)

            chunks_with_concepts = await conn.fetchval("""
                SELECT COUNT(DISTINCT chunk_id)
                FROM chunk_concepts
            """)

            total_chunks = await conn.fetchval("SELECT COUNT(*) FROM chunks")

        return {
            "concept_count": concept_count,
            "concepts_by_type": concepts_by_type,
            "relationship_count": relationship_count,
            "relationships_by_type": relationships_by_type,
            "validated_count": validated_count,
            "avg_confidence": avg_confidence,
            "confidence_dist": confidence_dist,
            "chunks_with_concepts": chunks_with_concepts,
            "total_chunks": total_chunks,
        }

    try:
        stats = asyncio.run(get_extraction_stats())

        typer.echo("Extraction Pipeline Status")
        typer.echo("=" * 60)
        typer.echo()

        typer.echo(f"Total concepts extracted: {stats['concept_count']}")
        typer.echo(
            f"Validated concepts:       {stats['validated_count']} ({stats['validated_count'] / max(stats['concept_count'], 1) * 100:.1f}%)"
        )
        typer.echo()

        typer.echo("Concepts by type:")
        for row in stats["concepts_by_type"]:
            typer.echo(f"  {row['concept_type']:15} {row['count']:5}")
        typer.echo()

        typer.echo(f"Total relationships: {stats['relationship_count']}")
        typer.echo()

        typer.echo("Relationships by type:")
        for row in stats["relationships_by_type"]:
            typer.echo(f"  {row['relationship_type']:15} {row['count']:5}")
        typer.echo()

        typer.echo("Extraction Quality:")
        if stats["avg_confidence"]:
            typer.echo(f"  Average confidence: {stats['avg_confidence']:.2f}")
        else:
            typer.echo("  Average confidence: N/A")

        typer.echo()
        typer.echo("  Confidence distribution:")
        for row in stats["confidence_dist"]:
            typer.echo(f"    {row['confidence_range']:20} {row['count']:5}")

        typer.echo()

        coverage_pct = stats["chunks_with_concepts"] / max(stats["total_chunks"], 1) * 100
        typer.echo(
            f"Chunk coverage: {stats['chunks_with_concepts']}/{stats['total_chunks']} ({coverage_pct:.1f}%)"
        )

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
