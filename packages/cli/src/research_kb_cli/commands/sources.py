"""Source management commands for research-kb.

Commands:
    list               List all ingested sources (filter via --role)
    add-manual         Register a metadata-only source (no PDF)
    set-role           Add or remove metadata.roles markers on a source
    stats              Show knowledge base statistics
    extraction-status  Show extraction pipeline statistics
"""

import asyncio
import hashlib
from typing import Optional
from uuid import UUID

import typer

from research_kb_storage import DatabaseConfig, get_connection_pool

app = typer.Typer(help="Manage and inspect sources")


def _synthetic_file_hash(title: str, year: Optional[int], first_author: str) -> str:
    """Build a deterministic synthetic file_hash for metadata-only sources.

    Same (title, year, first_author) → same hash, so the existing
    ``sources.file_hash UNIQUE`` constraint also acts as a fallback dedup
    if ``find_by_title_year`` misses (e.g. due to title-normalization edge
    cases).
    """
    payload = (
        f"{title.lower().strip()}|{year if year is not None else ''}|"
        f"{first_author.lower().strip()}"
    ).encode("utf-8")
    return f"synthetic:{hashlib.sha256(payload).hexdigest()}"


@app.command(name="list")
def list_sources(
    role: Optional[str] = typer.Option(
        None,
        "--role",
        help="Filter to sources whose metadata.roles contains this string.",
    ),
):
    """List ingested sources in the knowledge base.

    Examples:

        research-kb sources list
        research-kb sources list --role anchor.rl_optimal_control
    """

    async def _list_sources():
        from research_kb_storage import SourceStore

        config = DatabaseConfig()
        await get_connection_pool(config)
        if role is not None:
            return await SourceStore.list_by_role(role, limit=10000)
        return await SourceStore.list_all(limit=10000)

    try:
        sources = asyncio.run(_list_sources())

        if not sources:
            if role is not None:
                typer.echo(f"No sources matching role '{role}'.")
            else:
                typer.echo("No sources in knowledge base.")
            return

        if role is not None:
            typer.echo(f"Found {len(sources)} sources with role '{role}':\n")
        else:
            typer.echo(f"Found {len(sources)} sources:\n")

        for source in sources:
            type_badge = f"[{source.source_type.value}]"
            authors = ", ".join(source.authors[:2])
            if len(source.authors) > 2:
                authors += " et al."
            typer.echo(
                f"  {source.id}  {type_badge:12} "
                f"{source.title[:50]:50} ({authors}, {source.year})"
            )

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="add-manual")
def add_manual(
    title: str = typer.Option(..., "--title", help="Source title."),
    authors: str = typer.Option(
        ...,
        "--authors",
        help="Comma-separated author names (first author is used for dedup).",
    ),
    year: int = typer.Option(..., "--year", help="Publication year."),
    domain_id: str = typer.Option(
        ...,
        "--domain-id",
        help="Knowledge domain (e.g. reinforcement_learning, causal_inference).",
    ),
    source_type: str = typer.Option(
        "textbook",
        "--source-type",
        help="One of: textbook, paper, code_repo, blog.",
    ),
    isbn: Optional[str] = typer.Option(None, "--isbn"),
    doi: Optional[str] = typer.Option(None, "--doi"),
    url: Optional[str] = typer.Option(None, "--url"),
    kind: Optional[str] = typer.Option(
        None,
        "--kind",
        help=(
            "Optional sub-classifier stored as metadata.kind "
            "(e.g. 'thesis' for PhD theses inside source_type=paper)."
        ),
    ),
    role: list[str] = typer.Option(
        [],
        "--role",
        help=(
            "Pipeline-role marker stored in metadata.roles. May be repeated, "
            "e.g. --role anchor.rl_optimal_control --role quality.peer_reviewed."
        ),
    ),
    print_uuid: bool = typer.Option(
        True,
        "--print-uuid/--no-print-uuid",
        help="Print the resulting source UUID on stdout.",
    ),
):
    """Register a metadata-only source (no PDF needed).

    Idempotent: if a source already exists with the same (normalized title,
    year), this command merges any newly-supplied identifiers/roles into
    that existing source and prints ``EXISTING <uuid>`` instead of creating
    a duplicate.

    Examples:

        research-kb sources add-manual \\
            --title "Dynamic Programming" \\
            --authors "Richard Bellman" --year 1957 \\
            --domain-id reinforcement_learning --source-type textbook \\
            --role anchor.rl_optimal_control

        research-kb sources add-manual \\
            --title "Learning from Delayed Rewards" \\
            --authors "Christopher Watkins" --year 1989 \\
            --domain-id reinforcement_learning --source-type paper \\
            --kind thesis --role anchor.rl_optimal_control
    """
    from research_kb_contracts import SourceType
    from research_kb_storage import SourceStore

    author_list = [a.strip() for a in authors.split(",") if a.strip()]
    if not author_list:
        typer.echo("Error: --authors must list at least one author", err=True)
        raise typer.Exit(2)

    try:
        source_type_enum = SourceType(source_type)
    except ValueError:
        typer.echo(
            f"Error: invalid --source-type '{source_type}' "
            f"(allowed: {', '.join(t.value for t in SourceType)})",
            err=True,
        )
        raise typer.Exit(2)

    metadata: dict = {}
    if isbn:
        metadata["isbn"] = isbn
    if doi:
        metadata["doi"] = doi
    if url:
        metadata["url"] = url
    if kind:
        metadata["kind"] = kind

    async def _add_manual():
        config = DatabaseConfig()
        await get_connection_pool(config)

        # 1. Idempotency lookup: (title, year) match → merge into existing.
        existing = await SourceStore.find_by_title_year(title, year)
        if existing is not None:
            # Merge new metadata; add roles.
            if metadata:
                await SourceStore.update_metadata(existing.id, metadata)
            if role:
                await SourceStore.add_roles(existing.id, list(role))
            return ("EXISTING", existing.id)

        # 2. Create fresh metadata-only source.
        source = await SourceStore.create(
            source_type=source_type_enum,
            title=title,
            file_hash=_synthetic_file_hash(title, year, author_list[0]),
            domain_id=domain_id,
            authors=author_list,
            year=year,
            file_path=None,
            metadata=metadata if metadata else None,
            roles=list(role) if role else None,
        )
        return ("CREATED", source.id)

    try:
        kind_marker, source_id = asyncio.run(_add_manual())
        if print_uuid:
            typer.echo(f"{kind_marker} {source_id}")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="set-role")
def set_role(
    source_id: str = typer.Argument(..., help="Source UUID to modify."),
    add: list[str] = typer.Option(
        [],
        "--add",
        help="Role string to add. May be repeated.",
    ),
    remove: list[str] = typer.Option(
        [],
        "--remove",
        help="Role string to remove. May be repeated.",
    ),
):
    """Add or remove metadata.roles markers on an existing source.

    Examples:

        research-kb sources set-role 0123abcd-... --add anchor.rl_optimal_control
        research-kb sources set-role 0123abcd-... --remove anchor.rl_optimal_control
        research-kb sources set-role 0123abcd-... --add quality.peer_reviewed --remove draft
    """
    if not add and not remove:
        typer.echo(
            "Error: must specify at least one --add or --remove flag",
            err=True,
        )
        raise typer.Exit(2)

    try:
        source_uuid = UUID(source_id)
    except ValueError:
        typer.echo(f"Error: '{source_id}' is not a valid UUID", err=True)
        raise typer.Exit(2)

    async def _set_role():
        from research_kb_storage import SourceStore

        config = DatabaseConfig()
        await get_connection_pool(config)
        source = source_uuid  # noqa: F841 — keep variable name local
        if add:
            await SourceStore.add_roles(source_uuid, list(add))
        if remove:
            await SourceStore.remove_roles(source_uuid, list(remove))
        result = await SourceStore.get_by_id(source_uuid)
        return result

    try:
        result = asyncio.run(_set_role())
        if result is None:
            typer.echo(f"Error: source not found: {source_id}", err=True)
            raise typer.Exit(1)
        current_roles = result.metadata.get("roles", [])
        typer.echo(f"Updated source {source_uuid}: roles = {current_roles}")
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
