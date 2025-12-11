"""Research KB CLI - Main entry point.

Provides the `research-kb` command-line interface.

Usage:
    research-kb query "backdoor criterion" --limit 5 --format markdown

Master Plan Reference: Lines 588-590
"""

import asyncio
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

# Add packages to path (for development mode)
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent.parent / "pdf-tools" / "src")
)
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent.parent / "storage" / "src")
)
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent.parent / "contracts" / "src")
)
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent.parent / "common" / "src")
)
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent.parent / "s2-client" / "src")
)

from research_kb_pdf import EmbeddingClient
from research_kb_storage import (
    ConceptStore,
    DatabaseConfig,
    RelationshipStore,
    SearchQuery,
    find_shortest_path,
    get_connection_pool,
    get_neighborhood,
    search_hybrid,
    search_hybrid_v2,
    search_with_rerank,
    search_with_expansion,
    get_citing_sources,
    get_cited_sources,
    get_citation_stats,
    get_corpus_citation_summary,
    get_most_cited_sources,
)

from research_kb_cli.formatters import (
    format_results_agent,
    format_results_json,
    format_results_markdown,
)


class OutputFormat(str, Enum):
    """Output format options."""

    markdown = "markdown"
    json = "json"
    agent = "agent"


class ContextType(str, Enum):
    """Context type for search weighting.

    - building: Favor breadth, good for initial research
    - auditing: Favor precision, good for verification
    - balanced: Default balanced approach
    """

    building = "building"
    auditing = "auditing"
    balanced = "balanced"


# Create the Typer app
app = typer.Typer(
    name="research-kb",
    help="Query the research knowledge base for causal inference references.",
    add_completion=False,
)

# Add discover subcommand (for Semantic Scholar integration)
from research_kb_cli.discover import app as discover_app  # noqa: E402
from research_kb_cli.enrich import app as enrich_app  # noqa: E402

app.add_typer(discover_app, name="discover")
app.add_typer(enrich_app, name="enrich")


def get_context_weights(context_type: ContextType) -> tuple[float, float]:
    """Get FTS/vector weights based on context type.

    Args:
        context_type: The context mode

    Returns:
        Tuple of (fts_weight, vector_weight)
    """
    if context_type == ContextType.building:
        # Favor vector search for semantic breadth
        return 0.2, 0.8
    elif context_type == ContextType.auditing:
        # Favor FTS for precise term matching
        return 0.5, 0.5
    else:  # balanced
        return 0.3, 0.7


async def run_query(
    query_text: str,
    limit: int,
    context_type: ContextType,
    source_filter: Optional[str],
    use_graph: bool = True,
    graph_weight: float = 0.2,
    use_rerank: bool = True,
    use_expand: bool = True,
    use_llm_expand: bool = False,
    verbose: bool = False,
) -> tuple:
    """Execute the search query with graph-boosted search, expansion, and reranking.

    Args:
        query_text: The query string
        limit: Maximum results
        context_type: Context mode for weight tuning
        source_filter: Optional source type filter
        use_graph: Enable graph-boosted search (default: True)
        graph_weight: Graph score weight (default: 0.2)
        use_rerank: Enable cross-encoder reranking (default: True)
        use_expand: Enable query expansion (default: True)
        use_llm_expand: Enable LLM-based expansion (default: False)
        verbose: Show expansion details (default: False)

    Returns:
        Tuple of (SearchResult list, ExpandedQuery or None)
    """
    # Initialize database connection
    config = DatabaseConfig()
    await get_connection_pool(config)

    # Check if concepts exist when graph search requested
    if use_graph:
        concept_count = await ConceptStore.count()
        if concept_count == 0:
            # Gracefully fall back to non-graph search with warning
            import sys

            print(
                "Warning: Graph search requested but no concepts extracted.",
                file=sys.stderr,
            )
            print(
                "Falling back to standard search (FTS + vector only).", file=sys.stderr
            )
            print(
                "To extract concepts: python scripts/extract_concepts.py",
                file=sys.stderr,
            )
            print("", file=sys.stderr)
            use_graph = False

    # Generate embedding for query
    embed_client = EmbeddingClient()
    query_embedding = embed_client.embed(query_text)

    # Get weights based on context type
    fts_weight, vector_weight = get_context_weights(context_type)

    if use_graph:
        # Normalize weights to sum to 1.0
        total = fts_weight + vector_weight + graph_weight
        fts_weight = fts_weight / total
        vector_weight = vector_weight / total
        graph_weight = graph_weight / total

        # Build search query with graph
        search_query = SearchQuery(
            text=query_text,
            embedding=query_embedding,
            fts_weight=fts_weight,
            vector_weight=vector_weight,
            graph_weight=graph_weight,
            use_graph=True,
            max_hops=2,
            limit=limit,
            source_filter=source_filter,
        )
    else:
        # Build standard search query
        search_query = SearchQuery(
            text=query_text,
            embedding=query_embedding,
            fts_weight=fts_weight,
            vector_weight=vector_weight,
            limit=limit,
            source_filter=source_filter,
        )

    # Execute search with expansion if enabled
    expanded_query = None
    if use_expand or use_llm_expand:
        results, expanded_query = await search_with_expansion(
            search_query,
            use_synonyms=use_expand,
            use_graph_expansion=use_expand and use_graph,
            use_llm_expansion=use_llm_expand,
            use_rerank=use_rerank,
            rerank_top_k=limit,
        )
    elif use_rerank:
        results = await search_with_rerank(search_query, rerank_top_k=limit)
    elif use_graph:
        results = await search_hybrid_v2(search_query)
    else:
        results = await search_hybrid(search_query)

    return results, expanded_query


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
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show query expansion details",
    ),
):
    """Search the research knowledge base with graph-boosted search and reranking.

    Graph-boosted search, query expansion, and cross-encoder reranking are enabled by default:
    - Full-text search (keyword matching)
    - Vector similarity (semantic matching)
    - Knowledge graph signals (concept relationships)
    - Query expansion (synonyms + graph neighbors, improves recall)
    - Cross-encoder reranking (Phase 3, improves precision)

    Examples:

        research-kb query "backdoor criterion"

        research-kb query "instrumental variables" --graph-weight 0.3

        research-kb query "cross-fitting" --no-graph  # Fallback to FTS+vector only

        research-kb query "IV" --no-rerank  # Skip cross-encoder reranking

        research-kb query "IV" --expand --verbose  # Show expansion details

        research-kb query "IV" --llm-expand  # Use LLM for semantic expansion
    """
    try:
        # Run async query
        results, expanded_query = asyncio.run(
            run_query(
                query_text,
                limit,
                context_type,
                source_type,
                use_graph,
                graph_weight,
                use_rerank,
                use_expand,
                use_llm_expand,
                verbose,
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
            output = format_results_markdown(
                results, query_text, show_content=not no_content
            )
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


@app.command()
def sources():
    """List all ingested sources in the knowledge base."""

    async def list_sources():
        from research_kb_storage import SourceStore

        config = DatabaseConfig()
        await get_connection_pool(config)
        return await SourceStore.list_all()

    try:
        sources = asyncio.run(list_sources())

        if not sources:
            typer.echo("No sources in knowledge base.")
            return

        typer.echo(f"Found {len(sources)} sources:\n")

        for source in sources:
            type_badge = f"[{source.source_type.value}]"
            authors = ", ".join(source.authors[:2])
            if len(source.authors) > 2:
                authors += " et al."
            typer.echo(
                f"  {type_badge:12} {source.title[:50]:50} ({authors}, {source.year})"
            )

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def stats():
    """Show knowledge base statistics."""

    async def get_stats():
        from research_kb_storage.connection import get_connection_pool

        config = DatabaseConfig()
        pool = await get_connection_pool(config)

        async with pool.acquire() as conn:
            source_count = await conn.fetchval("SELECT COUNT(*) FROM sources")
            chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")

            by_type = await conn.fetch(
                """
                SELECT source_type, COUNT(*) as count
                FROM sources GROUP BY source_type
            """
            )

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


@app.command()
def concepts(
    query: str = typer.Argument(..., help="Concept name or search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of results"),
    show_relationships: bool = typer.Option(
        True, "--relationships/--no-relationships", help="Show related concepts"
    ),
):
    """Search for concepts in the knowledge graph.

    Searches by canonical name, aliases, and fuzzy matching.
    Shows concept definition, type, and related concepts.

    Examples:

        research-kb concepts "instrumental variables"

        research-kb concepts "IV" --limit 5

        research-kb concepts "matching" --no-relationships
    """

    async def search_concepts():
        config = DatabaseConfig()
        await get_connection_pool(config)

        # Try exact match first
        concept = await ConceptStore.get_by_canonical_name(query.lower())

        if concept:
            # Found exact match
            related = []
            if show_relationships:
                related = await RelationshipStore.list_all_for_concept(concept.id)
            return [concept], related

        # Fall back to fuzzy search across all concepts
        all_concepts = await ConceptStore.list_all(limit=1000)

        # Filter by name/alias matching
        matches = []
        query_lower = query.lower()
        for c in all_concepts:
            if (
                query_lower in c.canonical_name.lower()
                or query_lower in c.name.lower()
                or any(query_lower in alias.lower() for alias in c.aliases)
            ):
                matches.append(c)

        # Sort by confidence score
        matches.sort(key=lambda c: c.confidence_score or 0.0, reverse=True)
        matches = matches[:limit]

        # Get relationships for top matches
        related = []
        if show_relationships and matches:
            for match in matches[:3]:  # Only top 3 to avoid overload
                rels = await RelationshipStore.list_all_for_concept(match.id)
                related.extend(rels)

        return matches, related

    try:
        matches, related_rels = asyncio.run(search_concepts())

        if not matches:
            typer.echo(f"No concepts found for: {query}")
            return

        typer.echo(f"Found {len(matches)} concept(s) matching '{query}':\n")

        for i, concept in enumerate(matches, 1):
            # Concept header
            typer.echo(f"[{i}] {concept.name}")
            typer.echo(f"    Type: {concept.concept_type.value}")
            if concept.category:
                typer.echo(f"    Category: {concept.category}")
            if concept.aliases:
                typer.echo(f"    Aliases: {', '.join(concept.aliases)}")
            if concept.confidence_score:
                typer.echo(f"    Confidence: {concept.confidence_score:.2f}")
            if concept.definition:
                # Wrap definition
                def_lines = concept.definition[:200]
                if len(concept.definition) > 200:
                    def_lines += "..."
                typer.echo(f"    Definition: {def_lines}")

            # Show relationships
            if show_relationships:
                outgoing = [
                    r for r in related_rels if r.source_concept_id == concept.id
                ]
                incoming = [
                    r for r in related_rels if r.target_concept_id == concept.id
                ]

                if outgoing:
                    typer.echo(f"    Relationships ({len(outgoing)}):")
                    for rel in outgoing[:5]:  # Show top 5
                        typer.echo(f"      → {rel.relationship_type.value}")

                if incoming:
                    typer.echo(f"    Referenced by ({len(incoming)}):")
                    for rel in incoming[:5]:
                        typer.echo(f"      ← {rel.relationship_type.value}")

            typer.echo()

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def graph(
    concept_name: str = typer.Argument(..., help="Concept name to visualize"),
    hops: int = typer.Option(
        1, "--hops", "-h", help="Number of hops to traverse (1-3)"
    ),
    rel_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by relationship type"
    ),
):
    """Visualize concept neighborhood in the knowledge graph.

    Shows concepts and relationships within N hops of the target concept.

    Examples:

        research-kb graph "instrumental variables"

        research-kb graph "IV" --hops 2

        research-kb graph "matching" --type REQUIRES --hops 1
    """

    async def get_graph():
        config = DatabaseConfig()
        await get_connection_pool(config)

        # Find concept
        concept = await ConceptStore.get_by_canonical_name(concept_name.lower())
        if not concept:
            # Try fuzzy search
            all_concepts = await ConceptStore.list_all(limit=1000)
            query_lower = concept_name.lower()
            for c in all_concepts:
                if query_lower in c.canonical_name.lower() or any(
                    query_lower in alias.lower() for alias in c.aliases
                ):
                    concept = c
                    break

        if not concept:
            return None, None, None

        # Get neighborhood
        from research_kb_contracts import RelationshipType

        rel_filter = None
        if rel_type:
            try:
                rel_filter = RelationshipType(rel_type.upper())
            except ValueError:
                pass

        neighborhood = await get_neighborhood(
            concept.id, hops=min(hops, 3), relationship_type=rel_filter
        )

        return concept, neighborhood, rel_filter

    try:
        hops = max(1, min(hops, 3))  # Clamp to 1-3

        concept, neighborhood, rel_filter = asyncio.run(get_graph())

        if not concept:
            typer.echo(f"Concept not found: {concept_name}")
            return

        typer.echo(f"Graph neighborhood for: {concept.name}")
        typer.echo(f"Hops: {hops}")
        if rel_filter:
            typer.echo(f"Relationship type: {rel_filter.value}")
        typer.echo("=" * 60)
        typer.echo()

        # Show center
        typer.echo(f"CENTER: {concept.name} ({concept.concept_type.value})")
        typer.echo()

        # Show concepts
        typer.echo(f"Connected concepts ({len(neighborhood['concepts']) - 1}):")
        for i, c in enumerate(neighborhood["concepts"], 1):
            if c.id == concept.id:
                continue  # Skip center
            typer.echo(f"  [{i}] {c.name} ({c.concept_type.value})")

        typer.echo()

        # Show relationships
        typer.echo(f"Relationships ({len(neighborhood['relationships'])}):")

        # Build concept name lookup
        concept_lookup = {c.id: c.name for c in neighborhood["concepts"]}

        for rel in neighborhood["relationships"]:
            source_name = concept_lookup.get(rel.source_concept_id, "Unknown")
            target_name = concept_lookup.get(rel.target_concept_id, "Unknown")
            typer.echo(
                f"  {source_name} -[{rel.relationship_type.value}]-> {target_name}"
            )

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def path(
    start: str = typer.Argument(..., help="Starting concept name"),
    end: str = typer.Argument(..., help="Target concept name"),
    max_hops: int = typer.Option(5, "--max-hops", "-m", help="Maximum path length"),
):
    """Find shortest path between two concepts in the knowledge graph.

    Shows the chain of relationships connecting two concepts.

    Examples:

        research-kb path "double machine learning" "k-fold cross-validation"

        research-kb path "IV" "endogeneity" --max-hops 3

        research-kb path "matching" "propensity score" --max-hops 2
    """

    async def find_path():
        config = DatabaseConfig()
        await get_connection_pool(config)

        # Find start concept
        start_concept = await ConceptStore.get_by_canonical_name(start.lower())
        if not start_concept:
            # Fuzzy search
            all_concepts = await ConceptStore.list_all(limit=1000)
            query_lower = start.lower()
            for c in all_concepts:
                if query_lower in c.canonical_name.lower() or any(
                    query_lower in alias.lower() for alias in c.aliases
                ):
                    start_concept = c
                    break

        # Find end concept
        end_concept = await ConceptStore.get_by_canonical_name(end.lower())
        if not end_concept:
            all_concepts = await ConceptStore.list_all(limit=1000)
            query_lower = end.lower()
            for c in all_concepts:
                if query_lower in c.canonical_name.lower() or any(
                    query_lower in alias.lower() for alias in c.aliases
                ):
                    end_concept = c
                    break

        if not start_concept or not end_concept:
            return None, None, None

        # Find path
        path = await find_shortest_path(start_concept.id, end_concept.id, max_hops)

        return start_concept, end_concept, path

    try:
        start_concept, end_concept, path = asyncio.run(find_path())

        if not start_concept:
            typer.echo(f"Start concept not found: {start}")
            return

        if not end_concept:
            typer.echo(f"End concept not found: {end}")
            return

        typer.echo(f"Path from '{start_concept.name}' to '{end_concept.name}':")
        typer.echo("=" * 60)
        typer.echo()

        if not path:
            typer.echo("No path found (concepts not connected)")
            return

        # Display path
        typer.echo(f"Path length: {len(path) - 1} hop(s)\n")

        for i, (concept, relationship) in enumerate(path):
            if i == 0:
                # Starting point
                typer.echo(f"START: {concept.name} ({concept.concept_type.value})")
            else:
                # Show relationship and next concept
                if relationship:
                    typer.echo(f"  ↓ [{relationship.relationship_type.value}]")
                typer.echo(f"  {concept.name} ({concept.concept_type.value})")

        typer.echo()
        typer.echo(f"END: {end_concept.name}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def extraction_status():
    """Show extraction pipeline statistics.

    Displays:
    - Total extracted concepts by type
    - Total relationships by type
    - Concept validation status
    - Extraction quality metrics
    """

    async def get_extraction_stats():
        config = DatabaseConfig()
        pool = await get_connection_pool(config)

        async with pool.acquire() as conn:
            # Concept counts
            concept_count = await conn.fetchval("SELECT COUNT(*) FROM concepts")

            concepts_by_type = await conn.fetch(
                """
                SELECT concept_type, COUNT(*) as count
                FROM concepts
                GROUP BY concept_type
                ORDER BY count DESC
            """
            )

            # Relationship counts
            relationship_count = await conn.fetchval(
                "SELECT COUNT(*) FROM concept_relationships"
            )

            relationships_by_type = await conn.fetch(
                """
                SELECT relationship_type, COUNT(*) as count
                FROM concept_relationships
                GROUP BY relationship_type
                ORDER BY count DESC
            """
            )

            # Validation status
            validated_count = await conn.fetchval(
                "SELECT COUNT(*) FROM concepts WHERE validated = TRUE"
            )

            # Confidence distribution
            avg_confidence = await conn.fetchval(
                """
                SELECT AVG(confidence_score)
                FROM concepts
                WHERE confidence_score IS NOT NULL
            """
            )

            confidence_dist = await conn.fetch(
                """
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
            """
            )

            # Chunk coverage
            chunks_with_concepts = await conn.fetchval(
                """
                SELECT COUNT(DISTINCT chunk_id)
                FROM chunk_concepts
            """
            )

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

        # Concepts
        typer.echo(f"Total concepts extracted: {stats['concept_count']}")
        typer.echo(
            f"Validated concepts:       {stats['validated_count']} ({stats['validated_count'] / max(stats['concept_count'], 1) * 100:.1f}%)"
        )
        typer.echo()

        typer.echo("Concepts by type:")
        for row in stats["concepts_by_type"]:
            typer.echo(f"  {row['concept_type']:15} {row['count']:5}")
        typer.echo()

        # Relationships
        typer.echo(f"Total relationships: {stats['relationship_count']}")
        typer.echo()

        typer.echo("Relationships by type:")
        for row in stats["relationships_by_type"]:
            typer.echo(f"  {row['relationship_type']:15} {row['count']:5}")
        typer.echo()

        # Quality metrics
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

        # Chunk coverage
        coverage_pct = (
            stats["chunks_with_concepts"] / max(stats["total_chunks"], 1) * 100
        )
        typer.echo(
            f"Chunk coverage: {stats['chunks_with_concepts']}/{stats['total_chunks']} ({coverage_pct:.1f}%)"
        )

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def citations(
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

        research-kb citations "Pearl 2009"

        research-kb citations "DML" --type paper --limit 10
    """

    async def get_source_citations():
        from research_kb_storage import SourceStore

        config = DatabaseConfig()
        await get_connection_pool(config)

        # Find matching sources
        all_sources = await SourceStore.list_all()
        query_lower = source_query.lower()

        matches = []
        for s in all_sources:
            if query_lower in s.title.lower():
                if source_type is None or s.source_type.value == source_type:
                    matches.append(s)

        if not matches:
            return None, []

        # Use first match
        source = matches[0]

        # Get citations for this source
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

        research-kb cited-by "Pearl 2009"

        research-kb cited-by "instrumental variables" --type paper

        research-kb cited-by "DML" --limit 20
    """

    async def find_citing():
        from research_kb_storage import SourceStore
        from research_kb_contracts import SourceType

        config = DatabaseConfig()
        await get_connection_pool(config)

        # Find matching sources
        all_sources = await SourceStore.list_all()
        query_lower = source_query.lower()

        matches = [s for s in all_sources if query_lower in s.title.lower()]

        if not matches:
            return None, [], {}

        source = matches[0]

        # Get citing sources with type filter
        type_filter = None
        if source_type:
            type_filter = SourceType(source_type)

        citing = await get_citing_sources(source.id, source_type=type_filter, limit=limit)

        # Get citation stats
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

        # Stats breakdown
        typer.echo(f"Citation Authority Score: {stats.get('authority_score', 0):.4f}")
        typer.echo(f"Cited by {stats.get('cited_by_papers', 0)} papers, {stats.get('cited_by_textbooks', 0)} textbooks")
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

    Shows corpus sources referenced by the specified source,
    with breakdown by paper/textbook type.

    Examples:

        research-kb cites "DML"

        research-kb cites "econml" --type textbook

        research-kb cites "Pearl 2009" --limit 20
    """

    async def find_cited():
        from research_kb_storage import SourceStore
        from research_kb_contracts import SourceType

        config = DatabaseConfig()
        await get_connection_pool(config)

        # Find matching sources
        all_sources = await SourceStore.list_all()
        query_lower = source_query.lower()

        matches = [s for s in all_sources if query_lower in s.title.lower()]

        if not matches:
            return None, [], {}

        source = matches[0]

        # Get cited sources with type filter
        type_filter = None
        if source_type:
            type_filter = SourceType(source_type)

        cited = await get_cited_sources(source.id, source_type=type_filter, limit=limit)

        # Get citation stats
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

        # Stats breakdown
        typer.echo(f"Cites {stats.get('cites_papers', 0)} papers, {stats.get('cites_textbooks', 0)} textbooks in corpus")
        typer.echo()

        if not cited_sources:
            typer.echo("No corpus sources found in citations (or citation graph not built).")
            typer.echo("Run: python scripts/build_citation_graph.py")
            return

        typer.echo(f"Cited corpus sources ({len(cited_sources)}):\n")

        for i, s in enumerate(cited_sources, 1):
            type_badge = f"[{s.source_type.value}]"
            authority = s.citation_authority or 0.0 if hasattr(s, 'citation_authority') else 0.0
            typer.echo(f"{i:2}. {type_badge:12} {s.title[:50]}...")
            if s.authors:
                authors = ", ".join(s.authors[:2])
                typer.echo(f"    {authors} ({s.year}) | Authority: {authority:.4f}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="citation-stats")
def citation_stats_command():
    """Show corpus-wide citation graph statistics.

    Displays:
    - Total citations extracted
    - Internal vs external citation links
    - Citation type breakdown (paper→paper, textbook→paper, etc.)
    - Most cited sources in the corpus
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

        # Overview
        typer.echo(f"Total citations extracted:     {summary.get('total_citations', 0):,}")
        typer.echo(f"Total citation graph edges:    {summary.get('total_edges', 0):,}")
        typer.echo(f"  Internal (corpus→corpus):    {summary.get('internal_edges', 0):,}")
        typer.echo(f"  External (corpus→external):  {summary.get('external_edges', 0):,}")
        typer.echo()

        # Type breakdown
        typer.echo("Citation type breakdown:")
        typer.echo(f"  Paper → Paper:       {summary.get('paper_to_paper', 0):,}")
        typer.echo(f"  Paper → Textbook:    {summary.get('paper_to_textbook', 0):,}")
        typer.echo(f"  Textbook → Paper:    {summary.get('textbook_to_paper', 0):,}")
        typer.echo(f"  Textbook → Textbook: {summary.get('textbook_to_textbook', 0):,}")
        typer.echo()

        # Most cited
        if most_cited:
            typer.echo("Most cited sources (within corpus):")
            typer.echo("-" * 60)
            for i, source in enumerate(most_cited, 1):
                type_badge = f"[{source['source_type']}]"
                typer.echo(f"{i:2}. {type_badge:12} {source['title'][:45]}...")
                typer.echo(f"    Cited by: {source['cited_by_count']} | Authority: {source['citation_authority']:.4f}")
        else:
            typer.echo("No citation graph data. Run: python scripts/build_citation_graph.py")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
