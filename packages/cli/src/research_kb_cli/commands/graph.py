"""Knowledge graph commands for research-kb.

Commands:
    concepts       Search for concepts in the knowledge graph
    neighborhood   Visualize concept neighborhood (N-hop traversal)
    path           Find shortest path between two concepts
    explain        Explain connection between two concepts with evidence and synthesis
    export         Export a topic-filtered graph as JSON for downstream viz consumers
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer

from research_kb_storage import (
    ConceptStore,
    DatabaseConfig,
    RelationshipStore,
    find_shortest_path,
    format_citation_graph_export,
    get_connection_pool,
    get_neighborhood,
    explain_connection,
)

app = typer.Typer(help="Explore the knowledge graph")


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

        research-kb graph concepts "instrumental variables"

        research-kb graph concepts "IV" --limit 5

        research-kb graph concepts "matching" --no-relationships
    """

    async def search_concepts():
        config = DatabaseConfig()
        await get_connection_pool(config)

        # Try exact match first
        concept = await ConceptStore.get_by_canonical_name(query.lower())

        if concept:
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
            for match in matches[:3]:
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
            typer.echo(f"[{i}] {concept.name}")
            typer.echo(f"    Type: {concept.concept_type.value}")
            if concept.category:
                typer.echo(f"    Category: {concept.category}")
            if concept.aliases:
                typer.echo(f"    Aliases: {', '.join(concept.aliases)}")
            if concept.confidence_score:
                typer.echo(f"    Confidence: {concept.confidence_score:.2f}")
            if concept.definition:
                def_lines = concept.definition[:200]
                if len(concept.definition) > 200:
                    def_lines += "..."
                typer.echo(f"    Definition: {def_lines}")

            if show_relationships:
                outgoing = [r for r in related_rels if r.source_concept_id == concept.id]
                incoming = [r for r in related_rels if r.target_concept_id == concept.id]

                if outgoing:
                    typer.echo(f"    Relationships ({len(outgoing)}):")
                    for rel in outgoing[:5]:
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
def neighborhood(
    concept_name: str = typer.Argument(..., help="Concept name to visualize"),
    hops: int = typer.Option(1, "--hops", "-h", help="Number of hops to traverse (1-3)"),
    rel_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by relationship type"
    ),
):
    """Visualize concept neighborhood in the knowledge graph.

    Shows concepts and relationships within N hops of the target concept.

    Examples:

        research-kb graph neighborhood "instrumental variables"

        research-kb graph neighborhood "IV" --hops 2

        research-kb graph neighborhood "matching" --type REQUIRES --hops 1
    """

    async def get_graph():
        config = DatabaseConfig()
        await get_connection_pool(config)

        # Find concept
        concept = await ConceptStore.get_by_canonical_name(concept_name.lower())
        if not concept:
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

        n = await get_neighborhood(concept.id, hops=min(hops, 3), relationship_type=rel_filter)

        return concept, n, rel_filter

    try:
        hops = max(1, min(hops, 3))

        concept, n, rel_filter = asyncio.run(get_graph())

        if not concept:
            typer.echo(f"Concept not found: {concept_name}")
            return

        typer.echo(f"Graph neighborhood for: {concept.name}")
        typer.echo(f"Hops: {hops}")
        if rel_filter:
            typer.echo(f"Relationship type: {rel_filter.value}")
        typer.echo("=" * 60)
        typer.echo()

        typer.echo(f"CENTER: {concept.name} ({concept.concept_type.value})")
        typer.echo()

        typer.echo(f"Connected concepts ({len(n['concepts']) - 1}):")
        for i, c in enumerate(n["concepts"], 1):
            if c.id == concept.id:
                continue
            typer.echo(f"  [{i}] {c.name} ({c.concept_type.value})")

        typer.echo()

        typer.echo(f"Relationships ({len(n['relationships'])}):")
        concept_lookup = {c.id: c.name for c in n["concepts"]}
        for rel in n["relationships"]:
            source_name = concept_lookup.get(rel.source_concept_id, "Unknown")
            target_name = concept_lookup.get(rel.target_concept_id, "Unknown")
            typer.echo(f"  {source_name} -[{rel.relationship_type.value}]-> {target_name}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def path(
    start: str = typer.Argument(..., help="Starting concept name"),
    end: str = typer.Argument(..., help="Target concept name"),
    max_hops: int = typer.Option(5, "--max-hops", "-m", help="Maximum path length"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show concept definitions"),
    synthesis: bool = typer.Option(False, "--synthesis", "-s", help="Include synthesis prompt"),
    style: str = typer.Option(
        "educational",
        "--style",
        help="Synthesis style: educational, research, or implementation",
    ),
):
    """Find shortest path between two concepts in the knowledge graph.

    Shows the chain of relationships connecting two concepts.

    Examples:

        research-kb graph path "double machine learning" "k-fold cross-validation"

        research-kb graph path "IV" "endogeneity" --verbose

        research-kb graph path "matching" "propensity score" --synthesis --style research
    """
    from research_kb_storage import generate_synthesis_prompt, explain_path

    async def find_path():
        config = DatabaseConfig()
        await get_connection_pool(config)

        # Find start concept
        start_concept = await ConceptStore.get_by_canonical_name(start.lower())
        if not start_concept:
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

        found = await find_shortest_path(start_concept.id, end_concept.id, max_hops)

        return start_concept, end_concept, found

    try:
        start_concept, end_concept, found_path = asyncio.run(find_path())

        if not start_concept:
            typer.echo(f"Start concept not found: {start}")
            return

        if not end_concept:
            typer.echo(f"End concept not found: {end}")
            return

        typer.echo(f"Path from '{start_concept.name}' to '{end_concept.name}':")
        typer.echo("=" * 60)
        typer.echo()

        if not found_path:
            typer.echo("No path found (concepts not connected)")
            return

        typer.echo(f"Path length: {len(found_path) - 1} hop(s)\n")

        for i, (concept, relationship) in enumerate(found_path):
            if i == 0:
                typer.echo(f"START: {concept.name} ({concept.concept_type.value})")
                if verbose and concept.definition:
                    definition = (
                        concept.definition[:300] + "..."
                        if len(concept.definition) > 300
                        else concept.definition
                    )
                    typer.echo(f"       {definition}")
            else:
                if relationship:
                    typer.echo(f"  ↓ [{relationship.relationship_type.value}]")
                typer.echo(f"  {concept.name} ({concept.concept_type.value})")
                if verbose and concept.definition:
                    definition = (
                        concept.definition[:300] + "..."
                        if len(concept.definition) > 300
                        else concept.definition
                    )
                    typer.echo(f"       {definition}")

        typer.echo()
        typer.echo(f"END: {end_concept.name}")

        explanation = explain_path(found_path)
        typer.echo()
        typer.echo(f"Path: {explanation}")

        if synthesis:
            typer.echo()
            typer.echo("-" * 60)
            typer.echo(f"Synthesis Prompt ({style}):")
            typer.echo()
            prompt = generate_synthesis_prompt(found_path, style=style)
            typer.echo(prompt)

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def explain(
    concept_a: str = typer.Argument(..., help="First concept name"),
    concept_b: str = typer.Argument(..., help="Second concept name"),
    style: str = typer.Option(
        "educational",
        "--style",
        "-s",
        help="Synthesis style: educational, research, or implementation",
    ),
    max_evidence: int = typer.Option(
        2, "--max-evidence", "-e", help="Max evidence chunks per step (1-3)"
    ),
    no_llm: bool = typer.Option(
        False, "--no-llm", help="Skip LLM synthesis (graph + evidence only)"
    ),
    output_format: str = typer.Option(
        "markdown", "--format", "-f", help="Output format: markdown or json"
    ),
):
    """Explain how two concepts are connected with evidence and synthesis.

    Finds the shortest path, attaches evidence from the corpus, and
    optionally generates an LLM synthesis with source citations.

    Examples:

        research-kb graph explain "double machine learning" "cross-fitting"

        research-kb graph explain "IV" "endogeneity" --style research

        research-kb graph explain "DML" "overlap" --no-llm

        research-kb graph explain "RDD" "LATE" --format json
    """
    from research_kb_mcp.formatters import (
        format_connection_explanation,
        format_connection_explanation_json,
    )

    async def run_explain():
        config = DatabaseConfig()
        await get_connection_pool(config)

        return await explain_connection(
            concept_a=concept_a,
            concept_b=concept_b,
            style=style,
            max_evidence_per_step=max(1, min(3, max_evidence)),
            use_llm=not no_llm,
        )

    try:
        result = asyncio.run(run_explain())

        if output_format == "json":
            typer.echo(format_connection_explanation_json(result))
        else:
            typer.echo(format_connection_explanation(result))

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# graph export — generic node-link JSON for downstream visualization consumers
# ---------------------------------------------------------------------------
#
# Selector-resolution framework (D13b from planning):
#   Each selector flag (--arxiv-ids today; future --source-ids / --topic /
#   --concept-id) resolves to a `set[source_id]`-equivalent input. The
#   common graph-build + format layers consume the resolved set and emit
#   the versioned graph JSON. Adding a future selector requires:
#     1. One new CLI option below
#     2. One new branch in the resolver dispatch
#     3. (Possibly) one new builder for non-citation graph_types
#
# Anti-pattern to avoid: a `--manifest <yaml>` DSL. Composition happens in
# caller scripts (e.g., rl_and_control/scripts/build_graph_export.py), not
# in research-kb's CLI surface.


def _load_lines(path: Path) -> list[str]:
    """Load newline-delimited entries; strip `# ...` comments and blanks."""
    entries: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    return entries


@app.command()
def export(
    output: Path = typer.Option(
        ..., "--output", "-o", help="Output JSON file path (parent dirs created if missing)"
    ),
    arxiv_ids: Optional[Path] = typer.Option(
        None,
        "--arxiv-ids",
        help="Path to newline-delimited file of arXiv IDs to include in the graph",
    ),
    role: Optional[str] = typer.Option(
        None,
        "--role",
        help=(
            "metadata.roles marker to include (e.g. 'anchor.rl_optimal_control'). "
            "Every source whose metadata.roles array contains this string is "
            "added to the graph alongside any --arxiv-ids selection."
        ),
    ),
    output_format: str = typer.Option(
        "generic-node-link",
        "--format",
        "-f",
        help="Output format (currently only 'generic-node-link')",
    ),
    schema_version: str = typer.Option(
        "1.0", "--schema-version", help="Emitted schema_version in the JSON metadata"
    ),
    topic_label: str = typer.Option(
        "", "--topic-label", help="Human-readable topic label embedded in output metadata"
    ),
):
    """Export a topic-filtered graph as JSON for downstream viz consumers.

    Selector framework: at least one selector flag must be specified.
    Selectors COMBINE — the resulting graph is the union of all matched
    sources, deduplicated. Currently implemented selectors:

        --arxiv-ids <file>      newline-delimited arXiv IDs
        --role <string>         metadata.roles marker (anchors etc.)

    Planned selectors (not yet implemented; reserved):

        --source-ids <file>     newline-delimited source UUIDs
        --topic <name>          research-kb concept name -> matched sources
        --concept-id <uuid>     specific concept -> connected sources

    Output: JSON file matching the versioned graph schema (see
    `research_kb_storage.graph_queries.format_citation_graph_export` for
    the schema). brandon-behring.dev's `/lab/research-graph/` route is the
    first known consumer; the format is intentionally renderer-agnostic
    (5-line adapter to Cytoscape/Sigma at consume-time).

    Examples:

        research-kb graph export \\
            --arxiv-ids rl_ids.txt \\
            --output /tmp/rl_citation_graph.json

        research-kb graph export \\
            --arxiv-ids rl_ids.txt \\
            --role anchor.rl_optimal_control \\
            --output exports/rl_citation_graph.json \\
            --topic-label "RL and optimal control"
    """
    # Selector validation: at least one selector flag must be specified.
    # Selectors combine via set union in format_citation_graph_export.
    chosen_flags = []
    if arxiv_ids is not None:
        chosen_flags.append("--arxiv-ids")
    if role is not None:
        chosen_flags.append("--role")

    if not chosen_flags:
        typer.echo(
            "Error: must specify at least one selector "
            "(--arxiv-ids and/or --role; future: --source-ids/--topic/--concept-id)",
            err=True,
        )
        raise typer.Exit(2)

    if output_format != "generic-node-link":
        typer.echo(
            f"Error: format '{output_format}' not supported "
            f"(currently only 'generic-node-link')",
            err=True,
        )
        raise typer.Exit(2)

    async def run_export():
        config = DatabaseConfig()
        await get_connection_pool(config)

        # Resolve selectors to lists/values, then hand the union to the
        # storage-layer formatter (which dedups and unions internally).
        arxiv_list: list[str] = []
        if arxiv_ids is not None:
            arxiv_list = _load_lines(arxiv_ids)

        return await format_citation_graph_export(
            arxiv_ids=arxiv_list,
            role=role,
            schema_version=schema_version,
            topic_label=topic_label,
        )

    try:
        graph_data = asyncio.run(run_export())

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

        n_nodes = len(graph_data.get("nodes", []))
        n_edges = len(graph_data.get("edges", []))
        n_ingested = graph_data.get("metadata", {}).get("ingested_count", 0)
        n_missing = graph_data.get("metadata", {}).get("not_ingested_count", 0)
        typer.echo(
            f"✓ Wrote {output} ({n_nodes} nodes [{n_ingested} ingested + "
            f"{n_missing} not_ingested], {n_edges} edges, "
            f"schema_version={graph_data.get('schema_version')})"
        )
    except NotImplementedError as e:
        typer.echo(f"Not implemented: {e}", err=True)
        raise typer.Exit(2)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
