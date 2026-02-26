"""Response formatters for MCP tool outputs.

Converts service layer responses to markdown format optimized for
LLM consumption. Includes source IDs, page numbers, and relevance scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from research_kb_api.service import SearchResponse, SearchResultDetail
    from research_kb_contracts import (
        Source,
        Concept,
        ConceptRelationship,
        Chunk,
    )


# Maximum content length before truncation
MAX_CONTENT_LENGTH = 1500
TRUNCATION_SUFFIX = "..."


def truncate(text: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(TRUNCATION_SUFFIX)] + TRUNCATION_SUFFIX


def format_search_results(response: SearchResponse) -> str:
    """Format search response as markdown for LLM consumption.

    Args:
        response: SearchResponse from service layer

    Returns:
        Markdown-formatted search results
    """
    lines = [f"## Search Results for: {response.query}"]

    if response.expanded_query:
        lines.append(f"*Query expanded to: {response.expanded_query}*")

    lines.append(
        f"\n**Found {len(response.results)} results** (in {response.execution_time_ms:.0f}ms)\n"
    )

    if not response.results:
        lines.append("No results found.")
        return "\n".join(lines)

    for i, result in enumerate(response.results, 1):
        lines.append(format_search_result(result, i))

    return "\n".join(lines)


def format_search_result(result: SearchResultDetail, rank: int) -> str:
    """Format a single search result."""
    source = result.source
    chunk = result.chunk

    lines = [f"### {rank}. {source.title}"]

    # Source metadata
    meta_parts = []
    if source.authors:
        meta_parts.append(", ".join(source.authors[:3]))
        if len(source.authors) > 3:
            meta_parts.append("et al.")
    if source.year:
        meta_parts.append(f"({source.year})")
    if source.source_type:
        meta_parts.append(f"[{source.source_type}]")

    if meta_parts:
        lines.append(" ".join(meta_parts))

    # Page reference
    page_ref = ""
    if chunk.page_start:
        if chunk.page_end and chunk.page_end != chunk.page_start:
            page_ref = f"pp. {chunk.page_start}-{chunk.page_end}"
        else:
            page_ref = f"p. {chunk.page_start}"
    if chunk.section:
        page_ref = f"{chunk.section}, {page_ref}" if page_ref else chunk.section

    if page_ref:
        lines.append(f"*{page_ref}*")

    # Content
    lines.append(f"\n> {truncate(chunk.content)}\n")

    # Score breakdown
    scores = result.scores
    score_parts = [f"Score: {result.combined_score:.3f}"]
    if scores.fts > 0:
        score_parts.append(f"FTS: {scores.fts:.3f}")
    if scores.vector > 0:
        score_parts.append(f"Vector: {scores.vector:.3f}")
    if scores.graph > 0:
        score_parts.append(f"Graph: {scores.graph:.3f}")

    lines.append(f"*{' | '.join(score_parts)}*")
    lines.append(f"*Source ID: `{source.id}` | Chunk ID: `{chunk.id}`*\n")

    return "\n".join(lines)


def format_source_list(sources: list[Source]) -> str:
    """Format source list as markdown."""
    if not sources:
        return "No sources found."

    lines = [f"## Sources ({len(sources)} total)\n"]

    for source in sources:
        lines.append(format_source_summary(source))

    return "\n".join(lines)


def format_source_summary(source: Source) -> str:
    """Format a single source summary."""
    parts = [f"- **{source.title}**"]

    if source.authors:
        authors = ", ".join(source.authors[:2])
        if len(source.authors) > 2:
            authors += " et al."
        parts.append(f"  - Authors: {authors}")

    if source.year:
        parts.append(f"  - Year: {source.year}")

    if source.source_type:
        parts.append(
            f"  - Type: {source.source_type.value if hasattr(source.source_type, 'value') else source.source_type}"
        )

    parts.append(f"  - ID: `{source.id}`")

    return "\n".join(parts)


def format_source_detail(source: Source, chunks: Optional[list] = None) -> str:
    """Format detailed source information."""
    lines = [f"## {source.title}"]

    if source.authors:
        lines.append(f"**Authors:** {', '.join(source.authors)}")
    if source.year:
        lines.append(f"**Year:** {source.year}")
    if source.source_type:
        type_val = (
            source.source_type.value if hasattr(source.source_type, "value") else source.source_type
        )
        lines.append(f"**Type:** {type_val}")

    lines.append(f"**ID:** `{source.id}`")

    if source.metadata:
        lines.append("\n### Metadata")
        for key, value in source.metadata.items():
            if value:
                lines.append(f"- {key}: {value}")

    if chunks:
        lines.append(f"\n### Content Chunks ({len(chunks)} total)")
        for i, chunk in enumerate(chunks[:5], 1):  # Show first 5 chunks
            page_info = (
                f"p. {chunk.page_start}"
                if hasattr(chunk, "page_start") and chunk.page_start
                else ""
            )
            lines.append(f"\n**Chunk {i}** {page_info}")
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            lines.append(f"> {truncate(content, 500)}")

        if len(chunks) > 5:
            lines.append(f"\n*... and {len(chunks) - 5} more chunks*")

    return "\n".join(lines)


def format_citations(citations_data: dict) -> str:
    """Format citation information."""
    lines = [f"## Citations for Source `{citations_data['source_id']}`\n"]

    citing = citations_data.get("citing_sources", [])
    cited = citations_data.get("cited_sources", [])

    lines.append(f"### Papers Citing This Source ({len(citing)})")
    if citing:
        for s in citing:
            year = f" ({s['year']})" if s.get("year") else ""
            lines.append(f"- {s['title']}{year} [`{s['id']}`]")
    else:
        lines.append("*No citing papers found*")

    lines.append(f"\n### Papers Cited By This Source ({len(cited)})")
    if cited:
        for s in cited:
            year = f" ({s['year']})" if s.get("year") else ""
            lines.append(f"- {s['title']}{year} [`{s['id']}`]")
    else:
        lines.append("*No cited papers found*")

    return "\n".join(lines)


def format_citing_sources(sources: list, source_id: str) -> str:
    """Format sources that cite a given source."""
    lines = [f"## Sources Citing `{source_id}`\n"]
    lines.append(f"**{len(sources)} sources cite this paper**\n")

    if not sources:
        lines.append("*No citing sources found in the knowledge base*")
        return "\n".join(lines)

    for s in sources:
        year = f" ({s.year})" if s.year else ""
        authors = ", ".join(s.authors[:2]) if s.authors else "Unknown"
        if s.authors and len(s.authors) > 2:
            authors += " et al."
        lines.append(f"- **{s.title}**{year}")
        lines.append(f"  - {authors}")
        lines.append(f"  - ID: `{s.id}`")

    return "\n".join(lines)


def format_cited_sources(sources: list, source_id: str) -> str:
    """Format sources that are cited by a given source."""
    lines = [f"## Sources Cited By `{source_id}`\n"]
    lines.append(f"**{len(sources)} sources are cited**\n")

    if not sources:
        lines.append("*No cited sources found in the knowledge base*")
        return "\n".join(lines)

    for s in sources:
        year = f" ({s.year})" if s.year else ""
        authors = ", ".join(s.authors[:2]) if s.authors else "Unknown"
        if s.authors and len(s.authors) > 2:
            authors += " et al."
        lines.append(f"- **{s.title}**{year}")
        lines.append(f"  - {authors}")
        lines.append(f"  - ID: `{s.id}`")

    return "\n".join(lines)


def format_concept_list(concepts: list[Concept]) -> str:
    """Format concept list as markdown."""
    if not concepts:
        return "No concepts found."

    lines = [f"## Concepts ({len(concepts)} total)\n"]

    for concept in concepts:
        lines.append(format_concept_summary(concept))

    return "\n".join(lines)


def format_concept_summary(concept: Concept) -> str:
    """Format a single concept summary."""
    type_val = (
        concept.concept_type.value
        if hasattr(concept.concept_type, "value")
        else concept.concept_type
    )
    return f"- **{concept.name}** [{type_val}] `{concept.id}`"


def format_concept_detail(
    concept: Concept, relationships: Optional[list[ConceptRelationship]] = None
) -> str:
    """Format detailed concept information."""
    lines = [f"## {concept.name}"]

    type_val = (
        concept.concept_type.value
        if hasattr(concept.concept_type, "value")
        else concept.concept_type
    )
    lines.append(f"**Type:** {type_val}")
    lines.append(f"**ID:** `{concept.id}`")

    if concept.definition:
        lines.append(f"\n### Description\n{concept.definition}")

    if relationships:
        lines.append(f"\n### Relationships ({len(relationships)} total)")
        for rel in relationships[:20]:  # Show first 20
            rel_type = (
                rel.relationship_type.value
                if hasattr(rel.relationship_type, "value")
                else rel.relationship_type
            )
            lines.append(f"- {rel_type} → `{rel.target_concept_id}`")

        if len(relationships) > 20:
            lines.append(f"*... and {len(relationships) - 20} more relationships*")

    return "\n".join(lines)


def format_graph_neighborhood(neighborhood: dict) -> str:
    """Format graph neighborhood as markdown."""
    if "error" in neighborhood:
        return f"**Error:** {neighborhood['error']}"

    center = neighborhood.get("center") or {}
    nodes = neighborhood.get("nodes", [])
    edges = neighborhood.get("edges", [])

    if not center:
        return "**Error:** Concept not found"

    lines = [f"## Graph Neighborhood: {center.get('name', 'Unknown')}"]
    lines.append(f"*Type: {center.get('type', 'unknown')} | ID: `{center.get('id')}`*")
    lines.append(f"\n**{len(nodes)} connected concepts, {len(edges)} relationships**\n")

    if nodes:
        lines.append("### Connected Concepts")
        for node in nodes[:30]:  # Show first 30
            lines.append(f"- {node['name']} [{node.get('type', '?')}]")

        if len(nodes) > 30:
            lines.append(f"*... and {len(nodes) - 30} more*")

    if edges:
        lines.append("\n### Relationships")
        # Group by relationship type
        by_type: dict[str, int] = {}
        for edge in edges:
            rel_type = edge.get("type", "UNKNOWN")
            by_type[rel_type] = by_type.get(rel_type, 0) + 1

        for rel_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
            lines.append(f"- {rel_type}: {count}")

    return "\n".join(lines)


def format_graph_path(path_data: dict, include_definitions: bool = True) -> str:
    """Format graph path as markdown.

    Supports enhanced path format with definitions and synthesis prompts.

    Args:
        path_data: Dict with path info from get_graph_path()
        include_definitions: Show concept definitions (default True for MCP)

    Returns:
        Markdown-formatted path with optional definitions and synthesis prompt
    """
    if "error" in path_data:
        return f"**Error:** {path_data['error']}"

    from_concept = path_data.get("from", "?")
    to_concept = path_data.get("to", "?")
    path = path_data.get("path", [])
    relationships = path_data.get("relationships", [])
    explanation = path_data.get("explanation")
    path_length = path_data.get("path_length", len(path) - 1 if path else 0)

    lines = [f"## Conceptual Path: {from_concept} → {to_concept}"]

    if not path:
        lines.append("\n*No path found between these concepts*")
        return "\n".join(lines)

    lines.append(f"\n**Path length: {path_length} hops**\n")

    # Format each concept in the path
    for i, step in enumerate(path):
        if isinstance(step, dict):
            name = step.get("name", step.get("id", "?"))
            concept_type = step.get("type", "")
            definition = step.get("definition")

            # Numbered concept header with type
            type_str = f" ({concept_type})" if concept_type else ""
            lines.append(f"### {i + 1}. {name}{type_str}")

            # Include definition if present and requested
            if include_definitions and definition:
                # Truncate long definitions
                if len(definition) > 300:
                    definition = definition[:297] + "..."
                lines.append(f"> {definition}")

            # Add relationship arrow to next concept (if not last)
            if i < len(path) - 1:
                rel_type = relationships[i] if i < len(relationships) else "→"
                lines.append(f"**→ {rel_type} →**\n")
        else:
            # Fallback for simple string paths
            lines.append(f"{i + 1}. {step}")

    # Add explanation if provided
    if explanation:
        lines.append("\n---")
        lines.append(f"*Path: {explanation}*")

    # Add synthesis prompt if present
    if path_data.get("synthesis_prompt"):
        style = path_data.get("synthesis_style", "educational")
        lines.append("\n---")
        lines.append(f"## Synthesis Prompt ({style})")
        lines.append(f"\n> {path_data['synthesis_prompt']}")

    return "\n".join(lines)


def format_stats(stats: dict) -> str:
    """Format database statistics as markdown."""
    lines = ["## Research KB Statistics\n"]

    lines.append("| Entity | Count |")
    lines.append("|--------|------:|")

    for key, value in stats.items():
        label = key.replace("_", " ").title()
        lines.append(f"| {label} | {value:,} |")

    return "\n".join(lines)


def format_health(
    healthy: bool,
    details: Optional[dict] = None,
    degraded: bool = False,
) -> str:
    """Format health check result.

    Args:
        healthy: Whether core services (database, embed) are working
        details: Additional status information
        degraded: Whether optional services (kuzu, reranker) are down
    """
    if not healthy:
        status = "❌ Unhealthy"
    elif degraded:
        status = "⚠️ Degraded"
    else:
        status = "✅ Healthy"

    lines = [f"## Research KB Health: {status}"]

    if details:
        lines.append("\n### Details")
        for key, value in details.items():
            lines.append(f"- {key}: {value}")

    return "\n".join(lines)


def format_domains(domain_stats: list[dict]) -> str:
    """Format domain statistics as markdown.

    Args:
        domain_stats: List of domain statistics from DomainStore.get_all_stats()

    Returns:
        Markdown-formatted table with domain info and counts
    """
    if not domain_stats:
        return "## Knowledge Domains\n\n*No domains configured*"

    lines = ["## Knowledge Domains\n"]
    lines.append("| Domain ID | Name | Sources | Chunks | Concepts |")
    lines.append("|-----------|------|--------:|-------:|---------:|")

    for stat in domain_stats:
        lines.append(
            f"| `{stat['domain_id']}` | {stat['name']} | "
            f"{stat['source_count']:,} | {stat['chunk_count']:,} | "
            f"{stat['concept_count']:,} |"
        )

    lines.append("\n### Usage")
    lines.append("Use the `domain` parameter in `research_kb_search` to filter by domain:")
    lines.append("- `domain=None` — Search all domains (default)")
    for stat in domain_stats:
        lines.append(f"- `domain=\"{stat['domain_id']}\"` — {stat['name']} only")

    return "\n".join(lines)


def format_citation_network(citing: list, cited: list, source: Source) -> str:
    """Format bidirectional citation network as markdown.

    Args:
        citing: List of Source objects that cite this source
        cited: List of Source objects cited by this source
        source: The center source

    Returns:
        Markdown-formatted citation network
    """
    lines = [f"## Citation Network: {source.title}"]
    lines.append(f"*Source ID: `{source.id}`*\n")

    # Papers citing this source (downstream)
    lines.append(f"### Citing This Source ({len(citing)})")
    lines.append("*Papers that built on this work*\n")

    if citing:
        for s in citing:
            year = f" ({s.year})" if s.year else ""
            authors = ", ".join(s.authors[:2]) if s.authors else "Unknown"
            if s.authors and len(s.authors) > 2:
                authors += " et al."
            lines.append(f"- **{s.title}**{year}")
            lines.append(f"  - {authors}")
            lines.append(f"  - ID: `{s.id}`")
    else:
        lines.append("*No citing sources found in the knowledge base*")

    lines.append("")

    # Papers cited by this source (upstream)
    lines.append(f"### Cited By This Source ({len(cited)})")
    lines.append("*Foundations and context*\n")

    if cited:
        for s in cited:
            year = f" ({s.year})" if s.year else ""
            authors = ", ".join(s.authors[:2]) if s.authors else "Unknown"
            if s.authors and len(s.authors) > 2:
                authors += " et al."
            lines.append(f"- **{s.title}**{year}")
            lines.append(f"  - {authors}")
            lines.append(f"  - ID: `{s.id}`")
    else:
        lines.append("*No cited sources found in the knowledge base*")

    return "\n".join(lines)


def format_biblio_similar(similar_sources: list[dict], source: Source) -> str:
    """Format bibliographic coupling results as markdown.

    Args:
        similar_sources: List of dicts from BiblioStore.get_similar_sources()
        source: The query source

    Returns:
        Markdown-formatted similar sources with coupling percentages
    """
    lines = [f"## Bibliographically Similar: {source.title}"]
    lines.append(f"*Source ID: `{source.id}`*\n")

    if not similar_sources:
        lines.append("*No similar sources found (no shared references)*")
        return "\n".join(lines)

    lines.append(f"**{len(similar_sources)} similar sources** by shared references\n")

    for s in similar_sources:
        coupling_pct = s["coupling_strength"] * 100
        year = f" ({s['year']})" if s.get("year") else ""
        authors = s.get("authors", [])
        author_str = ", ".join(authors[:2]) if authors else "Unknown"
        if authors and len(authors) > 2:
            author_str += " et al."

        lines.append(f"- **{s['title']}**{year}")
        lines.append(f"  - {author_str}")
        lines.append(
            f"  - Coupling: **{coupling_pct:.1f}%** ({s['shared_references']} shared refs)"
        )
        lines.append(f"  - ID: `{s['source_id']}`")

    return "\n".join(lines)


def format_chunk_concepts(chunk: Chunk, concepts_with_links: list) -> str:
    """Format concepts linked to a chunk as markdown.

    Args:
        chunk: The Chunk object
        concepts_with_links: List of (Concept, ChunkConcept) tuples

    Returns:
        Markdown-formatted concept list with mention metadata
    """
    # Create page reference
    page_ref = ""
    if chunk.page_start:
        if chunk.page_end and chunk.page_end != chunk.page_start:
            page_ref = f"pp. {chunk.page_start}-{chunk.page_end}"
        else:
            page_ref = f"p. {chunk.page_start}"

    lines = ["## Concepts in Chunk"]
    lines.append(f"*Chunk ID: `{chunk.id}`*")
    if page_ref:
        lines.append(f"*Location: {page_ref}*")
    lines.append(f"\n**{len(concepts_with_links)} concepts linked**\n")

    # Group by mention type
    by_type: dict[str, list] = {"defines": [], "reference": [], "example": []}

    for concept, cc in concepts_with_links:
        mention_type = cc.mention_type or "reference"
        if mention_type not in by_type:
            by_type[mention_type] = []
        by_type[mention_type].append((concept, cc))

    # Format defines first (most important)
    if by_type.get("defines"):
        lines.append("### Defines")
        for concept, cc in by_type["defines"]:
            type_val = (
                concept.concept_type.value
                if hasattr(concept.concept_type, "value")
                else concept.concept_type
            )
            relevance = f" (relevance: {cc.relevance_score:.2f})" if cc.relevance_score else ""
            lines.append(f"- **{concept.name}** [{type_val}]{relevance}")
            lines.append(f"  - ID: `{concept.id}`")

    # Format references
    if by_type.get("reference"):
        lines.append("\n### References")
        for concept, cc in by_type["reference"]:
            type_val = (
                concept.concept_type.value
                if hasattr(concept.concept_type, "value")
                else concept.concept_type
            )
            relevance = f" (relevance: {cc.relevance_score:.2f})" if cc.relevance_score else ""
            lines.append(f"- **{concept.name}** [{type_val}]{relevance}")
            lines.append(f"  - ID: `{concept.id}`")

    # Format examples
    if by_type.get("example"):
        lines.append("\n### Examples")
        for concept, cc in by_type["example"]:
            type_val = (
                concept.concept_type.value
                if hasattr(concept.concept_type, "value")
                else concept.concept_type
            )
            relevance = f" (relevance: {cc.relevance_score:.2f})" if cc.relevance_score else ""
            lines.append(f"- **{concept.name}** [{type_val}]{relevance}")
            lines.append(f"  - ID: `{concept.id}`")

    return "\n".join(lines)
