"""Concept tools for MCP server.

Exposes concept listing, detail, and chunk-concept link functionality.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastmcp import FastMCP

from research_kb_api.service import (
    get_concepts,
    get_concept_by_id,
    get_concept_relationships,
)
from research_kb_storage import ChunkConceptStore, ChunkStore, ConceptStore
from research_kb_mcp.formatters import (
    format_concept_list,
    format_concept_detail,
    format_chunk_concepts,
)
from research_kb_common import get_logger

logger = get_logger(__name__)


def register_concept_tools(mcp: FastMCP) -> None:
    """Register concept tools with the MCP server."""

    @mcp.tool()
    async def research_kb_list_concepts(
        query: Optional[str] = None,
        limit: int = 50,
        concept_type: Optional[str] = None,
    ) -> str:
        """List or search concepts in the knowledge graph.

        Browse extracted concepts (methods, assumptions, theorems, etc.)
        from the causal inference literature.

        Args:
            query: Optional search query to filter concepts (default: list all)
            limit: Maximum number of concepts (1-100, default 50)
            concept_type: Filter by type (METHOD, ASSUMPTION, PROBLEM,
                         DEFINITION, THEOREM)

        Returns:
            Markdown-formatted list of concepts with:
            - Concept name
            - Type
            - Concept ID for detail queries

        Concept types:
            - METHOD: Statistical/ML methods (e.g., "causal forest", "DML")
            - ASSUMPTION: Identifying assumptions (e.g., "unconfoundedness")
            - PROBLEM: Research problems (e.g., "selection bias")
            - DEFINITION: Key terms (e.g., "average treatment effect")
            - THEOREM: Mathematical results (e.g., "Neyman orthogonality")
        """
        limit = max(1, min(100, limit))

        concepts = await get_concepts(
            query=query,
            limit=limit,
            concept_type=concept_type,
        )
        return format_concept_list(concepts)

    @mcp.tool()
    async def research_kb_get_concept(
        concept_id: str,
        include_relationships: bool = True,
    ) -> str:
        """Get detailed information about a specific concept.

        Retrieve concept details including description and relationships
        to other concepts in the knowledge graph.

        Args:
            concept_id: UUID of the concept (from search or list)
            include_relationships: Include relationships (default True)

        Returns:
            Markdown-formatted concept details with:
            - Name and type
            - Description (if available)
            - Relationships (REQUIRES, USES, ADDRESSES, etc.)

        Relationship types:
            - REQUIRES: This concept requires understanding of another
            - USES: This concept uses another (e.g., method uses assumption)
            - ADDRESSES: This concept addresses a problem
            - GENERALIZES/SPECIALIZES: Hierarchy relationships
            - ALTERNATIVE_TO: Competing approaches
            - EXTENDS: Extensions or improvements
        """
        concept = await get_concept_by_id(concept_id)

        if concept is None:
            return f"**Error:** Concept `{concept_id}` not found"

        relationships = None
        if include_relationships:
            relationships = await get_concept_relationships(concept_id)

        return format_concept_detail(concept, relationships)

    @mcp.tool()
    async def research_kb_chunk_concepts(chunk_id: str) -> str:
        """Get all concepts linked to a specific chunk.

        Shows concepts extracted from a text chunk, with mention type
        indicating how the concept appears (defines, references, or examples).
        Useful for understanding the conceptual content of a specific passage.

        Args:
            chunk_id: UUID of the chunk (from search results)

        Returns:
            Markdown-formatted list of concepts with:
            - Concept name and type
            - Mention type (defines, reference, example)
            - Relevance score (if available)
            - Concept IDs for follow-up queries

        Mention types:
            - defines: Chunk provides definition/explanation of concept
            - reference: Chunk mentions or uses the concept
            - example: Chunk provides an example of the concept
        """
        # Validate chunk exists
        chunk = await ChunkStore.get_by_id(UUID(chunk_id))
        if chunk is None:
            return f"**Error:** Chunk `{chunk_id}` not found"

        # Get chunk-concept links
        chunk_concepts = await ChunkConceptStore.list_concepts_for_chunk(UUID(chunk_id))

        if not chunk_concepts:
            return f"**Chunk `{chunk_id}`**: No concepts linked to this chunk."

        # Fetch concept details for each link
        concepts_with_links = []
        for cc in chunk_concepts:
            concept = await ConceptStore.get(cc.concept_id)
            if concept:
                concepts_with_links.append((concept, cc))

        return format_chunk_concepts(chunk, concepts_with_links)

    @mcp.tool()
    async def research_kb_find_similar_concepts(
        concept_id: str,
        limit: int = 10,
        threshold: float = 0.8,
    ) -> str:
        """Find concepts semantically similar to a given concept.

        Uses embedding similarity to find concepts related by meaning,
        even if they don't have explicit relationships in the knowledge graph.
        Useful for discovering conceptual connections and alternative terms.

        Args:
            concept_id: UUID of the source concept (from concept search/list)
            limit: Maximum number of similar concepts (1-50, default 10)
            threshold: Minimum similarity score (0.0-1.0, default 0.8)
                      Higher = more similar, 0.85+ for close matches

        Returns:
            Markdown-formatted list of similar concepts with:
            - Concept name and type
            - Similarity score
            - Concept ID for follow-up queries

        Example use cases:
            - Find alternative terms for a concept
            - Discover related concepts not explicitly linked
            - Explore semantic neighborhoods in the knowledge graph
        """
        limit = max(1, min(50, limit))
        threshold = max(0.0, min(1.0, threshold))

        try:
            # Get the source concept and its embedding
            source_concept = await ConceptStore.get_by_id(UUID(concept_id))
            if source_concept is None:
                return f"**Error:** Concept `{concept_id}` not found"

            if source_concept.embedding is None:
                return f"**Error:** Concept `{source_concept.name}` has no embedding"

            # Find similar concepts using the embedding
            similar = await ConceptStore.find_similar(
                embedding=source_concept.embedding,
                limit=limit + 1,  # +1 to exclude self
                threshold=threshold,
            )

            # Filter out the source concept itself
            similar = [(c, score) for c, score in similar if c.id != source_concept.id][:limit]

            if not similar:
                return f"**No similar concepts found** for `{source_concept.name}` at threshold {threshold:.2f}"

            # Format output
            lines = [f"## Concepts Similar to: {source_concept.name}"]
            type_val = (
                source_concept.concept_type.value
                if hasattr(source_concept.concept_type, "value")
                else source_concept.concept_type
            )
            lines.append(f"*Type: {type_val} | Threshold: {threshold:.2f}*\n")
            lines.append(f"**{len(similar)} similar concepts found**\n")

            for concept, score in similar:
                c_type = (
                    concept.concept_type.value
                    if hasattr(concept.concept_type, "value")
                    else concept.concept_type
                )
                lines.append(f"- **{concept.name}** [{c_type}]")
                lines.append(f"  - Similarity: {score:.3f}")
                lines.append(f"  - ID: `{concept.id}`")

            return "\n".join(lines)

        except ValueError as e:
            return f"**Error:** Invalid concept ID format: {e}"
        except Exception as e:
            logger.error("find_similar_concepts_failed", concept_id=concept_id, error=str(e))
            return f"**Error:** Failed to find similar concepts: {e}"
