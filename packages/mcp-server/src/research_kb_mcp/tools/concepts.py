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
