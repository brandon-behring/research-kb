"""Graph tools for MCP server.

Exposes knowledge graph exploration functionality.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastmcp import FastMCP

from research_kb_api.service import (
    get_graph_neighborhood,
    get_graph_path,
)
from research_kb_mcp.formatters import (
    format_graph_neighborhood,
    format_graph_path,
)
from research_kb_storage import CrossDomainStore, ConceptStore
from research_kb_common import get_logger

logger = get_logger(__name__)


def register_graph_tools(mcp: FastMCP) -> None:
    """Register graph tools with the MCP server."""

    @mcp.tool()
    async def research_kb_graph_neighborhood(
        concept_name: str,
        hops: int = 2,
        limit: int = 50,
    ) -> str:
        """Explore the neighborhood of a concept in the knowledge graph.

        Find concepts related to a given concept within N hops. Useful for
        understanding context, prerequisites, and related methods.

        Args:
            concept_name: Name of the concept to explore (fuzzy matched)
            hops: Number of relationship hops (1-3, default 2)
            limit: Maximum connected concepts to return (1-100, default 50)

        Returns:
            Markdown-formatted neighborhood with:
            - Center concept details
            - Connected concepts (names and types)
            - Relationship type distribution

        Example:
            Exploring "double machine learning" with 2 hops might reveal:
            - Direct: Neyman orthogonality, cross-fitting, nuisance estimation
            - 2-hop: propensity score, sample splitting, regularization
        """
        hops = max(1, min(3, hops))
        limit = max(1, min(100, limit))

        neighborhood = await get_graph_neighborhood(
            concept_name=concept_name,
            hops=hops,
            limit=limit,
        )
        return format_graph_neighborhood(neighborhood)

    @mcp.tool()
    async def research_kb_graph_path(
        concept_a: str,
        concept_b: str,
        include_definitions: bool = True,
        include_synthesis: bool = True,
        synthesis_style: str = "educational",
    ) -> str:
        """Find the shortest path between two concepts.

        Discover how two concepts are related through the knowledge graph.
        Useful for understanding connections between methods, assumptions,
        and problems.

        Args:
            concept_a: Name of the first concept (fuzzy matched)
            concept_b: Name of the second concept (fuzzy matched)
            include_definitions: Include concept definitions (default True)
            include_synthesis: Include synthesis prompt for LLM (default True)
            synthesis_style: Style of synthesis prompt:
                - "educational": Focus on understanding and learning
                - "research": Focus on assumptions and methodological rigor
                - "implementation": Focus on practical coding considerations

        Returns:
            Markdown-formatted path with:
            - Start and end concepts with types
            - Path length (number of hops)
            - Intermediate concepts along the path
            - Concept definitions (if include_definitions=True)
            - Relationship types between concepts
            - Synthesis prompt (if include_synthesis=True)

        Example:
            Path from "regression discontinuity" to "instrumental variables"
            might show: RD → local average treatment effect → IV
        """
        path = await get_graph_path(
            concept_a=concept_a,
            concept_b=concept_b,
            include_definitions=include_definitions,
            include_synthesis=include_synthesis,
            synthesis_style=synthesis_style,
        )
        return format_graph_path(path, include_definitions=include_definitions)

    @mcp.tool()
    async def research_kb_cross_domain_concepts(
        source_domain: str,
        target_domain: str,
        concept_id: Optional[str] = None,
        concept_name: Optional[str] = None,
        similarity_threshold: float = 0.85,
        limit: int = 10,
    ) -> str:
        """Find equivalent or related concepts across knowledge domains.

        Discovers semantic connections between concepts in different domains
        (e.g., finding time_series concepts related to causal_inference concepts).
        Uses embedding similarity to find matches.

        Args:
            concept_id: UUID of a specific concept to find cross-domain matches for
            concept_name: Name of concept to search (if concept_id not provided)
            source_domain: Domain to search from (required)
            target_domain: Domain to find matches in (required)
            similarity_threshold: Minimum similarity score (0.0-1.0, default 0.85)
            limit: Maximum matches to return (1-50, default 10)

        Returns:
            Markdown-formatted list of cross-domain matches with:
            - Source concept (from source domain)
            - Matched concepts (from target domain)
            - Similarity scores
            - Link type (EQUIVALENT, ANALOGOUS, RELATED)

        Link Types:
            - EQUIVALENT (>=0.95): Same concept, different terminology
            - ANALOGOUS (>=0.90): Similar role in different contexts
            - RELATED (<0.90): Semantically related but distinct

        Examples:
            - "DAG" in causal_inference → "Directed Acyclic Graph" in time_series
            - "selection bias" → "sample selection" across domains
            - "instrumental variables" → related estimation methods
        """
        limit = max(1, min(50, limit))
        similarity_threshold = max(0.5, min(1.0, similarity_threshold))

        try:
            # If concept_id provided, find cross-domain links for that concept
            if concept_id:
                links = await CrossDomainStore.get_cross_domain_concepts(
                    concept_id=UUID(concept_id),
                    direction="both",
                )

                if not links:
                    return f"**No cross-domain links found** for concept `{concept_id}`"

                lines = ["## Cross-Domain Concept Links"]
                lines.append(f"*Concept ID: `{concept_id}`*\n")

                for link in links[:limit]:
                    link_type = link.get("link_type", "UNKNOWN")
                    score = link.get("confidence_score", 0)
                    name = link.get("linked_concept_name", "Unknown")
                    domain = link.get("linked_domain", "Unknown")
                    lines.append(f"- **{name}** [{link_type}] ({domain})")
                    lines.append(f"  - Confidence: {score:.3f}")
                    lines.append(f"  - ID: `{link.get('linked_concept_id')}`")

                return "\n".join(lines)

            # If concept_name provided, find concept first, then get matches
            if concept_name:
                # Search for concept by name
                concepts = await ConceptStore.search_by_name(
                    name=concept_name,
                    domain_id=source_domain,
                    limit=1,
                )

                if not concepts:
                    return f"**Error:** Concept `{concept_name}` not found in {source_domain}"

                source_concept = concepts[0]

                # Check for existing cross-domain links
                existing = await CrossDomainStore.get_cross_domain_concepts(
                    concept_id=source_concept.id,
                    direction="both",
                )

                if existing:
                    lines = [f"## Cross-Domain Links: {source_concept.name}"]
                    lines.append(f"*Domain: {source_domain} | {len(existing)} links*\n")

                    for link in existing[:limit]:
                        link_type = link.get("link_type", "UNKNOWN")
                        score = link.get("confidence_score", 0)
                        name = link.get("linked_concept_name", "Unknown")
                        domain = link.get("linked_domain", "Unknown")
                        lines.append(f"- **{name}** [{link_type}] ({domain})")
                        lines.append(f"  - Confidence: {score:.3f}")

                    return "\n".join(lines)

                # If no existing links, try live discovery
                if source_concept.embedding is not None:
                    similar = await ConceptStore.find_similar(
                        embedding=source_concept.embedding,
                        limit=limit,
                        threshold=similarity_threshold,
                        domain_id=target_domain,
                    )

                    if similar:
                        lines = [f"## Similar Concepts in {target_domain}"]
                        lines.append(f"*Source: {source_concept.name} ({source_domain})*\n")

                        for concept, score in similar:
                            link_type = (
                                "EQUIVALENT"
                                if score >= 0.95
                                else "ANALOGOUS" if score >= 0.90 else "RELATED"
                            )
                            c_type = (
                                concept.concept_type.value
                                if hasattr(concept.concept_type, "value")
                                else concept.concept_type
                            )
                            lines.append(f"- **{concept.name}** [{link_type}] ({c_type})")
                            lines.append(f"  - Similarity: {score:.3f}")
                            lines.append(f"  - ID: `{concept.id}`")

                        return "\n".join(lines)

                return f"**No cross-domain matches found** for `{concept_name}` in {target_domain}"

            # General discovery between domains
            links = await CrossDomainStore.discover_links(
                source_domain=source_domain,
                target_domain=target_domain,
                similarity_threshold=similarity_threshold,
                limit_per_concept=3,
                source_concept_limit=limit * 2,
            )

            if not links:
                return f"**No cross-domain links discovered** between {source_domain} and {target_domain}"

            # Sort by similarity and take top results
            links.sort(key=lambda x: x["similarity"], reverse=True)
            links = links[:limit]

            lines = [f"## Cross-Domain Discovery: {source_domain} ↔ {target_domain}"]
            lines.append(f"*{len(links)} matches above {similarity_threshold:.2f} threshold*\n")

            for link in links:
                link_type = (
                    "EQUIVALENT"
                    if link["similarity"] >= 0.95
                    else "ANALOGOUS" if link["similarity"] >= 0.90 else "RELATED"
                )
                lines.append(
                    f"- **{link['source_name']}** ↔ **{link['target_name']}** [{link_type}]"
                )
                lines.append(f"  - Similarity: {link['similarity']:.3f}")

            return "\n".join(lines)

        except ValueError as e:
            return f"**Error:** Invalid parameter: {e}"
        except Exception as e:
            logger.error("cross_domain_tool_failed", error=str(e))
            return f"**Error:** Failed to find cross-domain concepts: {e}"
