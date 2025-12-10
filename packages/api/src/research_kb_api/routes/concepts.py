"""Concept endpoints.

Provides access to extracted concepts and their relationships in the knowledge graph.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from research_kb_api import schemas
from research_kb_api import service

router = APIRouter()


@router.get("", response_model=schemas.ConceptListResponse)
async def list_concepts(
    query: Optional[str] = Query(None, description="Search query for concept names"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    concept_type: Optional[str] = Query(None, description="Filter by concept type"),
) -> schemas.ConceptListResponse:
    """List or search concepts.

    Parameters
    ----------
    query : str, optional
        Search query for fuzzy matching concept names
    limit : int
        Maximum number of concepts to return
    concept_type : str, optional
        Filter by type (method, assumption, problem, definition, theorem)

    Returns
    -------
    ConceptListResponse
        List of concepts with metadata.
    """
    concepts = await service.get_concepts(
        query=query,
        limit=limit,
        concept_type=concept_type,
    )

    return schemas.ConceptListResponse(
        concepts=[
            schemas.ConceptDetail(
                id=str(c.id),
                name=c.name,
                canonical_name=c.canonical_name,
                concept_type=schemas.ConceptType(c.concept_type.value) if c.concept_type else None,
                definition=c.definition,
                aliases=c.aliases or [],
            )
            for c in concepts
        ],
        total=len(concepts),
    )


@router.get("/{concept_id}", response_model=schemas.ConceptWithRelationships)
async def get_concept(concept_id: str) -> schemas.ConceptWithRelationships:
    """Get concept details with relationships.

    Parameters
    ----------
    concept_id : str
        UUID of the concept

    Returns
    -------
    ConceptWithRelationships
        Concept details with all incoming and outgoing relationships.

    Raises
    ------
    HTTPException
        404 if concept not found
    """
    concept = await service.get_concept_by_id(concept_id)
    if not concept:
        raise HTTPException(status_code=404, detail=f"Concept {concept_id} not found")

    relationships = await service.get_concept_relationships(concept_id)

    return schemas.ConceptWithRelationships(
        concept=schemas.ConceptDetail(
            id=str(concept.id),
            name=concept.name,
            canonical_name=concept.canonical_name,
            concept_type=schemas.ConceptType(concept.concept_type.value) if concept.concept_type else None,
            definition=concept.definition,
            aliases=concept.aliases or [],
        ),
        relationships=[
            schemas.RelationshipDetail(
                id=str(r.id),
                source_id=str(r.source_concept_id),
                source_name=r.source_name if hasattr(r, "source_name") else "",
                target_id=str(r.target_concept_id),
                target_name=r.target_name if hasattr(r, "target_name") else "",
                relationship_type=schemas.RelationshipType(r.relationship_type.value) if r.relationship_type else None,
                confidence=r.confidence_score,
            )
            for r in relationships
        ],
    )
