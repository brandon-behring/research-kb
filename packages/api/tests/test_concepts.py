"""Tests for concepts endpoints."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock
from uuid import uuid4
from datetime import datetime

from research_kb_contracts import Concept, ConceptRelationship, ConceptType, RelationshipType


def make_concept(name: str = "instrumental variables") -> Concept:
    """Create a mock concept."""
    return Concept(
        id=uuid4(),
        name=name,
        canonical_name=name.lower().replace(" ", "_"),
        concept_type=ConceptType.METHOD,
        definition=f"Definition of {name}",
        aliases=[],
        created_at=datetime.now(),
    )


def make_relationship(source_id, target_id):
    """Create a mock relationship with extra attributes for route handlers."""
    from unittest.mock import MagicMock

    rel = MagicMock()
    rel.id = uuid4()
    rel.source_concept_id = source_id
    rel.target_concept_id = target_id
    rel.relationship_type = RelationshipType.REQUIRES
    rel.confidence_score = 0.9
    rel.source_name = "Source Concept"
    rel.target_name = "Target Concept"
    return rel


@pytest.mark.asyncio
async def test_list_concepts(app_client, mock_storage):
    """List concepts returns all concepts."""
    concepts = [make_concept(f"concept {i}") for i in range(3)]
    mock_storage["concept"].list_all.return_value = concepts

    response = await app_client.get("/concepts")

    assert response.status_code == 200
    data = response.json()
    assert len(data["concepts"]) == 3
    assert data["total"] == 3


@pytest.mark.asyncio
async def test_search_concepts(app_client, mock_storage):
    """Search concepts by query."""
    concepts = [make_concept("instrumental variables")]
    mock_storage["concept"].search.return_value = concepts

    response = await app_client.get("/concepts?query=instrumental")

    assert response.status_code == 200
    data = response.json()
    assert len(data["concepts"]) == 1
    assert "instrumental" in data["concepts"][0]["name"]


@pytest.mark.asyncio
async def test_get_concept(app_client, mock_storage):
    """Get concept by ID returns concept with relationships."""
    concept = make_concept()
    relationships = [make_relationship(concept.id, uuid4())]

    mock_storage["concept"].get.return_value = concept
    mock_storage["relationship"].get_for_concept.return_value = relationships

    response = await app_client.get(f"/concepts/{concept.id}")

    assert response.status_code == 200
    data = response.json()
    assert data["concept"]["name"] == "instrumental variables"
    assert len(data["relationships"]) == 1
    assert data["relationships"][0]["relationship_type"] == "REQUIRES"


@pytest.mark.asyncio
async def test_get_concept_not_found(app_client, mock_storage):
    """Get concept returns 404 for unknown ID."""
    mock_storage["concept"].get.return_value = None

    response = await app_client.get(f"/concepts/{uuid4()}")

    assert response.status_code == 404
