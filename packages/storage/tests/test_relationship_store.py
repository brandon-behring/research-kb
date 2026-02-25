"""Tests for RelationshipStore - CRUD operations for concept relationships."""

import pytest
from uuid import uuid4

from research_kb_common import StorageError
from research_kb_contracts import RelationshipType, ConceptType
from research_kb_storage import RelationshipStore, ConceptStore

pytestmark = pytest.mark.unit


async def test_create_relationship(test_db):
    """Test creating a relationship between concepts."""
    # Create two concepts first
    concept1 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Instrumental Variables",
        canonical_name="instrumental_variables",
        concept_type=ConceptType.METHOD,
    )
    concept2 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Endogeneity",
        canonical_name="endogeneity",
        concept_type=ConceptType.PROBLEM,
    )

    # Create relationship
    relationship = await RelationshipStore.create(
        source_concept_id=concept1.id,
        target_concept_id=concept2.id,
        relationship_type=RelationshipType.ADDRESSES,
        strength=0.9,
        confidence_score=0.85,
    )

    assert relationship.id is not None
    assert relationship.source_concept_id == concept1.id
    assert relationship.target_concept_id == concept2.id
    assert relationship.relationship_type == RelationshipType.ADDRESSES
    assert relationship.strength == pytest.approx(0.9, rel=1e-5)
    assert relationship.confidence_score == pytest.approx(0.85, rel=1e-5)
    assert relationship.is_directed is True
    assert relationship.created_at is not None


async def test_create_relationship_with_evidence(test_db):
    """Test creating a relationship with evidence chunks."""
    # Create concepts
    concept1 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Regression Discontinuity",
        canonical_name="regression_discontinuity",
        concept_type=ConceptType.METHOD,
    )
    concept2 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Treatment Effect",
        canonical_name="treatment_effect",
        concept_type=ConceptType.DEFINITION,
    )

    # Create relationship with evidence
    chunk_ids = [uuid4(), uuid4()]
    relationship = await RelationshipStore.create(
        source_concept_id=concept1.id,
        target_concept_id=concept2.id,
        relationship_type=RelationshipType.USES,
        evidence_chunk_ids=chunk_ids,
    )

    assert relationship.evidence_chunk_ids == chunk_ids


async def test_create_undirected_relationship(test_db):
    """Test creating an undirected relationship."""
    # Create concepts
    concept1 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Correlation",
        canonical_name="correlation",
        concept_type=ConceptType.DEFINITION,
    )
    concept2 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Causation",
        canonical_name="causation",
        concept_type=ConceptType.DEFINITION,
    )

    # Create undirected relationship
    relationship = await RelationshipStore.create(
        source_concept_id=concept1.id,
        target_concept_id=concept2.id,
        relationship_type=RelationshipType.ALTERNATIVE_TO,
        is_directed=False,
    )

    assert relationship.is_directed is False


async def test_create_duplicate_relationship_fails(test_db):
    """Test that creating duplicate relationships fails."""
    # Create concepts
    concept1 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Panel Data",
        canonical_name="panel_data",
        concept_type=ConceptType.DEFINITION,
    )
    concept2 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Fixed Effects",
        canonical_name="fixed_effects",
        concept_type=ConceptType.METHOD,
    )

    # Create first relationship
    await RelationshipStore.create(
        source_concept_id=concept1.id,
        target_concept_id=concept2.id,
        relationship_type=RelationshipType.USES,
    )

    # Try to create duplicate
    with pytest.raises(StorageError, match="already exists"):
        await RelationshipStore.create(
            source_concept_id=concept1.id,
            target_concept_id=concept2.id,
            relationship_type=RelationshipType.USES,
        )


async def test_create_relationship_missing_concept_fails(test_db):
    """Test that creating relationship with non-existent concept fails."""
    # Create one concept
    concept1 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Real Concept",
        canonical_name="real_concept",
        concept_type=ConceptType.DEFINITION,
    )

    # Try to create relationship with non-existent target
    fake_id = uuid4()
    with pytest.raises(StorageError, match="do not exist"):
        await RelationshipStore.create(
            source_concept_id=concept1.id,
            target_concept_id=fake_id,
            relationship_type=RelationshipType.USES,
        )


async def test_get_by_id(test_db):
    """Test retrieving relationship by ID."""
    # Create concepts and relationship
    concept1 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Difference in Differences",
        canonical_name="difference_in_differences",
        concept_type=ConceptType.METHOD,
    )
    concept2 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Parallel Trends",
        canonical_name="parallel_trends",
        concept_type=ConceptType.ASSUMPTION,
    )

    created = await RelationshipStore.create(
        source_concept_id=concept1.id,
        target_concept_id=concept2.id,
        relationship_type=RelationshipType.REQUIRES,
    )

    # Retrieve by ID
    retrieved = await RelationshipStore.get_by_id(created.id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.source_concept_id == concept1.id
    assert retrieved.target_concept_id == concept2.id
    assert retrieved.relationship_type == RelationshipType.REQUIRES


async def test_get_by_id_not_found(test_db):
    """Test retrieving non-existent relationship returns None."""
    fake_id = uuid4()
    result = await RelationshipStore.get_by_id(fake_id)
    assert result is None


async def test_get_by_concepts(test_db):
    """Test retrieving relationship by concept pair."""
    # Create concepts
    concept1 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Two Stage Least Squares",
        canonical_name="two_stage_least_squares",
        concept_type=ConceptType.METHOD,
    )
    concept2 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Instrumental Variables",
        canonical_name="instrumental_variables",
        concept_type=ConceptType.METHOD,
    )

    # Create relationship
    await RelationshipStore.create(
        source_concept_id=concept1.id,
        target_concept_id=concept2.id,
        relationship_type=RelationshipType.SPECIALIZES,
    )

    # Retrieve by concepts
    retrieved = await RelationshipStore.get_by_concepts(
        source_concept_id=concept1.id, target_concept_id=concept2.id
    )

    assert retrieved is not None
    assert retrieved.source_concept_id == concept1.id
    assert retrieved.target_concept_id == concept2.id


async def test_get_by_concepts_with_type(test_db):
    """Test retrieving relationship by concept pair and type."""
    # Create concepts
    concept1 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Matching",
        canonical_name="matching",
        concept_type=ConceptType.METHOD,
    )
    concept2 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Selection Bias",
        canonical_name="selection_bias",
        concept_type=ConceptType.PROBLEM,
    )

    # Create two different relationship types
    await RelationshipStore.create(
        source_concept_id=concept1.id,
        target_concept_id=concept2.id,
        relationship_type=RelationshipType.ADDRESSES,
    )
    await RelationshipStore.create(
        source_concept_id=concept1.id,
        target_concept_id=concept2.id,
        relationship_type=RelationshipType.USES,
    )

    # Retrieve specific type
    retrieved = await RelationshipStore.get_by_concepts(
        source_concept_id=concept1.id,
        target_concept_id=concept2.id,
        relationship_type=RelationshipType.ADDRESSES,
    )

    assert retrieved is not None
    assert retrieved.relationship_type == RelationshipType.ADDRESSES


async def test_list_from_concept(test_db):
    """Test listing outgoing relationships from a concept."""
    # Create concepts
    source = await ConceptStore.create(
        domain_id="causal_inference",
        name="Propensity Score",
        canonical_name="propensity_score",
        concept_type=ConceptType.METHOD,
    )
    target1 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Observational Data",
        canonical_name="observational_data",
        concept_type=ConceptType.DEFINITION,
    )
    target2 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Covariate Balance",
        canonical_name="covariate_balance",
        concept_type=ConceptType.DEFINITION,
    )

    # Create relationships
    await RelationshipStore.create(
        source_concept_id=source.id,
        target_concept_id=target1.id,
        relationship_type=RelationshipType.USES,
        strength=0.9,
    )
    await RelationshipStore.create(
        source_concept_id=source.id,
        target_concept_id=target2.id,
        relationship_type=RelationshipType.USES,
        strength=0.8,
    )

    # List outgoing
    relationships = await RelationshipStore.list_from_concept(source.id)

    assert len(relationships) == 2
    # Should be ordered by strength DESC
    assert relationships[0].strength == pytest.approx(0.9, rel=1e-5)
    assert relationships[1].strength == pytest.approx(0.8, rel=1e-5)


async def test_delete_relationship(test_db):
    """Test deleting a relationship."""
    # Create concepts and relationship
    concept1 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Autocorrelation",
        canonical_name="autocorrelation",
        concept_type=ConceptType.PROBLEM,
    )
    concept2 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Time Series",
        canonical_name="time_series",
        concept_type=ConceptType.DEFINITION,
    )

    relationship = await RelationshipStore.create(
        source_concept_id=concept1.id,
        target_concept_id=concept2.id,
        relationship_type=RelationshipType.USES,
    )

    # Delete
    deleted = await RelationshipStore.delete(relationship.id)
    assert deleted is True

    # Verify deleted
    result = await RelationshipStore.get_by_id(relationship.id)
    assert result is None


async def test_delete_nonexistent_relationship(test_db):
    """Test deleting non-existent relationship returns False."""
    fake_id = uuid4()
    deleted = await RelationshipStore.delete(fake_id)
    assert deleted is False


async def test_count_relationships(test_db):
    """Test counting total relationships."""
    # Initially zero
    count = await RelationshipStore.count()
    assert count == 0

    # Create some relationships
    concept1 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Test Concept 1",
        canonical_name="test_concept_1",
        concept_type=ConceptType.DEFINITION,
    )
    concept2 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Test Concept 2",
        canonical_name="test_concept_2",
        concept_type=ConceptType.DEFINITION,
    )

    await RelationshipStore.create(
        source_concept_id=concept1.id,
        target_concept_id=concept2.id,
        relationship_type=RelationshipType.USES,
    )

    count = await RelationshipStore.count()
    assert count == 1


async def test_batch_create_relationships(test_db):
    """Test batch creating multiple relationships."""
    # Create concepts
    concept1 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Concept A",
        canonical_name="concept_a",
        concept_type=ConceptType.DEFINITION,
    )
    concept2 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Concept B",
        canonical_name="concept_b",
        concept_type=ConceptType.DEFINITION,
    )
    concept3 = await ConceptStore.create(
        domain_id="causal_inference",
        name="Concept C",
        canonical_name="concept_c",
        concept_type=ConceptType.DEFINITION,
    )

    # Batch create
    relationships_data = [
        {
            "source_concept_id": concept1.id,
            "target_concept_id": concept2.id,
            "relationship_type": "USES",
            "strength": 0.9,
        },
        {
            "source_concept_id": concept2.id,
            "target_concept_id": concept3.id,
            "relationship_type": "EXTENDS",
            "strength": 0.8,
        },
    ]

    created = await RelationshipStore.batch_create(relationships_data)

    assert len(created) == 2
    assert created[0].strength == pytest.approx(0.9, rel=1e-5)
    assert created[1].strength == pytest.approx(0.8, rel=1e-5)


async def test_batch_create_empty_list(test_db):
    """Test batch create with empty list returns empty list."""
    created = await RelationshipStore.batch_create([])
    assert created == []
