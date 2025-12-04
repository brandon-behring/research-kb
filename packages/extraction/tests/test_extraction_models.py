"""Tests for extraction models."""

import pytest
from uuid import uuid4

from research_kb_extraction.models import (
    ChunkExtraction,
    ConceptMatch,
    ExtractedConcept,
    ExtractedRelationship,
    StoredConcept,
)


class TestExtractedConcept:
    """Tests for ExtractedConcept model."""

    def test_create_minimal(self):
        """Test creating concept with minimal fields."""
        concept = ExtractedConcept(
            name="instrumental variables",
            concept_type="method",
        )

        assert concept.name == "instrumental variables"
        assert concept.concept_type == "method"
        assert concept.definition is None
        assert concept.aliases == []
        assert concept.confidence == 0.8  # Default

    def test_create_full(self):
        """Test creating concept with all fields."""
        concept = ExtractedConcept(
            name="difference-in-differences",
            concept_type="method",
            definition="Compares treatment and control groups before and after",
            aliases=["DiD", "DD", "diff-in-diff"],
            confidence=0.95,
        )

        assert concept.name == "difference-in-differences"
        assert len(concept.aliases) == 3
        assert "DiD" in concept.aliases
        assert concept.confidence == 0.95

    def test_confidence_bounds(self):
        """Test confidence validation."""
        # Valid bounds
        concept = ExtractedConcept(name="test", concept_type="method", confidence=0.0)
        assert concept.confidence == 0.0

        concept = ExtractedConcept(name="test", concept_type="method", confidence=1.0)
        assert concept.confidence == 1.0

        # Invalid bounds should raise
        with pytest.raises(ValueError):
            ExtractedConcept(name="test", concept_type="method", confidence=-0.1)

        with pytest.raises(ValueError):
            ExtractedConcept(name="test", concept_type="method", confidence=1.1)

    def test_invalid_concept_type(self):
        """Test invalid concept type raises error."""
        with pytest.raises(ValueError):
            ExtractedConcept(name="test", concept_type="invalid_type")


class TestExtractedRelationship:
    """Tests for ExtractedRelationship model."""

    def test_create_minimal(self):
        """Test creating relationship with minimal fields."""
        rel = ExtractedRelationship(
            source_concept="IV",
            target_concept="relevance",
            relationship_type="REQUIRES",
        )

        assert rel.source_concept == "IV"
        assert rel.target_concept == "relevance"
        assert rel.relationship_type == "REQUIRES"
        assert rel.confidence == 0.7  # Default

    def test_all_relationship_types(self):
        """Test all valid relationship types."""
        valid_types = [
            "REQUIRES",
            "USES",
            "ADDRESSES",
            "GENERALIZES",
            "SPECIALIZES",
            "ALTERNATIVE_TO",
            "EXTENDS",
        ]

        for rel_type in valid_types:
            rel = ExtractedRelationship(
                source_concept="A",
                target_concept="B",
                relationship_type=rel_type,
            )
            assert rel.relationship_type == rel_type

    def test_invalid_relationship_type(self):
        """Test invalid relationship type raises error."""
        with pytest.raises(ValueError):
            ExtractedRelationship(
                source_concept="A",
                target_concept="B",
                relationship_type="INVALID",
            )


class TestChunkExtraction:
    """Tests for ChunkExtraction model."""

    def test_empty_extraction(self):
        """Test empty extraction."""
        extraction = ChunkExtraction()

        assert extraction.concepts == []
        assert extraction.relationships == []
        assert extraction.concept_count == 0
        assert extraction.relationship_count == 0

    def test_with_concepts(self, sample_extraction):
        """Test extraction with concepts."""
        assert sample_extraction.concept_count == 3
        assert sample_extraction.relationship_count == 2

    def test_get_concepts_by_type(self, sample_extraction):
        """Test filtering concepts by type."""
        methods = sample_extraction.get_concepts_by_type("method")
        assumptions = sample_extraction.get_concepts_by_type("assumption")

        assert len(methods) == 1
        assert methods[0].name == "instrumental variables"
        assert len(assumptions) == 2

    def test_get_high_confidence_concepts(self, sample_extraction):
        """Test filtering by confidence threshold."""
        high_conf = sample_extraction.get_high_confidence_concepts(threshold=0.87)

        # Only concepts with confidence >= 0.87
        assert len(high_conf) == 2
        assert all(c.confidence >= 0.87 for c in high_conf)


class TestConceptMatch:
    """Tests for ConceptMatch model."""

    def test_new_concept_match(self):
        """Test match result for new concept."""
        concept = ExtractedConcept(name="new concept", concept_type="method")
        match = ConceptMatch(
            extracted=concept,
            matched_concept_id=None,
            matched_canonical_name="new concept",
            similarity_score=0.0,
            is_new=True,
        )

        assert match.is_new
        assert not match.should_merge
        assert match.matched_concept_id is None

    def test_existing_concept_match(self):
        """Test match result for existing concept."""
        concept = ExtractedConcept(name="IV", concept_type="method")
        existing_id = uuid4()
        match = ConceptMatch(
            extracted=concept,
            matched_concept_id=existing_id,
            matched_canonical_name="instrumental variables",
            similarity_score=0.98,
            is_new=False,
        )

        assert not match.is_new
        assert match.should_merge
        assert match.matched_concept_id == existing_id

    def test_should_merge_threshold(self):
        """Test merge threshold (0.95)."""
        concept = ExtractedConcept(name="test", concept_type="method")

        # Below threshold - should not merge
        match = ConceptMatch(
            extracted=concept,
            similarity_score=0.94,
            is_new=False,
        )
        assert not match.should_merge

        # At threshold - should not merge (> not >=)
        match = ConceptMatch(
            extracted=concept,
            similarity_score=0.95,
            is_new=False,
        )
        assert not match.should_merge

        # Above threshold - should merge
        match = ConceptMatch(
            extracted=concept,
            similarity_score=0.96,
            is_new=False,
        )
        assert match.should_merge


class TestStoredConcept:
    """Tests for StoredConcept model."""

    def test_create_stored_concept(self):
        """Test creating stored concept with DB fields."""
        concept_id = uuid4()
        concept = StoredConcept(
            id=concept_id,
            name="instrumental variables",
            canonical_name="instrumental variables",
            concept_type="method",
            definition="IV estimation method",
            validated=True,
        )

        assert concept.id == concept_id
        assert concept.canonical_name == "instrumental variables"
        assert concept.validated
