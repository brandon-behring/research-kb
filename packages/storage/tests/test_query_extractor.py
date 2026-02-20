"""Tests for query_extractor - extract concepts from user queries."""

import pytest
from uuid import uuid4

from research_kb_contracts import ConceptType
from research_kb_storage import ConceptStore
from research_kb_storage.query_extractor import (
    extract_query_concepts,
    extract_query_concepts_by_similarity,
    extract_query_concepts_unified,
)

pytestmark = pytest.mark.unit


class TestExtractQueryConcepts:
    """Tests for extract_query_concepts function."""

    async def test_extract_empty_query(self, test_db):
        """Empty query returns empty list."""
        result = await extract_query_concepts("")
        assert result == []

    async def test_extract_whitespace_query(self, test_db):
        """Whitespace-only query returns empty list."""
        result = await extract_query_concepts("   ")
        assert result == []

    async def test_extract_exact_match(self, test_db):
        """Exact canonical_name match returns concept ID."""
        # Create a concept
        concept = await ConceptStore.create(
            name="Instrumental Variables",
            canonical_name="instrumental variables",
            concept_type=ConceptType.METHOD,
        )

        # Query with exact match
        result = await extract_query_concepts("instrumental variables")

        assert len(result) == 1
        assert result[0] == concept.id

    async def test_extract_case_insensitive(self, test_db):
        """Match is case-insensitive."""
        concept = await ConceptStore.create(
            name="Double Machine Learning",
            canonical_name="double machine learning",
            concept_type=ConceptType.METHOD,
        )

        result = await extract_query_concepts("DOUBLE MACHINE LEARNING")

        assert len(result) == 1
        assert result[0] == concept.id

    async def test_extract_substring_match(self, test_db):
        """Substring matching for longer concept names."""
        concept = await ConceptStore.create(
            name="Propensity Score",
            canonical_name="propensity score",
            concept_type=ConceptType.METHOD,
        )

        # Query containing the concept name
        result = await extract_query_concepts("what is propensity score matching")

        assert len(result) >= 1
        assert concept.id in result

    async def test_extract_respects_max_concepts(self, test_db):
        """Respects max_concepts parameter."""
        # Create multiple concepts
        for i in range(10):
            await ConceptStore.create(
                name=f"Method {i}",
                canonical_name=f"method_{i}",
                concept_type=ConceptType.METHOD,
            )

        # Request with low limit
        result = await extract_query_concepts("method", max_concepts=3)

        assert len(result) <= 3

    async def test_extract_no_match(self, test_db):
        """Query with no matching concepts returns empty list."""
        await ConceptStore.create(
            name="Regression",
            canonical_name="regression",
            concept_type=ConceptType.METHOD,
        )

        result = await extract_query_concepts("xyzzy nonexistent")

        assert result == []

    async def test_extract_multiple_concepts(self, test_db):
        """Query matching multiple concepts returns all."""
        concept1 = await ConceptStore.create(
            name="IV",
            canonical_name="iv",
            concept_type=ConceptType.METHOD,
        )
        concept2 = await ConceptStore.create(
            name="Endogeneity",
            canonical_name="endogeneity",
            concept_type=ConceptType.PROBLEM,
        )

        result = await extract_query_concepts("iv for endogeneity")

        assert len(result) == 2
        assert concept1.id in result
        assert concept2.id in result

    async def test_extract_graceful_failure(self, test_db):
        """Handles errors gracefully - returns empty list."""
        # Should not raise even with unusual input
        result = await extract_query_concepts(None)  # type: ignore
        # This might return [] due to the 'not query_text' check
        assert result == [] or isinstance(result, list)


class TestExtractQueryConceptsBySimilarity:
    """Tests for extract_query_concepts_by_similarity function."""

    async def test_invalid_embedding_empty(self, test_db):
        """Empty embedding returns empty list."""
        result = await extract_query_concepts_by_similarity([])
        assert result == []

    async def test_invalid_embedding_wrong_dimension(self, test_db):
        """Wrong dimension embedding returns empty list."""
        result = await extract_query_concepts_by_similarity([0.1] * 512)
        assert result == []

    async def test_valid_embedding_with_concepts(self, test_db):
        """Valid embedding finds similar concepts."""
        # Create concept with embedding
        embedding = [0.1] * 1024
        concept = await ConceptStore.create(
            name="Test Concept",
            canonical_name=f"test_concept_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
            embedding=embedding,
        )

        # Query with similar embedding (same vector should have similarity ~1.0)
        result = await extract_query_concepts_by_similarity(
            query_embedding=embedding,
            min_similarity=0.9,
        )

        assert len(result) >= 1
        assert concept.id in result

    async def test_similarity_threshold_filtering(self, test_db):
        """Low similarity concepts are filtered out."""
        # Create concept with one embedding
        embedding1 = [1.0] + [0.0] * 1023
        concept = await ConceptStore.create(
            name="Different Concept",
            canonical_name=f"different_concept_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
            embedding=embedding1,
        )

        # Query with very different embedding
        embedding2 = [0.0] * 1023 + [1.0]
        result = await extract_query_concepts_by_similarity(
            query_embedding=embedding2,
            min_similarity=0.99,  # Very high threshold
        )

        # Should not match due to low similarity
        assert concept.id not in result

    async def test_respects_max_concepts(self, test_db):
        """Respects max_concepts parameter."""
        embedding = [0.5] * 1024

        # Create multiple concepts with same embedding
        for i in range(10):
            await ConceptStore.create(
                name=f"Similar {i}",
                canonical_name=f"similar_{i}_{uuid4().hex[:8]}",
                concept_type=ConceptType.METHOD,
                embedding=embedding,
            )

        result = await extract_query_concepts_by_similarity(
            query_embedding=embedding,
            min_similarity=0.9,
            max_concepts=3,
        )

        assert len(result) <= 3

    async def test_no_concepts_with_embeddings(self, test_db):
        """Returns empty when no concepts have embeddings."""
        # Create concept without embedding
        await ConceptStore.create(
            name="No Embedding",
            canonical_name=f"no_embedding_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
            # No embedding parameter
        )

        result = await extract_query_concepts_by_similarity(
            query_embedding=[0.1] * 1024,
            min_similarity=0.5,
        )

        assert result == []


class TestExtractQueryConceptsUnified:
    """Tests for extract_query_concepts_unified function."""

    async def test_unified_empty_query(self, test_db):
        """Empty query returns empty list."""
        result = await extract_query_concepts_unified("")
        assert result == []

    async def test_unified_text_only(self, test_db):
        """Unified extraction with text matching only."""
        concept = await ConceptStore.create(
            name="Propensity Score Matching",
            canonical_name="propensity score",
            concept_type=ConceptType.METHOD,
        )

        result = await extract_query_concepts_unified(
            "propensity score estimation",
            use_text_match=True,
            use_semantic=False,
        )

        assert len(result) >= 1
        assert concept.id in result

    async def test_unified_semantic_only(self, test_db):
        """Unified extraction with semantic matching only."""
        embedding = [0.5] * 1024
        concept = await ConceptStore.create(
            name="Semantic Test",
            canonical_name=f"semantic_test_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
            embedding=embedding,
        )

        result = await extract_query_concepts_unified(
            "semantic test query",
            query_embedding=embedding,
            use_text_match=False,
            use_semantic=True,
            min_semantic_similarity=0.9,
        )

        assert len(result) >= 1
        assert concept.id in result

    async def test_unified_both_strategies(self, test_db):
        """Unified extraction combines both strategies."""
        embedding = [0.3] * 1024

        # Text-matchable concept
        concept1 = await ConceptStore.create(
            name="IV Method",
            canonical_name="instrumental variables",
            concept_type=ConceptType.METHOD,
        )

        # Semantically similar concept (different name)
        concept2 = await ConceptStore.create(
            name="Different Name",
            canonical_name=f"different_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
            embedding=embedding,
        )

        result = await extract_query_concepts_unified(
            "instrumental variables usage",
            query_embedding=embedding,
            use_text_match=True,
            use_semantic=True,
            min_semantic_similarity=0.9,
        )

        # Should find both via different strategies
        assert concept1.id in result
        assert concept2.id in result

    async def test_unified_deduplication(self, test_db):
        """Unified extraction deduplicates results."""
        embedding = [0.6] * 1024

        # Concept matching both text AND semantic
        concept = await ConceptStore.create(
            name="Matching Both",
            canonical_name="matching both",
            concept_type=ConceptType.METHOD,
            embedding=embedding,
        )

        result = await extract_query_concepts_unified(
            "matching both strategies",
            query_embedding=embedding,
            use_text_match=True,
            use_semantic=True,
            min_semantic_similarity=0.9,
        )

        # Should only appear once despite matching both
        assert result.count(concept.id) == 1

    async def test_unified_respects_max_concepts(self, test_db):
        """Unified extraction respects max_concepts."""
        embedding = [0.4] * 1024

        for i in range(10):
            await ConceptStore.create(
                name=f"Test {i}",
                canonical_name=f"test_{i}",
                concept_type=ConceptType.METHOD,
                embedding=embedding,
            )

        result = await extract_query_concepts_unified(
            "test concepts",
            query_embedding=embedding,
            max_concepts=3,
        )

        assert len(result) <= 3

    async def test_unified_graceful_without_embedding(self, test_db):
        """Unified extraction works without embedding (text only)."""
        concept = await ConceptStore.create(
            name="Text Only",
            canonical_name="text only concept",
            concept_type=ConceptType.METHOD,
        )

        # No embedding provided, should fall back to text only
        result = await extract_query_concepts_unified(
            "text only concept search",
            query_embedding=None,
            use_text_match=True,
            use_semantic=True,  # Ignored without embedding
        )

        assert len(result) >= 1
        assert concept.id in result
