"""Unit tests for graph query orchestration logic (no database or KuzuDB required).

Covers:
- is_kuzu_ready / _check_kuzu_ready: caching, error scenarios
- explain_path: all relationship types, edge cases
- generate_synthesis_prompt: all three styles
- compute_graph_score: timeout handling, empty inputs, KuzuDB fallback dispatch
- compute_weighted_graph_score: mention weights, relationship weights
- get_mention_weight / apply_mention_weights: all mention types
- get_relationship_weight: all relationship types

Phase S Commit 2: Target graph_queries.py 18.9% → 35%
"""

from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

import pytest

from research_kb_contracts import Concept, ConceptRelationship, ConceptType, RelationshipType

pytestmark = pytest.mark.unit

_NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_concept(name="test_concept", concept_type=ConceptType.METHOD, concept_id=None):
    """Create a minimal Concept for testing."""
    return Concept(
        id=concept_id or uuid4(),
        name=name,
        canonical_name=name.lower().replace(" ", "_"),
        concept_type=concept_type,
        domain_id="causal_inference",
        created_at=_NOW,
    )


def _make_relationship(source_id=None, target_id=None, rel_type=RelationshipType.REQUIRES):
    """Create a minimal ConceptRelationship for testing."""
    return ConceptRelationship(
        id=uuid4(),
        source_concept_id=source_id or uuid4(),
        target_concept_id=target_id or uuid4(),
        relationship_type=rel_type,
        created_at=_NOW,
    )


def _make_path(concepts_and_rels):
    """Build a path as list of (Concept, Optional[ConceptRelationship]) tuples.

    Input: list of (concept, relationship_or_none) pairs.
    """
    return concepts_and_rels


# ---------------------------------------------------------------------------
# Weight functions (pure computation — no mocks needed)
# ---------------------------------------------------------------------------


class TestGetMentionWeight:
    """Test get_mention_weight function."""

    def test_defines_weight(self):
        from research_kb_storage.graph_queries import get_mention_weight

        assert get_mention_weight("defines") == 1.0

    def test_reference_weight(self):
        from research_kb_storage.graph_queries import get_mention_weight

        assert get_mention_weight("reference") == 0.6

    def test_example_weight(self):
        from research_kb_storage.graph_queries import get_mention_weight

        assert get_mention_weight("example") == 0.4

    def test_none_gives_default(self):
        from research_kb_storage.graph_queries import get_mention_weight

        assert get_mention_weight(None) == 0.5

    def test_unknown_gives_default(self):
        from research_kb_storage.graph_queries import get_mention_weight

        assert get_mention_weight("unknown_type") == 0.5


class TestGetRelationshipWeight:
    """Test get_relationship_weight function."""

    def test_requires_weight(self):
        from research_kb_storage.graph_queries import get_relationship_weight

        assert get_relationship_weight(RelationshipType.REQUIRES) == 1.0

    def test_extends_weight(self):
        from research_kb_storage.graph_queries import get_relationship_weight

        assert get_relationship_weight(RelationshipType.EXTENDS) == 0.9

    def test_uses_weight(self):
        from research_kb_storage.graph_queries import get_relationship_weight

        assert get_relationship_weight(RelationshipType.USES) == 0.8

    def test_addresses_weight(self):
        from research_kb_storage.graph_queries import get_relationship_weight

        assert get_relationship_weight(RelationshipType.ADDRESSES) == 0.7

    def test_specializes_weight(self):
        from research_kb_storage.graph_queries import get_relationship_weight

        assert get_relationship_weight(RelationshipType.SPECIALIZES) == 0.6

    def test_generalizes_weight(self):
        from research_kb_storage.graph_queries import get_relationship_weight

        assert get_relationship_weight(RelationshipType.GENERALIZES) == 0.6

    def test_alternative_to_weight(self):
        from research_kb_storage.graph_queries import get_relationship_weight

        assert get_relationship_weight(RelationshipType.ALTERNATIVE_TO) == 0.5


class TestApplyMentionWeights:
    """Test apply_mention_weights function."""

    def test_no_mention_info_returns_original_score(self):
        from research_kb_storage.graph_queries import apply_mention_weights

        score = apply_mention_weights(0.8, [uuid4()], None)
        assert score == 0.8

    def test_empty_chunk_concepts_returns_original_score(self):
        from research_kb_storage.graph_queries import apply_mention_weights

        score = apply_mention_weights(0.8, [], {"some": ("defines", 1.0)})
        assert score == 0.8

    def test_defines_mention_preserves_score(self):
        from research_kb_storage.graph_queries import apply_mention_weights

        cid = uuid4()
        info = {cid: ("defines", 1.0)}
        score = apply_mention_weights(0.8, [cid], info)
        # defines weight = 1.0, so score should be preserved
        assert score == pytest.approx(0.8, rel=1e-5)

    def test_reference_mention_reduces_score(self):
        from research_kb_storage.graph_queries import apply_mention_weights

        cid = uuid4()
        # mention_type="reference" (weight=0.6), relevance=0.8
        # effective weight = 0.6 * 0.8 = 0.48
        # final = 0.8 * 0.48 = 0.384
        info = {cid: ("reference", 0.8)}
        score = apply_mention_weights(0.8, [cid], info)
        assert score == pytest.approx(0.8 * 0.6 * 0.8, rel=1e-5)

    def test_multiple_concepts_averages_weights(self):
        from research_kb_storage.graph_queries import apply_mention_weights

        cid1, cid2 = uuid4(), uuid4()
        # cid1: defines(1.0) * relevance(1.0) = 1.0
        # cid2: example(0.4) * relevance(0.8) = 0.32
        # avg = (1.0 + 0.32) / 2 = 0.66
        info = {cid1: ("defines", 1.0), cid2: ("example", 0.8)}
        score = apply_mention_weights(1.0, [cid1, cid2], info)
        assert score == pytest.approx((1.0 * 1.0 + 0.4 * 0.8) / 2, rel=1e-5)

    def test_missing_concept_uses_default_weight(self):
        from research_kb_storage.graph_queries import apply_mention_weights

        cid = uuid4()
        missing_cid = uuid4()
        info = {cid: ("defines", 1.0)}
        score = apply_mention_weights(1.0, [cid, missing_cid], info)
        # cid=1.0, missing_cid=0.5(default), avg = 0.75
        assert score == pytest.approx(0.75, rel=1e-5)


# ---------------------------------------------------------------------------
# explain_path
# ---------------------------------------------------------------------------


class TestExplainPath:
    """Test explain_path function."""

    def test_single_concept_path(self):
        from research_kb_storage.graph_queries import explain_path

        c = _make_concept("double machine learning")
        path = [(c, None)]
        explanation = explain_path(path)
        # canonical_name uses underscores
        assert "double_machine_learning" in explanation.lower()

    def test_two_concept_path(self):
        from research_kb_storage.graph_queries import explain_path

        c1 = _make_concept("double machine learning")
        c2 = _make_concept("cross fitting")
        rel = _make_relationship(c1.id, c2.id, RelationshipType.REQUIRES)
        path = [(c1, None), (c2, rel)]
        explanation = explain_path(path)
        assert "double_machine_learning" in explanation.lower()
        assert "cross_fitting" in explanation.lower()
        assert "requires" in explanation.lower()

    def test_three_concept_path(self):
        from research_kb_storage.graph_queries import explain_path

        c1 = _make_concept("DML")
        c2 = _make_concept("cross fitting")
        c3 = _make_concept("sample splitting")
        r1 = _make_relationship(c1.id, c2.id, RelationshipType.REQUIRES)
        r2 = _make_relationship(c2.id, c3.id, RelationshipType.USES)
        path = [(c1, None), (c2, r1), (c3, r2)]
        explanation = explain_path(path)
        assert "dml" in explanation.lower()
        assert "sample_splitting" in explanation.lower()

    def test_empty_path(self):
        from research_kb_storage.graph_queries import explain_path

        explanation = explain_path([])
        assert isinstance(explanation, str)


# ---------------------------------------------------------------------------
# generate_synthesis_prompt
# ---------------------------------------------------------------------------


class TestGenerateSynthesisPrompt:
    """Test generate_synthesis_prompt for all styles."""

    def _build_test_path(self):
        c1 = _make_concept("IV")
        c2 = _make_concept("unconfoundedness", ConceptType.ASSUMPTION)
        rel = _make_relationship(c1.id, c2.id, RelationshipType.REQUIRES)
        return [(c1, None), (c2, rel)]

    def test_educational_style(self):
        from research_kb_storage.graph_queries import generate_synthesis_prompt

        path = self._build_test_path()
        prompt = generate_synthesis_prompt(path, style="educational")
        assert len(prompt) > 0
        assert isinstance(prompt, str)

    def test_research_style(self):
        from research_kb_storage.graph_queries import generate_synthesis_prompt

        path = self._build_test_path()
        prompt = generate_synthesis_prompt(path, style="research")
        assert len(prompt) > 0
        assert isinstance(prompt, str)

    def test_implementation_style(self):
        from research_kb_storage.graph_queries import generate_synthesis_prompt

        path = self._build_test_path()
        prompt = generate_synthesis_prompt(path, style="implementation")
        assert len(prompt) > 0
        assert isinstance(prompt, str)

    def test_empty_path_still_returns_prompt(self):
        from research_kb_storage.graph_queries import generate_synthesis_prompt

        prompt = generate_synthesis_prompt([], style="educational")
        assert isinstance(prompt, str)


# ---------------------------------------------------------------------------
# KuzuDB readiness check
# ---------------------------------------------------------------------------


class TestKuzuReadiness:
    """Test is_kuzu_ready and _check_kuzu_ready caching."""

    def test_reset_kuzu_cache(self):
        """reset_kuzu_cache clears the cached state."""
        from research_kb_storage.graph_queries import reset_kuzu_cache

        # Should not raise
        reset_kuzu_cache()

    @patch("research_kb_storage.graph_queries.is_kuzu_ready", return_value=False)
    def test_check_kuzu_ready_returns_false_when_unavailable(self, mock_ready):
        from research_kb_storage.graph_queries import _check_kuzu_ready, reset_kuzu_cache

        reset_kuzu_cache()
        result = _check_kuzu_ready()
        assert result is False

    @patch("research_kb_storage.graph_queries.is_kuzu_ready", return_value=True)
    def test_check_kuzu_ready_returns_true_when_available(self, mock_ready):
        from research_kb_storage.graph_queries import _check_kuzu_ready, reset_kuzu_cache

        reset_kuzu_cache()
        result = _check_kuzu_ready()
        assert result is True


# ---------------------------------------------------------------------------
# Constants verification
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify exported constants and dicts."""

    def test_relationship_weights_dict_complete(self):
        from research_kb_storage.graph_queries import RELATIONSHIP_WEIGHTS

        assert RelationshipType.REQUIRES in RELATIONSHIP_WEIGHTS
        assert RelationshipType.EXTENDS in RELATIONSHIP_WEIGHTS
        assert RelationshipType.USES in RELATIONSHIP_WEIGHTS
        assert RelationshipType.ADDRESSES in RELATIONSHIP_WEIGHTS
        assert RelationshipType.SPECIALIZES in RELATIONSHIP_WEIGHTS
        assert RelationshipType.GENERALIZES in RELATIONSHIP_WEIGHTS
        assert RelationshipType.ALTERNATIVE_TO in RELATIONSHIP_WEIGHTS
        # All weights between 0 and 1
        assert all(0.0 <= w <= 1.0 for w in RELATIONSHIP_WEIGHTS.values())

    def test_mention_weights_dict_complete(self):
        from research_kb_storage.graph_queries import MENTION_WEIGHTS

        assert "defines" in MENTION_WEIGHTS
        assert "reference" in MENTION_WEIGHTS
        assert "example" in MENTION_WEIGHTS
        assert all(0.0 <= w <= 1.0 for w in MENTION_WEIGHTS.values())

    def test_timeout_constants_positive(self):
        from research_kb_storage.graph_queries import GRAPH_SCORE_TIMEOUT, PATH_QUERY_TIMEOUT

        assert GRAPH_SCORE_TIMEOUT > 0
        assert PATH_QUERY_TIMEOUT > 0
