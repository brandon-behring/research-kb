"""Tests for MethodAssumptionAuditor - the North Star assumption audit feature.

Covers the full audit pipeline:
- Dataclass construction and serialization
- Method lookup by name/canonical_name/alias (case-insensitive)
- Graph query for REQUIRES/USES -> ASSUMPTION relationships
- Domain filtering to prevent cross-domain contamination
- Ollama LLM fallback extraction (mocked)
- Cache round-trip (requires method_assumption_cache table)
- Main audit_assumptions orchestration flow
- Docstring snippet generation with importance grouping

Uses real PostgreSQL (research_kb_test) for DB tests.
Mocks httpx for Ollama interaction tests.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from research_kb_contracts import ConceptType, RelationshipType
from research_kb_storage import (
    ConceptStore,
    RelationshipStore,
)
from research_kb_storage.assumption_audit import (
    AssumptionDetail,
    MethodAssumptions,
    MethodAssumptionAuditor,
    _generate_docstring_snippet,
)

pytestmark = pytest.mark.integration


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def method_with_assumptions(db_pool):
    """Create a method concept with 4 linked assumption concepts.

    Graph structure:
        DML --REQUIRES--> unconfoundedness (strength=0.95, confidence=0.9)
        DML --REQUIRES--> overlap (strength=0.90, confidence=0.85)
        DML --USES------> convergence_rate (strength=0.80, confidence=0.8)
        DML --USES------> sample_splitting (strength=0.70, confidence=0.75)

    All concepts share domain_id='causal_inference'.
    """
    method = await ConceptStore.create(
        name="Double Machine Learning",
        canonical_name=f"double_machine_learning_{uuid4().hex[:8]}",
        concept_type=ConceptType.METHOD,
        aliases=["DML", "debiased ML"],
        definition="Uses cross-fitting to debias ML-based nuisance estimates",
        domain_id="causal_inference",
    )

    assumptions = {}
    assumption_data = [
        ("unconfoundedness", "No unmeasured confounders", "REQUIRES", 0.95, 0.9),
        (
            "overlap",
            "Positive probability of treatment for all X",
            "REQUIRES",
            0.90,
            0.85,
        ),
        (
            "convergence_rate",
            "Nuisance estimators converge at n^{-1/4}",
            "USES",
            0.80,
            0.8,
        ),
        (
            "sample_splitting",
            "Independent samples for nuisance and target",
            "USES",
            0.70,
            0.75,
        ),
    ]

    for name, definition, rel_type, strength, confidence in assumption_data:
        concept = await ConceptStore.create(
            name=name,
            canonical_name=f"{name}_{uuid4().hex[:8]}",
            concept_type=ConceptType.ASSUMPTION,
            definition=definition,
            domain_id="causal_inference",
        )
        assumptions[name] = concept

        await RelationshipStore.create(
            source_concept_id=method.id,
            target_concept_id=concept.id,
            relationship_type=RelationshipType(rel_type),
            strength=strength,
            confidence_score=confidence,
        )

    return method, assumptions


@pytest.fixture
async def method_sparse(db_pool):
    """Create a method with only 1 assumption (below MIN_ASSUMPTIONS_THRESHOLD).

    Used to test Ollama fallback triggering.
    """
    method = await ConceptStore.create(
        name="Synthetic Control",
        canonical_name=f"synthetic_control_{uuid4().hex[:8]}",
        concept_type=ConceptType.METHOD,
        definition="Constructs a weighted combination of control units",
        domain_id="causal_inference",
    )

    assumption = await ConceptStore.create(
        name="parallel_trends_synth",
        canonical_name=f"parallel_trends_synth_{uuid4().hex[:8]}",
        concept_type=ConceptType.ASSUMPTION,
        definition="Treatment unit would follow control trend absent treatment",
        domain_id="causal_inference",
    )

    await RelationshipStore.create(
        source_concept_id=method.id,
        target_concept_id=assumption.id,
        relationship_type=RelationshipType.REQUIRES,
        strength=0.95,
        confidence_score=0.9,
    )

    return method, {"parallel_trends_synth": assumption}


@pytest.fixture
async def cross_domain_method(db_pool):
    """Create a method with assumptions across two domains.

    Tests domain filtering:
    - DML method in 'causal_inference'
    - unconfoundedness assumption in 'causal_inference' (should appear)
    - stationarity assumption in 'time_series' (should be filtered)
    """
    method = await ConceptStore.create(
        name="Cross Domain Method",
        canonical_name=f"cross_domain_method_{uuid4().hex[:8]}",
        concept_type=ConceptType.METHOD,
        domain_id="causal_inference",
    )

    same_domain = await ConceptStore.create(
        name="same_domain_assumption",
        canonical_name=f"same_domain_assumption_{uuid4().hex[:8]}",
        concept_type=ConceptType.ASSUMPTION,
        definition="Within same domain",
        domain_id="causal_inference",
    )

    other_domain = await ConceptStore.create(
        name="other_domain_assumption",
        canonical_name=f"other_domain_assumption_{uuid4().hex[:8]}",
        concept_type=ConceptType.ASSUMPTION,
        definition="From a different domain",
        domain_id="time_series",
    )

    await RelationshipStore.create(
        source_concept_id=method.id,
        target_concept_id=same_domain.id,
        relationship_type=RelationshipType.REQUIRES,
        strength=0.9,
    )
    await RelationshipStore.create(
        source_concept_id=method.id,
        target_concept_id=other_domain.id,
        relationship_type=RelationshipType.REQUIRES,
        strength=0.85,
    )

    return method, same_domain, other_domain


# =============================================================================
# 1. TestAssumptionDetailDataclass
# =============================================================================


class TestAssumptionDetailDataclass:
    """Tests for AssumptionDetail dataclass construction and defaults."""

    def test_minimal_construction(self):
        """AssumptionDetail requires only a name; other fields have defaults."""
        detail = AssumptionDetail(name="unconfoundedness")

        assert detail.name == "unconfoundedness"
        assert detail.concept_id is None
        assert detail.formal_statement is None
        assert detail.plain_english is None
        assert detail.importance == "standard"
        assert detail.violation_consequence is None
        assert detail.verification_approaches == []
        assert detail.source_citation is None
        assert detail.relationship_type is None
        assert detail.confidence is None

    def test_full_construction(self):
        """AssumptionDetail with all fields populated."""
        uid = uuid4()
        detail = AssumptionDetail(
            name="overlap",
            concept_id=uid,
            formal_statement="0 < P(T=1|X) < 1",
            plain_english="Positive treatment probability for all covariate values",
            importance="critical",
            violation_consequence="Extreme weights destabilize estimation",
            verification_approaches=["propensity score histogram", "trimming bounds"],
            source_citation="Rosenbaum & Rubin (1983)",
            relationship_type="REQUIRES",
            confidence=0.92,
        )

        assert detail.concept_id == uid
        assert detail.formal_statement == "0 < P(T=1|X) < 1"
        assert detail.importance == "critical"
        assert len(detail.verification_approaches) == 2
        assert "trimming bounds" in detail.verification_approaches
        assert detail.confidence == pytest.approx(0.92, rel=1e-5)


# =============================================================================
# 2. TestMethodAssumptionsDataclass
# =============================================================================


class TestMethodAssumptionsDataclass:
    """Tests for MethodAssumptions dataclass and to_dict serialization."""

    def test_minimal_construction(self):
        """MethodAssumptions with only method name."""
        result = MethodAssumptions(method="IV")

        assert result.method == "IV"
        assert result.method_id is None
        assert result.method_aliases == []
        assert result.definition is None
        assert result.assumptions == []
        assert result.source == "graph"
        assert result.code_docstring_snippet is None

    def test_to_dict_empty_assumptions(self):
        """to_dict with no assumptions produces valid dict structure."""
        result = MethodAssumptions(method="IV", source="not_found")
        d = result.to_dict()

        assert d["method"] == "IV"
        assert d["method_id"] is None
        assert d["assumptions"] == []
        assert d["source"] == "not_found"

    def test_to_dict_with_assumptions(self):
        """to_dict serializes AssumptionDetail objects correctly."""
        uid_method = uuid4()
        uid_assumption = uuid4()

        detail = AssumptionDetail(
            name="SUTVA",
            concept_id=uid_assumption,
            formal_statement="Y_i(t) is independent of T_j for i != j",
            plain_english="No interference between units",
            importance="critical",
            violation_consequence="Treatment effect not well-defined",
            verification_approaches=["cluster analysis"],
            source_citation="Rubin (1980)",
            relationship_type="REQUIRES",
            confidence=0.95,
        )

        result = MethodAssumptions(
            method="IV",
            method_id=uid_method,
            method_aliases=["instrumental variables", "2SLS"],
            definition="Uses instrument to identify causal effect",
            assumptions=[detail],
            source="graph",
            code_docstring_snippet="Assumptions:\n    [CRITICAL] - SUTVA",
        )

        d = result.to_dict()

        assert d["method_id"] == str(uid_method)
        assert d["method_aliases"] == ["instrumental variables", "2SLS"]
        assert len(d["assumptions"]) == 1

        a = d["assumptions"][0]
        assert a["name"] == "SUTVA"
        assert a["concept_id"] == str(uid_assumption)
        assert a["importance"] == "critical"
        assert a["verification_approaches"] == ["cluster analysis"]
        assert a["confidence"] == pytest.approx(0.95, rel=1e-5)

    def test_to_dict_none_concept_id(self):
        """to_dict renders None concept_id as None (not 'None' string)."""
        detail = AssumptionDetail(name="test", concept_id=None)
        result = MethodAssumptions(method="test", assumptions=[detail])
        d = result.to_dict()

        assert d["assumptions"][0]["concept_id"] is None


# =============================================================================
# 3. TestFindMethod
# =============================================================================


class TestFindMethod:
    """Tests for MethodAssumptionAuditor.find_method()."""

    async def test_find_by_name(self, method_with_assumptions):
        """Find method by exact name (case-insensitive)."""
        method, _ = method_with_assumptions

        found = await MethodAssumptionAuditor.find_method("Double Machine Learning")
        assert found is not None
        assert found.id == method.id
        assert found.concept_type == ConceptType.METHOD

    async def test_find_by_canonical_name(self, method_with_assumptions):
        """Find method by canonical_name (case-insensitive)."""
        method, _ = method_with_assumptions

        found = await MethodAssumptionAuditor.find_method(method.canonical_name)
        assert found is not None
        assert found.id == method.id

    async def test_find_by_alias(self, method_with_assumptions):
        """Find method via aliases array entry."""
        method, _ = method_with_assumptions

        found = await MethodAssumptionAuditor.find_method("DML")
        assert found is not None
        assert found.id == method.id

    async def test_find_case_insensitive(self, method_with_assumptions):
        """Lookup is case-insensitive for name, canonical_name, and aliases."""
        method, _ = method_with_assumptions

        found_lower = await MethodAssumptionAuditor.find_method("double machine learning")
        found_upper = await MethodAssumptionAuditor.find_method("DOUBLE MACHINE LEARNING")
        found_alias = await MethodAssumptionAuditor.find_method("dml")

        assert found_lower is not None
        assert found_upper is not None
        assert found_alias is not None
        assert found_lower.id == method.id
        assert found_upper.id == method.id
        assert found_alias.id == method.id

    async def test_find_not_found(self, db_pool):
        """Non-existent method returns None."""
        found = await MethodAssumptionAuditor.find_method("nonexistent_method_xyz_12345")
        assert found is None

    async def test_find_ignores_non_method_concepts(self, db_pool):
        """find_method only returns concepts with type='method'."""
        await ConceptStore.create(
            name="unconfoundedness",
            canonical_name=f"unconfoundedness_{uuid4().hex[:8]}",
            concept_type=ConceptType.ASSUMPTION,
        )

        found = await MethodAssumptionAuditor.find_method("unconfoundedness")
        assert found is None


# =============================================================================
# 4. TestGetAssumptionsFromGraph
# =============================================================================


class TestGetAssumptionsFromGraph:
    """Tests for MethodAssumptionAuditor.get_assumptions_from_graph()."""

    async def test_returns_all_linked_assumptions(self, method_with_assumptions):
        """Retrieves all REQUIRES/USES -> ASSUMPTION relationships."""
        method, assumptions = method_with_assumptions

        results = await MethodAssumptionAuditor.get_assumptions_from_graph(method.id)

        assert len(results) == 4
        names = {a.name for a in results}
        assert "unconfoundedness" in names
        assert "overlap" in names
        assert "convergence_rate" in names
        assert "sample_splitting" in names

    async def test_requires_marked_critical(self, method_with_assumptions):
        """REQUIRES relationships produce importance='critical'."""
        method, _ = method_with_assumptions

        results = await MethodAssumptionAuditor.get_assumptions_from_graph(method.id)

        requires = [a for a in results if a.relationship_type == "REQUIRES"]
        assert len(requires) == 2
        assert all(a.importance == "critical" for a in requires)

    async def test_uses_marked_standard(self, method_with_assumptions):
        """USES relationships produce importance='standard'."""
        method, _ = method_with_assumptions

        results = await MethodAssumptionAuditor.get_assumptions_from_graph(method.id)

        uses = [a for a in results if a.relationship_type == "USES"]
        assert len(uses) == 2
        assert all(a.importance == "standard" for a in uses)

    async def test_ordered_requires_before_uses(self, method_with_assumptions):
        """Results are ordered: REQUIRES first, then USES (by strength DESC)."""
        method, _ = method_with_assumptions

        results = await MethodAssumptionAuditor.get_assumptions_from_graph(method.id)

        relationship_types = [a.relationship_type for a in results]
        # REQUIRES should come before USES
        requires_indices = [i for i, t in enumerate(relationship_types) if t == "REQUIRES"]
        uses_indices = [i for i, t in enumerate(relationship_types) if t == "USES"]
        assert max(requires_indices) < min(uses_indices)

    async def test_domain_filtering_enabled(self, cross_domain_method):
        """With filter_by_domain=True, only same-domain assumptions returned."""
        method, same_domain, other_domain = cross_domain_method

        results = await MethodAssumptionAuditor.get_assumptions_from_graph(
            method.id, filter_by_domain=True
        )

        names = {a.name for a in results}
        assert "same_domain_assumption" in names
        assert "other_domain_assumption" not in names

    async def test_domain_filtering_disabled(self, cross_domain_method):
        """With filter_by_domain=False, all assumptions returned regardless of domain."""
        method, same_domain, other_domain = cross_domain_method

        results = await MethodAssumptionAuditor.get_assumptions_from_graph(
            method.id, filter_by_domain=False
        )

        names = {a.name for a in results}
        assert "same_domain_assumption" in names
        assert "other_domain_assumption" in names
        assert len(results) == 2

    async def test_empty_results_for_unknown_method(self, db_pool):
        """Method with no assumptions returns empty list."""
        method = await ConceptStore.create(
            name="Lonely Method",
            canonical_name=f"lonely_method_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
        )

        results = await MethodAssumptionAuditor.get_assumptions_from_graph(method.id)
        assert results == []

    async def test_ignores_non_assumption_targets(self, db_pool):
        """Only targets with concept_type='assumption' are included."""
        method = await ConceptStore.create(
            name="Method With Problem",
            canonical_name=f"method_with_problem_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
        )
        problem = await ConceptStore.create(
            name="Selection Bias",
            canonical_name=f"selection_bias_{uuid4().hex[:8]}",
            concept_type=ConceptType.PROBLEM,
        )
        await RelationshipStore.create(
            source_concept_id=method.id,
            target_concept_id=problem.id,
            relationship_type=RelationshipType.REQUIRES,
            strength=0.9,
        )

        results = await MethodAssumptionAuditor.get_assumptions_from_graph(method.id)
        assert results == []


# =============================================================================
# 5. TestExtractAssumptionsWithOllama
# =============================================================================


class TestExtractAssumptionsWithOllama:
    """Tests for Ollama LLM extraction (mocked HTTP calls).

    httpx is imported locally inside extract_assumptions_with_ollama,
    so we patch httpx.AsyncClient at the httpx module level.
    """

    async def test_successful_extraction(self):
        """Successful Ollama call returns parsed AssumptionDetail objects."""
        ollama_response = {
            "response": json.dumps(
                {
                    "assumptions": [
                        {
                            "name": "unconfoundedness",
                            "formal_statement": "Y(t) \\perp T | X",
                            "plain_english": "No unmeasured confounders",
                            "importance": "critical",
                            "violation_consequence": "Biased causal estimates",
                            "verification_approaches": [
                                "DAG review",
                                "sensitivity analysis",
                            ],
                        },
                        {
                            "name": "overlap",
                            "formal_statement": "0 < P(T=1|X) < 1",
                            "plain_english": "Positive treatment probability",
                            "importance": "critical",
                            "violation_consequence": "Extreme weights",
                            "verification_approaches": ["propensity score histogram"],
                        },
                    ]
                }
            )
        }

        mock_client = AsyncMock()

        # Mock tags check (GET /api/tags)
        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_tags_response)

        # Mock generate (POST /api/generate)
        mock_gen_response = MagicMock()
        mock_gen_response.status_code = 200
        mock_gen_response.raise_for_status = MagicMock()
        mock_gen_response.json.return_value = ollama_response
        mock_client.post = AsyncMock(return_value=mock_gen_response)

        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("httpx.Timeout", return_value=MagicMock()),
        ):
            results = await MethodAssumptionAuditor.extract_assumptions_with_ollama(
                method_name="DML",
                definition="Double Machine Learning",
            )

        assert len(results) == 2
        assert results[0].name == "unconfoundedness"
        assert results[0].importance == "critical"
        assert results[0].formal_statement == "Y(t) \\perp T | X"
        assert results[0].confidence == pytest.approx(0.7, rel=1e-5)  # LLM confidence
        assert len(results[0].verification_approaches) == 2
        assert results[1].name == "overlap"

    async def test_connection_failure_returns_empty(self):
        """Ollama connection failure returns empty list gracefully."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("httpx.Timeout", return_value=MagicMock()),
        ):
            results = await MethodAssumptionAuditor.extract_assumptions_with_ollama(
                method_name="DML"
            )

        assert results == []

    async def test_invalid_json_returns_empty(self):
        """Invalid JSON from Ollama returns empty list gracefully."""
        mock_client = AsyncMock()

        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_tags_response)

        mock_gen_response = MagicMock()
        mock_gen_response.status_code = 200
        mock_gen_response.raise_for_status = MagicMock()
        mock_gen_response.json.return_value = {"response": "not valid {json}}"}
        mock_client.post = AsyncMock(return_value=mock_gen_response)

        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("httpx.Timeout", return_value=MagicMock()),
        ):
            results = await MethodAssumptionAuditor.extract_assumptions_with_ollama(
                method_name="DML"
            )

        assert results == []

    async def test_invalid_importance_normalized(self):
        """Unrecognized importance value is normalized to 'standard'."""
        ollama_response = {
            "response": json.dumps(
                {
                    "assumptions": [
                        {
                            "name": "test_assumption",
                            "importance": "high",  # Invalid value
                            "plain_english": "test",
                        }
                    ]
                }
            )
        }

        mock_client = AsyncMock()

        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_tags_response)

        mock_gen_response = MagicMock()
        mock_gen_response.status_code = 200
        mock_gen_response.raise_for_status = MagicMock()
        mock_gen_response.json.return_value = ollama_response
        mock_client.post = AsyncMock(return_value=mock_gen_response)

        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("httpx.Timeout", return_value=MagicMock()),
        ):
            results = await MethodAssumptionAuditor.extract_assumptions_with_ollama(
                method_name="test"
            )

        assert len(results) == 1
        assert results[0].importance == "standard"

    async def test_ollama_unavailable_status_returns_empty(self):
        """Non-200 status from Ollama /api/tags returns empty list."""
        mock_client = AsyncMock()

        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 503
        mock_client.get = AsyncMock(return_value=mock_tags_response)

        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("httpx.Timeout", return_value=MagicMock()),
        ):
            results = await MethodAssumptionAuditor.extract_assumptions_with_ollama(
                method_name="test"
            )

        assert results == []


# =============================================================================
# 6. TestCacheAssumptions
# =============================================================================


class TestCacheAssumptions:
    """Tests for cache_assumptions and get_cached_assumptions.

    These tests handle the case where method_assumption_cache table may
    or may not exist in the test database.
    """

    async def test_cache_returns_zero_when_table_missing(self, db_pool):
        """cache_assumptions returns 0 when method_assumption_cache table doesn't exist."""
        assumptions = [
            AssumptionDetail(name="test_assumption", importance="critical"),
        ]

        cached = await MethodAssumptionAuditor.cache_assumptions(
            method_id=uuid4(),
            assumptions=assumptions,
        )

        # Table doesn't exist in test DB, so graceful fallback
        assert cached == 0

    async def test_get_cached_returns_empty_when_table_missing(self, db_pool):
        """get_cached_assumptions returns [] when table doesn't exist."""
        results = await MethodAssumptionAuditor.get_cached_assumptions(uuid4())
        assert results == []

    async def test_cache_empty_list_returns_zero(self, db_pool):
        """Caching empty list returns 0 immediately without DB call."""
        cached = await MethodAssumptionAuditor.cache_assumptions(
            method_id=uuid4(),
            assumptions=[],
        )
        assert cached == 0

    async def test_cache_round_trip_with_table(self, method_with_assumptions):
        """Cache and retrieve round-trip when method_assumption_cache table exists.

        Creates the table in the test database if needed, validates round-trip,
        then cleans up.
        """
        method, _ = method_with_assumptions
        pool = method_with_assumptions  # We need the pool from fixture
        # Get pool from the module's connection manager
        from research_kb_storage.connection import get_connection_pool

        pool = await get_connection_pool()

        # Create the cache table for this test
        async with pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS method_assumption_cache (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    method_concept_id UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
                    assumption_name TEXT NOT NULL,
                    assumption_concept_id UUID REFERENCES concepts(id) ON DELETE SET NULL,
                    formal_statement TEXT,
                    plain_english TEXT,
                    importance TEXT CHECK (importance IN ('critical', 'standard', 'technical')),
                    violation_consequence TEXT,
                    verification_approaches TEXT[],
                    source_citation TEXT,
                    extraction_method TEXT,
                    confidence_score REAL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(method_concept_id, assumption_name)
                )
            """
            )

        try:
            assumptions = [
                AssumptionDetail(
                    name="unconfoundedness",
                    formal_statement="Y(t) \\perp T | X",
                    plain_english="No unmeasured confounders",
                    importance="critical",
                    violation_consequence="Biased estimates",
                    verification_approaches=["DAG review", "sensitivity analysis"],
                    source_citation="Imbens (2004)",
                    confidence=0.9,
                ),
                AssumptionDetail(
                    name="overlap",
                    plain_english="Positive treatment probability",
                    importance="standard",
                    confidence=0.85,
                ),
            ]

            cached_count = await MethodAssumptionAuditor.cache_assumptions(
                method_id=method.id,
                assumptions=assumptions,
                extraction_method="ollama:llama3.1:8b",
            )

            assert cached_count == 2

            # Retrieve cached assumptions
            retrieved = await MethodAssumptionAuditor.get_cached_assumptions(method.id)

            assert len(retrieved) == 2
            # Ordered by importance: critical first, then standard
            assert retrieved[0].importance == "critical"
            assert retrieved[0].name == "unconfoundedness"
            assert retrieved[0].formal_statement == "Y(t) \\perp T | X"
            assert retrieved[0].plain_english == "No unmeasured confounders"
            assert retrieved[0].violation_consequence == "Biased estimates"
            assert "DAG review" in retrieved[0].verification_approaches
            assert retrieved[0].source_citation == "Imbens (2004)"

            assert retrieved[1].importance == "standard"
            assert retrieved[1].name == "overlap"

        finally:
            # Clean up
            async with pool.acquire() as conn:
                await conn.execute("DROP TABLE IF EXISTS method_assumption_cache CASCADE")


# =============================================================================
# 7. TestAuditAssumptions
# =============================================================================


class TestAuditAssumptions:
    """Tests for the main audit_assumptions orchestration flow."""

    async def test_method_not_found(self, db_pool):
        """Non-existent method returns source='not_found'."""
        result = await MethodAssumptionAuditor.audit_assumptions(
            "completely_nonexistent_method_xyz",
            use_ollama_fallback=False,
        )

        assert result.method == "completely_nonexistent_method_xyz"
        assert result.source == "not_found"
        assert result.assumptions == []
        assert "not found" in result.code_docstring_snippet.lower()

    async def test_method_with_sufficient_graph_assumptions(self, method_with_assumptions):
        """Method with >=3 graph assumptions: no Ollama fallback triggered."""
        method, _ = method_with_assumptions

        result = await MethodAssumptionAuditor.audit_assumptions(
            method.name,
            use_ollama_fallback=False,
        )

        assert result.method_id == method.id
        assert result.source == "graph"
        assert len(result.assumptions) >= 4
        assert result.code_docstring_snippet is not None
        assert "Assumptions:" in result.code_docstring_snippet

    async def test_sparse_method_triggers_ollama_fallback(self, method_sparse):
        """Method with <3 assumptions triggers Ollama fallback when enabled."""
        method, _ = method_sparse

        ollama_response = {
            "response": json.dumps(
                {
                    "assumptions": [
                        {
                            "name": "no_anticipation",
                            "plain_english": "No anticipation of treatment",
                            "importance": "critical",
                        },
                        {
                            "name": "convex_hull",
                            "plain_english": "Treated unit in convex hull of controls",
                            "importance": "standard",
                        },
                    ]
                }
            )
        }

        mock_client = AsyncMock()

        mock_tags = MagicMock()
        mock_tags.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_tags)

        mock_gen = MagicMock()
        mock_gen.status_code = 200
        mock_gen.raise_for_status = MagicMock()
        mock_gen.json.return_value = ollama_response
        mock_client.post = AsyncMock(return_value=mock_gen)

        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("httpx.Timeout", return_value=MagicMock()),
        ):
            result = await MethodAssumptionAuditor.audit_assumptions(
                method.name,
                use_ollama_fallback=True,
            )

        assert len(result.assumptions) >= 3  # 1 from graph + 2 from Ollama
        names = {a.name for a in result.assumptions}
        assert "parallel_trends_synth" in names  # Graph result
        assert "no_anticipation" in names  # Ollama result
        assert "graph+ollama" in result.source or result.source == "graph+ollama"

    async def test_ollama_fallback_disabled(self, method_sparse):
        """When use_ollama_fallback=False, sparse results are returned as-is."""
        method, _ = method_sparse

        result = await MethodAssumptionAuditor.audit_assumptions(
            method.name,
            use_ollama_fallback=False,
        )

        assert len(result.assumptions) == 1
        assert result.source == "graph"

    async def test_audit_returns_method_aliases(self, method_with_assumptions):
        """Audit result includes method aliases from the concept."""
        method, _ = method_with_assumptions

        result = await MethodAssumptionAuditor.audit_assumptions(
            method.name,
            use_ollama_fallback=False,
        )

        assert "DML" in result.method_aliases
        assert "debiased ML" in result.method_aliases

    async def test_audit_returns_definition(self, method_with_assumptions):
        """Audit result includes method definition."""
        method, _ = method_with_assumptions

        result = await MethodAssumptionAuditor.audit_assumptions(
            method.name,
            use_ollama_fallback=False,
        )

        assert result.definition is not None
        assert "cross-fitting" in result.definition


# =============================================================================
# 8. TestGenerateDocstringSnippet
# =============================================================================


class TestGenerateDocstringSnippet:
    """Tests for _generate_docstring_snippet helper."""

    def test_empty_assumptions(self):
        """No assumptions produces 'No assumptions documented' message."""
        result = _generate_docstring_snippet("IV", [])
        assert "No assumptions documented" in result
        assert "IV" in result

    def test_critical_assumptions_labeled(self):
        """Critical assumptions get [CRITICAL] prefix."""
        assumptions = [
            AssumptionDetail(name="unconfoundedness", importance="critical"),
        ]

        result = _generate_docstring_snippet("DML", assumptions)

        assert "[CRITICAL]" in result
        assert "unconfoundedness" in result

    def test_standard_assumptions_no_label(self):
        """Standard assumptions have no prefix label."""
        assumptions = [
            AssumptionDetail(name="convergence", importance="standard"),
        ]

        result = _generate_docstring_snippet("DML", assumptions)

        assert "[CRITICAL]" not in result
        assert "[technical]" not in result
        assert "convergence" in result

    def test_technical_assumptions_labeled(self):
        """Technical assumptions get [technical] prefix."""
        assumptions = [
            AssumptionDetail(name="regularity", importance="technical"),
        ]

        result = _generate_docstring_snippet("DML", assumptions)

        assert "[technical]" in result
        assert "regularity" in result

    def test_mixed_importance_ordering(self):
        """Docstring groups: critical first, then standard, then technical."""
        assumptions = [
            AssumptionDetail(name="regularity", importance="technical"),
            AssumptionDetail(name="overlap", importance="critical"),
            AssumptionDetail(name="convergence", importance="standard"),
        ]

        result = _generate_docstring_snippet("DML", assumptions)

        # Critical should appear before standard, standard before technical
        lines = result.split("\n")
        critical_idx = next(i for i, l in enumerate(lines) if "overlap" in l)
        standard_idx = next(i for i, l in enumerate(lines) if "convergence" in l)
        technical_idx = next(i for i, l in enumerate(lines) if "regularity" in l)

        assert critical_idx < standard_idx < technical_idx

    def test_plain_english_included(self):
        """When plain_english is set, it appears after the name."""
        assumptions = [
            AssumptionDetail(
                name="overlap",
                plain_english="Positive treatment probability for all X",
                importance="standard",
            ),
        ]

        result = _generate_docstring_snippet("DML", assumptions)

        assert "overlap: Positive treatment probability" in result

    def test_header_line(self):
        """Docstring starts with 'Assumptions:' header."""
        assumptions = [
            AssumptionDetail(name="test", importance="standard"),
        ]

        result = _generate_docstring_snippet("DML", assumptions)

        assert result.startswith("Assumptions:")


# =============================================================================
# 9. TestExtractAssumptionsWithAnthropic
# =============================================================================


class TestExtractAssumptionsWithAnthropic:
    """Tests for Anthropic LLM extraction (mocked API calls).

    Follows the same mocking pattern as TestExtractAssumptionsWithOllama
    but mocks the anthropic library.
    """

    async def test_successful_extraction(self):
        """Successful Anthropic call returns parsed AssumptionDetail objects."""
        response_json = json.dumps(
            {
                "assumptions": [
                    {
                        "name": "unconfoundedness",
                        "formal_statement": "Y(t) \\perp T | X",
                        "plain_english": "No unmeasured confounders",
                        "importance": "critical",
                        "violation_consequence": "Biased causal estimates",
                        "verification_approaches": [
                            "DAG review",
                            "sensitivity analysis",
                        ],
                    },
                    {
                        "name": "overlap",
                        "formal_statement": "0 < P(T=1|X) < 1",
                        "plain_english": "Positive treatment probability",
                        "importance": "critical",
                        "violation_consequence": "Extreme weights",
                        "verification_approaches": ["propensity score histogram"],
                    },
                    {
                        "name": "convergence_rate",
                        "formal_statement": "n^{-1/4} rate",
                        "plain_english": "Nuisance estimators converge fast enough",
                        "importance": "technical",
                        "violation_consequence": "Biased debiased estimator",
                        "verification_approaches": ["cross-validation"],
                    },
                ]
            }
        )

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=response_json)]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"}),
            patch.dict("sys.modules", {"anthropic": mock_anthropic_module}),
        ):
            results = await MethodAssumptionAuditor.extract_assumptions_with_anthropic(
                method_name="DML",
                definition="Double Machine Learning",
            )

        assert len(results) == 3
        assert results[0].name == "unconfoundedness"
        assert results[0].importance == "critical"
        assert results[0].formal_statement == "Y(t) \\perp T | X"
        assert results[0].confidence == pytest.approx(0.85, rel=1e-5)
        assert len(results[0].verification_approaches) == 2
        assert results[1].name == "overlap"
        assert results[2].name == "convergence_rate"
        assert results[2].importance == "technical"

    async def test_no_api_key_returns_empty(self):
        """Missing ANTHROPIC_API_KEY returns empty list gracefully."""
        with patch.dict("os.environ", {}, clear=False):
            # Ensure ANTHROPIC_API_KEY is not set
            import os

            original = os.environ.pop("ANTHROPIC_API_KEY", None)
            try:
                results = await MethodAssumptionAuditor.extract_assumptions_with_anthropic(
                    method_name="DML",
                )
            finally:
                if original is not None:
                    os.environ["ANTHROPIC_API_KEY"] = original

        assert results == []

    async def test_import_error_returns_empty(self):
        """Missing anthropic package returns empty list gracefully."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("No module named 'anthropic'")
            return original_import(name, *args, **kwargs)

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"}),
            patch("builtins.__import__", side_effect=mock_import),
        ):
            results = await MethodAssumptionAuditor.extract_assumptions_with_anthropic(
                method_name="DML",
            )

        assert results == []

    async def test_invalid_json_returns_empty(self):
        """Malformed JSON response returns empty list gracefully."""
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="not valid {json}}")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"}),
            patch.dict("sys.modules", {"anthropic": mock_anthropic_module}),
        ):
            results = await MethodAssumptionAuditor.extract_assumptions_with_anthropic(
                method_name="DML",
            )

        assert results == []

    async def test_api_error_returns_empty(self):
        """API exception returns empty list gracefully."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API rate limit exceeded")

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"}),
            patch.dict("sys.modules", {"anthropic": mock_anthropic_module}),
        ):
            results = await MethodAssumptionAuditor.extract_assumptions_with_anthropic(
                method_name="DML",
            )

        assert results == []

    async def test_invalid_importance_normalized(self):
        """Unrecognized importance value is normalized to 'standard'."""
        response_json = json.dumps(
            {
                "assumptions": [
                    {
                        "name": "test_assumption",
                        "importance": "high",  # Invalid value
                        "plain_english": "test",
                    }
                ]
            }
        )

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=response_json)]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"}),
            patch.dict("sys.modules", {"anthropic": mock_anthropic_module}),
        ):
            results = await MethodAssumptionAuditor.extract_assumptions_with_anthropic(
                method_name="test",
            )

        assert len(results) == 1
        assert results[0].importance == "standard"

    async def test_code_fenced_json_parsed_correctly(self):
        """JSON wrapped in markdown code fences is parsed correctly."""
        inner_json = json.dumps(
            {
                "assumptions": [
                    {
                        "name": "overlap",
                        "plain_english": "Positive treatment probability",
                        "importance": "critical",
                    }
                ]
            }
        )
        # Wrap in code fences like Haiku actually does
        fenced_response = f"```json\n{inner_json}\n```"

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=fenced_response)]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"}),
            patch.dict("sys.modules", {"anthropic": mock_anthropic_module}),
        ):
            results = await MethodAssumptionAuditor.extract_assumptions_with_anthropic(
                method_name="IPW",
            )

        assert len(results) == 1
        assert results[0].name == "overlap"
        assert results[0].importance == "critical"


# =============================================================================
# 10. TestAuditAssumptionsBackendDispatch
# =============================================================================


class TestAuditAssumptionsBackendDispatch:
    """Tests for llm_backend parameter in audit_assumptions().

    Verifies that the correct extraction function is called based on
    the llm_backend parameter ("ollama" vs "anthropic").
    """

    async def test_anthropic_backend_calls_anthropic_extractor(self, method_sparse):
        """llm_backend='anthropic' dispatches to extract_assumptions_with_anthropic."""
        method, _ = method_sparse

        anthropic_response = [
            AssumptionDetail(
                name="no_anticipation",
                plain_english="No anticipation of treatment",
                importance="critical",
                confidence=0.85,
            ),
            AssumptionDetail(
                name="convex_hull",
                plain_english="Treated unit in convex hull of controls",
                importance="standard",
                confidence=0.85,
            ),
            AssumptionDetail(
                name="no_interference",
                plain_english="No spillover between units",
                importance="standard",
                confidence=0.85,
            ),
        ]

        with (
            patch.object(
                MethodAssumptionAuditor,
                "extract_assumptions_with_anthropic",
                new_callable=AsyncMock,
                return_value=anthropic_response,
            ) as mock_anthropic,
            patch.object(
                MethodAssumptionAuditor,
                "extract_assumptions_with_ollama",
                new_callable=AsyncMock,
            ) as mock_ollama,
            patch.object(
                MethodAssumptionAuditor,
                "cache_assumptions",
                new_callable=AsyncMock,
                return_value=3,
            ),
        ):
            result = await MethodAssumptionAuditor.audit_assumptions(
                method.name,
                use_llm_fallback=True,
                llm_backend="anthropic",
            )

        mock_anthropic.assert_called_once()
        mock_ollama.assert_not_called()
        assert "anthropic" in result.source

    async def test_ollama_backend_calls_ollama_extractor(self, method_sparse):
        """llm_backend='ollama' dispatches to extract_assumptions_with_ollama."""
        method, _ = method_sparse

        ollama_response = [
            AssumptionDetail(
                name="no_anticipation",
                plain_english="No anticipation of treatment",
                importance="critical",
                confidence=0.7,
            ),
            AssumptionDetail(
                name="convex_hull",
                plain_english="Treated unit in convex hull of controls",
                importance="standard",
                confidence=0.7,
            ),
            AssumptionDetail(
                name="no_interference",
                plain_english="No spillover between units",
                importance="standard",
                confidence=0.7,
            ),
        ]

        with (
            patch.object(
                MethodAssumptionAuditor,
                "extract_assumptions_with_ollama",
                new_callable=AsyncMock,
                return_value=ollama_response,
            ) as mock_ollama,
            patch.object(
                MethodAssumptionAuditor,
                "extract_assumptions_with_anthropic",
                new_callable=AsyncMock,
            ) as mock_anthropic,
            patch.object(
                MethodAssumptionAuditor,
                "cache_assumptions",
                new_callable=AsyncMock,
                return_value=3,
            ),
        ):
            result = await MethodAssumptionAuditor.audit_assumptions(
                method.name,
                use_llm_fallback=True,
                llm_backend="ollama",
            )

        mock_ollama.assert_called_once()
        mock_anthropic.assert_not_called()
        assert "ollama" in result.source

    async def test_llm_fallback_disabled_skips_extraction(self, method_sparse):
        """use_llm_fallback=False skips LLM extraction entirely."""
        method, _ = method_sparse

        with (
            patch.object(
                MethodAssumptionAuditor,
                "extract_assumptions_with_anthropic",
                new_callable=AsyncMock,
            ) as mock_anthropic,
            patch.object(
                MethodAssumptionAuditor,
                "extract_assumptions_with_ollama",
                new_callable=AsyncMock,
            ) as mock_ollama,
        ):
            result = await MethodAssumptionAuditor.audit_assumptions(
                method.name,
                use_llm_fallback=False,
                llm_backend="anthropic",
            )

        mock_anthropic.assert_not_called()
        mock_ollama.assert_not_called()
        assert len(result.assumptions) == 1  # Only graph result

    async def test_backward_compat_use_ollama_fallback(self, method_sparse):
        """Legacy use_ollama_fallback=True still works with default llm_backend."""
        method, _ = method_sparse

        with (
            patch.object(
                MethodAssumptionAuditor,
                "extract_assumptions_with_ollama",
                new_callable=AsyncMock,
                return_value=[
                    AssumptionDetail(name="a1", importance="critical", confidence=0.7),
                    AssumptionDetail(name="a2", importance="standard", confidence=0.7),
                    AssumptionDetail(name="a3", importance="standard", confidence=0.7),
                ],
            ) as mock_ollama,
            patch.object(
                MethodAssumptionAuditor,
                "cache_assumptions",
                new_callable=AsyncMock,
                return_value=3,
            ),
        ):
            result = await MethodAssumptionAuditor.audit_assumptions(
                method.name,
                use_ollama_fallback=True,
                # llm_backend defaults to "ollama"
            )

        mock_ollama.assert_called_once()
        assert len(result.assumptions) >= 3

    async def test_sufficient_assumptions_skip_llm(self, method_with_assumptions):
        """Methods with >=3 graph assumptions don't trigger LLM fallback."""
        method, _ = method_with_assumptions

        with (
            patch.object(
                MethodAssumptionAuditor,
                "extract_assumptions_with_anthropic",
                new_callable=AsyncMock,
            ) as mock_anthropic,
            patch.object(
                MethodAssumptionAuditor,
                "extract_assumptions_with_ollama",
                new_callable=AsyncMock,
            ) as mock_ollama,
        ):
            result = await MethodAssumptionAuditor.audit_assumptions(
                method.name,
                use_llm_fallback=True,
                llm_backend="anthropic",
            )

        mock_anthropic.assert_not_called()
        mock_ollama.assert_not_called()
        assert len(result.assumptions) >= 4
        assert result.source == "graph"
