"""Tests for query expansion functionality.

Tests the QueryExpander class and its expansion strategies:
- Synonym expansion (deterministic lookup)
- Graph expansion (knowledge graph relationships)
- Combined expansion (multiple strategies)
- FTS query building

Uses fixtures from fixtures/concepts/synonym_map.yaml for realistic testing.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from research_kb_storage.query_expander import (
    QueryExpander,
    ExpandedQuery,
    expand_query,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synonym_map() -> dict[str, list[str]]:
    """Minimal synonym map for testing."""
    return {
        "iv": [
            "instrumental variables",
            "instrumental variable",
            "2sls",
            "two-stage least squares",
        ],
        "did": [
            "difference-in-differences",
            "difference in differences",
            "diff-in-diff",
        ],
        "ate": ["average treatment effect", "treatment effect"],
        "dml": ["double machine learning", "debiased machine learning"],
        "dag": ["directed acyclic graph", "causal diagram", "causal graph"],
    }


@pytest.fixture
def expander(synonym_map) -> QueryExpander:
    """QueryExpander with test synonym map."""
    return QueryExpander(synonym_map=synonym_map)


@pytest.fixture
def real_synonym_map_path() -> Path:
    """Path to actual synonym map fixture file."""
    return Path(__file__).parent.parent.parent.parent / "fixtures" / "concepts" / "synonym_map.yaml"


# =============================================================================
# Synonym Expansion Tests
# =============================================================================


class TestSynonymExpansion:
    """Tests for deterministic synonym lookup expansion."""

    def test_synonym_expansion_single_term(self, expander):
        """Verify IV → instrumental variables mapping."""
        # Given: A query with a known synonym key
        query = "IV"

        # When: We expand with synonyms
        expansions = expander.expand_with_synonyms(query)

        # Then: All synonyms are returned
        assert "instrumental variables" in expansions
        assert "2sls" in expansions
        assert "two-stage least squares" in expansions

    def test_synonym_expansion_case_insensitive(self, expander):
        """Verify case-insensitive matching."""
        # Given: Various case forms of the same term
        queries = ["IV", "iv", "Iv", "iV"]

        # When: We expand each form
        for query in queries:
            expansions = expander.expand_with_synonyms(query)

            # Then: All should return the same synonyms
            assert "instrumental variables" in expansions
            assert len(expansions) >= 3

    def test_synonym_expansion_multiple_terms(self, expander):
        """Verify expansion works with multi-word queries."""
        # Given: Query with multiple expandable terms
        query = "IV and DID estimation"

        # When: We expand
        expansions = expander.expand_with_synonyms(query)

        # Then: Both IV and DID synonyms are included
        # IV synonyms
        iv_found = any("instrumental" in e for e in expansions)
        # DID synonyms
        did_found = any("difference" in e for e in expansions)

        assert iv_found, "Should find IV synonyms"
        assert did_found, "Should find DID synonyms"

    def test_synonym_expansion_no_match(self, expander):
        """Verify empty result for unknown terms."""
        # Given: Query with no matching synonym keys
        query = "unknown methodology"

        # When: We expand
        expansions = expander.expand_with_synonyms(query)

        # Then: Empty list returned
        assert expansions == []

    def test_synonym_expansion_avoids_duplicates(self, expander):
        """Verify no duplicate expansions returned."""
        # Given: Query that might match multiple ways
        query = "IV instrumental"

        # When: We expand
        expansions = expander.expand_with_synonyms(query)

        # Then: No duplicates, and "instrumental" not in expansions (already in query)
        assert len(expansions) == len(set(e.lower() for e in expansions))
        # Should not include "instrumental variables" since "instrumental" is in query
        # (implementation detail - depends on exact logic)

    def test_from_yaml_loads_real_file(self, real_synonym_map_path):
        """Verify QueryExpander.from_yaml loads real synonym file."""
        # Given: Path to actual synonym map
        if not real_synonym_map_path.exists():
            pytest.skip(f"Synonym map not found: {real_synonym_map_path}")

        # When: We create expander from YAML
        expander = QueryExpander.from_yaml(real_synonym_map_path)

        # Then: Synonyms are loaded
        assert len(expander.synonym_map) > 0
        assert "iv" in expander.synonym_map
        assert "instrumental variables" in expander.synonym_map["iv"]


# =============================================================================
# Graph Expansion Tests
# =============================================================================


class TestGraphExpansion:
    """Tests for knowledge graph relationship expansion."""

    async def test_graph_expansion_finds_related_concepts(self, expander):
        """Verify graph expansion returns related concept names."""
        # Given: Mocked graph query results
        mock_neighborhood = {
            "concepts": [
                MagicMock(canonical_name="endogeneity", name="Endogeneity"),
                MagicMock(canonical_name="exogeneity", name="Exogeneity"),
            ]
        }

        # Patch at the source modules (imports are done inside the method)
        with (
            patch(
                "research_kb_storage.query_extractor.extract_query_concepts",
                new_callable=AsyncMock,
                return_value=["concept-uuid-1"],
            ),
            patch(
                "research_kb_storage.graph_queries.get_neighborhood",
                new_callable=AsyncMock,
                return_value=mock_neighborhood,
            ),
        ):
            # When: We expand with graph
            expansions = await expander.expand_with_graph("IV estimation")

            # Then: Related concepts are returned
            assert "endogeneity" in expansions
            assert "exogeneity" in expansions

    async def test_graph_expansion_respects_max_concepts(self, expander):
        """Verify max_concepts parameter is respected."""
        # Given: Many concepts in neighborhood
        mock_neighborhood = {
            "concepts": [
                MagicMock(canonical_name=f"concept_{i}", name=f"Concept {i}") for i in range(10)
            ]
        }

        with (
            patch(
                "research_kb_storage.query_extractor.extract_query_concepts",
                new_callable=AsyncMock,
                return_value=["concept-uuid-1"],
            ),
            patch(
                "research_kb_storage.graph_queries.get_neighborhood",
                new_callable=AsyncMock,
                return_value=mock_neighborhood,
            ),
        ):
            # When: We expand with max_concepts=3
            expansions = await expander.expand_with_graph("test query", max_concepts=3)

            # Then: At most 3 concepts returned
            assert len(expansions) <= 3

    async def test_graph_expansion_handles_no_concepts(self, expander):
        """Verify graceful handling when no concepts found."""
        with patch(
            "research_kb_storage.query_extractor.extract_query_concepts",
            new_callable=AsyncMock,
            return_value=[],
        ):
            # When: No concepts are found in query
            expansions = await expander.expand_with_graph("random text")

            # Then: Empty list returned
            assert expansions == []

    async def test_graph_expansion_handles_errors_gracefully(self, expander):
        """Verify errors in graph queries don't crash expansion."""
        with patch(
            "research_kb_storage.query_extractor.extract_query_concepts",
            new_callable=AsyncMock,
            side_effect=Exception("Database error"),
        ):
            # When: Graph query fails
            expansions = await expander.expand_with_graph("IV estimation")

            # Then: Empty list returned, no exception raised
            assert expansions == []


# =============================================================================
# FTS Query Building Tests
# =============================================================================


class TestFTSQueryBuilding:
    """Tests for PostgreSQL FTS query construction."""

    def test_build_fts_query_basic(self, expander):
        """Verify FTS query format with boosting."""
        # Given: Original query and expansions
        original = "IV estimation"
        expansions = ["instrumental variables"]

        # When: We build FTS query
        fts_query = expander.build_fts_query(original, expansions)

        # Then: Format is correct with weights
        assert "IV:A" in fts_query
        assert "estimation:A" in fts_query
        assert "instrumental:B" in fts_query
        assert "variables:B" in fts_query
        assert "|" in fts_query  # OR operator

    def test_build_fts_query_escapes_special_chars(self, expander):
        """Verify special characters are escaped."""
        # Given: Query with special characters
        original = "test:query"
        expansions = ["term!with@chars"]

        # When: We build FTS query
        fts_query = expander.build_fts_query(original, expansions)

        # Then: Special chars are removed/escaped
        assert ":" not in fts_query.replace(":A", "").replace(":B", "")
        assert "!" not in fts_query
        assert "@" not in fts_query

    def test_build_fts_query_empty_expansions(self, expander):
        """Verify handling of empty expansion list."""
        # Given: Original query, no expansions
        original = "simple query"
        expansions = []

        # When: We build FTS query
        fts_query = expander.build_fts_query(original, expansions)

        # Then: Only original terms with A weight
        assert "simple:A" in fts_query
        assert "query:A" in fts_query
        assert ":B" not in fts_query

    def test_build_fts_query_custom_weights(self, expander):
        """Verify custom weight parameters work."""
        # Given: Custom weights
        original = "test"
        expansions = ["expansion"]

        # When: We build with custom weights
        fts_query = expander.build_fts_query(
            original, expansions, original_weight="C", expansion_weight="D"
        )

        # Then: Custom weights used
        assert "test:C" in fts_query
        assert "expansion:D" in fts_query


# =============================================================================
# Combined Expansion Tests
# =============================================================================


class TestCombinedExpansion:
    """Tests for full expand() method combining strategies."""

    async def test_expand_with_synonyms_only(self, expander):
        """Verify expansion with synonyms only."""
        # When: We expand with only synonyms enabled
        result = await expander.expand(
            "IV estimation",
            use_synonyms=True,
            use_graph=False,
            use_llm=False,
        )

        # Then: Result contains synonym expansions
        assert isinstance(result, ExpandedQuery)
        assert result.original == "IV estimation"
        assert len(result.expanded_terms) > 0
        assert "synonyms" in result.expansion_sources
        assert "graph" not in result.expansion_sources

    async def test_expand_combined_synonym_and_graph(self, expander):
        """Verify combined synonym + graph expansion."""
        # Given: Mocked graph expansion
        mock_neighborhood = {
            "concepts": [
                MagicMock(canonical_name="endogeneity", name="Endogeneity"),
            ]
        }

        # Patch at the source modules
        with (
            patch(
                "research_kb_storage.query_extractor.extract_query_concepts",
                new_callable=AsyncMock,
                return_value=["concept-uuid-1"],
            ),
            patch(
                "research_kb_storage.graph_queries.get_neighborhood",
                new_callable=AsyncMock,
                return_value=mock_neighborhood,
            ),
        ):
            # When: We expand with synonyms and graph
            result = await expander.expand(
                "IV estimation",
                use_synonyms=True,
                use_graph=True,
                use_llm=False,
            )

            # Then: Both sources contribute
            assert "synonyms" in result.expansion_sources
            assert "graph" in result.expansion_sources
            # Synonym expansions present
            assert any("instrumental" in t for t in result.expanded_terms)
            # Graph expansion present
            assert "endogeneity" in result.expanded_terms

    async def test_expand_empty_query(self, expander):
        """Verify empty query handling."""
        # When: We expand empty/whitespace queries
        for query in ["", "   ", None]:
            if query is None:
                continue  # Skip None - would fail type check
            result = await expander.expand(query)

            # Then: Empty result returned
            assert result.original == query
            assert result.expanded_terms == []

    async def test_expand_generates_valid_fts_query(self, expander):
        """Verify FTS query is generated in result."""
        # When: We expand a query
        result = await expander.expand(
            "DML",
            use_synonyms=True,
            use_graph=False,
            use_llm=False,
        )

        # Then: FTS query is populated
        assert result.fts_query != ""
        assert "DML:A" in result.fts_query  # Original term
        assert ":B" in result.fts_query  # Expansion terms

    async def test_expand_deduplicates_across_sources(self, expander):
        """Verify no duplicate terms across expansion sources."""
        # Given: Graph returns same term as synonyms
        mock_neighborhood = {
            "concepts": [
                # This matches a synonym
                MagicMock(canonical_name="instrumental variables", name="IV"),
            ]
        }

        # Patch at the source modules
        with (
            patch(
                "research_kb_storage.query_extractor.extract_query_concepts",
                new_callable=AsyncMock,
                return_value=["concept-uuid-1"],
            ),
            patch(
                "research_kb_storage.graph_queries.get_neighborhood",
                new_callable=AsyncMock,
                return_value=mock_neighborhood,
            ),
        ):
            # When: Both sources would add same term
            result = await expander.expand(
                "IV",
                use_synonyms=True,
                use_graph=True,
            )

            # Then: No duplicates in expanded_terms
            term_counts = {}
            for term in result.expanded_terms:
                lower = term.lower()
                term_counts[lower] = term_counts.get(lower, 0) + 1

            for term, count in term_counts.items():
                assert count == 1, f"Duplicate term found: {term}"


# =============================================================================
# Module Function Tests
# =============================================================================


class TestModuleFunction:
    """Tests for module-level expand_query convenience function."""

    async def test_expand_query_function(self, real_synonym_map_path):
        """Verify module-level function works."""
        if not real_synonym_map_path.exists():
            pytest.skip(f"Synonym map not found: {real_synonym_map_path}")

        # When: We use the convenience function
        result = await expand_query(
            "IV estimation",
            use_synonyms=True,
            use_graph=False,
            use_llm=False,
            synonym_map_path=real_synonym_map_path,
        )

        # Then: Result is valid
        assert isinstance(result, ExpandedQuery)
        assert result.original == "IV estimation"
        assert len(result.expanded_terms) > 0


# =============================================================================
# ExpandedQuery Dataclass Tests
# =============================================================================


class TestExpandedQuery:
    """Tests for ExpandedQuery dataclass."""

    def test_all_terms_property(self):
        """Verify all_terms includes original + expanded."""
        # Given: ExpandedQuery with terms
        eq = ExpandedQuery(
            original="IV",
            expanded_terms=["instrumental variables", "2sls"],
        )

        # When: We access all_terms
        all_terms = eq.all_terms

        # Then: Original + expanded
        assert all_terms == ["IV", "instrumental variables", "2sls"]

    def test_expansion_count_property(self):
        """Verify expansion_count returns correct count."""
        # Given: ExpandedQuery with 3 expansions
        eq = ExpandedQuery(
            original="test",
            expanded_terms=["a", "b", "c"],
        )

        # When: We check count
        count = eq.expansion_count

        # Then: Correct count
        assert count == 3

    def test_default_values(self):
        """Verify default values for optional fields."""
        # Given: Minimal ExpandedQuery
        eq = ExpandedQuery(original="test")

        # Then: Defaults are set
        assert eq.expanded_terms == []
        assert eq.fts_query == ""
        assert eq.expansion_sources == {}
        assert eq.expansion_count == 0


# =============================================================================
# HyDE Anthropic Backend Tests
# =============================================================================


class TestHydeAnthropic:
    """Tests for _generate_hyde_anthropic with mocked anthropic module."""

    async def test_hyde_anthropic_success(self):
        """Mock anthropic module via sys.modules, verify returns stripped text."""
        from research_kb_storage.query_expander import _generate_hyde_anthropic

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="  A hypothetical document about IV.  ")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"}),
            patch.dict("sys.modules", {"anthropic": mock_anthropic_module}),
        ):
            result = await _generate_hyde_anthropic(
                prompt="Write about instrumental variables",
                model="claude-3-5-haiku-20241022",
            )

        assert result == "A hypothetical document about IV."
        mock_client.messages.create.assert_called_once()

    async def test_hyde_anthropic_no_api_key(self):
        """No ANTHROPIC_API_KEY returns None gracefully."""
        from research_kb_storage.query_expander import _generate_hyde_anthropic

        import os

        original = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = await _generate_hyde_anthropic(
                prompt="Write about IV",
                model="claude-3-5-haiku-20241022",
            )
            assert result is None
        finally:
            if original is not None:
                os.environ["ANTHROPIC_API_KEY"] = original

    async def test_hyde_anthropic_api_error(self):
        """Mock raises exception, returns None gracefully."""
        from research_kb_storage.query_expander import _generate_hyde_anthropic

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value.messages.create.side_effect = RuntimeError(
            "API rate limit"
        )

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"}),
            patch.dict("sys.modules", {"anthropic": mock_anthropic_module}),
        ):
            result = await _generate_hyde_anthropic(
                prompt="Write about IV",
                model="claude-3-5-haiku-20241022",
            )

        assert result is None
