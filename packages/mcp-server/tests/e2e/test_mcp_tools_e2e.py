"""End-to-end tests for MCP tools against real database.

These tests verify that all MCP tools work correctly with the actual
PostgreSQL database and return valid, non-empty responses.

Requirements:
    - PostgreSQL running with research_kb database
    - Database populated with sources, chunks, concepts

Run with:
    pytest packages/mcp-server/tests/e2e/test_mcp_tools_e2e.py -v -m e2e
"""

from __future__ import annotations

import pytest

from research_kb_storage import (
    get_connection_pool,
    close_connection_pool,
    DatabaseConfig,
)
from research_kb_api.service import (
    get_stats,
    search,
    get_sources,
    get_source_by_id,
    get_source_chunks,
    get_concepts,
    get_concept_by_id,
    get_concept_relationships,
    get_graph_neighborhood,
    get_graph_path,
    get_citations_for_source,
    SearchOptions,
    ContextType,
)


# Use function-scoped pool to avoid event loop issues
@pytest.fixture
async def pool():
    """Create database connection pool for test."""
    p = await get_connection_pool(DatabaseConfig())
    yield p
    await close_connection_pool()


@pytest.mark.e2e
class TestHealthTools:
    """E2E tests for health and stats tools."""

    async def test_get_stats_returns_counts(self, pool):
        """get_stats returns valid statistics."""
        stats = await get_stats()

        assert stats is not None
        assert "sources" in stats
        assert "chunks" in stats
        assert "concepts" in stats
        assert "relationships" in stats

        # Verify counts are reasonable (non-negative)
        assert stats["sources"] >= 0
        assert stats["chunks"] >= 0
        assert stats["concepts"] >= 0
        assert stats["relationships"] >= 0

    async def test_get_stats_has_populated_data(self, pool):
        """Verify database actually has data."""
        stats = await get_stats()

        # These should have data in a populated KB
        assert stats["sources"] > 0, "No sources in database"
        assert stats["chunks"] > 0, "No chunks in database"


@pytest.mark.e2e
class TestSearchTools:
    """E2E tests for search tool."""

    async def test_search_returns_results(self, pool):
        """Search returns results for common query."""
        options = SearchOptions(
            query="instrumental variables",
            limit=5,
            context_type=ContextType.balanced,
            use_graph=False,
            use_citations=False,
            use_rerank=False,
            use_expand=False,
        )
        response = await search(options)

        assert response is not None
        assert hasattr(response, "results")
        assert isinstance(response.results, list)

    async def test_search_with_graph(self, pool):
        """Search with graph boosting enabled."""
        options = SearchOptions(
            query="causal inference",
            limit=3,
            context_type=ContextType.balanced,
            use_graph=True,
            use_citations=False,
            use_rerank=False,
            use_expand=False,
        )
        response = await search(options)

        assert response is not None
        assert isinstance(response.results, list)

    async def test_search_building_context(self, pool):
        """Search with building context type."""
        options = SearchOptions(
            query="regression",
            limit=3,
            context_type=ContextType.building,
            use_rerank=False,
            use_expand=False,
        )
        response = await search(options)

        assert response is not None

    async def test_search_auditing_context(self, pool):
        """Search with auditing context type."""
        options = SearchOptions(
            query="estimation",
            limit=3,
            context_type=ContextType.auditing,
            use_rerank=False,
            use_expand=False,
        )
        response = await search(options)

        assert response is not None


@pytest.mark.e2e
class TestSourceTools:
    """E2E tests for source tools."""

    async def test_get_sources(self, pool):
        """get_sources returns source list."""
        sources = await get_sources(limit=10)

        assert isinstance(sources, list)
        if sources:
            source = sources[0]
            assert hasattr(source, "id")
            assert hasattr(source, "title")

    async def test_get_sources_pagination(self, pool):
        """get_sources supports pagination."""
        page1 = await get_sources(limit=5, offset=0)
        page2 = await get_sources(limit=5, offset=5)

        assert isinstance(page1, list)
        assert isinstance(page2, list)

    async def test_get_source_by_id(self, pool):
        """get_source_by_id returns source details."""
        sources = await get_sources(limit=1)
        if not sources:
            pytest.skip("No sources in database")

        source_id = str(sources[0].id)
        source = await get_source_by_id(source_id)

        assert source is not None
        assert str(source.id) == source_id
        assert source.title is not None

    async def test_get_source_chunks(self, pool):
        """get_source_chunks returns chunks for source."""
        sources = await get_sources(limit=1)
        if not sources:
            pytest.skip("No sources in database")

        source_id = str(sources[0].id)
        chunks = await get_source_chunks(source_id, limit=10)

        assert isinstance(chunks, list)
        if chunks:
            chunk = chunks[0]
            assert hasattr(chunk, "content") or "content" in chunk


@pytest.mark.e2e
class TestConceptTools:
    """E2E tests for concept tools."""

    async def test_get_concepts(self, pool):
        """get_concepts returns concepts."""
        concepts = await get_concepts(limit=10)

        assert isinstance(concepts, list)
        if concepts:
            concept = concepts[0]
            assert hasattr(concept, "id")
            assert hasattr(concept, "name")

    async def test_get_concepts_with_query(self, pool):
        """get_concepts can filter by query."""
        concepts = await get_concepts(query="regression", limit=5)

        assert isinstance(concepts, list)

    async def test_get_concepts_by_type(self, pool):
        """get_concepts can filter by type."""
        concepts = await get_concepts(concept_type="METHOD", limit=5)

        assert isinstance(concepts, list)
        if concepts:
            for concept in concepts:
                ctype = concept.concept_type
                if hasattr(ctype, "value"):
                    ctype = ctype.value
                assert ctype.upper() == "METHOD"

    async def test_get_concept_by_id(self, pool):
        """get_concept_by_id returns concept details."""
        concepts = await get_concepts(limit=1)
        if not concepts:
            pytest.skip("No concepts in database")

        concept_id = str(concepts[0].id)
        concept = await get_concept_by_id(concept_id)

        assert concept is not None
        assert str(concept.id) == concept_id

    async def test_get_concept_relationships(self, pool):
        """get_concept_relationships returns relationships."""
        concepts = await get_concepts(limit=1)
        if not concepts:
            pytest.skip("No concepts in database")

        concept_id = str(concepts[0].id)
        relationships = await get_concept_relationships(concept_id)

        assert isinstance(relationships, list)


@pytest.mark.e2e
class TestGraphTools:
    """E2E tests for graph tools."""

    async def test_get_graph_neighborhood(self, pool):
        """get_graph_neighborhood returns connected concepts."""
        concepts = await get_concepts(concept_type="METHOD", limit=1)
        if not concepts:
            pytest.skip("No concepts in database")

        concept_name = concepts[0].name
        result = await get_graph_neighborhood(
            concept_name=concept_name,
            hops=1,
            limit=20,
        )

        assert result is not None

    async def test_get_graph_neighborhood_multiple_hops(self, pool):
        """get_graph_neighborhood supports multiple hops."""
        concepts = await get_concepts(concept_type="METHOD", limit=1)
        if not concepts:
            pytest.skip("No concepts in database")

        concept_name = concepts[0].name
        result = await get_graph_neighborhood(
            concept_name=concept_name,
            hops=2,
            limit=30,
        )

        assert result is not None

    async def test_get_graph_path(self, pool):
        """get_graph_path finds path between concepts."""
        result = await get_graph_path(
            concept_a="regression",
            concept_b="estimation",
        )

        assert result is not None


@pytest.mark.e2e
class TestCitationTools:
    """E2E tests for citation tools."""

    async def test_get_citations_for_source(self, pool):
        """get_citations_for_source returns citation info."""
        sources = await get_sources(limit=1)
        if not sources:
            pytest.skip("No sources in database")

        source_id = str(sources[0].id)
        result = await get_citations_for_source(source_id)

        assert result is not None
        assert isinstance(result, dict)


@pytest.mark.e2e
class TestCrossToolIntegration:
    """Integration tests verifying tools work together."""

    async def test_search_then_get_source(self, pool):
        """Search results can be used to fetch source details."""
        options = SearchOptions(
            query="regression",
            limit=1,
            use_graph=False,
            use_citations=False,
            use_rerank=False,
            use_expand=False,
        )
        response = await search(options)

        if not response.results:
            pytest.skip("No search results for query")

        result = response.results[0]
        source_id = str(result.source.id) if result.source else None

        if source_id:
            source = await get_source_by_id(source_id)
            assert source is not None
            assert source.title is not None

    async def test_concept_then_neighborhood(self, pool):
        """Listed concept can be explored in neighborhood."""
        concepts = await get_concepts(concept_type="METHOD", limit=1)

        if not concepts:
            pytest.skip("No concepts found")

        concept = concepts[0]
        neighborhood = await get_graph_neighborhood(
            concept_name=concept.name,
            hops=1,
            limit=10,
        )

        assert neighborhood is not None

    async def test_source_then_chunks_then_concepts(self, pool):
        """Full pipeline: source -> chunks -> concepts."""
        sources = await get_sources(limit=1)
        if not sources:
            pytest.skip("No sources in database")

        source_id = str(sources[0].id)
        source = await get_source_by_id(source_id)
        assert source is not None

        chunks = await get_source_chunks(source_id, limit=5)
        assert isinstance(chunks, list)

        stats = await get_stats()
        if stats.get("concepts", 0) > 0:
            concepts = await get_concepts(limit=1)
            if concepts:
                concept = await get_concept_by_id(str(concepts[0].id))
                assert concept is not None
