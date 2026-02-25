"""Tests for the service layer business logic.

Tests cover:
- SearchOptions and SearchResponse dataclasses
- Embedding cache operations
- Context weight calculations
- Search orchestration
- Source/concept/graph operations
"""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from research_kb_contracts import (
    Chunk,
    Concept,
    ConceptRelationship,
    ConceptType,
    RelationshipType,
    SearchResult,
    Source,
    SourceType,
)
from research_kb_api.service import (
    ChunkSummary,
    ContextType,
    ScoreBreakdown,
    SearchOptions,
    SearchResponse,
    SearchResultDetail,
    SourceSummary,
    get_cached_embedding,
    get_context_weights,
    get_embedding_client,
    search,
    get_sources,
    get_source_by_id,
    get_source_chunks,
    get_concepts,
    get_concept_by_id,
    get_concept_relationships,
    get_graph_neighborhood,
    get_graph_path,
    get_stats,
    get_citations_for_source,
    _embedding_cache,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_embedding():
    """Mock embedding vector."""
    return [0.1] * 1024


@pytest.fixture
def sample_source():
    """Create a sample source for testing."""
    now = datetime.now()
    return Source(
        id=uuid4(),
        title="Causal Inference in Statistics",
        authors=["Judea Pearl", "Madelyn Glymour"],
        year=2016,
        source_type=SourceType.TEXTBOOK,
        file_hash="abc123",
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def sample_chunk(sample_source):
    """Create a sample chunk for testing."""
    return Chunk(
        id=uuid4(),
        source_id=sample_source.id,
        content="The backdoor criterion provides a graphical test for identifying causal effects.",
        content_hash="chunk123",
        page_start=42,
        page_end=43,
        metadata={"section_header": "Chapter 3"},
        created_at=datetime.now(),
    )


@pytest.fixture
def sample_search_result(sample_source, sample_chunk):
    """Create a sample search result."""
    result = SearchResult(
        source=sample_source,
        chunk=sample_chunk,
        combined_score=0.85,
        rank=1,
    )
    # Add score components
    result.fts_score = 0.3
    result.vector_score = 0.6
    result.graph_score = 0.1
    result.citation_score = 0.05
    return result


@pytest.fixture
def sample_concept():
    """Create a sample concept for testing."""
    return Concept(
        id=uuid4(),
        name="backdoor criterion",
        canonical_name="backdoor_criterion",
        concept_type=ConceptType.THEOREM,
        definition="A graphical test for identifying adjustment sets",
        aliases=["backdoor", "back-door criterion"],
        created_at=datetime.now(),
    )


# =============================================================================
# Test Dataclasses
# =============================================================================


class TestSearchOptions:
    """Test SearchOptions dataclass."""

    def test_default_values(self):
        """Test SearchOptions has correct defaults."""
        options = SearchOptions(query="test query")

        assert options.query == "test query"
        assert options.limit == 10
        assert options.context_type == ContextType.balanced
        assert options.source_filter is None
        assert options.use_graph is True
        assert options.graph_weight == pytest.approx(0.2)
        assert options.use_rerank is True
        assert options.use_expand is True
        assert options.use_llm_expand is False

    def test_custom_values(self):
        """Test SearchOptions with custom values."""
        options = SearchOptions(
            query="instrumental variables",
            limit=20,
            context_type=ContextType.auditing,
            source_filter="PAPER",
            use_graph=False,
            graph_weight=0.3,
            use_rerank=False,
            use_expand=False,
            use_llm_expand=True,
        )

        assert options.query == "instrumental variables"
        assert options.limit == 20
        assert options.context_type == ContextType.auditing
        assert options.source_filter == "PAPER"
        assert options.use_graph is False
        assert options.graph_weight == pytest.approx(0.3)
        assert options.use_rerank is False
        assert options.use_expand is False
        assert options.use_llm_expand is True


class TestSearchResponse:
    """Test SearchResponse dataclass."""

    def test_default_values(self):
        """Test SearchResponse has correct defaults."""
        response = SearchResponse(query="test")

        assert response.query == "test"
        assert response.expanded_query is None
        assert response.results == []
        assert response.execution_time_ms == pytest.approx(0.0)
        assert response.embedding_time_ms == pytest.approx(0.0)
        assert response.search_time_ms == pytest.approx(0.0)

    def test_with_results(self):
        """Test SearchResponse with results populated."""
        result = SearchResultDetail(
            source=SourceSummary(id="src1", title="Test"),
            chunk=ChunkSummary(id="chk1", content="Test content"),
            concepts=["concept1"],
            scores=ScoreBreakdown(fts=0.3, vector=0.7, combined=0.5),
            combined_score=0.5,
        )

        response = SearchResponse(
            query="test",
            expanded_query="expanded test query",
            results=[result],
            execution_time_ms=150.5,
            embedding_time_ms=50.2,
            search_time_ms=100.3,
        )

        assert len(response.results) == 1
        assert response.expanded_query == "expanded test query"
        assert response.execution_time_ms == pytest.approx(150.5)


class TestScoreBreakdown:
    """Test ScoreBreakdown dataclass."""

    def test_default_values(self):
        """Test ScoreBreakdown has zero defaults."""
        scores = ScoreBreakdown()

        assert scores.fts == pytest.approx(0.0)
        assert scores.vector == pytest.approx(0.0)
        assert scores.graph == pytest.approx(0.0)
        assert scores.citation == pytest.approx(0.0)
        assert scores.combined == pytest.approx(0.0)

    def test_custom_values(self):
        """Test ScoreBreakdown with custom scores."""
        scores = ScoreBreakdown(
            fts=0.25,
            vector=0.65,
            graph=0.05,
            citation=0.05,
            combined=0.8,
        )

        assert scores.fts == pytest.approx(0.25)
        assert scores.vector == pytest.approx(0.65)
        assert scores.graph == pytest.approx(0.05)
        assert scores.citation == pytest.approx(0.05)
        assert scores.combined == pytest.approx(0.8)


class TestContextType:
    """Test ContextType enum."""

    def test_enum_values(self):
        """Test ContextType has expected values."""
        assert ContextType.building.value == "building"
        assert ContextType.auditing.value == "auditing"
        assert ContextType.balanced.value == "balanced"

    def test_context_type_from_string(self):
        """Test ContextType can be created from string."""
        assert ContextType("building") == ContextType.building
        assert ContextType("auditing") == ContextType.auditing
        assert ContextType("balanced") == ContextType.balanced


# =============================================================================
# Test Context Weights
# =============================================================================


class TestGetContextWeights:
    """Test get_context_weights function."""

    def test_building_weights(self):
        """Test building context favors vector search."""
        fts, vector = get_context_weights(ContextType.building)

        assert fts == pytest.approx(0.2)
        assert vector == pytest.approx(0.8)

    def test_auditing_weights(self):
        """Test auditing context has balanced weights."""
        fts, vector = get_context_weights(ContextType.auditing)

        assert fts == pytest.approx(0.5)
        assert vector == pytest.approx(0.5)

    def test_balanced_weights(self):
        """Test balanced context slightly favors vector."""
        fts, vector = get_context_weights(ContextType.balanced)

        assert fts == pytest.approx(0.3)
        assert vector == pytest.approx(0.7)


# =============================================================================
# Test Embedding Client
# =============================================================================


class TestGetEmbeddingClient:
    """Test embedding client lazy loading."""

    def test_lazy_loading(self):
        """Test embedding client is lazily loaded."""
        import research_kb_api.service as service_module

        # Reset the global client
        original = service_module._embedding_client
        service_module._embedding_client = None

        with patch("research_kb_api.service.EmbeddingClient") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            # First call should create the client
            client1 = get_embedding_client()
            assert mock_class.call_count == 1

            # Second call should return the same instance
            service_module._embedding_client = mock_instance
            client2 = get_embedding_client()
            # Should still be the same mock, not creating a new one
            assert client2 is mock_instance

        # Restore
        service_module._embedding_client = original


class TestGetCachedEmbedding:
    """Test embedding caching."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear embedding cache before each test."""
        _embedding_cache.clear()
        yield
        _embedding_cache.clear()

    async def test_cache_miss_generates_embedding(self, mock_embedding):
        """Test cache miss triggers embedding generation."""
        with patch("research_kb_api.service.get_embedding_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.embed_query.return_value = mock_embedding
            mock_get_client.return_value = mock_client

            result = await get_cached_embedding("test query")

            assert result == mock_embedding
            mock_client.embed_query.assert_called_once_with("test query")

    async def test_cache_hit_returns_cached(self, mock_embedding):
        """Test cache hit returns cached embedding without regeneration."""
        # Pre-populate cache
        _embedding_cache["cached query"] = mock_embedding

        with patch("research_kb_api.service.get_embedding_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            result = await get_cached_embedding("cached query")

            assert result == mock_embedding
            # Should not call embed_query since it's cached
            mock_client.embed_query.assert_not_called()

    async def test_cache_eviction_when_full(self, mock_embedding):
        """Test cache evicts old entries when over limit."""
        # Fill cache with more than 1000 entries to trigger eviction
        for i in range(1010):
            _embedding_cache[f"query_{i}"] = [0.1] * 1024

        with patch("research_kb_api.service.get_embedding_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.embed_query.return_value = mock_embedding
            mock_get_client.return_value = mock_client

            # Add a new entry - this should trigger eviction
            await get_cached_embedding("new_query")

            # Cache should have been trimmed
            # After adding new entry and evicting 500, should be ~511
            assert len(_embedding_cache) <= 600


# =============================================================================
# Test Search Function
# =============================================================================


class TestSearch:
    """Test main search function."""

    @pytest.fixture
    def mock_search_deps(self, mock_embedding, sample_search_result):
        """Mock all search dependencies."""
        with (
            patch("research_kb_api.service.get_cached_embedding") as embed_mock,
            patch("research_kb_api.service.ConceptStore") as concept_mock,
            patch("research_kb_api.service.search_with_expansion") as expand_mock,
            patch("research_kb_api.service.search_with_rerank") as rerank_mock,
            patch("research_kb_api.service.search_hybrid_v2") as hybrid_v2_mock,
            patch("research_kb_api.service.search_hybrid") as hybrid_mock,
        ):

            embed_mock.return_value = mock_embedding
            concept_mock.count = AsyncMock(return_value=100)
            expand_mock.return_value = ([sample_search_result], None)
            rerank_mock.return_value = [sample_search_result]
            hybrid_v2_mock.return_value = [sample_search_result]
            hybrid_mock.return_value = [sample_search_result]

            yield {
                "embed": embed_mock,
                "concept": concept_mock,
                "expand": expand_mock,
                "rerank": rerank_mock,
                "hybrid_v2": hybrid_v2_mock,
                "hybrid": hybrid_mock,
            }

    async def test_basic_search(self, mock_search_deps, sample_source):
        """Test basic search returns results."""
        options = SearchOptions(query="backdoor criterion")

        response = await search(options)

        assert response.query == "backdoor criterion"
        assert len(response.results) == 1
        assert response.results[0].source.title == sample_source.title
        assert response.execution_time_ms > 0

    async def test_search_timing_recorded(self, mock_search_deps):
        """Test search records timing metrics."""
        options = SearchOptions(query="test")

        response = await search(options)

        assert response.embedding_time_ms >= 0
        assert response.search_time_ms >= 0
        assert response.execution_time_ms >= response.embedding_time_ms + response.search_time_ms

    async def test_search_uses_expansion_by_default(self, mock_search_deps):
        """Test search uses expansion when enabled."""
        options = SearchOptions(query="test", use_expand=True)

        await search(options)

        mock_search_deps["expand"].assert_called_once()

    async def test_search_uses_rerank_without_expansion(
        self, mock_search_deps, sample_search_result
    ):
        """Test search uses rerank when expansion disabled."""
        mock_search_deps["rerank"].return_value = [sample_search_result]
        options = SearchOptions(query="test", use_expand=False, use_rerank=True)

        await search(options)

        mock_search_deps["rerank"].assert_called_once()

    async def test_search_uses_hybrid_v2_with_graph(self, mock_search_deps, sample_search_result):
        """Test search uses hybrid_v2 with graph enabled."""
        mock_search_deps["hybrid_v2"].return_value = [sample_search_result]
        options = SearchOptions(
            query="test",
            use_expand=False,
            use_rerank=False,
            use_graph=True,
        )

        await search(options)

        mock_search_deps["hybrid_v2"].assert_called_once()

    async def test_search_uses_hybrid_without_graph(self, mock_search_deps, sample_search_result):
        """Test search uses basic hybrid without graph."""
        mock_search_deps["hybrid"].return_value = [sample_search_result]
        options = SearchOptions(
            query="test",
            use_expand=False,
            use_rerank=False,
            use_graph=False,
        )

        await search(options)

        mock_search_deps["hybrid"].assert_called_once()

    async def test_search_falls_back_when_no_concepts(self, mock_search_deps, sample_search_result):
        """Test search falls back to non-graph when no concepts exist."""
        mock_search_deps["concept"].count = AsyncMock(return_value=0)
        mock_search_deps["hybrid"].return_value = [sample_search_result]
        options = SearchOptions(
            query="test",
            use_expand=False,
            use_rerank=False,
            use_graph=True,  # Requested but should fall back
        )

        await search(options)

        # Should fall back to basic hybrid since no concepts
        mock_search_deps["hybrid"].assert_called_once()

    async def test_search_normalizes_weights_with_graph(self, mock_search_deps):
        """Test search normalizes weights when graph is enabled."""
        options = SearchOptions(
            query="test",
            context_type=ContextType.balanced,  # 0.3 FTS, 0.7 vector
            use_graph=True,
            graph_weight=0.2,
        )

        await search(options)

        # Total should be 0.3 + 0.7 + 0.2 = 1.2
        # Normalized: FTS = 0.3/1.2 = 0.25, vector = 0.7/1.2 = 0.583, graph = 0.2/1.2 = 0.167
        call_args = mock_search_deps["expand"].call_args
        query = call_args[0][0]  # First positional argument is SearchQuery

        # Weights should be normalized
        total = query.fts_weight + query.vector_weight + query.graph_weight
        assert total == pytest.approx(1.0, rel=0.01)

    async def test_search_populates_expanded_query(self, mock_search_deps, sample_search_result):
        """Test search includes expanded query when available."""
        expanded = MagicMock()
        # Service uses expanded_terms (list), not expanded_text
        expanded.expanded_terms = ["test", "query", "expanded", "with", "synonyms"]
        mock_search_deps["expand"].return_value = ([sample_search_result], expanded)

        options = SearchOptions(query="test")

        response = await search(options)

        # Service joins expanded_terms with ", "
        assert response.expanded_query == "test, query, expanded, with, synonyms"


# =============================================================================
# Test Source Operations
# =============================================================================


class TestSourceOperations:
    """Test source-related service functions."""

    async def test_get_sources(self, sample_source):
        """Test get_sources returns list of sources."""
        with patch("research_kb_api.service.SourceStore") as mock_store:
            mock_store.list_all = AsyncMock(return_value=[sample_source])

            result = await get_sources(limit=10, offset=0)

            assert len(result) == 1
            assert result[0].title == sample_source.title
            mock_store.list_all.assert_called_once_with(limit=10, offset=0, source_type=None)

    async def test_get_sources_with_filter(self, sample_source):
        """Test get_sources with source type filter."""
        with patch("research_kb_api.service.SourceStore") as mock_store:
            mock_store.list_all = AsyncMock(return_value=[sample_source])

            await get_sources(limit=50, offset=10, source_type="TEXTBOOK")

            mock_store.list_all.assert_called_once_with(limit=50, offset=10, source_type="TEXTBOOK")

    async def test_get_source_by_id(self, sample_source):
        """Test get_source_by_id returns source."""
        with patch("research_kb_api.service.SourceStore") as mock_store:
            mock_store.get = AsyncMock(return_value=sample_source)
            source_id = str(sample_source.id)

            result = await get_source_by_id(source_id)

            assert result is not None
            assert result.title == sample_source.title

    async def test_get_source_by_id_not_found(self):
        """Test get_source_by_id returns None when not found."""
        with patch("research_kb_api.service.SourceStore") as mock_store:
            mock_store.get = AsyncMock(return_value=None)

            result = await get_source_by_id(str(uuid4()))

            assert result is None

    async def test_get_source_chunks(self, sample_source, sample_chunk):
        """Test get_source_chunks returns chunks for a source."""
        with patch("research_kb_api.service.ChunkStore") as mock_store:
            mock_store.get_by_source = AsyncMock(return_value=[sample_chunk])

            result = await get_source_chunks(str(sample_source.id), limit=50)

            assert len(result) == 1
            assert result[0].content == sample_chunk.content


# =============================================================================
# Test Concept Operations
# =============================================================================


class TestConceptOperations:
    """Test concept-related service functions."""

    async def test_get_concepts_no_query(self, sample_concept):
        """Test get_concepts without query returns all concepts."""
        with patch("research_kb_api.service.ConceptStore") as mock_store:
            mock_store.list_all = AsyncMock(return_value=[sample_concept])

            result = await get_concepts(limit=50)

            assert len(result) == 1
            mock_store.list_all.assert_called_once()

    async def test_get_concepts_with_query(self, sample_concept):
        """Test get_concepts with query searches concepts."""
        with patch("research_kb_api.service.ConceptStore") as mock_store:
            mock_store.search = AsyncMock(return_value=[sample_concept])

            result = await get_concepts(query="backdoor", limit=50)

            assert len(result) == 1
            mock_store.search.assert_called_once_with("backdoor", limit=50)

    async def test_get_concept_by_id(self, sample_concept):
        """Test get_concept_by_id returns concept."""
        with patch("research_kb_api.service.ConceptStore") as mock_store:
            mock_store.get = AsyncMock(return_value=sample_concept)

            result = await get_concept_by_id(str(sample_concept.id))

            assert result is not None
            assert result.name == sample_concept.name

    async def test_get_concept_relationships(self, sample_concept):
        """Test get_concept_relationships returns relationships."""
        relationship = ConceptRelationship(
            id=uuid4(),
            source_concept_id=sample_concept.id,
            target_concept_id=uuid4(),
            relationship_type=RelationshipType.REQUIRES,
            created_at=datetime.now(),
        )

        with patch("research_kb_api.service.RelationshipStore") as mock_store:
            mock_store.get_for_concept = AsyncMock(return_value=[relationship])

            result = await get_concept_relationships(str(sample_concept.id))

            assert len(result) == 1
            assert result[0].relationship_type == RelationshipType.REQUIRES


# =============================================================================
# Test Graph Operations
# =============================================================================


class TestGraphOperations:
    """Test graph-related service functions."""

    async def test_get_graph_neighborhood(self, sample_concept):
        """Test get_graph_neighborhood returns neighborhood data."""
        with (
            patch("research_kb_api.service.ConceptStore") as concept_mock,
            patch("research_kb_api.service.get_neighborhood") as neighbor_mock,
        ):

            concept_mock.search = AsyncMock(return_value=[sample_concept])
            neighbor_mock.return_value = {
                "concepts": [sample_concept],
                "relationships": [],
            }

            result = await get_graph_neighborhood("backdoor criterion", hops=2)

            assert "center" in result
            assert result["center"]["name"] == sample_concept.name
            assert "nodes" in result
            assert "edges" in result

    async def test_get_graph_neighborhood_not_found(self):
        """Test get_graph_neighborhood returns error when concept not found."""
        with patch("research_kb_api.service.ConceptStore") as concept_mock:
            concept_mock.search = AsyncMock(return_value=[])

            result = await get_graph_neighborhood("nonexistent concept")

            assert "error" in result
            assert "not found" in result["error"]

    async def test_get_graph_path(self, sample_concept):
        """Test get_graph_path finds path between concepts."""
        concept_a = sample_concept
        concept_b = Concept(
            id=uuid4(),
            name="instrumental variables",
            canonical_name="instrumental_variables",
            concept_type=ConceptType.METHOD,
            created_at=datetime.now(),
        )

        with (
            patch("research_kb_api.service.ConceptStore") as concept_mock,
            patch("research_kb_api.service.find_shortest_path") as path_mock,
        ):

            concept_mock.search = AsyncMock(side_effect=[[concept_a], [concept_b]])
            # Service expects list of (Concept, Relationship) tuples
            path_mock.return_value = [(concept_a, None), (concept_b, None)]

            result = await get_graph_path("backdoor criterion", "instrumental variables")

            assert result["from"] == "backdoor criterion"
            assert result["to"] == "instrumental variables"
            assert "path" in result

    async def test_get_graph_path_concept_a_not_found(self, sample_concept):
        """Test get_graph_path returns error when first concept not found."""
        with patch("research_kb_api.service.ConceptStore") as concept_mock:
            concept_mock.search = AsyncMock(return_value=[])

            result = await get_graph_path("nonexistent", "backdoor criterion")

            assert "error" in result
            assert "nonexistent" in result["error"]

    async def test_get_graph_path_concept_b_not_found(self, sample_concept):
        """Test get_graph_path returns error when second concept not found."""
        with patch("research_kb_api.service.ConceptStore") as concept_mock:
            concept_mock.search = AsyncMock(side_effect=[[sample_concept], []])

            result = await get_graph_path("backdoor criterion", "nonexistent")

            assert "error" in result
            assert "nonexistent" in result["error"]


# =============================================================================
# Test Stats and Citations
# =============================================================================


class TestStatsAndCitations:
    """Test stats and citation service functions."""

    async def test_get_stats(self):
        """Test get_stats returns database statistics."""
        # Create a proper async context manager mock
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=[100, 5000, 200, 500, 300, 1000])

        # Create an async context manager that returns the connection
        class MockAcquire:
            async def __aenter__(self):
                return mock_conn

            async def __aexit__(self, *args):
                pass

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = MockAcquire()

        async def mock_get_pool(config):
            return mock_pool

        # Patch at the storage module level where it's imported from
        with (
            patch("research_kb_storage.get_connection_pool", mock_get_pool),
            patch("research_kb_storage.DatabaseConfig"),
        ):

            result = await get_stats()

            assert result["sources"] == 100
            assert result["chunks"] == 5000
            assert result["concepts"] == 200
            assert result["relationships"] == 500
            assert result["citations"] == 300
            assert result["chunk_concepts"] == 1000

    async def test_get_citations_for_source(self, sample_source):
        """Test get_citations_for_source returns citation info."""
        citing_source = Source(
            id=uuid4(),
            title="Citing Paper",
            source_type=SourceType.PAPER,
            year=2020,
            file_hash="cite1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        cited_source = Source(
            id=uuid4(),
            title="Cited Paper",
            source_type=SourceType.PAPER,
            year=2010,
            file_hash="cite2",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with (
            patch("research_kb_api.service.get_citing_sources") as citing_mock,
            patch("research_kb_api.service.get_cited_sources") as cited_mock,
        ):

            citing_mock.return_value = [citing_source]
            cited_mock.return_value = [cited_source]

            result = await get_citations_for_source(str(sample_source.id))

            assert result["source_id"] == str(sample_source.id)
            assert len(result["citing_sources"]) == 1
            assert result["citing_sources"][0]["title"] == "Citing Paper"
            assert len(result["cited_sources"]) == 1
            assert result["cited_sources"][0]["title"] == "Cited Paper"


# =============================================================================
# Test Summary Dataclasses
# =============================================================================


class TestSummaryDataclasses:
    """Test summary dataclasses used in responses."""

    def test_source_summary(self):
        """Test SourceSummary dataclass."""
        summary = SourceSummary(
            id="src123",
            title="Test Paper",
            authors=["Author 1", "Author 2"],
            year=2023,
            source_type="PAPER",
        )

        assert summary.id == "src123"
        assert summary.title == "Test Paper"
        assert len(summary.authors) == 2
        assert summary.year == 2023
        assert summary.source_type == "PAPER"

    def test_source_summary_defaults(self):
        """Test SourceSummary default values."""
        summary = SourceSummary(id="src1", title="Test")

        assert summary.authors == []
        assert summary.year is None
        assert summary.source_type is None

    def test_chunk_summary(self):
        """Test ChunkSummary dataclass."""
        summary = ChunkSummary(
            id="chk123",
            content="Test content here",
            page_start=10,
            page_end=11,
            section="Introduction",
        )

        assert summary.id == "chk123"
        assert summary.content == "Test content here"
        assert summary.page_start == 10
        assert summary.page_end == 11
        assert summary.section == "Introduction"

    def test_chunk_summary_defaults(self):
        """Test ChunkSummary default values."""
        summary = ChunkSummary(id="chk1", content="Content")

        assert summary.page_start is None
        assert summary.page_end is None
        assert summary.section is None

    def test_search_result_detail(self):
        """Test SearchResultDetail dataclass."""
        detail = SearchResultDetail(
            source=SourceSummary(id="src1", title="Test"),
            chunk=ChunkSummary(id="chk1", content="Content"),
            concepts=["concept1", "concept2"],
            scores=ScoreBreakdown(fts=0.3, vector=0.7, combined=0.5),
            combined_score=0.5,
        )

        assert detail.source.id == "src1"
        assert detail.chunk.id == "chk1"
        assert len(detail.concepts) == 2
        assert detail.scores.fts == pytest.approx(0.3)
        assert detail.combined_score == pytest.approx(0.5)

    def test_search_result_detail_defaults(self):
        """Test SearchResultDetail default values."""
        detail = SearchResultDetail(
            source=SourceSummary(id="src1", title="Test"),
            chunk=ChunkSummary(id="chk1", content="Content"),
        )

        assert detail.concepts == []
        assert detail.scores.fts == pytest.approx(0.0)
        assert detail.combined_score == pytest.approx(0.0)
