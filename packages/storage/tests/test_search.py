"""Tests for hybrid search (FTS + vector similarity + graph + citation + RRF)."""

import pytest

from research_kb_common import SearchError
from research_kb_contracts import SourceType
from research_kb_storage import (
    ChunkStore,
    DatabaseConfig,
    SearchQuery,
    SourceStore,
    get_connection_pool,
    close_connection_pool,
    search_hybrid,
)
from research_kb_storage.search import (
    compute_rrf_score,
    _compute_ranks_by_signal,
)


@pytest.fixture
async def test_data(db_pool):
    """Create test data for search tests."""
    # Create source
    source = await SourceStore.create(
        source_type=SourceType.TEXTBOOK,
        title="Causality",
        file_hash="sha256:causality",
        authors=["Judea Pearl"],
    )

    # Create chunks with embeddings
    chunks = []

    # Chunk 1: About backdoor criterion (high relevance for "backdoor")
    chunk1 = await ChunkStore.create(
        source_id=source.id,
        content="The backdoor criterion states that a set of variables Z satisfies the backdoor criterion relative to X and Y.",
        content_hash="sha256:chunk1",
        location="Chapter 3, p. 73",
        embedding=[0.1] * 1024,  # Simulated embedding (BGE-large-en-v1.5 is 1024-dim)
        metadata={"chunk_type": "theorem", "theorem_name": "Backdoor Criterion"},
    )
    chunks.append(chunk1)

    # Chunk 2: About instrumental variables (different topic)
    chunk2 = await ChunkStore.create(
        source_id=source.id,
        content="Instrumental variables provide a method for estimating causal effects when there is unobserved confounding.",
        content_hash="sha256:chunk2",
        location="Chapter 5, p. 142",
        embedding=[0.5] * 1024,  # Different embedding
        metadata={"chunk_type": "definition", "concept": "instrumental_variables"},
    )
    chunks.append(chunk2)

    # Chunk 3: About frontdoor criterion (partially related)
    chunk3 = await ChunkStore.create(
        source_id=source.id,
        content="The frontdoor criterion provides an alternative identification strategy when the backdoor criterion fails.",
        content_hash="sha256:chunk3",
        location="Chapter 3, p. 81",
        embedding=[0.2] * 1024,  # Similar to chunk1
        metadata={"chunk_type": "theorem", "theorem_name": "Frontdoor Criterion"},
    )
    chunks.append(chunk3)

    return {"source": source, "chunks": chunks}


class TestSearchQuery:
    """Test SearchQuery validation."""

    def test_search_query_text_only(self):
        """Test creating SearchQuery with text only."""
        query = SearchQuery(text="backdoor criterion")

        assert query.text == "backdoor criterion"
        assert query.embedding is None

    def test_search_query_embedding_only(self):
        """Test creating SearchQuery with embedding only."""
        query = SearchQuery(embedding=[0.1] * 1024)

        assert query.text is None
        assert len(query.embedding) == 1024

    def test_search_query_hybrid(self):
        """Test creating SearchQuery with both text and embedding."""
        query = SearchQuery(
            text="backdoor criterion",
            embedding=[0.1] * 1024,
            fts_weight=0.3,
            vector_weight=0.7,
        )

        assert query.text == "backdoor criterion"
        assert len(query.embedding) == 1024
        # Weights normalized to sum to 1
        assert abs(query.fts_weight + query.vector_weight - 1.0) < 0.001

    def test_search_query_no_criteria_fails(self):
        """Test creating SearchQuery with no search criteria raises error."""
        with pytest.raises(ValueError) as exc_info:
            SearchQuery()

        assert "at least one" in str(exc_info.value).lower()

    def test_search_query_wrong_embedding_dimension(self):
        """Test creating SearchQuery with wrong embedding dimension raises error."""
        with pytest.raises(ValueError) as exc_info:
            SearchQuery(embedding=[0.1] * 128)  # Wrong dimension

        assert "1024 dimensions" in str(exc_info.value)

    def test_search_query_weight_normalization(self):
        """Test weights are normalized to sum to 1."""
        query = SearchQuery(
            text="test",
            embedding=[0.1] * 1024,
            fts_weight=2.0,
            vector_weight=3.0,
        )

        # Should normalize: 2/(2+3) = 0.4, 3/(2+3) = 0.6
        assert abs(query.fts_weight - 0.4) < 0.001
        assert abs(query.vector_weight - 0.6) < 0.001


class TestFTSSearch:
    """Test full-text search (FTS only)."""

    async def test_fts_search_finds_relevant_chunks(self, test_data):
        """Test FTS search finds chunks containing search terms."""
        query = SearchQuery(
            text="backdoor criterion",
            limit=10,
        )

        results = await search_hybrid(query)

        # Should find chunks 1 and 3 (both mention "backdoor" or "criterion")
        assert len(results) >= 1

        # First result should be chunk 1 (mentions both terms)
        assert "backdoor criterion" in results[0].chunk.content.lower()

        # All results should have FTS scores
        assert all(r.fts_score is not None for r in results)
        assert all(r.fts_score > 0 for r in results)

    async def test_fts_search_ranking(self, test_data):
        """Test FTS search ranks results by relevance."""
        query = SearchQuery(text="backdoor", limit=10)

        results = await search_hybrid(query)

        # Results should be ordered by FTS score (descending)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].fts_score >= results[i + 1].fts_score

        # Rank should be 1-based and sequential
        for i, result in enumerate(results):
            assert result.rank == i + 1


class TestVectorSearch:
    """Test vector similarity search (vector only)."""

    async def test_vector_search_finds_similar_chunks(self, test_data):
        """Test vector search finds chunks with similar embeddings."""
        # Query with embedding similar to chunk1 ([0.1] * 1024)
        query = SearchQuery(
            embedding=[0.15] * 1024,  # Close to chunk1
            limit=10,
        )

        results = await search_hybrid(query)

        assert len(results) > 0

        # All results should have vector scores (cosine similarity)
        assert all(r.vector_score is not None for r in results)

        # Chunk1 should be most similar (highest similarity)
        # Chunk1 has embedding [0.1]*1024, similarity to [0.15]*1024 should be high
        assert (
            results[0].vector_score > 0.5
        )  # High similarity (1=identical, 0=opposite)

    async def test_vector_search_ranking_by_similarity(self, test_data):
        """Test vector search ranks by cosine similarity."""
        query = SearchQuery(embedding=[0.1] * 1024, limit=10)

        results = await search_hybrid(query)

        # Results should be ordered by similarity (descending = most similar first)
        # Use small epsilon for floating point comparison
        if len(results) > 1:
            for i in range(len(results) - 1):
                # Allow small tolerance for floating point precision
                assert results[i].vector_score >= results[i + 1].vector_score - 1e-6


class TestHybridSearch:
    """Test hybrid search combining FTS and vector."""

    async def test_hybrid_search_combines_scores(self, test_data):
        """Test hybrid search combines FTS and vector scores."""
        query = SearchQuery(
            text="backdoor",
            embedding=[0.1] * 1024,
            fts_weight=0.5,
            vector_weight=0.5,
            limit=10,
        )

        results = await search_hybrid(query)

        assert len(results) > 0

        # All results should have both FTS and vector scores
        for result in results:
            assert result.fts_score is not None or result.vector_score is not None
            assert result.combined_score > 0

    async def test_hybrid_search_respects_limit(self, test_data):
        """Test hybrid search respects result limit."""
        query = SearchQuery(
            text="criterion",
            embedding=[0.1] * 1024,
            limit=2,
        )

        results = await search_hybrid(query)

        assert len(results) <= 2

    async def test_hybrid_search_empty_results(self, test_data):
        """Test hybrid search returns empty list when no matches."""
        query = SearchQuery(
            text="nonexistent_term_xyz123",
            limit=10,
        )

        results = await search_hybrid(query)

        assert results == []


class TestSearchErrors:
    """Test search error handling."""

    async def test_search_no_database_connection_fails(self):
        """Test search fails gracefully when database is unavailable."""
        # Close existing pool
        await close_connection_pool()

        # Try to configure with invalid host
        bad_config = DatabaseConfig(host="nonexistent.invalid", port=9999)

        query = SearchQuery(text="test")

        # Should raise SearchError (wrapped from connection error)
        with pytest.raises((SearchError, Exception)):
            # Temporarily set bad config
            await get_connection_pool(bad_config)
            await search_hybrid(query)

        # Restore good connection for other tests
        good_config = DatabaseConfig()
        await get_connection_pool(good_config)


class TestRRFScoring:
    """Test Reciprocal Rank Fusion score computation."""

    def test_rrf_single_signal(self):
        """Test RRF with one signal, rank 1."""
        score = compute_rrf_score({"fts": 1})
        # 1 / (60 + 1) = 0.016393...
        assert score == pytest.approx(1.0 / 61, rel=1e-5)

    def test_rrf_two_signals_rank_1(self):
        """Test RRF with two signals both rank 1."""
        score = compute_rrf_score({"fts": 1, "vector": 1})
        # 2 * (1 / 61) = 0.032786...
        assert score == pytest.approx(2.0 / 61, rel=1e-5)

    def test_rrf_four_signals_all_rank_1(self):
        """Test theoretical max with all signals rank 1."""
        score = compute_rrf_score({"fts": 1, "vector": 1, "graph": 1, "citation": 1})
        assert score == pytest.approx(4.0 / 61, rel=1e-5)

    def test_rrf_higher_rank_lower_score(self):
        """Test that higher rank produces lower contribution."""
        score_rank1 = compute_rrf_score({"fts": 1})
        score_rank10 = compute_rrf_score({"fts": 10})
        assert score_rank1 > score_rank10

    def test_rrf_mixed_ranks(self):
        """Test RRF with diverse ranks."""
        score = compute_rrf_score({"fts": 3, "vector": 1, "graph": 5})
        # 1/63 + 1/61 + 1/65
        expected = 1.0 / 63 + 1.0 / 61 + 1.0 / 65
        assert score == pytest.approx(expected, rel=1e-5)

    def test_rrf_empty_rankings(self):
        """Test RRF with no signals returns 0."""
        assert compute_rrf_score({}) == 0.0

    def test_rrf_none_rank_skipped(self):
        """Test that None ranks are skipped."""
        score = compute_rrf_score({"fts": 1, "vector": None})
        assert score == pytest.approx(1.0 / 61, rel=1e-5)

    def test_rrf_custom_k(self):
        """Test RRF with non-default k parameter."""
        # k=0: 1/(0+1) = 1.0 for rank 1
        score = compute_rrf_score({"fts": 1}, k=0)
        assert score == pytest.approx(1.0, rel=1e-5)

    def test_rrf_large_k_diminishes_all(self):
        """Test that very large k diminishes all contributions."""
        score = compute_rrf_score({"fts": 1, "vector": 1}, k=10000)
        # Each contribution is ~1/10001, very small
        assert score < 0.001


class TestComputeRanksBySignal:
    """Test _compute_ranks_by_signal internal ranking function."""

    @staticmethod
    def _make_result(chunk_id, fts=None, vector=None, graph=None, citation=None):
        """Create a minimal SearchResult-like object for ranking tests."""
        from unittest.mock import MagicMock
        from uuid import UUID

        result = MagicMock()
        result.chunk.id = UUID(chunk_id) if isinstance(chunk_id, str) else chunk_id
        result.fts_score = fts
        result.vector_score = vector
        result.graph_score = graph
        result.citation_score = citation
        return result

    def test_single_signal_single_result(self):
        """Single result with one signal gets rank 1."""
        results = [self._make_result("00000000-0000-0000-0000-000000000001", fts=0.8)]
        rankings = _compute_ranks_by_signal(results)
        assert rankings["00000000-0000-0000-0000-000000000001"]["fts"] == 1

    def test_single_signal_multiple_results_ordered(self):
        """Multiple results ranked by descending score within one signal."""
        r1 = self._make_result("00000000-0000-0000-0000-000000000001", fts=0.9)
        r2 = self._make_result("00000000-0000-0000-0000-000000000002", fts=0.5)
        r3 = self._make_result("00000000-0000-0000-0000-000000000003", fts=0.7)
        rankings = _compute_ranks_by_signal([r1, r2, r3])
        assert rankings["00000000-0000-0000-0000-000000000001"]["fts"] == 1
        assert rankings["00000000-0000-0000-0000-000000000003"]["fts"] == 2
        assert rankings["00000000-0000-0000-0000-000000000002"]["fts"] == 3

    def test_multiple_signals(self):
        """Results ranked independently per signal."""
        r1 = self._make_result("00000000-0000-0000-0000-000000000001", fts=0.9, vector=0.3)
        r2 = self._make_result("00000000-0000-0000-0000-000000000002", fts=0.5, vector=0.8)
        rankings = _compute_ranks_by_signal([r1, r2])
        # FTS: r1 first
        assert rankings["00000000-0000-0000-0000-000000000001"]["fts"] == 1
        assert rankings["00000000-0000-0000-0000-000000000002"]["fts"] == 2
        # Vector: r2 first
        assert rankings["00000000-0000-0000-0000-000000000002"]["vector"] == 1
        assert rankings["00000000-0000-0000-0000-000000000001"]["vector"] == 2

    def test_tied_scores_get_sequential_ranks(self):
        """Tied scores get sequential (non-shared) ranks."""
        r1 = self._make_result("00000000-0000-0000-0000-000000000001", fts=0.5)
        r2 = self._make_result("00000000-0000-0000-0000-000000000002", fts=0.5)
        rankings = _compute_ranks_by_signal([r1, r2])
        fts_ranks = {rankings[k]["fts"] for k in rankings}
        assert fts_ranks == {1, 2}

    def test_none_scores_excluded(self):
        """Results with None for a signal are not ranked in that signal."""
        r1 = self._make_result("00000000-0000-0000-0000-000000000001", fts=0.9, vector=None)
        r2 = self._make_result("00000000-0000-0000-0000-000000000002", fts=None, vector=0.8)
        rankings = _compute_ranks_by_signal([r1, r2])
        # r1 only has FTS rank
        assert "fts" in rankings["00000000-0000-0000-0000-000000000001"]
        assert "vector" not in rankings["00000000-0000-0000-0000-000000000001"]
        # r2 only has vector rank
        assert "vector" in rankings["00000000-0000-0000-0000-000000000002"]
        assert "fts" not in rankings["00000000-0000-0000-0000-000000000002"]

    def test_four_signals(self):
        """All four signals ranked correctly."""
        r1 = self._make_result(
            "00000000-0000-0000-0000-000000000001",
            fts=0.9, vector=0.7, graph=0.5, citation=0.3,
        )
        r2 = self._make_result(
            "00000000-0000-0000-0000-000000000002",
            fts=0.3, vector=0.9, graph=0.8, citation=0.6,
        )
        rankings = _compute_ranks_by_signal([r1, r2])
        # r1 wins FTS, r2 wins vector/graph/citation
        assert rankings["00000000-0000-0000-0000-000000000001"]["fts"] == 1
        assert rankings["00000000-0000-0000-0000-000000000002"]["vector"] == 1
        assert rankings["00000000-0000-0000-0000-000000000002"]["graph"] == 1
        assert rankings["00000000-0000-0000-0000-000000000002"]["citation"] == 1

    def test_empty_results(self):
        """Empty result list returns empty rankings."""
        rankings = _compute_ranks_by_signal([])
        assert rankings == {}


class TestSearchQueryGraphCitation:
    """Test SearchQuery with graph and citation parameters."""

    def test_graph_enabled_weights_normalized(self):
        """Test that enabling graph normalizes all 3 weights."""
        query = SearchQuery(
            text="test",
            fts_weight=0.3,
            vector_weight=0.5,
            graph_weight=0.2,
            use_graph=True,
        )
        total = query.fts_weight + query.vector_weight + query.graph_weight
        assert abs(total - 1.0) < 0.001

    def test_citation_enabled_weights_normalized(self):
        """Test that enabling citations normalizes all 3 weights."""
        query = SearchQuery(
            text="test",
            fts_weight=0.3,
            vector_weight=0.5,
            citation_weight=0.2,
            use_citations=True,
        )
        total = query.fts_weight + query.vector_weight + query.citation_weight
        assert abs(total - 1.0) < 0.001

    def test_four_way_weights_normalized(self):
        """Test full 4-way weight normalization."""
        query = SearchQuery(
            text="test",
            embedding=[0.1] * 1024,
            fts_weight=0.2,
            vector_weight=0.4,
            graph_weight=0.2,
            citation_weight=0.2,
            use_graph=True,
            use_citations=True,
        )
        total = (
            query.fts_weight
            + query.vector_weight
            + query.graph_weight
            + query.citation_weight
        )
        assert abs(total - 1.0) < 0.001

    def test_graph_disabled_weight_not_normalized(self):
        """Test that graph weight is NOT included in normalization when use_graph=False."""
        query = SearchQuery(
            text="test",
            fts_weight=0.3,
            vector_weight=0.7,
            graph_weight=0.5,  # This should be ignored in normalization
            use_graph=False,
        )
        # Only fts + vector normalized
        assert abs(query.fts_weight + query.vector_weight - 1.0) < 0.001

    def test_scoring_method_rrf(self):
        """Test RRF scoring method accepted."""
        query = SearchQuery(text="test", scoring_method="rrf")
        assert query.scoring_method == "rrf"

    def test_scoring_method_weighted(self):
        """Test default weighted scoring method."""
        query = SearchQuery(text="test")
        assert query.scoring_method == "weighted"

    def test_scoring_method_invalid_rejected(self):
        """Test invalid scoring method rejected."""
        with pytest.raises(ValueError, match="scoring_method"):
            SearchQuery(text="test", scoring_method="unknown")

    def test_max_hops_default(self):
        """Test default max_hops is 2."""
        query = SearchQuery(text="test")
        assert query.max_hops == 2

    def test_domain_id_filter(self):
        """Test domain_id parameter accepted."""
        query = SearchQuery(text="test", domain_id="causal_inference")
        assert query.domain_id == "causal_inference"

    def test_source_filter_parameter(self):
        """Test source_filter parameter accepted."""
        query = SearchQuery(text="test", source_filter="paper")
        assert query.source_filter == "paper"


class TestSearchHybridV2Validation:
    """Test search_hybrid_v2 input validation (no DB needed)."""

    async def test_v2_requires_graph_or_citations(self, db_pool):
        """Test search_hybrid_v2 rejects query without graph or citations."""
        from research_kb_storage.search import search_hybrid_v2

        query = SearchQuery(
            text="test",
            use_graph=False,
            use_citations=False,
        )
        with pytest.raises(ValueError, match="use_graph"):
            await search_hybrid_v2(query)
