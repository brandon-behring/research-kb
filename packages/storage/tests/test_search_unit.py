"""Unit tests for search orchestration logic (no database required).

Covers:
- search_hybrid: RRF reranking path, mode dispatch, error wrapping
- search_hybrid_v2: graph scoring pipeline, citation authority, weight renormalization
- search_with_rerank: reranker unavailable fallback, reranker failure fallback
- search_with_expansion: query expansion paths, HyDE embedding, branch selection
- search_vector_only: embedding validation, error wrapping

Phase S Commit 1: Target search.py 17.8% → 45%
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from research_kb_common import SearchError
from research_kb_contracts import Chunk, SearchResult, Source, SourceType
from research_kb_storage.search import SearchQuery

pytestmark = pytest.mark.unit

_NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool_mock(conn=None):
    """Create a properly structured asyncpg pool mock.

    asyncpg's pool.acquire() returns a context manager synchronously
    (PoolAcquireContext) that supports ``async with``. So pool must be
    a regular MagicMock with acquire() returning something that has
    async __aenter__/__aexit__.
    """
    if conn is None:
        conn = AsyncMock()
        conn.set_type_codec = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])

    # pool.acquire() is a sync call returning an async-context-manager
    pool = MagicMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool, conn


def _make_chunk(chunk_id=None, source_id=None, domain_id="causal_inference"):
    """Create a minimal Chunk for testing."""
    return Chunk(
        id=chunk_id or uuid4(),
        source_id=source_id or uuid4(),
        domain_id=domain_id,
        content="Test content about instrumental variables.",
        content_hash=f"hash_{uuid4().hex[:8]}",
        location="Chapter 1, p. 1",
        created_at=_NOW,
    )


def _make_source(source_id=None, domain_id="causal_inference"):
    """Create a minimal Source for testing."""
    return Source(
        id=source_id or uuid4(),
        source_type=SourceType.PAPER,
        title="Test Paper",
        domain_id=domain_id,
        file_hash=f"hash_{uuid4().hex[:8]}",
        metadata={},
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_search_result(
    fts_score=None,
    vector_score=None,
    graph_score=None,
    citation_score=None,
    combined_score=0.5,
    rank=1,
    source_id=None,
):
    """Create a minimal SearchResult for testing."""
    sid = source_id or uuid4()
    return SearchResult(
        chunk=_make_chunk(source_id=sid),
        source=_make_source(source_id=sid),
        fts_score=fts_score,
        vector_score=vector_score,
        graph_score=graph_score,
        citation_score=citation_score,
        combined_score=combined_score,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# search_hybrid: RRF reranking + error wrapping
# ---------------------------------------------------------------------------


class TestSearchHybridUnit:
    """Unit tests for search_hybrid orchestration."""

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_rrf_reranking_reorders_results(self, mock_get_pool):
        """RRF path recomputes combined_score and re-sorts results."""
        from research_kb_storage.search import search_hybrid

        r1 = _make_search_result(fts_score=0.9, vector_score=0.1, combined_score=0.5, rank=1)
        r2 = _make_search_result(fts_score=0.1, vector_score=0.9, combined_score=0.5, rank=2)

        pool, conn = _make_pool_mock()
        mock_get_pool.return_value = pool

        with patch("research_kb_storage.search._hybrid_search", return_value=[r1, r2]):
            query = SearchQuery(text="test", embedding=[0.1] * 1024, scoring_method="rrf")
            results = await search_hybrid(query)

        assert len(results) == 2
        for r in results:
            assert r.combined_score > 0
        assert results[0].rank == 1
        assert results[1].rank == 2

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_fts_only_dispatch(self, mock_get_pool):
        """Text-only query dispatches to _fts_search."""
        from research_kb_storage.search import search_hybrid

        pool, conn = _make_pool_mock()
        mock_get_pool.return_value = pool

        expected = [_make_search_result(fts_score=0.8)]
        with patch("research_kb_storage.search._fts_search", return_value=expected) as fts_mock:
            results = await search_hybrid(SearchQuery(text="test"))

        fts_mock.assert_called_once()
        assert results == expected

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_vector_only_dispatch(self, mock_get_pool):
        """Embedding-only query dispatches to _vector_search."""
        from research_kb_storage.search import search_hybrid

        pool, conn = _make_pool_mock()
        mock_get_pool.return_value = pool

        expected = [_make_search_result(vector_score=0.9)]
        with patch("research_kb_storage.search._vector_search", return_value=expected) as vec_mock:
            results = await search_hybrid(SearchQuery(embedding=[0.1] * 1024))

        vec_mock.assert_called_once()
        assert results == expected

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_hybrid_dispatch(self, mock_get_pool):
        """Text + embedding dispatches to _hybrid_search."""
        from research_kb_storage.search import search_hybrid

        pool, conn = _make_pool_mock()
        mock_get_pool.return_value = pool

        expected = [_make_search_result(fts_score=0.5, vector_score=0.7)]
        with patch(
            "research_kb_storage.search._hybrid_search", return_value=expected
        ) as hybrid_mock:
            results = await search_hybrid(SearchQuery(text="test", embedding=[0.1] * 1024))

        hybrid_mock.assert_called_once()
        assert results == expected

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_generic_exception_wrapped_as_search_error(self, mock_get_pool):
        """Non-SearchError exceptions are wrapped in SearchError."""
        from research_kb_storage.search import search_hybrid

        pool, conn = _make_pool_mock()
        mock_get_pool.return_value = pool

        with patch(
            "research_kb_storage.search._hybrid_search",
            side_effect=RuntimeError("db exploded"),
        ):
            with pytest.raises(SearchError, match="Search failed"):
                await search_hybrid(SearchQuery(text="test", embedding=[0.1] * 1024))

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_search_error_not_rewrapped(self, mock_get_pool):
        """SearchError propagates unchanged (not double-wrapped)."""
        from research_kb_storage.search import search_hybrid

        pool, conn = _make_pool_mock()
        mock_get_pool.return_value = pool

        original = SearchError("original error")
        with patch("research_kb_storage.search._fts_search", side_effect=original):
            with pytest.raises(SearchError, match="original error"):
                await search_hybrid(SearchQuery(text="test"))

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_rrf_empty_results_returns_empty(self, mock_get_pool):
        """RRF path with no results returns empty list."""
        from research_kb_storage.search import search_hybrid

        pool, conn = _make_pool_mock()
        mock_get_pool.return_value = pool

        with patch("research_kb_storage.search._fts_search", return_value=[]):
            results = await search_hybrid(SearchQuery(text="nothing", scoring_method="rrf"))

        assert results == []


# ---------------------------------------------------------------------------
# search_hybrid_v2: graph + citation pipeline
# ---------------------------------------------------------------------------


class TestSearchHybridV2Unit:
    """Unit tests for search_hybrid_v2 orchestration logic."""

    async def test_v2_rejects_no_graph_no_citations(self):
        """Raises ValueError when neither graph nor citations enabled."""
        from research_kb_storage.search import search_hybrid_v2

        query = SearchQuery(text="test", use_graph=False, use_citations=False)
        with pytest.raises(ValueError, match="use_graph"):
            await search_hybrid_v2(query)

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_v2_weighted_scoring_combines_four_signals(self, mock_get_pool):
        """Weighted scoring path correctly combines fts+vector+graph+citation."""
        from research_kb_storage.search import search_hybrid_v2

        sid = uuid4()
        r1 = _make_search_result(
            fts_score=0.8, vector_score=0.6, combined_score=0.7, rank=1, source_id=sid
        )
        r2 = _make_search_result(
            fts_score=0.3, vector_score=0.9, combined_score=0.6, rank=2, source_id=sid
        )

        # Mock connection pool for citation authority fetch
        conn = AsyncMock()
        conn.set_type_codec = AsyncMock()
        conn.fetch = AsyncMock(return_value=[{"id": sid, "citation_authority": 0.75}])
        pool, _ = _make_pool_mock(conn)
        mock_get_pool.return_value = pool

        with (
            patch(
                "research_kb_storage.query_extractor.extract_query_concepts",
                new_callable=AsyncMock,
                return_value=[uuid4()],
            ),
            patch("research_kb_storage.chunk_concept_store.ChunkConceptStore") as mock_cc_store,
            patch(
                "research_kb_storage.graph_queries.compute_weighted_graph_score",
                new_callable=AsyncMock,
                return_value=(0.5, ["path explanation"]),
            ),
            patch("research_kb_storage.graph_queries._check_kuzu_ready", return_value=False),
            patch(
                "research_kb_storage.search._hybrid_search_for_rerank",
                new_callable=AsyncMock,
                return_value=[r1, r2],
            ),
        ):
            mock_cc_store.get_concept_info_for_chunks = AsyncMock(
                return_value={
                    r1.chunk.id: [(uuid4(), "reference", 0.8)],
                    r2.chunk.id: [(uuid4(), "defines", 1.0)],
                }
            )

            query = SearchQuery(
                text="test",
                embedding=[0.1] * 1024,
                fts_weight=0.2,
                vector_weight=0.4,
                graph_weight=0.2,
                citation_weight=0.2,
                use_graph=True,
                use_citations=True,
                limit=10,
            )
            results = await search_hybrid_v2(query)

        assert len(results) == 2
        for r in results:
            assert r.graph_score is not None
            assert r.citation_score is not None
            assert r.combined_score > 0
        assert results[0].rank == 1
        assert results[1].rank == 2

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_v2_rrf_scoring_path(self, mock_get_pool):
        """RRF scoring in v2 computes rank-based scores."""
        from research_kb_storage.search import search_hybrid_v2

        sid = uuid4()
        r1 = _make_search_result(
            fts_score=0.9, vector_score=0.3, combined_score=0.6, rank=1, source_id=sid
        )

        conn = AsyncMock()
        conn.set_type_codec = AsyncMock()
        conn.fetch = AsyncMock(return_value=[{"id": sid, "citation_authority": 0.5}])
        pool, _ = _make_pool_mock(conn)
        mock_get_pool.return_value = pool

        with (
            patch(
                "research_kb_storage.query_extractor.extract_query_concepts",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "research_kb_storage.search._hybrid_search_for_rerank",
                new_callable=AsyncMock,
                return_value=[r1],
            ),
        ):
            query = SearchQuery(
                text="test",
                embedding=[0.1] * 1024,
                use_graph=True,
                scoring_method="rrf",
                limit=5,
            )
            results = await search_hybrid_v2(query)

        assert len(results) == 1
        assert results[0].combined_score > 0

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_v2_no_query_concepts_zeros_graph_scores(self, mock_get_pool):
        """When no query concepts found, all graph scores are 0."""
        from research_kb_storage.search import search_hybrid_v2

        r1 = _make_search_result(fts_score=0.8, vector_score=0.6, combined_score=0.7)

        pool, _ = _make_pool_mock()
        mock_get_pool.return_value = pool

        with (
            patch(
                "research_kb_storage.query_extractor.extract_query_concepts",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("research_kb_storage.chunk_concept_store.ChunkConceptStore") as mock_cc,
            patch(
                "research_kb_storage.search._hybrid_search_for_rerank",
                new_callable=AsyncMock,
                return_value=[r1],
            ),
        ):
            mock_cc.get_concept_info_for_chunks = AsyncMock(return_value={})

            query = SearchQuery(text="test", embedding=[0.1] * 1024, use_graph=True, limit=5)
            results = await search_hybrid_v2(query)

        assert len(results) == 1
        assert results[0].graph_score == 0.0

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_v2_weight_renormalization_when_no_graph_contribution(self, mock_get_pool):
        """Weights renormalize when graph signal contributes nothing."""
        from research_kb_storage.search import search_hybrid_v2

        sid = uuid4()
        r1 = _make_search_result(fts_score=0.8, vector_score=0.6, combined_score=0.7, source_id=sid)

        conn = AsyncMock()
        conn.set_type_codec = AsyncMock()
        conn.fetch = AsyncMock(return_value=[{"id": sid, "citation_authority": 0.0}])
        pool, _ = _make_pool_mock(conn)
        mock_get_pool.return_value = pool

        with (
            patch(
                "research_kb_storage.query_extractor.extract_query_concepts",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("research_kb_storage.chunk_concept_store.ChunkConceptStore") as mock_cc,
            patch(
                "research_kb_storage.search._hybrid_search_for_rerank",
                new_callable=AsyncMock,
                return_value=[r1],
            ),
        ):
            mock_cc.get_concept_info_for_chunks = AsyncMock(return_value={})

            query = SearchQuery(
                text="test",
                embedding=[0.1] * 1024,
                fts_weight=0.2,
                vector_weight=0.4,
                graph_weight=0.2,
                citation_weight=0.2,
                use_graph=True,
                use_citations=True,
                limit=5,
            )
            results = await search_hybrid_v2(query)

        # Graph and citation both 0 → renormalized to fts+vector only
        assert len(results) == 1
        expected_fts_w = 0.2 / 0.6
        expected_vec_w = 0.4 / 0.6
        expected_combined = expected_fts_w * 0.8 + expected_vec_w * 0.6
        assert results[0].combined_score == pytest.approx(expected_combined, rel=1e-4)

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_v2_fts_only_query_dispatches_to_fts_search(self, mock_get_pool):
        """Text-only v2 query (no embedding) dispatches to _fts_search."""
        from research_kb_storage.search import search_hybrid_v2

        r1 = _make_search_result(fts_score=0.8, combined_score=0.8)

        pool, conn = _make_pool_mock()
        mock_get_pool.return_value = pool

        with (
            patch(
                "research_kb_storage.query_extractor.extract_query_concepts",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("research_kb_storage.chunk_concept_store.ChunkConceptStore") as mock_cc,
            patch(
                "research_kb_storage.search._fts_search",
                new_callable=AsyncMock,
                return_value=[r1],
            ) as fts_mock,
        ):
            mock_cc.get_concept_info_for_chunks = AsyncMock(return_value={})
            query = SearchQuery(text="test", use_graph=True, limit=5)
            await search_hybrid_v2(query)

        fts_mock.assert_called_once()

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_v2_exception_wrapped_as_search_error(self, mock_get_pool):
        """Non-SearchError in v2 is wrapped as SearchError."""
        from research_kb_storage.search import search_hybrid_v2

        pool, _ = _make_pool_mock()
        mock_get_pool.return_value = pool

        with patch(
            "research_kb_storage.query_extractor.extract_query_concepts",
            new_callable=AsyncMock,
            side_effect=RuntimeError("extraction failed"),
        ):
            query = SearchQuery(text="test", use_graph=True, limit=5)
            with pytest.raises(SearchError, match="Graph-boosted search failed"):
                await search_hybrid_v2(query)

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_v2_limit_applied_to_final_results(self, mock_get_pool):
        """Final results are trimmed to query.limit after re-ranking."""
        from research_kb_storage.search import search_hybrid_v2

        results_in = [
            _make_search_result(
                fts_score=0.9 - i * 0.1, vector_score=0.5, combined_score=0.7 - i * 0.1
            )
            for i in range(5)
        ]

        pool, _ = _make_pool_mock()
        mock_get_pool.return_value = pool

        with (
            patch(
                "research_kb_storage.query_extractor.extract_query_concepts",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("research_kb_storage.chunk_concept_store.ChunkConceptStore") as mock_cc,
            patch(
                "research_kb_storage.search._hybrid_search_for_rerank",
                new_callable=AsyncMock,
                return_value=results_in,
            ),
        ):
            mock_cc.get_concept_info_for_chunks = AsyncMock(return_value={})

            query = SearchQuery(text="test", embedding=[0.1] * 1024, use_graph=True, limit=2)
            final = await search_hybrid_v2(query)

        assert len(final) == 2
        assert final[0].rank == 1
        assert final[1].rank == 2

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_v2_citations_only_no_graph(self, mock_get_pool):
        """v2 works with use_citations=True but use_graph=False."""
        from research_kb_storage.search import search_hybrid_v2

        sid = uuid4()
        r1 = _make_search_result(
            fts_score=0.8, vector_score=0.7, combined_score=0.75, source_id=sid
        )

        conn = AsyncMock()
        conn.set_type_codec = AsyncMock()
        conn.fetch = AsyncMock(return_value=[{"id": sid, "citation_authority": 0.9}])
        pool, _ = _make_pool_mock(conn)
        mock_get_pool.return_value = pool

        with (
            patch(
                "research_kb_storage.query_extractor.extract_query_concepts",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "research_kb_storage.search._hybrid_search_for_rerank",
                new_callable=AsyncMock,
                return_value=[r1],
            ),
        ):
            query = SearchQuery(
                text="test",
                embedding=[0.1] * 1024,
                citation_weight=0.15,
                use_citations=True,
                limit=5,
            )
            results = await search_hybrid_v2(query)

        assert len(results) == 1
        assert results[0].citation_score == 0.9


# ---------------------------------------------------------------------------
# search_with_rerank
# ---------------------------------------------------------------------------


class TestSearchWithRerankUnit:
    """Unit tests for search_with_rerank orchestration."""

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_reranker_unavailable_returns_unreranked(self, mock_get_pool):
        """When reranker is unavailable, returns top candidates without reranking."""
        from research_kb_storage.search import search_with_rerank

        candidates = [
            _make_search_result(fts_score=0.9, combined_score=0.9, rank=i + 1) for i in range(10)
        ]

        pool, _ = _make_pool_mock()
        mock_get_pool.return_value = pool

        mock_rerank_client = MagicMock()
        mock_rerank_client.is_available.return_value = False

        with (
            patch(
                "research_kb_storage.search.search_hybrid",
                new_callable=AsyncMock,
                return_value=candidates,
            ),
            patch(
                "research_kb_pdf.rerank_client.RerankClient",
                return_value=mock_rerank_client,
            ),
        ):
            results = await search_with_rerank(SearchQuery(text="test"), rerank_top_k=5)

        assert len(results) == 5
        mock_rerank_client.rerank_search_results.assert_not_called()

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_reranker_failure_returns_fallback(self, mock_get_pool):
        """When reranker raises exception, falls back to unreranked results."""
        from research_kb_storage.search import search_with_rerank

        candidates = [
            _make_search_result(fts_score=0.9, combined_score=0.9, rank=i + 1) for i in range(10)
        ]

        pool, _ = _make_pool_mock()
        mock_get_pool.return_value = pool

        mock_rerank_client = MagicMock()
        mock_rerank_client.is_available.return_value = True
        mock_rerank_client.rerank_search_results.side_effect = ConnectionError("server down")

        with (
            patch(
                "research_kb_storage.search.search_hybrid",
                new_callable=AsyncMock,
                return_value=candidates,
            ),
            patch(
                "research_kb_pdf.rerank_client.RerankClient",
                return_value=mock_rerank_client,
            ),
        ):
            results = await search_with_rerank(SearchQuery(text="test"), rerank_top_k=3)

        assert len(results) == 3

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_reranker_success_returns_reranked(self, mock_get_pool):
        """When reranker succeeds, returns reranked results."""
        from research_kb_storage.search import search_with_rerank

        candidates = [
            _make_search_result(fts_score=0.9 - i * 0.1, combined_score=0.9 - i * 0.1, rank=i + 1)
            for i in range(10)
        ]
        reranked = candidates[5:8]

        pool, _ = _make_pool_mock()
        mock_get_pool.return_value = pool

        mock_rerank_client = MagicMock()
        mock_rerank_client.is_available.return_value = True
        mock_rerank_client.rerank_search_results.return_value = reranked

        with (
            patch(
                "research_kb_storage.search.search_hybrid",
                new_callable=AsyncMock,
                return_value=candidates,
            ),
            patch(
                "research_kb_pdf.rerank_client.RerankClient",
                return_value=mock_rerank_client,
            ),
        ):
            results = await search_with_rerank(SearchQuery(text="test"), rerank_top_k=3)

        assert results == reranked

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_rerank_no_candidates_returns_empty(self, mock_get_pool):
        """When base search returns no candidates, returns empty immediately."""
        from research_kb_storage.search import search_with_rerank

        pool, _ = _make_pool_mock()
        mock_get_pool.return_value = pool

        with patch(
            "research_kb_storage.search.search_hybrid",
            new_callable=AsyncMock,
            return_value=[],
        ):
            results = await search_with_rerank(SearchQuery(text="nonexistent"), rerank_top_k=5)

        assert results == []

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_rerank_uses_graph_search_when_enabled(self, mock_get_pool):
        """When use_graph=True, delegates to search_hybrid_v2."""
        from research_kb_storage.search import search_with_rerank

        candidates = [_make_search_result(combined_score=0.8)]

        pool, _ = _make_pool_mock()
        mock_get_pool.return_value = pool

        mock_rerank_client = MagicMock()
        mock_rerank_client.is_available.return_value = False

        with (
            patch(
                "research_kb_storage.search.search_hybrid_v2",
                new_callable=AsyncMock,
                return_value=candidates,
            ) as v2_mock,
            patch(
                "research_kb_pdf.rerank_client.RerankClient",
                return_value=mock_rerank_client,
            ),
        ):
            query = SearchQuery(
                text="test", embedding=[0.1] * 1024, use_graph=True, graph_weight=0.15
            )
            await search_with_rerank(query, rerank_top_k=5)

        v2_mock.assert_called_once()


# ---------------------------------------------------------------------------
# search_vector_only
# ---------------------------------------------------------------------------


class TestSearchVectorOnlyUnit:
    """Unit tests for search_vector_only."""

    async def test_no_embedding_raises_search_error(self):
        """Raises SearchError when no embedding provided."""
        from research_kb_storage.search import search_vector_only

        with pytest.raises(SearchError, match="requires an embedding"):
            await search_vector_only(SearchQuery(text="test"))

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_delegates_to_vector_search(self, mock_get_pool):
        """Delegates to _vector_search and returns results."""
        from research_kb_storage.search import search_vector_only

        pool, conn = _make_pool_mock()
        mock_get_pool.return_value = pool

        expected = [_make_search_result(vector_score=0.95)]
        with patch(
            "research_kb_storage.search._vector_search",
            new_callable=AsyncMock,
            return_value=expected,
        ):
            results = await search_vector_only(SearchQuery(embedding=[0.1] * 1024))

        assert results == expected

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_exception_wrapped_as_search_error(self, mock_get_pool):
        """Non-SearchError is wrapped as SearchError."""
        from research_kb_storage.search import search_vector_only

        pool, conn = _make_pool_mock()
        mock_get_pool.return_value = pool

        with patch(
            "research_kb_storage.search._vector_search",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(SearchError, match="Vector search failed"):
                await search_vector_only(SearchQuery(embedding=[0.1] * 1024))


# ---------------------------------------------------------------------------
# search_with_expansion
# ---------------------------------------------------------------------------


class TestSearchWithExpansionUnit:
    """Unit tests for search_with_expansion orchestration."""

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_expansion_failure_proceeds_with_original_query(self, mock_get_pool):
        """Query expansion failure logs warning and proceeds."""
        from research_kb_storage.search import search_with_expansion

        expected = [_make_search_result(fts_score=0.7)]
        pool, _ = _make_pool_mock()
        mock_get_pool.return_value = pool

        mock_expander = MagicMock()
        mock_expander.expand = AsyncMock(side_effect=RuntimeError("yaml missing"))

        with (
            patch("research_kb_storage.query_expander.QueryExpander") as mock_qe_class,
            patch(
                "research_kb_storage.search.search_hybrid",
                new_callable=AsyncMock,
                return_value=expected,
            ),
        ):
            mock_qe_class.from_yaml.return_value = mock_expander
            results, expanded = await search_with_expansion(
                SearchQuery(text="test"), use_rerank=False
            )

        assert results == expected
        assert expanded is None

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_hyde_failure_proceeds_with_original_embedding(self, mock_get_pool):
        """HyDE failure logs warning and proceeds with original embedding."""
        from research_kb_storage.search import search_with_expansion

        expected = [_make_search_result(vector_score=0.8)]
        original_embedding = [0.1] * 1024
        pool, _ = _make_pool_mock()
        mock_get_pool.return_value = pool

        mock_expander = MagicMock()
        mock_expander.expand = AsyncMock(return_value=MagicMock(expanded_terms=[]))

        mock_hyde_config = MagicMock()
        mock_hyde_config.enabled = True

        with (
            patch("research_kb_storage.query_expander.QueryExpander") as mock_qe_class,
            patch(
                "research_kb_storage.query_expander.get_hyde_embedding",
                new_callable=AsyncMock,
                side_effect=RuntimeError("ollama down"),
            ),
            patch(
                "research_kb_storage.search.search_hybrid",
                new_callable=AsyncMock,
                return_value=expected,
            ),
        ):
            mock_qe_class.from_yaml.return_value = mock_expander
            query = SearchQuery(text="test", embedding=original_embedding.copy())
            results, expanded = await search_with_expansion(
                query, use_rerank=False, hyde_config=mock_hyde_config
            )

        assert results == expected
        assert query.embedding == original_embedding

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_rerank_path_selected(self, mock_get_pool):
        """When use_rerank=True, delegates to search_with_rerank."""
        from research_kb_storage.search import search_with_expansion

        expected = [_make_search_result(fts_score=0.9)]
        pool, _ = _make_pool_mock()
        mock_get_pool.return_value = pool

        mock_expander = MagicMock()
        mock_expander.expand = AsyncMock(return_value=MagicMock(expanded_terms=[]))

        with (
            patch("research_kb_storage.query_expander.QueryExpander") as mock_qe_class,
            patch(
                "research_kb_storage.search.search_with_rerank",
                new_callable=AsyncMock,
                return_value=expected,
            ) as rerank_mock,
        ):
            mock_qe_class.from_yaml.return_value = mock_expander
            results, _ = await search_with_expansion(
                SearchQuery(text="test"), use_rerank=True, rerank_top_k=5
            )

        rerank_mock.assert_called_once()
        assert results == expected

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_graph_path_selected_when_no_rerank(self, mock_get_pool):
        """When use_rerank=False and use_graph=True, delegates to search_hybrid_v2."""
        from research_kb_storage.search import search_with_expansion

        expected = [_make_search_result(fts_score=0.9)]
        pool, _ = _make_pool_mock()
        mock_get_pool.return_value = pool

        mock_expander = MagicMock()
        mock_expander.expand = AsyncMock(return_value=MagicMock(expanded_terms=[]))

        with (
            patch("research_kb_storage.query_expander.QueryExpander") as mock_qe_class,
            patch(
                "research_kb_storage.search.search_hybrid_v2",
                new_callable=AsyncMock,
                return_value=expected,
            ) as v2_mock,
        ):
            mock_qe_class.from_yaml.return_value = mock_expander
            query = SearchQuery(
                text="test", embedding=[0.1] * 1024, use_graph=True, graph_weight=0.15
            )
            results, _ = await search_with_expansion(query, use_rerank=False)

        v2_mock.assert_called_once()

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_basic_hybrid_path_selected(self, mock_get_pool):
        """When use_rerank=False and use_graph=False, delegates to search_hybrid."""
        from research_kb_storage.search import search_with_expansion

        expected = [_make_search_result(fts_score=0.9)]
        pool, _ = _make_pool_mock()
        mock_get_pool.return_value = pool

        mock_expander = MagicMock()
        mock_expander.expand = AsyncMock(return_value=MagicMock(expanded_terms=[]))

        with (
            patch("research_kb_storage.query_expander.QueryExpander") as mock_qe_class,
            patch(
                "research_kb_storage.search.search_hybrid",
                new_callable=AsyncMock,
                return_value=expected,
            ) as hybrid_mock,
        ):
            mock_qe_class.from_yaml.return_value = mock_expander
            results, _ = await search_with_expansion(SearchQuery(text="test"), use_rerank=False)

        hybrid_mock.assert_called_once()

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_no_expansion_when_no_text(self, mock_get_pool):
        """Expansion skipped when query has no text."""
        from research_kb_storage.search import search_with_expansion

        expected = [_make_search_result(vector_score=0.9)]
        pool, _ = _make_pool_mock()
        mock_get_pool.return_value = pool

        with patch(
            "research_kb_storage.search.search_hybrid",
            new_callable=AsyncMock,
            return_value=expected,
        ):
            results, expanded = await search_with_expansion(
                SearchQuery(embedding=[0.1] * 1024), use_rerank=False
            )

        assert expanded is None
        assert results == expected
