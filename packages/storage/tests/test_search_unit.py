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
    """Unit tests for search_hybrid_v2 orchestration logic.

    RS4 (ADR-0001) retired the chunk-level concept graph: search_hybrid_v2 now
    combines FTS + vector + citation only. It no longer extracts query concepts,
    computes graph scores, or raises on use_graph. These tests cover the
    surviving citation-authority + scoring pipeline.
    """

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_v2_weighted_scoring_combines_signals(self, mock_get_pool):
        """Weighted scoring path correctly combines fts+vector+citation."""
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

        with patch(
            "research_kb_storage.search._hybrid_search_for_rerank",
            new_callable=AsyncMock,
            return_value=[r1, r2],
        ):
            query = SearchQuery(
                text="test",
                embedding=[0.1] * 1024,
                fts_weight=0.3,
                vector_weight=0.4,
                citation_weight=0.3,
                use_citations=True,
                limit=10,
            )
            results = await search_hybrid_v2(query)

        assert len(results) == 2
        for r in results:
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

        with patch(
            "research_kb_storage.search._hybrid_search_for_rerank",
            new_callable=AsyncMock,
            return_value=[r1],
        ):
            query = SearchQuery(
                text="test",
                embedding=[0.1] * 1024,
                use_citations=True,
                scoring_method="rrf",
                limit=5,
            )
            results = await search_hybrid_v2(query)

        assert len(results) == 1
        assert results[0].combined_score > 0

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_v2_weight_renormalization_when_no_citation_contribution(self, mock_get_pool):
        """Weights renormalize when citation signal contributes nothing."""
        from research_kb_storage.search import search_hybrid_v2

        sid = uuid4()
        r1 = _make_search_result(fts_score=0.8, vector_score=0.6, combined_score=0.7, source_id=sid)

        conn = AsyncMock()
        conn.set_type_codec = AsyncMock()
        conn.fetch = AsyncMock(return_value=[{"id": sid, "citation_authority": 0.0}])
        pool, _ = _make_pool_mock(conn)
        mock_get_pool.return_value = pool

        with patch(
            "research_kb_storage.search._hybrid_search_for_rerank",
            new_callable=AsyncMock,
            return_value=[r1],
        ):
            query = SearchQuery(
                text="test",
                embedding=[0.1] * 1024,
                fts_weight=0.2,
                vector_weight=0.4,
                citation_weight=0.2,
                use_citations=True,
                limit=5,
            )
            results = await search_hybrid_v2(query)

        # Citation contributes nothing → renormalized to fts+vector only.
        # SearchQuery.__post_init__ already normalized the input weights so
        # fts+vector+citation sum to 1 (0.25/0.5/0.25); with no citation
        # contribution the effective weights renormalize over fts+vector.
        assert len(results) == 1
        fts_w = 0.25 / 0.75
        vec_w = 0.5 / 0.75
        expected_combined = fts_w * 0.8 + vec_w * 0.6
        assert results[0].combined_score == pytest.approx(expected_combined, rel=1e-4)

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_v2_fts_only_query_dispatches_to_fts_search(self, mock_get_pool):
        """Text-only v2 query (no embedding) dispatches to _fts_search."""
        from research_kb_storage.search import search_hybrid_v2

        r1 = _make_search_result(fts_score=0.8, combined_score=0.8)

        pool, conn = _make_pool_mock()
        mock_get_pool.return_value = pool

        with patch(
            "research_kb_storage.search._fts_search",
            new_callable=AsyncMock,
            return_value=[r1],
        ) as fts_mock:
            query = SearchQuery(text="test", use_citations=True, limit=5)
            await search_hybrid_v2(query)

        fts_mock.assert_called_once()

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_v2_exception_wrapped_as_search_error(self, mock_get_pool):
        """Non-SearchError in v2 is wrapped as SearchError."""
        from research_kb_storage.search import search_hybrid_v2

        # Make the base search raise a generic error to exercise the wrapper.
        pool, _ = _make_pool_mock()
        mock_get_pool.return_value = pool

        with patch(
            "research_kb_storage.search._hybrid_search_for_rerank",
            new_callable=AsyncMock,
            side_effect=RuntimeError("base search failed"),
        ):
            query = SearchQuery(text="test", embedding=[0.1] * 1024, use_citations=True, limit=5)
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

        with patch(
            "research_kb_storage.search._hybrid_search_for_rerank",
            new_callable=AsyncMock,
            return_value=results_in,
        ):
            query = SearchQuery(text="test", embedding=[0.1] * 1024, use_citations=True, limit=2)
            final = await search_hybrid_v2(query)

        assert len(final) == 2
        assert final[0].rank == 1
        assert final[1].rank == 2

    @patch("research_kb_storage.search.get_connection_pool")
    async def test_v2_citations_only(self, mock_get_pool):
        """v2 works with use_citations=True (citation authority applied)."""
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

        with patch(
            "research_kb_storage.search._hybrid_search_for_rerank",
            new_callable=AsyncMock,
            return_value=[r1],
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


# ---------------------------------------------------------------------------
# Issue #3: fast_search vs search_hybrid field parity
# ---------------------------------------------------------------------------


class TestSearchResultFieldParity:
    """Regression tests guaranteeing fast_search and hybrid search return
    identical SearchResult field shapes.

    Issue #3: literature_search.py uses a shared parser across both paths.
    If fast_search and search_hybrid diverge, deduplication would silently
    fail. These tests pin down the contract so any future divergence fails
    loudly in CI.
    """

    def test_search_result_has_required_identity_fields(self):
        """SearchResult exposes source.id and chunk.id for dedup key."""
        result = _make_search_result(vector_score=0.9)

        assert hasattr(result.source, "id")
        assert hasattr(result.chunk, "id")
        assert result.source.id is not None
        assert result.chunk.id is not None

    def test_search_result_model_dump_has_stable_field_set(self):
        """SearchResult.model_dump() produces a stable, documented field set.

        If new optional fields are added to SearchResult, update the expected
        set intentionally. This guards against accidental field drift.
        """
        result = _make_search_result(fts_score=0.5, vector_score=0.7, combined_score=0.6, rank=1)
        dumped = result.model_dump()

        required = {
            "chunk",
            "source",
            "fts_score",
            "vector_score",
            "graph_score",
            "citation_score",
            "combined_score",
            "rank",
        }
        assert required.issubset(dumped.keys()), f"Missing fields: {required - dumped.keys()}"

    @patch("research_kb_storage.search.get_connection_pool")
    @patch("research_kb_storage.search.register_vector", new_callable=AsyncMock)
    async def test_fast_and_hybrid_return_same_identity_fields(self, mock_register, mock_get_pool):
        """A result surfaced by both fast_search and search_hybrid has
        byte-identical source.id, chunk.id, and field-set when dumped.

        Regression guard for Issue #3: if the two paths ever diverge on
        identity fields, dedup in literature_search.py breaks silently.
        """
        from research_kb_storage.search import search_hybrid, search_vector_only

        shared_sid = uuid4()
        shared_cid = uuid4()

        # Build a single SearchResult reused as the "hit" in both paths
        def _build_shared_result():
            chunk = _make_chunk(chunk_id=shared_cid, source_id=shared_sid)
            source = _make_source(source_id=shared_sid)
            return SearchResult(
                chunk=chunk,
                source=source,
                fts_score=None,
                vector_score=0.88,
                graph_score=None,
                citation_score=None,
                combined_score=0.88,
                rank=1,
            )

        pool, _ = _make_pool_mock()
        mock_get_pool.return_value = pool

        # Path 1: search_hybrid with embedding-only query -> _vector_search
        with patch(
            "research_kb_storage.search._vector_search",
            new_callable=AsyncMock,
            return_value=[_build_shared_result()],
        ):
            hybrid_results = await search_hybrid(SearchQuery(embedding=[0.1] * 1024, limit=5))

        # Path 2: search_vector_only with same embedding
        with patch(
            "research_kb_storage.search._vector_search",
            new_callable=AsyncMock,
            return_value=[_build_shared_result()],
        ):
            fast_results = await search_vector_only(SearchQuery(embedding=[0.1] * 1024, limit=5))

        assert len(hybrid_results) == 1
        assert len(fast_results) == 1

        h = hybrid_results[0]
        f = fast_results[0]

        # Identity fields must match — dedup key
        assert h.source.id == f.source.id == shared_sid
        assert h.chunk.id == f.chunk.id == shared_cid

        # Both paths must return the same class (no subclassing drift)
        assert type(h) is type(f) is SearchResult

        # Field sets must be identical
        h_keys = set(h.model_dump().keys())
        f_keys = set(f.model_dump().keys())
        assert h_keys == f_keys, (
            f"Field divergence: hybrid-only={h_keys - f_keys}, " f"fast-only={f_keys - h_keys}"
        )


# ---------------------------------------------------------------------------
# _apply_priority_multiplier: ingestion_priority downweight + ablation gate
# ---------------------------------------------------------------------------


def _make_priority_result(priority, combined_score=1.0):
    """Build a SearchResult with metadata.ingestion_priority set to ``priority``.

    Parameters
    ----------
    priority : str | None
        Value for ``source.metadata['ingestion_priority']``. None leaves metadata empty.
    combined_score : float
        Initial combined_score before _apply_priority_multiplier mutates it.

    Returns
    -------
    SearchResult
    """
    sid = uuid4()
    chunk = _make_chunk(source_id=sid)
    source = Source(
        id=sid,
        source_type=SourceType.PAPER,
        title="Test Source",
        domain_id="mathematics",
        file_hash=f"hash_{uuid4().hex[:8]}",
        metadata={"ingestion_priority": priority} if priority is not None else {},
        created_at=_NOW,
        updated_at=_NOW,
    )
    return SearchResult(
        chunk=chunk,
        source=source,
        fts_score=None,
        vector_score=None,
        graph_score=None,
        citation_score=None,
        combined_score=combined_score,
        rank=1,
    )


class TestPriorityMultiplier:
    """Unit tests for _apply_priority_multiplier (sync helper)."""

    def test_low_redundant_gets_half_multiplier(self):
        """``low_redundant`` priority halves the combined_score (0.5x)."""
        from research_kb_storage.search import _apply_priority_multiplier

        results = [_make_priority_result("low_redundant", combined_score=1.0)]
        _apply_priority_multiplier(results)

        assert results[0].combined_score == pytest.approx(0.5)

    def test_low_review_pending_gets_three_quarters_multiplier(self):
        """``low_review_pending`` priority applies 0.75x."""
        from research_kb_storage.search import _apply_priority_multiplier

        results = [_make_priority_result("low_review_pending", combined_score=1.0)]
        _apply_priority_multiplier(results)

        assert results[0].combined_score == pytest.approx(0.75)

    def test_normal_priority_unchanged(self):
        """Sources with no ingestion_priority marker are not downweighted."""
        from research_kb_storage.search import _apply_priority_multiplier

        results = [_make_priority_result(None, combined_score=0.8)]
        _apply_priority_multiplier(results)

        assert results[0].combined_score == pytest.approx(0.8)

    def test_unknown_priority_value_unchanged(self):
        """Unknown priority strings (typo, future tier) are no-ops, not errors."""
        from research_kb_storage.search import _apply_priority_multiplier

        results = [_make_priority_result("brand_new_tier", combined_score=0.6)]
        _apply_priority_multiplier(results)

        assert results[0].combined_score == pytest.approx(0.6)

    def test_env_var_disable_skips_multiplier(self, monkeypatch):
        """RKB_PRIORITY_MULTIPLIERS_DISABLED=1 short-circuits the helper.

        Used by ``scripts/eval_v2.py --disable-priority`` to measure marker
        effect via A/B comparison without code changes.
        """
        from research_kb_storage.search import _apply_priority_multiplier

        monkeypatch.setenv("RKB_PRIORITY_MULTIPLIERS_DISABLED", "1")
        results = [
            _make_priority_result("low_redundant", combined_score=1.0),
            _make_priority_result("low_review_pending", combined_score=0.8),
        ]
        _apply_priority_multiplier(results)

        # Scores must be untouched when the ablation flag is set
        assert results[0].combined_score == pytest.approx(1.0)
        assert results[1].combined_score == pytest.approx(0.8)

    def test_env_var_other_values_still_apply_multiplier(self, monkeypatch):
        """Only the literal string '1' disables; other truthy values must NOT.

        Guards against accidental disables — e.g., ``RKB_PRIORITY_MULTIPLIERS_DISABLED=true``
        is a misconfiguration that should still apply downweight (fail safe to default).
        """
        from research_kb_storage.search import _apply_priority_multiplier

        monkeypatch.setenv("RKB_PRIORITY_MULTIPLIERS_DISABLED", "true")
        results = [_make_priority_result("low_redundant", combined_score=1.0)]
        _apply_priority_multiplier(results)

        assert results[0].combined_score == pytest.approx(0.5)

    def test_env_var_unset_applies_multiplier(self, monkeypatch):
        """When env var is unset, default downweight behavior is preserved."""
        from research_kb_storage.search import _apply_priority_multiplier

        monkeypatch.delenv("RKB_PRIORITY_MULTIPLIERS_DISABLED", raising=False)
        results = [_make_priority_result("low_redundant", combined_score=1.0)]
        _apply_priority_multiplier(results)

        assert results[0].combined_score == pytest.approx(0.5)
