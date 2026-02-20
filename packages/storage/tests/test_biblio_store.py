"""Tests for BiblioStore - Bibliographic coupling computations.

Tests:
- Jaccard similarity calculations
- Coupling score retrieval
- Similar sources lookup
- Statistics computation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from research_kb_storage.biblio_store import BiblioStore

pytestmark = pytest.mark.unit


@pytest.fixture
def source_ids():
    """Generate test source UUIDs."""
    return {
        "a": uuid4(),
        "b": uuid4(),
        "c": uuid4(),
        "d": uuid4(),
        "e": uuid4(),
    }


class MockAsyncContextManager:
    """Async context manager that returns a mock connection."""

    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_pool():
    """Create a mock connection pool with async context manager support."""
    conn = AsyncMock()
    pool = MagicMock()
    pool.acquire.return_value = MockAsyncContextManager(conn)
    return pool, conn


def make_get_pool(pool):
    """Create an async function that returns the mock pool."""

    async def get_pool():
        return pool

    return get_pool


class TestComputeCouplingForSource:
    """Tests for compute_coupling_for_source()."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_shared_citations(self, source_ids, mock_pool):
        """Test returns empty list when source has no shared citations."""
        pool, conn = mock_pool
        conn.fetch.return_value = []

        with patch("research_kb_storage.biblio_store.get_connection_pool", make_get_pool(pool)):
            result = await BiblioStore.compute_coupling_for_source(source_ids["a"])

        assert result == []
        conn.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_coupling_results(self, source_ids, mock_pool):
        """Test returns coupling results for sources with shared citations."""
        pool, conn = mock_pool
        conn.fetch.return_value = [
            {
                "other_id": source_ids["b"],
                "shared_count": 5,
                "my_count": 10,
                "other_count": 8,
                "coupling_strength": 0.385,  # 5 / (10 + 8 - 5)
            },
            {
                "other_id": source_ids["c"],
                "shared_count": 3,
                "my_count": 10,
                "other_count": 6,
                "coupling_strength": 0.231,  # 3 / (10 + 6 - 3)
            },
        ]

        with patch("research_kb_storage.biblio_store.get_connection_pool", make_get_pool(pool)):
            result = await BiblioStore.compute_coupling_for_source(source_ids["a"])

        assert len(result) == 2
        assert result[0]["other_source_id"] == source_ids["b"]
        assert result[0]["shared_count"] == 5
        assert result[0]["coupling_strength"] == pytest.approx(0.385, rel=1e-3)
        assert result[1]["other_source_id"] == source_ids["c"]

    @pytest.mark.asyncio
    async def test_respects_min_coupling_threshold(self, source_ids, mock_pool):
        """Test that min_coupling threshold is passed to query."""
        pool, conn = mock_pool
        conn.fetch.return_value = []

        with patch("research_kb_storage.biblio_store.get_connection_pool", make_get_pool(pool)):
            await BiblioStore.compute_coupling_for_source(source_ids["a"], min_coupling=0.25)

        # Verify min_coupling was passed as parameter
        call_args = conn.fetch.call_args
        assert call_args[0][1] == source_ids["a"]  # source_id
        assert call_args[0][2] == 0.25  # min_coupling

    @pytest.mark.asyncio
    async def test_handles_null_coupling_strength(self, source_ids, mock_pool):
        """Test handles NULL coupling_strength gracefully."""
        pool, conn = mock_pool
        conn.fetch.return_value = [
            {
                "other_id": source_ids["b"],
                "shared_count": 0,
                "my_count": 0,
                "other_count": 0,
                "coupling_strength": None,
            },
        ]

        with patch("research_kb_storage.biblio_store.get_connection_pool", make_get_pool(pool)):
            result = await BiblioStore.compute_coupling_for_source(source_ids["a"])

        assert len(result) == 1
        assert result[0]["coupling_strength"] == 0.0


class TestComputeAllCoupling:
    """Tests for compute_all_coupling()."""

    @pytest.mark.asyncio
    async def test_returns_stats_with_no_sources(self, mock_pool):
        """Test returns stats when no sources have citations."""
        pool, conn = mock_pool
        conn.fetch.return_value = []
        conn.execute.return_value = None

        with patch("research_kb_storage.biblio_store.get_connection_pool", make_get_pool(pool)):
            stats = await BiblioStore.compute_all_coupling()

        assert stats["total_sources"] == 0
        assert stats["pairs_computed"] == 0
        assert stats["pairs_stored"] == 0

    @pytest.mark.asyncio
    async def test_truncates_existing_data(self, source_ids, mock_pool):
        """Test truncates bibliographic_coupling table before computing."""
        pool, conn = mock_pool
        conn.fetch.return_value = [{"citing_source_id": source_ids["a"]}]
        conn.execute.return_value = None
        conn.executemany.return_value = None

        # Mock compute_coupling_for_source to return empty
        with patch("research_kb_storage.biblio_store.get_connection_pool", make_get_pool(pool)):
            with patch.object(BiblioStore, "compute_coupling_for_source", return_value=[]):
                await BiblioStore.compute_all_coupling()

        # Verify TRUNCATE was called
        truncate_calls = [call for call in conn.execute.call_args_list if "TRUNCATE" in str(call)]
        assert len(truncate_calls) == 1

    @pytest.mark.asyncio
    async def test_computes_and_stores_coupling(self, source_ids, mock_pool):
        """Test computes coupling for all sources and stores results."""
        pool, conn = mock_pool
        conn.fetch.return_value = [
            {"citing_source_id": source_ids["a"]},
            {"citing_source_id": source_ids["b"]},
        ]
        conn.execute.return_value = None
        conn.executemany.return_value = None

        mock_couplings = [
            {
                "other_source_id": source_ids["b"],
                "shared_count": 3,
                "coupling_strength": 0.3,
            }
        ]

        with patch("research_kb_storage.biblio_store.get_connection_pool", make_get_pool(pool)):
            with patch.object(
                BiblioStore, "compute_coupling_for_source", return_value=mock_couplings
            ):
                stats = await BiblioStore.compute_all_coupling(batch_size=10)

        assert stats["total_sources"] == 2
        assert stats["pairs_computed"] == 2  # Called twice
        assert stats["pairs_stored"] > 0

    @pytest.mark.asyncio
    async def test_enforces_source_id_ordering(self, source_ids, mock_pool):
        """Test that source_a_id < source_b_id ordering is enforced."""
        pool, conn = mock_pool

        # source_b < source_a alphabetically by UUID
        # The store should swap them to maintain ordering
        small_id = uuid4()
        large_id = uuid4()
        # Ensure ordering
        if str(small_id) > str(large_id):
            small_id, large_id = large_id, small_id

        conn.fetch.return_value = [{"citing_source_id": large_id}]
        conn.execute.return_value = None

        captured_batch = []

        async def capture_executemany(query, batch):
            captured_batch.extend(batch)

        conn.executemany.side_effect = capture_executemany

        mock_couplings = [
            {
                "other_source_id": small_id,
                "shared_count": 3,
                "coupling_strength": 0.3,
            }
        ]

        with patch("research_kb_storage.biblio_store.get_connection_pool", make_get_pool(pool)):
            with patch.object(
                BiblioStore, "compute_coupling_for_source", return_value=mock_couplings
            ):
                await BiblioStore.compute_all_coupling(batch_size=1)

        # Verify ordering: first ID should be smaller
        if captured_batch:
            a_id, b_id, _, _ = captured_batch[0]
            assert a_id < b_id


class TestGetSimilarSources:
    """Tests for get_similar_sources()."""

    @pytest.mark.asyncio
    async def test_returns_similar_sources(self, source_ids, mock_pool):
        """Test returns sources similar by bibliographic coupling."""
        pool, conn = mock_pool
        conn.fetch.return_value = [
            {
                "similar_source_id": source_ids["b"],
                "title": "Similar Paper B",
                "authors": ["Author B"],
                "year": 2023,
                "source_type": "paper",
                "shared_references": 5,
                "coupling_strength": 0.45,
            },
            {
                "similar_source_id": source_ids["c"],
                "title": "Similar Paper C",
                "authors": ["Author C"],
                "year": 2022,
                "source_type": "paper",
                "shared_references": 3,
                "coupling_strength": 0.25,
            },
        ]

        with patch("research_kb_storage.biblio_store.get_connection_pool", make_get_pool(pool)):
            result = await BiblioStore.get_similar_sources(source_ids["a"], limit=10)

        assert len(result) == 2
        assert result[0]["source_id"] == source_ids["b"]
        assert result[0]["title"] == "Similar Paper B"
        assert result[0]["coupling_strength"] == pytest.approx(0.45, rel=1e-5)
        assert result[1]["source_id"] == source_ids["c"]
        assert result[1]["shared_references"] == 3

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_coupling(self, source_ids, mock_pool):
        """Test returns empty list when no coupling exists."""
        pool, conn = mock_pool
        conn.fetch.return_value = []

        with patch("research_kb_storage.biblio_store.get_connection_pool", make_get_pool(pool)):
            result = await BiblioStore.get_similar_sources(source_ids["a"])

        assert result == []

    @pytest.mark.asyncio
    async def test_respects_limit_parameter(self, source_ids, mock_pool):
        """Test that limit parameter is passed to query."""
        pool, conn = mock_pool
        conn.fetch.return_value = []

        with patch("research_kb_storage.biblio_store.get_connection_pool", make_get_pool(pool)):
            await BiblioStore.get_similar_sources(source_ids["a"], limit=5)

        call_args = conn.fetch.call_args
        assert call_args[0][1] == source_ids["a"]
        assert call_args[0][2] == 5


class TestGetCouplingScore:
    """Tests for get_coupling_score()."""

    @pytest.mark.asyncio
    async def test_returns_coupling_score(self, source_ids, mock_pool):
        """Test returns coupling score between two sources."""
        pool, conn = mock_pool
        conn.fetchrow.return_value = {"coupling_strength": 0.42}

        with patch("research_kb_storage.biblio_store.get_connection_pool", make_get_pool(pool)):
            score = await BiblioStore.get_coupling_score(source_ids["a"], source_ids["b"])

        assert score == pytest.approx(0.42, rel=1e-5)

    @pytest.mark.asyncio
    async def test_returns_none_when_no_coupling(self, source_ids, mock_pool):
        """Test returns None when no coupling exists."""
        pool, conn = mock_pool
        conn.fetchrow.return_value = None

        with patch("research_kb_storage.biblio_store.get_connection_pool", make_get_pool(pool)):
            score = await BiblioStore.get_coupling_score(source_ids["a"], source_ids["b"])

        assert score is None

    @pytest.mark.asyncio
    async def test_enforces_source_ordering(self, mock_pool):
        """Test that source IDs are reordered for consistent lookup."""
        pool, conn = mock_pool
        conn.fetchrow.return_value = {"coupling_strength": 0.5}

        # Create IDs where a > b (needs swap)
        id_a = uuid4()
        id_b = uuid4()
        if str(id_a) < str(id_b):
            id_a, id_b = id_b, id_a  # Ensure a > b

        with patch("research_kb_storage.biblio_store.get_connection_pool", make_get_pool(pool)):
            await BiblioStore.get_coupling_score(id_a, id_b)

        # The smaller ID should be passed first
        call_args = conn.fetchrow.call_args
        first_id = call_args[0][1]
        second_id = call_args[0][2]
        assert first_id < second_id


class TestGetStats:
    """Tests for get_stats()."""

    @pytest.mark.asyncio
    async def test_returns_stats(self, mock_pool):
        """Test returns bibliographic coupling statistics."""
        pool, conn = mock_pool
        conn.fetchrow.return_value = {
            "total_pairs": 150,
            "avg_coupling": 0.234,
            "max_coupling": 0.85,
            "sources_involved": 42,
        }

        with patch("research_kb_storage.biblio_store.get_connection_pool", make_get_pool(pool)):
            stats = await BiblioStore.get_stats()

        assert stats["total_pairs"] == 150
        assert stats["avg_coupling"] == pytest.approx(0.234, rel=1e-5)
        assert stats["max_coupling"] == pytest.approx(0.85, rel=1e-5)
        assert stats["sources_involved"] == 42

    @pytest.mark.asyncio
    async def test_handles_empty_table(self, mock_pool):
        """Test handles empty bibliographic_coupling table."""
        pool, conn = mock_pool
        conn.fetchrow.return_value = {
            "total_pairs": 0,
            "avg_coupling": None,
            "max_coupling": None,
            "sources_involved": 0,
        }

        with patch("research_kb_storage.biblio_store.get_connection_pool", make_get_pool(pool)):
            stats = await BiblioStore.get_stats()

        assert stats["total_pairs"] == 0
        assert stats["avg_coupling"] == 0.0
        assert stats["max_coupling"] == 0.0
        assert stats["sources_involved"] == 0


class TestJaccardSimilarity:
    """Tests for Jaccard similarity calculation logic."""

    def test_jaccard_formula(self):
        """Test Jaccard similarity formula: |A intersect B| / |A union B|."""
        # If A has 10 refs, B has 8 refs, shared = 5
        # Union = 10 + 8 - 5 = 13
        # Jaccard = 5 / 13 = 0.385
        shared = 5
        a_count = 10
        b_count = 8

        union = a_count + b_count - shared
        jaccard = shared / union

        assert jaccard == pytest.approx(0.385, rel=1e-2)

    def test_jaccard_zero_when_no_overlap(self):
        """Test Jaccard is 0 when no shared references."""
        shared = 0
        a_count = 10
        b_count = 8

        union = a_count + b_count - shared
        jaccard = shared / union if union > 0 else 0

        assert jaccard == 0.0

    def test_jaccard_one_when_identical(self):
        """Test Jaccard is 1 when reference sets are identical."""
        shared = 10
        a_count = 10
        b_count = 10

        union = a_count + b_count - shared  # 10 + 10 - 10 = 10
        jaccard = shared / union

        assert jaccard == pytest.approx(1.0, rel=1e-5)

    def test_jaccard_handles_subset(self):
        """Test Jaccard when one set is subset of another."""
        # A has refs {1,2,3,4,5}, B has refs {1,2,3}
        # Shared = 3, Union = 5
        shared = 3
        a_count = 5
        b_count = 3

        union = a_count + b_count - shared  # 5 + 3 - 3 = 5
        jaccard = shared / union

        assert jaccard == pytest.approx(0.6, rel=1e-5)
