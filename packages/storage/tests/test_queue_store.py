"""Tests for QueueStore — ingestion queue operations.

Tests cover:
- add: insert with all fields, duplicate handling, metadata defaults
- add_batch: multiple papers, skip duplicates, return counts
- get_pending: ordering, domain filter, limit
- update_status: status transitions, error message, pdf path, retry count increment
- get_by_s2_paper_id: found and not-found cases
- delete_completed: cleanup older items, parse DELETE result
- get_stats: by_status, by_domain, retry stats
- retry_failed: reset failed under max retries, domain filter
- Error propagation as StorageError
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from research_kb_common import StorageError
from research_kb_storage.queue_store import QueueStatus, QueueStore

pytestmark = pytest.mark.unit


def _make_mock_pool(conn_mock):
    """Create a mock connection pool wrapping the given connection mock."""
    pool = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn_mock)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool


class TestQueueStatusConstants:
    """Tests for QueueStatus constants."""

    def test_status_constants_are_strings(self):
        """All statuses are non-empty strings."""
        statuses = [
            QueueStatus.PENDING,
            QueueStatus.DOWNLOADING,
            QueueStatus.EXTRACTING,
            QueueStatus.EMBEDDING,
            QueueStatus.COMPLETED,
            QueueStatus.FAILED,
        ]
        for s in statuses:
            assert isinstance(s, str)
            assert len(s) > 0

    def test_status_constants_are_unique(self):
        """All statuses are distinct."""
        statuses = [
            QueueStatus.PENDING,
            QueueStatus.DOWNLOADING,
            QueueStatus.EXTRACTING,
            QueueStatus.EMBEDDING,
            QueueStatus.COMPLETED,
            QueueStatus.FAILED,
        ]
        assert len(set(statuses)) == len(statuses)


class TestAdd:
    """Tests for QueueStore.add()."""

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_add_returns_uuid(self, mock_get_pool):
        """add returns UUID of created queue item."""
        expected_id = str(uuid4())
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=expected_id)
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await QueueStore.add(
            s2_paper_id="abc123",
            title="Test Paper",
            domain_id="causal_inference",
        )

        assert isinstance(result, UUID)
        assert str(result) == expected_id

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_add_passes_all_params(self, mock_get_pool):
        """add forwards all parameters to SQL INSERT."""
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=str(uuid4()))
        mock_get_pool.return_value = _make_mock_pool(conn)

        await QueueStore.add(
            s2_paper_id="abc123",
            title="DML Paper",
            domain_id="causal_inference",
            pdf_url="https://example.com/paper.pdf",
            doi="10.1111/test",
            arxiv_id="2001.12345",
            authors=["Author A", "Author B"],
            year=2020,
            venue="NIPS",
            priority=100,
            metadata={"citation_count": 500},
        )

        args = conn.fetchval.call_args[0]
        assert "abc123" in args
        assert "DML Paper" in args
        assert "https://example.com/paper.pdf" in args
        assert "10.1111/test" in args
        assert "2001.12345" in args
        assert ["Author A", "Author B"] in args
        assert 2020 in args
        assert "NIPS" in args
        assert "causal_inference" in args
        assert QueueStatus.PENDING in args
        assert 100 in args

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_add_defaults_metadata_to_empty(self, mock_get_pool):
        """None metadata and authors default to empty."""
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=str(uuid4()))
        mock_get_pool.return_value = _make_mock_pool(conn)

        await QueueStore.add(
            s2_paper_id="def456",
            title="Paper",
            domain_id="statistics",
        )

        args = conn.fetchval.call_args[0]
        assert [] in args  # authors default
        assert {} in args  # metadata default

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_add_duplicate_raises_storage_error(self, mock_get_pool):
        """Duplicate s2_paper_id raises StorageError."""
        conn = AsyncMock()
        conn.fetchval = AsyncMock(side_effect=Exception("unique constraint violation"))
        mock_get_pool.return_value = _make_mock_pool(conn)

        with pytest.raises(StorageError, match="already in queue"):
            await QueueStore.add(
                s2_paper_id="abc123",
                title="Duplicate",
                domain_id="causal_inference",
            )


class TestAddBatch:
    """Tests for QueueStore.add_batch()."""

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_add_batch_counts_added_and_skipped(self, mock_get_pool):
        """add_batch returns (added, skipped) tuple."""
        call_count = 0

        async def mock_fetchval(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("unique constraint")
            return str(uuid4())

        conn = AsyncMock()
        conn.fetchval = mock_fetchval
        mock_get_pool.return_value = _make_mock_pool(conn)

        papers = [
            {"s2_paper_id": "p1", "title": "Paper 1", "domain_id": "ci"},
            {"s2_paper_id": "p2", "title": "Paper 2", "domain_id": "ci"},
            {"s2_paper_id": "p3", "title": "Paper 3", "domain_id": "ci"},
        ]

        added, skipped = await QueueStore.add_batch(papers)

        assert added == 2
        assert skipped == 1

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_add_batch_empty_list(self, mock_get_pool):
        """Empty list returns (0, 0)."""
        conn = AsyncMock()
        mock_get_pool.return_value = _make_mock_pool(conn)

        added, skipped = await QueueStore.add_batch([])

        assert added == 0
        assert skipped == 0


class TestGetPending:
    """Tests for QueueStore.get_pending()."""

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_get_pending_returns_list(self, mock_get_pool):
        """get_pending returns list of queue item dicts."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(
            return_value=[
                {"id": str(uuid4()), "title": "Paper 1", "status": "pending"},
            ]
        )
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await QueueStore.get_pending(limit=10)

        assert len(result) == 1
        assert result[0]["title"] == "Paper 1"

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_get_pending_orders_by_priority_desc(self, mock_get_pool):
        """SQL orders by priority DESC, created_at ASC."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        mock_get_pool.return_value = _make_mock_pool(conn)

        await QueueStore.get_pending()

        sql = conn.fetch.call_args[0][0]
        assert "priority DESC" in sql
        assert "created_at ASC" in sql

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_get_pending_with_domain_filter(self, mock_get_pool):
        """Passing domain_id adds WHERE clause."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        mock_get_pool.return_value = _make_mock_pool(conn)

        await QueueStore.get_pending(domain_id="causal_inference")

        sql = conn.fetch.call_args[0][0]
        assert "domain_id = $3" in sql


class TestUpdateStatus:
    """Tests for QueueStore.update_status()."""

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_update_status_basic(self, mock_get_pool):
        """update_status changes status field."""
        queue_id = uuid4()
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")
        mock_get_pool.return_value = _make_mock_pool(conn)

        await QueueStore.update_status(queue_id, QueueStatus.DOWNLOADING)

        sql = conn.execute.call_args[0][0]
        assert "status = $1" in sql

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_update_status_with_error_message(self, mock_get_pool):
        """Error message is added to SET clause."""
        queue_id = uuid4()
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")
        mock_get_pool.return_value = _make_mock_pool(conn)

        await QueueStore.update_status(queue_id, QueueStatus.FAILED, error_message="PDF corrupt")

        sql = conn.execute.call_args[0][0]
        assert "error_message" in sql

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_update_status_failure_increments_retry(self, mock_get_pool):
        """FAILED status increments retry_count."""
        queue_id = uuid4()
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")
        mock_get_pool.return_value = _make_mock_pool(conn)

        await QueueStore.update_status(queue_id, QueueStatus.FAILED)

        sql = conn.execute.call_args[0][0]
        assert "retry_count = retry_count + 1" in sql

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_update_status_non_failure_no_retry_increment(self, mock_get_pool):
        """Non-FAILED status does NOT increment retry_count."""
        queue_id = uuid4()
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")
        mock_get_pool.return_value = _make_mock_pool(conn)

        await QueueStore.update_status(queue_id, QueueStatus.COMPLETED)

        sql = conn.execute.call_args[0][0]
        assert "retry_count" not in sql

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_update_status_not_found_raises_error(self, mock_get_pool):
        """UPDATE 0 (not found) raises StorageError."""
        queue_id = uuid4()
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 0")
        mock_get_pool.return_value = _make_mock_pool(conn)

        with pytest.raises(StorageError, match="Queue item not found"):
            await QueueStore.update_status(queue_id, QueueStatus.COMPLETED)

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_update_status_with_pdf_path(self, mock_get_pool):
        """pdf_path is added to SET clause."""
        queue_id = uuid4()
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")
        mock_get_pool.return_value = _make_mock_pool(conn)

        await QueueStore.update_status(queue_id, QueueStatus.EXTRACTING, pdf_path="/data/paper.pdf")

        sql = conn.execute.call_args[0][0]
        assert "pdf_path" in sql


class TestGetByS2PaperId:
    """Tests for QueueStore.get_by_s2_paper_id()."""

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_returns_dict_when_found(self, mock_get_pool):
        """Returns dict when paper found."""
        row = {"s2_paper_id": "abc123", "title": "Paper", "status": "pending"}
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            return_value=MagicMock(
                **{
                    "__iter__": lambda s: iter(row.items()),
                    "keys": lambda s: row.keys(),
                    "__getitem__": lambda s, k: row[k],
                }
            )
        )
        # Simpler: just return a dict-like
        conn.fetchrow = AsyncMock(return_value=row)
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await QueueStore.get_by_s2_paper_id("abc123")

        assert result is not None
        assert result["s2_paper_id"] == "abc123"

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_returns_none_when_not_found(self, mock_get_pool):
        """Returns None when paper not in queue."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await QueueStore.get_by_s2_paper_id("nonexistent")

        assert result is None


class TestDeleteCompleted:
    """Tests for QueueStore.delete_completed()."""

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_delete_completed_returns_count(self, mock_get_pool):
        """Returns number of deleted rows."""
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="DELETE 5")
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await QueueStore.delete_completed(older_than_days=7)

        assert result == 5

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_delete_completed_zero(self, mock_get_pool):
        """Returns 0 when no rows match."""
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="DELETE 0")
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await QueueStore.delete_completed(older_than_days=30)

        assert result == 0

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_delete_completed_filters_status_and_age(self, mock_get_pool):
        """SQL filters by status=completed AND age > older_than_days."""
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="DELETE 0")
        mock_get_pool.return_value = _make_mock_pool(conn)

        await QueueStore.delete_completed(older_than_days=14)

        sql = conn.execute.call_args[0][0]
        assert "status = $1" in sql
        assert "make_interval" in sql
        args = conn.execute.call_args[0]
        assert QueueStatus.COMPLETED in args
        assert 14 in args


class TestGetQueueStats:
    """Tests for QueueStore.get_stats()."""

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_get_stats_returns_structure(self, mock_get_pool):
        """get_stats returns dict with expected keys."""
        status_rows = [
            {"status": "pending", "count": 10},
            {"status": "completed", "count": 50},
        ]
        domain_rows = [
            {"domain_id": "causal_inference", "count": 40},
            {"domain_id": "time_series", "count": 20},
        ]
        retry_row = {
            "with_retries": 3,
            "max_retries": 2,
            "avg_retries": 1.5,
        }

        conn = AsyncMock()
        conn.fetch = AsyncMock(side_effect=[status_rows, domain_rows])
        conn.fetchrow = AsyncMock(return_value=retry_row)
        mock_get_pool.return_value = _make_mock_pool(conn)

        stats = await QueueStore.get_stats()

        assert stats["by_status"] == {"pending": 10, "completed": 50}
        assert stats["by_domain"] == {"causal_inference": 40, "time_series": 20}
        assert stats["total"] == 60
        assert stats["with_retries"] == 3
        assert stats["max_retries"] == 2
        assert stats["avg_retries"] == pytest.approx(1.5)

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_get_stats_handles_empty(self, mock_get_pool):
        """Empty queue returns zeros."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(side_effect=[[], []])
        conn.fetchrow = AsyncMock(
            return_value={"with_retries": None, "max_retries": None, "avg_retries": None}
        )
        mock_get_pool.return_value = _make_mock_pool(conn)

        stats = await QueueStore.get_stats()

        assert stats["total"] == 0
        assert stats["with_retries"] == 0
        assert stats["max_retries"] == 0


class TestRetryFailed:
    """Tests for QueueStore.retry_failed()."""

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_retry_failed_returns_reset_count(self, mock_get_pool):
        """retry_failed returns count of reset items."""
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 3")
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await QueueStore.retry_failed(max_retries=3)

        assert result == 3

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_retry_failed_resets_to_pending(self, mock_get_pool):
        """SQL sets status to PENDING and clears error_message."""
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 0")
        mock_get_pool.return_value = _make_mock_pool(conn)

        await QueueStore.retry_failed()

        sql = conn.execute.call_args[0][0]
        assert "status = $1" in sql
        assert "error_message = NULL" in sql

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_retry_failed_respects_max_retries(self, mock_get_pool):
        """SQL filters by retry_count < max_retries."""
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 0")
        mock_get_pool.return_value = _make_mock_pool(conn)

        await QueueStore.retry_failed(max_retries=5)

        sql = conn.execute.call_args[0][0]
        assert "retry_count < $3" in sql

    @patch("research_kb_storage.queue_store.get_connection_pool", new_callable=AsyncMock)
    async def test_retry_failed_with_domain_filter(self, mock_get_pool):
        """Passing domain_id adds filter."""
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 0")
        mock_get_pool.return_value = _make_mock_pool(conn)

        await QueueStore.retry_failed(domain_id="causal_inference")

        sql = conn.execute.call_args[0][0]
        assert "domain_id = $4" in sql
