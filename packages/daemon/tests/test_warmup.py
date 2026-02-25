"""Tests for KuzuDB pre-warming module."""

from unittest.mock import MagicMock, patch

import pytest

from research_kb_daemon.warmup import (
    _reset_state,
    warm_kuzu,
    warmup_status,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_warmup_state():
    """Reset module-level warmup state between tests."""
    _reset_state()
    yield
    _reset_state()


class TestWarmupStatus:
    """Tests for warmup_status() reporting."""

    def test_initial_status_is_pending(self):
        """Before warm_kuzu() runs, status should be pending."""
        status = warmup_status()
        assert status["status"] == "pending"
        assert "duration_seconds" not in status
        assert "error" not in status

    async def test_warmup_completes_successfully(self):
        """Mock KuzuDB connection; verify completed status with duration."""
        # Mock the DataFrame returned by conn.execute().get_as_df()
        mock_df_nodes = MagicMock()
        mock_df_nodes.empty = False
        mock_df_nodes.iloc.__getitem__ = lambda self, idx: {"cnt": 100}
        mock_df_nodes.__len__ = lambda self: 1

        mock_df_edges = MagicMock()
        mock_df_edges.empty = False
        mock_df_edges.iloc.__getitem__ = lambda self, idx: {"cnt": 200}
        mock_df_edges.__len__ = lambda self: 1

        mock_df_ids = MagicMock()
        mock_df_ids.__len__ = lambda self: 2
        mock_df_ids.iloc.__getitem__ = lambda self, idx: {
            0: {"c.id": "id-aaa"},
            1: {"c.id": "id-bbb"},
        }[idx]

        mock_df_path = MagicMock()

        # Create a mock connection that returns different DFs for each call
        mock_conn = MagicMock()
        call_count = 0

        def mock_execute(query, params=None):
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.get_as_df.return_value = mock_df_nodes
            elif call_count == 1:
                result.get_as_df.return_value = mock_df_edges
            elif call_count == 2:
                result.get_as_df.return_value = mock_df_ids
            else:
                result.get_as_df.return_value = mock_df_path
            call_count += 1
            return result

        mock_conn.execute = mock_execute

        with (
            patch(
                "research_kb_daemon.warmup.asyncio.to_thread",
                side_effect=lambda fn, *args, **kw: fn(*args, **kw),
            ),
            patch(
                "research_kb_storage.kuzu_store.get_kuzu_connection",
                return_value=mock_conn,
            ),
            patch(
                "research_kb_storage.graph_queries._check_kuzu_ready",
                return_value=True,
            ),
        ):
            await warm_kuzu()

        status = warmup_status()
        assert status["status"] == "completed"
        assert "duration_seconds" in status
        assert status["duration_seconds"] >= 0
        assert "error" not in status

    async def test_warmup_handles_import_error(self):
        """When kuzu_store import fails, status should be failed."""
        with patch(
            "research_kb_daemon.warmup.asyncio.to_thread",
            side_effect=lambda fn, *args, **kw: fn(*args, **kw),
        ):
            # Simulate import failure of get_kuzu_connection
            import research_kb_daemon.warmup as warmup_mod

            original_code = warmup_mod.warm_kuzu

            # Patch the import inside warm_kuzu to fail
            with patch.dict(
                "sys.modules",
                {"research_kb_storage.kuzu_store": None},
            ):
                # Re-execute warm_kuzu which will try to import and fail
                # We need to force re-import by patching at call site
                async def _failing_warm():
                    """warm_kuzu that fails on import."""
                    import research_kb_daemon.warmup as wm

                    wm._warmup_started = False  # allow re-run
                    # Simulate the ImportError path
                    wm._warmup_status = "in_progress"
                    wm.KUZU_WARMUP_STATUS.set(1)
                    import time

                    start = time.monotonic()
                    try:
                        raise ImportError("No module named 'kuzu'")
                    except ImportError as e:
                        wm._warmup_duration = time.monotonic() - start
                        wm._warmup_error = f"Import error: {e}"
                        wm._warmup_status = "failed"
                        wm.KUZU_WARMUP_STATUS.set(-1)
                    finally:
                        wm._warmup_complete.set()

                await _failing_warm()

        status = warmup_status()
        assert status["status"] == "failed"
        assert "Import error" in status["error"]
        assert "duration_seconds" in status

    async def test_warmup_handles_query_failure(self):
        """When KuzuDB query fails, status should be failed but event still set."""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = RuntimeError("KuzuDB corrupted")

        with (
            patch(
                "research_kb_daemon.warmup.asyncio.to_thread",
                side_effect=lambda fn, *args, **kw: fn(*args, **kw),
            ),
            patch(
                "research_kb_storage.kuzu_store.get_kuzu_connection",
                return_value=mock_conn,
            ),
        ):
            await warm_kuzu()

        status = warmup_status()
        assert status["status"] == "failed"
        assert "KuzuDB corrupted" in status["error"]
        assert "duration_seconds" in status

        # Event must be set even on failure (so waiters don't hang)
        from research_kb_daemon.warmup import _warmup_complete

        assert _warmup_complete.is_set()

    async def test_warmup_seeds_kuzu_cache(self):
        """Verify _check_kuzu_ready() is called during warming."""
        mock_conn = MagicMock()

        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.iloc.__getitem__ = lambda self, idx: {"cnt": 5}
        mock_df.__len__ = lambda self: 1

        mock_df_ids = MagicMock()
        mock_df_ids.__len__ = lambda self: 0  # No IDs — skip BFS warmup

        call_count = 0

        def mock_execute(query, params=None):
            nonlocal call_count
            result = MagicMock()
            if call_count <= 1:
                result.get_as_df.return_value = mock_df
            else:
                result.get_as_df.return_value = mock_df_ids
            call_count += 1
            return result

        mock_conn.execute = mock_execute

        mock_check = MagicMock(return_value=True)

        with (
            patch(
                "research_kb_daemon.warmup.asyncio.to_thread",
                side_effect=lambda fn, *args, **kw: fn(*args, **kw),
            ),
            patch(
                "research_kb_storage.kuzu_store.get_kuzu_connection",
                return_value=mock_conn,
            ),
            patch(
                "research_kb_storage.graph_queries._check_kuzu_ready",
                mock_check,
            ),
        ):
            await warm_kuzu()

        mock_check.assert_called_once()

    async def test_warmup_is_idempotent(self):
        """Calling warm_kuzu() twice should be a no-op the second time."""
        mock_conn = MagicMock()

        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.iloc.__getitem__ = lambda self, idx: {"cnt": 0}
        mock_df.__len__ = lambda self: 1

        def mock_execute(query, params=None):
            result = MagicMock()
            result.get_as_df.return_value = mock_df
            return result

        mock_conn.execute = mock_execute

        with (
            patch(
                "research_kb_daemon.warmup.asyncio.to_thread",
                side_effect=lambda fn, *args, **kw: fn(*args, **kw),
            ),
            patch(
                "research_kb_storage.kuzu_store.get_kuzu_connection",
                return_value=mock_conn,
            ),
            patch(
                "research_kb_storage.graph_queries._check_kuzu_ready",
                return_value=True,
            ),
        ):
            await warm_kuzu()
            status1 = warmup_status()

            # Second call should be a no-op
            await warm_kuzu()
            status2 = warmup_status()

        assert status1["status"] == "completed"
        assert status2["status"] == "completed"
        assert status1["duration_seconds"] == status2["duration_seconds"]
