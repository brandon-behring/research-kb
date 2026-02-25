"""KuzuDB pre-warming on daemon startup.

Eliminates ~60s cold-start penalty by paging in the KuzuDB file
and warming BFS indexes before the first real graph query arrives.

Non-blocking: health and fast_search work immediately while warming runs
in a background asyncio task.

Warming queries (in order):
1. Count all Concept nodes — scans node table (~284K rows)
2. Count all RELATES edges — scans edge table (~726K rows)
3. Run a SHORTEST path query — initializes BFS index

Typical warming time: 5-15s depending on OS page cache state.
"""

import asyncio
import time
from typing import Optional

from research_kb_common import get_logger

from research_kb_daemon.metrics import KUZU_WARMUP_DURATION, KUZU_WARMUP_STATUS

logger = get_logger(__name__)

# Module-level state — reset via _reset_state() in tests
_warmup_started: bool = False
_warmup_complete: asyncio.Event = asyncio.Event()
_warmup_status: str = "pending"  # pending | in_progress | completed | failed
_warmup_duration: Optional[float] = None
_warmup_error: Optional[str] = None


def _reset_state() -> None:
    """Reset module state for testing.

    Must be called between tests that exercise warm_kuzu() since the module
    uses globals to track one-shot warmup state.
    """
    global _warmup_started, _warmup_complete, _warmup_status
    global _warmup_duration, _warmup_error

    _warmup_started = False
    _warmup_complete = asyncio.Event()
    _warmup_status = "pending"
    _warmup_duration = None
    _warmup_error = None
    KUZU_WARMUP_STATUS.set(0)


def warmup_status() -> dict:
    """Return current warmup status for the health endpoint.

    Returns:
        Dict with at least {"status": str}. Additional keys:
        - "duration_seconds" when status is "completed"
        - "error" when status is "failed"
    """
    result: dict = {"status": _warmup_status}
    if _warmup_duration is not None:
        result["duration_seconds"] = round(_warmup_duration, 2)
    if _warmup_error is not None:
        result["error"] = _warmup_error
    return result


async def warm_kuzu() -> None:
    """Pre-warm KuzuDB by paging in data and initializing BFS indexes.

    This coroutine is designed to run as a fire-and-forget background task.
    It never raises — all errors are caught, logged, and reflected in
    warmup_status().

    Warming sequence:
    1. Open KuzuDB connection (triggers 110MB file open)
    2. MATCH (c:Concept) RETURN count(c) — scan node table
    3. MATCH ()-[r:RELATES]->() RETURN count(r) — scan edge table
    4. Grab 2 concept IDs, run SHORTEST path — warm BFS indexes
    5. Seed _check_kuzu_ready() cache so first real query skips the check
    """
    global _warmup_started, _warmup_status, _warmup_duration, _warmup_error

    if _warmup_started:
        logger.warning("kuzu_warmup_already_started")
        return

    _warmup_started = True
    _warmup_status = "in_progress"
    KUZU_WARMUP_STATUS.set(1)

    start = time.monotonic()
    logger.info("kuzu_warmup_starting")

    try:
        # Import here to handle missing kuzu gracefully
        from research_kb_storage.graph_queries import _check_kuzu_ready
        from research_kb_storage.kuzu_store import get_kuzu_connection

        # Step 1: Open connection (triggers file open + schema check)
        conn = await asyncio.to_thread(get_kuzu_connection)

        # Step 2: Scan node table
        result = await asyncio.to_thread(conn.execute, "MATCH (c:Concept) RETURN count(c) AS cnt")
        assert not isinstance(result, list)
        df = result.get_as_df()
        node_count = int(df.iloc[0]["cnt"]) if not df.empty else 0
        logger.info("kuzu_warmup_nodes_scanned", count=node_count)

        # Step 3: Scan edge table
        result = await asyncio.to_thread(
            conn.execute, "MATCH ()-[r:RELATES]->() RETURN count(r) AS cnt"
        )
        assert not isinstance(result, list)
        df = result.get_as_df()
        edge_count = int(df.iloc[0]["cnt"]) if not df.empty else 0
        logger.info("kuzu_warmup_edges_scanned", count=edge_count)

        # Step 4: Warm BFS indexes with a shortest path query
        if node_count >= 2:
            result = await asyncio.to_thread(
                conn.execute,
                "MATCH (c:Concept) RETURN c.id LIMIT 2",
            )
            assert not isinstance(result, list)
            df = result.get_as_df()
            if len(df) >= 2:
                id_a = df.iloc[0]["c.id"]
                id_b = df.iloc[1]["c.id"]
                try:
                    await asyncio.to_thread(
                        conn.execute,
                        (
                            "MATCH p = (a:Concept)-[r:RELATES* SHORTEST 1..3]-(b:Concept) "
                            "WHERE a.id = $a_id AND b.id = $b_id "
                            "RETURN length(p) AS hops LIMIT 1"
                        ),
                        {"a_id": id_a, "b_id": id_b},
                    )
                    logger.info("kuzu_warmup_bfs_warmed")
                except Exception as e:
                    # BFS warming is best-effort — path may not exist
                    logger.debug("kuzu_warmup_bfs_no_path", error=str(e))

        # Step 5: Seed the _check_kuzu_ready() cache
        await asyncio.to_thread(_check_kuzu_ready)

        # Success
        _warmup_duration = time.monotonic() - start
        _warmup_status = "completed"
        KUZU_WARMUP_STATUS.set(2)
        KUZU_WARMUP_DURATION.observe(_warmup_duration)
        logger.info(
            "kuzu_warmup_completed",
            duration_seconds=round(_warmup_duration, 2),
            nodes=node_count,
            edges=edge_count,
        )

    except ImportError as e:
        _warmup_duration = time.monotonic() - start
        _warmup_error = f"Import error: {e}"
        _warmup_status = "failed"
        KUZU_WARMUP_STATUS.set(-1)
        logger.error("kuzu_warmup_import_failed", error=str(e))

    except Exception as e:
        _warmup_duration = time.monotonic() - start
        _warmup_error = str(e)
        _warmup_status = "failed"
        KUZU_WARMUP_STATUS.set(-1)
        logger.error(
            "kuzu_warmup_failed",
            error=str(e),
            duration_seconds=round(_warmup_duration, 2),
        )

    finally:
        _warmup_complete.set()
