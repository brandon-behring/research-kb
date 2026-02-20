"""Tests for KuzuDB concurrency fixes.

Validates:
1. _kuzu_lock serializes concurrent access to the singleton connection
2. Batch graph scores match individual scores (within tolerance)
3. _graph_semaphore limits concurrent graph-boosted search operations
4. apply_mention_weights helper produces correct results
"""

import asyncio
from uuid import uuid4

import pytest

import research_kb_storage.kuzu_store as kuzu_mod
from research_kb_storage.kuzu_store import (
    bulk_insert_concepts,
    bulk_insert_relationships,
    compute_batch_graph_scores,
    compute_single_graph_score,
    get_kuzu_connection,
)
from research_kb_storage.graph_queries import apply_mention_weights

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_concept(concept_id=None, name="test", concept_type="METHOD"):
    cid = concept_id or uuid4()
    return {
        "id": str(cid),
        "name": name,
        "canonical_name": name.lower().replace(" ", "_"),
        "concept_type": concept_type,
    }


def _make_rel(source_id, target_id, rel_type="USES", strength=1.0):
    return {
        "source_id": source_id,
        "target_id": target_id,
        "relationship_type": rel_type,
        "strength": strength,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_kuzu():
    """Reset module-level singleton and lock before each test."""
    kuzu_mod._db = None
    kuzu_mod._conn = None
    kuzu_mod._kuzu_lock = asyncio.Lock()
    yield
    kuzu_mod._db = None
    kuzu_mod._conn = None
    kuzu_mod._kuzu_lock = asyncio.Lock()


@pytest.fixture
async def populated_graph(tmp_path):
    """Insert a 4-node test graph and return IDs.

    Graph:
        A --REQUIRES--> B
        A --USES------> C
        B --EXTENDS---> D
    """
    db_path = tmp_path / "concurrency_test.kuzu"
    get_kuzu_connection(db_path=db_path)

    id_a, id_b, id_c, id_d = uuid4(), uuid4(), uuid4(), uuid4()
    concepts = [
        _make_concept(id_a, "Method A", "METHOD"),
        _make_concept(id_b, "Assumption B", "ASSUMPTION"),
        _make_concept(id_c, "Technique C", "TECHNIQUE"),
        _make_concept(id_d, "Theorem D", "THEOREM"),
    ]
    rels = [
        _make_rel(str(id_a), str(id_b), "REQUIRES", 1.0),
        _make_rel(str(id_a), str(id_c), "USES", 1.0),
        _make_rel(str(id_b), str(id_d), "EXTENDS", 1.0),
    ]
    await bulk_insert_concepts(concepts)
    await bulk_insert_relationships(rels)

    return {"a": id_a, "b": id_b, "c": id_c, "d": id_d}


# ===================================================================
# 1. Lock Serialization
# ===================================================================


class TestLockSerialization:
    """Verify _kuzu_lock prevents concurrent KuzuDB access."""

    async def test_concurrent_batch_scores_do_not_overlap(self, populated_graph):
        """Launch multiple batch scoring tasks concurrently.

        All should succeed without errors — the lock serializes access.
        Prior to the fix, concurrent asyncio.to_thread calls on the same
        connection caused segfaults or undefined behavior.
        """
        ids = populated_graph

        async def score_task():
            return await compute_batch_graph_scores(
                query_concept_ids=[ids["a"]],
                chunk_concept_ids_list=[[ids["b"]], [ids["c"]]],
                max_hops=3,
            )

        # Run 5 concurrent scoring tasks
        results = await asyncio.gather(*[score_task() for _ in range(5)])

        # All should return identical results
        for scores in results:
            assert len(scores) == 2
            assert all(s >= 0.0 for s in scores)
        # Results should be deterministic
        assert results[0] == results[1] == results[2]

    async def test_lock_exists_at_module_level(self):
        """Verify _kuzu_lock is an asyncio.Lock instance."""
        assert isinstance(kuzu_mod._kuzu_lock, asyncio.Lock)

    async def test_concurrent_single_scores(self, populated_graph):
        """Multiple compute_single_graph_score calls should not race."""
        ids = populated_graph

        async def single_task():
            return await compute_single_graph_score(
                query_concept_ids=[ids["a"]],
                chunk_concept_ids=[ids["b"]],
                max_hops=3,
            )

        results = await asyncio.gather(*[single_task() for _ in range(5)])
        scores = [r[0] for r in results]
        # All should be identical (deterministic graph)
        assert all(s == scores[0] for s in scores)


# ===================================================================
# 2. Batch Scoring Parity
# ===================================================================


class TestBatchScoringParity:
    """Batch scores should match individual scores (within tolerance)."""

    async def test_batch_matches_individual_direct(self, populated_graph):
        """A->B (direct, 1-hop): batch[0] ≈ single(A, B)."""
        ids = populated_graph

        # Individual score
        single_score, _ = await compute_single_graph_score([ids["a"]], [ids["b"]], max_hops=3)

        # Batch score
        batch_scores = await compute_batch_graph_scores([ids["a"]], [[ids["b"]]], max_hops=3)

        assert batch_scores[0] == pytest.approx(single_score, abs=0.05)

    async def test_batch_matches_individual_indirect(self, populated_graph):
        """A->D (2-hop): batch[0] ≈ single(A, D)."""
        ids = populated_graph

        single_score, _ = await compute_single_graph_score([ids["a"]], [ids["d"]], max_hops=3)

        batch_scores = await compute_batch_graph_scores([ids["a"]], [[ids["d"]]], max_hops=3)

        assert batch_scores[0] == pytest.approx(single_score, abs=0.05)

    async def test_batch_matches_individual_multiple_chunks(self, populated_graph):
        """Multi-chunk batch: each score matches its individual counterpart."""
        ids = populated_graph

        # Individual scores
        score_b, _ = await compute_single_graph_score([ids["a"]], [ids["b"]], max_hops=3)
        score_c, _ = await compute_single_graph_score([ids["a"]], [ids["c"]], max_hops=3)
        score_d, _ = await compute_single_graph_score([ids["a"]], [ids["d"]], max_hops=3)

        # Batch scores
        batch_scores = await compute_batch_graph_scores(
            [ids["a"]],
            [[ids["b"]], [ids["c"]], [ids["d"]]],
            max_hops=3,
        )

        assert batch_scores[0] == pytest.approx(score_b, abs=0.05)
        assert batch_scores[1] == pytest.approx(score_c, abs=0.05)
        assert batch_scores[2] == pytest.approx(score_d, abs=0.05)

    async def test_batch_preserves_ordering(self, populated_graph):
        """Direct connections score higher than indirect ones in batch."""
        ids = populated_graph

        batch_scores = await compute_batch_graph_scores(
            [ids["a"]],
            [[ids["b"]], [ids["d"]]],  # B=1-hop, D=2-hop
            max_hops=3,
        )

        # B is closer to A than D, so should score higher
        assert batch_scores[0] > batch_scores[1]


# ===================================================================
# 3. Graph Semaphore Backpressure
# ===================================================================


class TestGraphSemaphore:
    """Verify _graph_semaphore limits concurrent graph operations."""

    def test_semaphore_exists(self):
        from research_kb_storage.search import _graph_semaphore

        assert isinstance(_graph_semaphore, asyncio.Semaphore)

    def test_semaphore_limit_is_3(self):
        from research_kb_storage.search import _graph_semaphore

        # asyncio.Semaphore stores its value as _value
        assert _graph_semaphore._value == 3

    async def test_semaphore_limits_concurrent_access(self):
        """Demonstrate that _graph_semaphore limits to 3 concurrent operations."""

        # Reset for this test
        import research_kb_storage.search as search_mod

        search_mod._graph_semaphore = asyncio.Semaphore(3)
        sem = search_mod._graph_semaphore

        concurrent_count = 0
        max_concurrent = 0

        async def tracked_task():
            nonlocal concurrent_count, max_concurrent
            async with sem:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                await asyncio.sleep(0.05)
                concurrent_count -= 1

        # Launch 10 tasks that all try to acquire the semaphore
        await asyncio.gather(*[tracked_task() for _ in range(10)])

        # Max concurrent should be at most 3
        assert max_concurrent <= 3
        assert max_concurrent >= 1  # At least one ran


# ===================================================================
# 4. apply_mention_weights Helper
# ===================================================================


class TestApplyMentionWeights:
    """Unit tests for the extracted mention-weight helper."""

    def test_no_mention_info_returns_original(self):
        score = apply_mention_weights(0.8, [uuid4()], None)
        assert score == 0.8

    def test_zero_score_returns_zero(self):
        cid = uuid4()
        info = {cid: ("defines", 0.9)}
        score = apply_mention_weights(0.0, [cid], info)
        assert score == 0.0

    def test_empty_chunk_concepts_returns_original(self):
        score = apply_mention_weights(0.5, [], {uuid4(): ("defines", 0.9)})
        assert score == 0.5

    def test_defines_boosts_more_than_example(self):
        cid = uuid4()
        score_defines = apply_mention_weights(0.8, [cid], {cid: ("defines", 1.0)})
        score_example = apply_mention_weights(0.8, [cid], {cid: ("example", 1.0)})
        # "defines" weight=1.0, "example" weight=0.4
        assert score_defines > score_example

    def test_relevance_multiplier_applied(self):
        cid = uuid4()
        # Full relevance
        score_full = apply_mention_weights(0.8, [cid], {cid: ("reference", 1.0)})
        # Half relevance
        score_half = apply_mention_weights(0.8, [cid], {cid: ("reference", 0.5)})
        assert score_full > score_half

    def test_missing_concept_uses_default_weight(self):
        cid_present = uuid4()
        cid_missing = uuid4()
        info = {cid_present: ("defines", 1.0)}
        score = apply_mention_weights(0.8, [cid_present, cid_missing], info)
        # avg of 1.0 (defines) and 0.5 (default) = 0.75
        # 0.8 * 0.75 = 0.6
        assert score == pytest.approx(0.6, abs=0.01)

    def test_none_relevance_uses_base_weight(self):
        cid = uuid4()
        score = apply_mention_weights(0.8, [cid], {cid: ("defines", None)})
        # defines weight=1.0, no relevance multiplier
        # 0.8 * 1.0 = 0.8
        assert score == pytest.approx(0.8, abs=0.01)
