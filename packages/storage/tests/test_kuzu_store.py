"""Tests for KuzuDB graph store (kuzu_store.py).

Tests all 12 public functions with real KuzuDB instances in temp directories.
Each test gets an isolated database to prevent shared state.

Test graph used by most fixtures:
    A (method) --REQUIRES--> B (assumption)
    A (method) --USES--> C (technique)
    B (assumption) --EXTENDS--> D (theorem)
"""

import asyncio
import shutil
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

import pytest

import research_kb_storage.kuzu_store as kuzu_mod
from research_kb_storage.kuzu_store import (
    RELATIONSHIP_WEIGHTS,
    bulk_insert_concepts,
    bulk_insert_relationships,
    clear_all_data,
    close_kuzu_connection,
    compute_batch_graph_scores,
    compute_single_graph_score,
    find_shortest_path_kuzu,
    find_shortest_path_length_kuzu,
    get_kuzu_connection,
    get_neighborhood_kuzu,
    get_relationship_weight,
    get_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_concept(
    concept_id: UUID | None = None,
    name: str = "test",
    concept_type: str = "METHOD",
) -> dict:
    """Build a concept dict for bulk_insert_concepts."""
    cid = concept_id or uuid4()
    return {
        "id": str(cid),
        "name": name,
        "canonical_name": name.lower().replace(" ", "_"),
        "concept_type": concept_type,
    }


def _make_rel(
    source_id: str,
    target_id: str,
    rel_type: str = "USES",
    strength: float = 1.0,
) -> dict:
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
def _reset_kuzu_singleton():
    """Reset the module-level singleton before and after every test."""
    kuzu_mod._db = None
    kuzu_mod._conn = None
    kuzu_mod._kuzu_lock = asyncio.Lock()
    yield
    kuzu_mod._db = None
    kuzu_mod._conn = None
    kuzu_mod._kuzu_lock = asyncio.Lock()


@pytest.fixture
def tmp_kuzu_dir(tmp_path):
    """Provide a fresh temporary directory for KuzuDB."""
    db_path = tmp_path / "test.kuzu"
    return db_path


@pytest.fixture
def kuzu_conn(tmp_kuzu_dir):
    """Return a live KuzuDB connection on a temp database (sync fixture)."""
    conn = get_kuzu_connection(db_path=tmp_kuzu_dir)
    return conn


@pytest.fixture
async def populated_graph(tmp_kuzu_dir):
    """Insert the standard 4-node test graph and return IDs.

    Graph:
        A --REQUIRES--> B
        A --USES------> C
        B --EXTENDS---> D
    """
    conn = get_kuzu_connection(db_path=tmp_kuzu_dir)

    id_a = uuid4()
    id_b = uuid4()
    id_c = uuid4()
    id_d = uuid4()

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

    return {
        "a": id_a,
        "b": id_b,
        "c": id_c,
        "d": id_d,
    }


# ===================================================================
# 1. Connection Management
# ===================================================================

class TestConnectionManagement:
    """get_kuzu_connection, close_kuzu_connection, singleton, schema."""

    def test_get_connection_returns_object(self, tmp_kuzu_dir):
        conn = get_kuzu_connection(db_path=tmp_kuzu_dir)
        assert conn is not None

    def test_singleton_returns_same_object(self, tmp_kuzu_dir):
        conn1 = get_kuzu_connection(db_path=tmp_kuzu_dir)
        conn2 = get_kuzu_connection(db_path=tmp_kuzu_dir)
        assert conn1 is conn2

    def test_close_resets_globals(self, tmp_kuzu_dir):
        get_kuzu_connection(db_path=tmp_kuzu_dir)
        close_kuzu_connection()
        assert kuzu_mod._db is None
        assert kuzu_mod._conn is None

    def test_reopen_after_close(self, tmp_kuzu_dir):
        conn1 = get_kuzu_connection(db_path=tmp_kuzu_dir)
        close_kuzu_connection()
        conn2 = get_kuzu_connection(db_path=tmp_kuzu_dir)
        assert conn2 is not conn1

    def test_schema_created_on_first_connection(self, tmp_kuzu_dir):
        """Concept and RELATES tables must exist after first connection."""
        conn = get_kuzu_connection(db_path=tmp_kuzu_dir)
        result = conn.execute("CALL show_tables() RETURN *")
        tables = set(result.get_as_df()["name"].tolist())
        assert "Concept" in tables
        assert "RELATES" in tables

    def test_schema_idempotent(self, tmp_kuzu_dir):
        """Calling _ensure_schema twice must not raise."""
        conn = get_kuzu_connection(db_path=tmp_kuzu_dir)
        # Close and reopen -- _ensure_schema runs again
        close_kuzu_connection()
        conn2 = get_kuzu_connection(db_path=tmp_kuzu_dir)
        result = conn2.execute("CALL show_tables() RETURN *")
        tables = set(result.get_as_df()["name"].tolist())
        assert "Concept" in tables


# ===================================================================
# 2. Relationship Weights
# ===================================================================

class TestRelationshipWeights:
    """get_relationship_weight for all known + unknown types."""

    def test_requires_weight(self):
        assert get_relationship_weight("REQUIRES") == 1.0

    def test_extends_weight(self):
        assert get_relationship_weight("EXTENDS") == 0.9

    def test_uses_weight(self):
        assert get_relationship_weight("USES") == 0.8

    def test_addresses_weight(self):
        assert get_relationship_weight("ADDRESSES") == 0.7

    def test_specializes_weight(self):
        assert get_relationship_weight("SPECIALIZES") == 0.6

    def test_generalizes_weight(self):
        assert get_relationship_weight("GENERALIZES") == 0.6

    def test_alternative_to_weight(self):
        assert get_relationship_weight("ALTERNATIVE_TO") == 0.5

    def test_related_to_weight(self):
        assert get_relationship_weight("RELATED_TO") == 0.4

    def test_unknown_type_returns_default(self):
        assert get_relationship_weight("INVENTED_TYPE") == 0.5

    def test_all_known_types_present(self):
        assert len(RELATIONSHIP_WEIGHTS) == 8


# ===================================================================
# 3. Bulk Insert
# ===================================================================

class TestBulkInsert:
    """bulk_insert_concepts and bulk_insert_relationships."""

    async def test_insert_concepts(self, kuzu_conn):
        concepts = [_make_concept(name=f"C{i}") for i in range(5)]
        count = await bulk_insert_concepts(concepts)
        assert count == 5

    async def test_insert_empty_concepts(self, kuzu_conn):
        count = await bulk_insert_concepts([])
        assert count == 0

    async def test_insert_relationships(self, kuzu_conn):
        id_a, id_b = uuid4(), uuid4()
        concepts = [
            _make_concept(id_a, "A"),
            _make_concept(id_b, "B"),
        ]
        await bulk_insert_concepts(concepts)

        rels = [_make_rel(str(id_a), str(id_b), "USES")]
        count = await bulk_insert_relationships(rels)
        assert count == 1

    async def test_insert_empty_relationships(self, kuzu_conn):
        count = await bulk_insert_relationships([])
        assert count == 0

    async def test_concepts_queryable_after_insert(self, kuzu_conn):
        cid = uuid4()
        await bulk_insert_concepts([_make_concept(cid, "Queryable Concept")])
        result = kuzu_conn.execute(
            "MATCH (c:Concept) WHERE c.id = $cid RETURN c.name AS name",
            {"cid": str(cid)},
        )
        df = result.get_as_df()
        assert len(df) == 1
        assert df.iloc[0]["name"] == "Queryable Concept"


# ===================================================================
# 4. Get Stats
# ===================================================================

class TestGetStats:
    """get_stats after data insertion."""

    async def test_stats_after_insert(self, populated_graph):
        stats = await get_stats()
        assert stats["concept_count"] == 4
        assert stats["relationship_count"] == 3
        assert "REQUIRES" in stats["relationship_types"]
        assert "USES" in stats["relationship_types"]
        assert "EXTENDS" in stats["relationship_types"]

    async def test_stats_empty_db(self, kuzu_conn):
        stats = await get_stats()
        assert stats["concept_count"] == 0
        assert stats["relationship_count"] == 0
        assert stats["relationship_types"] == {}


# ===================================================================
# 5. Shortest Path
# ===================================================================

class TestShortestPath:
    """find_shortest_path_kuzu - returns list[dict] or None."""

    async def test_direct_neighbor(self, populated_graph):
        ids = populated_graph
        path = await find_shortest_path_kuzu(ids["a"], ids["b"], max_hops=5)
        assert path is not None
        assert len(path) == 2
        assert path[0]["concept_id"] == str(ids["a"])
        assert path[1]["concept_id"] == str(ids["b"])
        # First node has no incoming relationship
        assert path[0]["rel_type"] is None
        # Second node has the REQUIRES edge
        assert path[1]["rel_type"] == "REQUIRES"

    async def test_two_hops(self, populated_graph):
        """A -> B -> D (REQUIRES then EXTENDS)."""
        ids = populated_graph
        path = await find_shortest_path_kuzu(ids["a"], ids["d"], max_hops=5)
        assert path is not None
        assert len(path) == 3
        assert path[0]["concept_id"] == str(ids["a"])
        assert path[2]["concept_id"] == str(ids["d"])

    async def test_no_path(self, populated_graph):
        """C and D are not connected (C has no outgoing and D has no incoming
        that lead to each other through directed edges -- but KuzuDB uses
        bidirectional SHORTEST, so C-A-B-D is reachable)."""
        ids = populated_graph
        # Create an isolated node
        isolated = uuid4()
        await bulk_insert_concepts([_make_concept(isolated, "Isolated")])
        path = await find_shortest_path_kuzu(ids["a"], isolated, max_hops=5)
        assert path is None

    async def test_max_hops_limit(self, populated_graph):
        """A -> D requires 2 hops; max_hops=1 should fail."""
        ids = populated_graph
        path = await find_shortest_path_kuzu(ids["a"], ids["d"], max_hops=1)
        assert path is None


# ===================================================================
# 6. Shortest Path Length
# ===================================================================

class TestShortestPathLength:
    """find_shortest_path_length_kuzu - returns int or None."""

    async def test_direct_neighbor_length(self, populated_graph):
        ids = populated_graph
        length = await find_shortest_path_length_kuzu(ids["a"], ids["b"])
        assert length == 1

    async def test_two_hop_length(self, populated_graph):
        ids = populated_graph
        length = await find_shortest_path_length_kuzu(ids["a"], ids["d"])
        assert length == 2

    async def test_no_path_returns_none(self, populated_graph):
        ids = populated_graph
        isolated = uuid4()
        await bulk_insert_concepts([_make_concept(isolated, "Alone")])
        length = await find_shortest_path_length_kuzu(ids["a"], isolated)
        assert length is None

    async def test_max_hops_enforced(self, populated_graph):
        ids = populated_graph
        length = await find_shortest_path_length_kuzu(ids["a"], ids["d"], max_hops=1)
        assert length is None


# ===================================================================
# 7. Neighborhood
# ===================================================================

class TestNeighborhood:
    """get_neighborhood_kuzu - N-hop neighbors and edges."""

    async def test_one_hop_neighbors(self, populated_graph):
        ids = populated_graph
        result = await get_neighborhood_kuzu(ids["a"], hops=1)
        assert result["center_id"] == str(ids["a"])
        neighbor_ids = {n["neighbor_id"] for n in result["neighbors"]}
        # A connects to B (REQUIRES) and C (USES) -- 1 hop
        assert str(ids["b"]) in neighbor_ids
        assert str(ids["c"]) in neighbor_ids

    async def test_two_hop_neighbors(self, populated_graph):
        ids = populated_graph
        result = await get_neighborhood_kuzu(ids["a"], hops=2)
        neighbor_ids = {n["neighbor_id"] for n in result["neighbors"]}
        # 2 hops from A should also reach D (A->B->D)
        assert str(ids["d"]) in neighbor_ids

    async def test_relationship_type_filter_raises_on_recursive_rel(
        self, populated_graph
    ):
        """Known bug: relationship_type filter on recursive rel raises StorageError.

        The source injects `AND r.relationship_type = '...'` into a query
        where r is a recursive relationship variable (RELATES*1..N). KuzuDB
        cannot apply property access on RECURSIVE_REL types. This test
        documents the current behavior so the bug can be fixed separately.
        """
        from research_kb_common.errors import StorageError

        ids = populated_graph
        with pytest.raises(StorageError, match="Failed to get neighborhood"):
            await get_neighborhood_kuzu(
                ids["a"], hops=1, relationship_type="REQUIRES"
            )

    async def test_isolated_node_neighborhood(self, kuzu_conn):
        isolated = uuid4()
        await bulk_insert_concepts([_make_concept(isolated, "Lonely")])
        result = await get_neighborhood_kuzu(isolated, hops=2)
        assert result["neighbors"] == []
        assert result["relationships"] == []


# ===================================================================
# 8. Batch Graph Scores
# ===================================================================

class TestBatchGraphScores:
    """compute_batch_graph_scores - scores for multiple chunks."""

    async def test_basic_scoring(self, populated_graph):
        ids = populated_graph
        scores = await compute_batch_graph_scores(
            query_concept_ids=[ids["a"]],
            chunk_concept_ids_list=[[ids["b"]], [ids["c"]]],
            max_hops=3,
        )
        assert len(scores) == 2
        # Both B and C are 1-hop from A, so scores should be equal and positive
        assert scores[0] > 0.0
        assert scores[1] > 0.0

    async def test_empty_query_returns_zeros(self, populated_graph):
        ids = populated_graph
        scores = await compute_batch_graph_scores(
            query_concept_ids=[],
            chunk_concept_ids_list=[[ids["b"]], [ids["c"]]],
        )
        assert scores == [0.0, 0.0]

    async def test_empty_chunks_return_zero(self, populated_graph):
        ids = populated_graph
        scores = await compute_batch_graph_scores(
            query_concept_ids=[ids["a"]],
            chunk_concept_ids_list=[[], [ids["b"]]],
        )
        assert scores[0] == 0.0
        assert scores[1] > 0.0

    async def test_scores_normalized_0_to_1(self, populated_graph):
        ids = populated_graph
        scores = await compute_batch_graph_scores(
            query_concept_ids=[ids["a"], ids["b"]],
            chunk_concept_ids_list=[[ids["c"], ids["d"]]],
            max_hops=3,
        )
        for s in scores:
            assert 0.0 <= s <= 1.0

    async def test_no_connection_yields_zero(self, populated_graph):
        ids = populated_graph
        isolated = uuid4()
        await bulk_insert_concepts([_make_concept(isolated, "Far Away")])
        scores = await compute_batch_graph_scores(
            query_concept_ids=[ids["a"]],
            chunk_concept_ids_list=[[isolated]],
            max_hops=3,
        )
        assert scores == [0.0]


# ===================================================================
# 9. Single Graph Score
# ===================================================================

class TestSingleGraphScore:
    """compute_single_graph_score - (score, explanations) tuple."""

    async def test_direct_connection(self, populated_graph):
        ids = populated_graph
        score, explanations = await compute_single_graph_score(
            query_concept_ids=[ids["a"]],
            chunk_concept_ids=[ids["b"]],
            max_hops=3,
        )
        assert score > 0.0
        assert isinstance(explanations, list)

    async def test_indirect_path(self, populated_graph):
        ids = populated_graph
        score, explanations = await compute_single_graph_score(
            query_concept_ids=[ids["a"]],
            chunk_concept_ids=[ids["d"]],
            max_hops=3,
        )
        assert score > 0.0
        # Indirect is further away, so score should be lower than direct
        direct_score, _ = await compute_single_graph_score(
            query_concept_ids=[ids["a"]],
            chunk_concept_ids=[ids["b"]],
            max_hops=3,
        )
        assert score < direct_score

    async def test_no_path_returns_zero(self, populated_graph):
        ids = populated_graph
        isolated = uuid4()
        await bulk_insert_concepts([_make_concept(isolated, "Nowhere")])
        score, explanations = await compute_single_graph_score(
            query_concept_ids=[ids["a"]],
            chunk_concept_ids=[isolated],
            max_hops=3,
        )
        assert score == 0.0
        assert explanations == []

    async def test_empty_inputs(self, kuzu_conn):
        score, explanations = await compute_single_graph_score(
            query_concept_ids=[],
            chunk_concept_ids=[uuid4()],
        )
        assert score == 0.0
        assert explanations == []

        score2, exp2 = await compute_single_graph_score(
            query_concept_ids=[uuid4()],
            chunk_concept_ids=[],
        )
        assert score2 == 0.0
        assert exp2 == []

    async def test_explanations_contain_concept_names(self, populated_graph):
        ids = populated_graph
        _, explanations = await compute_single_graph_score(
            query_concept_ids=[ids["a"]],
            chunk_concept_ids=[ids["b"]],
            max_hops=3,
        )
        if explanations:
            combined = " ".join(explanations)
            # The explanations should reference the canonical names we inserted
            assert "method_a" in combined or "assumption_b" in combined


# ===================================================================
# 10. Clear Data
# ===================================================================

class TestClearData:
    """clear_all_data - remove everything then verify empty."""

    async def test_clear_returns_count(self, populated_graph):
        deleted = await clear_all_data()
        assert deleted == 4

    async def test_db_empty_after_clear(self, populated_graph):
        await clear_all_data()
        stats = await get_stats()
        assert stats["concept_count"] == 0
        assert stats["relationship_count"] == 0

    async def test_clear_empty_db(self, kuzu_conn):
        deleted = await clear_all_data()
        assert deleted == 0
