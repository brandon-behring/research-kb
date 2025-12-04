"""Tests for graph query utilities.

Tests:
- Shortest path finding
- N-hop neighborhood traversal
- Graph score computation
"""

import pytest
from uuid import uuid4

from research_kb_contracts import ConceptType, RelationshipType
from research_kb_storage import (
    ConceptStore,
    RelationshipStore,
    find_shortest_path,
    find_shortest_path_length,
    get_neighborhood,
    compute_graph_score,
)


@pytest.fixture
async def test_graph(db_pool):
    """Create a small test graph.

    Graph structure:
        A -[USES]-> B -[REQUIRES]-> C
        A -[ADDRESSES]-> D
        B -[ADDRESSES]-> D
    """
    # Create concepts
    concept_a = await ConceptStore.create(
        name="Concept A",
        canonical_name=f"concept_a_{uuid4().hex[:8]}",
        concept_type=ConceptType.METHOD,
    )

    concept_b = await ConceptStore.create(
        name="Concept B",
        canonical_name=f"concept_b_{uuid4().hex[:8]}",
        concept_type=ConceptType.METHOD,
    )

    concept_c = await ConceptStore.create(
        name="Concept C",
        canonical_name=f"concept_c_{uuid4().hex[:8]}",
        concept_type=ConceptType.ASSUMPTION,
    )

    concept_d = await ConceptStore.create(
        name="Concept D",
        canonical_name=f"concept_d_{uuid4().hex[:8]}",
        concept_type=ConceptType.PROBLEM,
    )

    # Create relationships
    rel_ab = await RelationshipStore.create(
        source_concept_id=concept_a.id,
        target_concept_id=concept_b.id,
        relationship_type=RelationshipType.USES,
    )

    rel_bc = await RelationshipStore.create(
        source_concept_id=concept_b.id,
        target_concept_id=concept_c.id,
        relationship_type=RelationshipType.REQUIRES,
    )

    rel_ad = await RelationshipStore.create(
        source_concept_id=concept_a.id,
        target_concept_id=concept_d.id,
        relationship_type=RelationshipType.ADDRESSES,
    )

    rel_bd = await RelationshipStore.create(
        source_concept_id=concept_b.id,
        target_concept_id=concept_d.id,
        relationship_type=RelationshipType.ADDRESSES,
    )

    return {
        "concepts": {
            "a": concept_a,
            "b": concept_b,
            "c": concept_c,
            "d": concept_d,
        },
        "relationships": {
            "ab": rel_ab,
            "bc": rel_bc,
            "ad": rel_ad,
            "bd": rel_bd,
        },
    }


class TestShortestPath:
    """Tests for find_shortest_path()."""

    async def test_direct_path(self, test_graph):
        """Test finding direct 1-hop path."""
        concepts = test_graph["concepts"]

        path = await find_shortest_path(concepts["a"].id, concepts["b"].id)

        assert path is not None
        assert len(path) == 2  # Start + 1 hop

        # Check path structure
        assert path[0][0].id == concepts["a"].id
        assert path[0][1] is None  # No incoming relationship

        assert path[1][0].id == concepts["b"].id
        assert path[1][1] is not None  # Has relationship
        assert path[1][1].relationship_type == RelationshipType.USES

    async def test_two_hop_path(self, test_graph):
        """Test finding 2-hop path."""
        concepts = test_graph["concepts"]

        path = await find_shortest_path(concepts["a"].id, concepts["c"].id)

        assert path is not None
        assert len(path) == 3  # Start + 2 hops

        # Check path: A -> B -> C
        assert path[0][0].id == concepts["a"].id
        assert path[1][0].id == concepts["b"].id
        assert path[2][0].id == concepts["c"].id

        assert path[1][1].relationship_type == RelationshipType.USES
        assert path[2][1].relationship_type == RelationshipType.REQUIRES

    async def test_no_path_exists(self, test_graph):
        """Test when no path exists between concepts."""
        concepts = test_graph["concepts"]

        # Try reverse path (D -> A) - should fail since graph is directed
        path = await find_shortest_path(concepts["d"].id, concepts["a"].id)

        assert path is None

    async def test_path_with_max_hops(self, test_graph):
        """Test path finding with hop limit."""
        concepts = test_graph["concepts"]

        # Try A -> C with max 1 hop (should fail, requires 2)
        path = await find_shortest_path(concepts["a"].id, concepts["c"].id, max_hops=1)

        assert path is None

        # Try with max 2 hops (should succeed)
        path = await find_shortest_path(concepts["a"].id, concepts["c"].id, max_hops=2)

        assert path is not None
        assert len(path) == 3

    async def test_same_start_end(self, test_graph):
        """Test path from concept to itself."""
        concepts = test_graph["concepts"]

        path = await find_shortest_path(concepts["a"].id, concepts["a"].id)

        # Should return immediate result
        assert path is not None
        assert len(path) == 1
        assert path[0][0].id == concepts["a"].id


class TestShortestPathLength:
    """Tests for find_shortest_path_length()."""

    async def test_direct_path_length(self, test_graph):
        """Test length of direct path."""
        concepts = test_graph["concepts"]

        length = await find_shortest_path_length(concepts["a"].id, concepts["b"].id)

        assert length == 1

    async def test_two_hop_path_length(self, test_graph):
        """Test length of 2-hop path."""
        concepts = test_graph["concepts"]

        length = await find_shortest_path_length(concepts["a"].id, concepts["c"].id)

        assert length == 2

    async def test_no_path_length(self, test_graph):
        """Test when no path exists."""
        concepts = test_graph["concepts"]

        length = await find_shortest_path_length(concepts["d"].id, concepts["a"].id)

        assert length is None

    async def test_same_concept_length(self, test_graph):
        """Test distance from concept to itself."""
        concepts = test_graph["concepts"]

        length = await find_shortest_path_length(concepts["a"].id, concepts["a"].id)

        assert length == 0


class TestNeighborhood:
    """Tests for get_neighborhood()."""

    async def test_one_hop_neighborhood(self, test_graph):
        """Test 1-hop neighborhood."""
        concepts = test_graph["concepts"]

        neighborhood = await get_neighborhood(concepts["a"].id, hops=1)

        assert neighborhood["center"].id == concepts["a"].id

        # A connects to B and D (1-hop)
        concept_ids = {c.id for c in neighborhood["concepts"]}
        assert concepts["a"].id in concept_ids
        assert concepts["b"].id in concept_ids
        assert concepts["d"].id in concept_ids
        assert len(neighborhood["concepts"]) == 3  # A, B, D

        # Should have 3 relationships (A->B, A->D, B->D)
        # B->D is included because both B and D are in the neighborhood
        assert len(neighborhood["relationships"]) == 3

    async def test_two_hop_neighborhood(self, test_graph):
        """Test 2-hop neighborhood."""
        concepts = test_graph["concepts"]

        neighborhood = await get_neighborhood(concepts["a"].id, hops=2)

        # A connects to B, D (1-hop) and C (2-hop via B)
        concept_ids = {c.id for c in neighborhood["concepts"]}
        assert concepts["a"].id in concept_ids
        assert concepts["b"].id in concept_ids
        assert concepts["c"].id in concept_ids
        assert concepts["d"].id in concept_ids
        assert len(neighborhood["concepts"]) == 4

    async def test_neighborhood_with_relationship_filter(self, test_graph):
        """Test neighborhood with relationship type filter."""
        concepts = test_graph["concepts"]

        neighborhood = await get_neighborhood(
            concepts["a"].id, hops=1, relationship_type=RelationshipType.USES
        )

        # Only B should be included (via USES), not D (via ADDRESSES)
        concept_ids = {c.id for c in neighborhood["concepts"]}
        assert concepts["b"].id in concept_ids
        assert concepts["d"].id not in concept_ids

    async def test_isolated_concept_neighborhood(self, db_pool):
        """Test neighborhood of concept with no relationships."""
        isolated = await ConceptStore.create(
            name="Isolated",
            canonical_name=f"isolated_{uuid4().hex[:8]}",
            concept_type=ConceptType.DEFINITION,
        )

        neighborhood = await get_neighborhood(isolated.id, hops=1)

        assert neighborhood["center"].id == isolated.id
        assert len(neighborhood["concepts"]) == 1  # Only itself
        assert len(neighborhood["relationships"]) == 0


class TestGraphScore:
    """Tests for compute_graph_score()."""

    async def test_graph_score_direct_connection(self, test_graph):
        """Test score with direct connection."""
        concepts = test_graph["concepts"]

        # A and B are directly connected
        score = await compute_graph_score(
            query_concept_ids=[concepts["a"].id],
            chunk_concept_ids=[concepts["b"].id],
        )

        # Direct connection (1 hop): 1 / (1 + 1) = 0.5
        assert score == 0.5

    async def test_graph_score_two_hop(self, test_graph):
        """Test score with 2-hop connection."""
        concepts = test_graph["concepts"]

        # A -> B -> C (2 hops)
        score = await compute_graph_score(
            query_concept_ids=[concepts["a"].id],
            chunk_concept_ids=[concepts["c"].id],
        )

        # 2-hop connection: 1 / (2 + 1) = 0.333
        assert 0.32 < score < 0.34

    async def test_graph_score_multiple_pairs(self, test_graph):
        """Test score with multiple query/chunk pairs."""
        concepts = test_graph["concepts"]

        # Query: A, Chunks: B, D
        # A->B: 0.5 (1 hop), A->D: 0.5 (1 hop)
        # Total: 1.0, Max: 2.0, Score: 0.5
        score = await compute_graph_score(
            query_concept_ids=[concepts["a"].id],
            chunk_concept_ids=[concepts["b"].id, concepts["d"].id],
        )

        assert score == 0.5

    async def test_graph_score_no_connection(self, test_graph):
        """Test score when concepts not connected."""
        concepts = test_graph["concepts"]

        # D -> A has no path (reversed direction)
        score = await compute_graph_score(
            query_concept_ids=[concepts["d"].id],
            chunk_concept_ids=[concepts["a"].id],
        )

        assert score == 0.0

    async def test_graph_score_empty_inputs(self, test_graph):
        """Test score with empty concept lists."""
        concepts = test_graph["concepts"]

        # Empty query
        score = await compute_graph_score(
            query_concept_ids=[],
            chunk_concept_ids=[concepts["a"].id],
        )
        assert score == 0.0

        # Empty chunk
        score = await compute_graph_score(
            query_concept_ids=[concepts["a"].id],
            chunk_concept_ids=[],
        )
        assert score == 0.0

    async def test_graph_score_normalized(self, test_graph):
        """Test that score is properly normalized."""
        concepts = test_graph["concepts"]

        # Multiple queries and chunks
        score = await compute_graph_score(
            query_concept_ids=[concepts["a"].id, concepts["b"].id],
            chunk_concept_ids=[concepts["c"].id, concepts["d"].id],
        )

        # Score should be between 0 and 1
        assert 0.0 <= score <= 1.0


class TestGraphIntegration:
    """Integration tests for graph operations."""

    async def test_path_and_neighborhood_consistency(self, test_graph):
        """Test that path finding and neighborhood queries are consistent."""
        concepts = test_graph["concepts"]

        # Get neighborhood of A
        neighborhood = await get_neighborhood(concepts["a"].id, hops=1)

        # All concepts in 1-hop neighborhood should have path length <= 1
        for c in neighborhood["concepts"]:
            if c.id == concepts["a"].id:
                continue  # Skip center

            length = await find_shortest_path_length(concepts["a"].id, c.id)
            assert length is not None
            assert length <= 1

    async def test_complex_graph_traversal(self, db_pool):
        """Test traversal in more complex graph structure."""
        # Create chain: A -> B -> C -> D -> E
        concepts = []
        for i, letter in enumerate(["A", "B", "C", "D", "E"]):
            c = await ConceptStore.create(
                name=f"Concept {letter}",
                canonical_name=f"concept_{letter.lower()}_{uuid4().hex[:8]}",
                concept_type=ConceptType.METHOD,
            )
            concepts.append(c)

        # Create chain relationships
        for i in range(len(concepts) - 1):
            await RelationshipStore.create(
                source_concept_id=concepts[i].id,
                target_concept_id=concepts[i + 1].id,
                relationship_type=RelationshipType.USES,
            )

        # Test: A -> E should be 4 hops
        length = await find_shortest_path_length(concepts[0].id, concepts[4].id)
        assert length == 4

        # Test: Path should go through all intermediate nodes
        path = await find_shortest_path(concepts[0].id, concepts[4].id, max_hops=5)
        assert path is not None
        assert len(path) == 5
        assert all(path[i][0].id == concepts[i].id for i in range(5))

        # Test: 2-hop neighborhood of B should include C, D (forward only)
        # Note: Neighborhood traversal is directed, so A won't be included
        neighborhood = await get_neighborhood(concepts[1].id, hops=2)
        neighbor_ids = {c.id for c in neighborhood["concepts"]}
        assert concepts[2].id in neighbor_ids  # C (1-hop forward)
        assert concepts[3].id in neighbor_ids  # D (2-hop forward)
        assert concepts[4].id not in neighbor_ids  # E (3-hop, out of range)
