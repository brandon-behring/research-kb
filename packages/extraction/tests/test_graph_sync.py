"""Tests for Neo4j graph synchronization service."""

import pytest
from uuid import uuid4

from research_kb_extraction.graph_sync import GraphSyncService


# Mark all tests as requiring special setup
pytestmark = pytest.mark.skipif(
    "not config.getoption('--run-neo4j')",
    reason="Requires Neo4j (run with --run-neo4j flag)",
)


@pytest.fixture
async def graph_sync():
    """Create GraphSyncService instance for testing."""
    service = GraphSyncService(
        uri="bolt://localhost:7687", username="neo4j", password="research_kb_dev"
    )

    # Check if Neo4j is available
    if not await service.is_available():
        pytest.skip("Neo4j not available")

    # Clear graph before tests
    await service.clear_all()

    yield service

    # Cleanup after tests
    await service.clear_all()
    await service.close()


@pytest.mark.asyncio
async def test_graph_sync_service_initialization():
    """Test GraphSyncService initialization."""
    service = GraphSyncService(
        uri="bolt://localhost:7687", username="test_user", password="test_pass"
    )

    assert service.uri == "bolt://localhost:7687"
    assert service.username == "test_user"
    assert service.password == "test_pass"
    assert service._driver is None

    await service.close()


@pytest.mark.asyncio
async def test_is_available(graph_sync):
    """Test checking Neo4j availability."""
    available = await graph_sync.is_available()
    assert available is True


@pytest.mark.asyncio
async def test_sync_concept(graph_sync):
    """Test syncing a concept to Neo4j."""
    concept_id = uuid4()

    await graph_sync.sync_concept(
        concept_id=concept_id,
        name="Instrumental Variables",
        canonical_name="instrumental_variables",
        concept_type="method",
        definition="A method for estimating causal effects",
    )

    # Verify concept was created
    stats = await graph_sync.get_stats()
    assert stats["concepts"] == 1


@pytest.mark.asyncio
async def test_sync_concept_upsert(graph_sync):
    """Test that syncing same concept twice updates (upsert)."""
    concept_id = uuid4()

    # First sync
    await graph_sync.sync_concept(
        concept_id=concept_id,
        name="Original Name",
        canonical_name="concept",
        concept_type="method",
        definition="Original definition",
    )

    # Second sync with same ID
    await graph_sync.sync_concept(
        concept_id=concept_id,
        name="Updated Name",
        canonical_name="concept",
        concept_type="method",
        definition="Updated definition",
    )

    # Should still have only 1 concept (upserted)
    stats = await graph_sync.get_stats()
    assert stats["concepts"] == 1


@pytest.mark.asyncio
async def test_sync_relationship(graph_sync):
    """Test syncing a relationship to Neo4j."""
    # Create two concepts first
    source_id = uuid4()
    target_id = uuid4()

    await graph_sync.sync_concept(
        concept_id=source_id,
        name="IV",
        canonical_name="instrumental_variables",
        concept_type="method",
    )

    await graph_sync.sync_concept(
        concept_id=target_id,
        name="Endogeneity",
        canonical_name="endogeneity",
        concept_type="problem",
    )

    # Create relationship
    rel_id = uuid4()
    await graph_sync.sync_relationship(
        relationship_id=rel_id,
        source_concept_id=source_id,
        target_concept_id=target_id,
        relationship_type="ADDRESSES",
        strength=0.9,
    )

    # Verify relationship was created
    stats = await graph_sync.get_stats()
    assert stats["relationships"] == 1


@pytest.mark.asyncio
async def test_find_related_concepts(graph_sync):
    """Test finding related concepts within N hops."""
    # Create a chain: A -> B -> C
    concept_a = uuid4()
    concept_b = uuid4()
    concept_c = uuid4()

    await graph_sync.sync_concept(concept_a, "A", "a", "concept")
    await graph_sync.sync_concept(concept_b, "B", "b", "concept")
    await graph_sync.sync_concept(concept_c, "C", "c", "concept")

    await graph_sync.sync_relationship(uuid4(), concept_a, concept_b, "RELATED_TO")
    await graph_sync.sync_relationship(uuid4(), concept_b, concept_c, "RELATED_TO")

    # Find concepts related to A within 1 hop
    related_1hop = await graph_sync.find_related_concepts(concept_a, max_hops=1)
    assert len(related_1hop) == 1  # Should find B

    # Find concepts related to A within 2 hops
    related_2hop = await graph_sync.find_related_concepts(concept_a, max_hops=2)
    assert len(related_2hop) == 2  # Should find B and C


@pytest.mark.asyncio
async def test_find_shortest_path(graph_sync):
    """Test finding shortest path between concepts."""
    # Create a path: IV -> Endogeneity -> Bias
    iv_id = uuid4()
    endo_id = uuid4()
    bias_id = uuid4()

    await graph_sync.sync_concept(iv_id, "IV", "instrumental_variables", "method")
    await graph_sync.sync_concept(endo_id, "Endogeneity", "endogeneity", "problem")
    await graph_sync.sync_concept(bias_id, "Bias", "bias", "concept")

    await graph_sync.sync_relationship(uuid4(), iv_id, endo_id, "ADDRESSES")
    await graph_sync.sync_relationship(uuid4(), endo_id, bias_id, "CAUSES")

    # Find shortest path
    path = await graph_sync.find_shortest_path(
        "instrumental_variables", "bias", max_hops=5
    )

    assert path is not None
    assert path["path_length"] == 2
    assert len(path["concept_path"]) == 3


@pytest.mark.asyncio
async def test_find_shortest_path_no_path(graph_sync):
    """Test finding shortest path when no path exists."""
    # Create two unconnected concepts
    concept_a = uuid4()
    concept_b = uuid4()

    await graph_sync.sync_concept(concept_a, "A", "a", "concept")
    await graph_sync.sync_concept(concept_b, "B", "b", "concept")

    # Try to find path (should return None)
    path = await graph_sync.find_shortest_path("a", "b", max_hops=5)

    assert path is None


@pytest.mark.asyncio
async def test_compute_graph_score_direct_links(graph_sync):
    """Test computing graph score with direct links."""
    # Create query concepts and chunk concepts
    query_c1 = uuid4()
    query_c2 = uuid4()
    chunk_c1 = uuid4()
    chunk_c2 = uuid4()

    await graph_sync.sync_concept(query_c1, "Q1", "q1", "concept")
    await graph_sync.sync_concept(query_c2, "Q2", "q2", "concept")
    await graph_sync.sync_concept(chunk_c1, "C1", "c1", "concept")
    await graph_sync.sync_concept(chunk_c2, "C2", "c2", "concept")

    # Create direct links: Q1->C1, Q2->C2
    await graph_sync.sync_relationship(uuid4(), query_c1, chunk_c1, "RELATED_TO")
    await graph_sync.sync_relationship(uuid4(), query_c2, chunk_c2, "RELATED_TO")

    # Compute score
    score = await graph_sync.compute_graph_score(
        query_concept_ids=[query_c1, query_c2],
        chunk_concept_ids=[chunk_c1, chunk_c2],
        max_hops=2,
    )

    # Direct links contribute 1.0 each, 2 links / 4 pairs = 0.5
    assert score > 0.0


@pytest.mark.asyncio
async def test_compute_graph_score_no_connections(graph_sync):
    """Test graph score with no connections."""
    # Create unconnected concepts
    query_c = uuid4()
    chunk_c = uuid4()

    await graph_sync.sync_concept(query_c, "Q", "q", "concept")
    await graph_sync.sync_concept(chunk_c, "C", "c", "concept")

    # Compute score (should be 0.0)
    score = await graph_sync.compute_graph_score(
        query_concept_ids=[query_c], chunk_concept_ids=[chunk_c], max_hops=2
    )

    assert score == 0.0


@pytest.mark.asyncio
async def test_compute_graph_score_empty_lists(graph_sync):
    """Test graph score with empty concept lists."""
    score = await graph_sync.compute_graph_score(
        query_concept_ids=[], chunk_concept_ids=[], max_hops=2
    )

    assert score == 0.0


@pytest.mark.asyncio
async def test_clear_all(graph_sync):
    """Test clearing all nodes and relationships."""
    # Create some data
    concept_id = uuid4()
    await graph_sync.sync_concept(concept_id, "Test", "test", "concept")

    # Clear
    await graph_sync.clear_all()

    # Verify empty
    stats = await graph_sync.get_stats()
    assert stats["concepts"] == 0
    assert stats["relationships"] == 0


@pytest.mark.asyncio
async def test_get_stats(graph_sync):
    """Test getting graph statistics."""
    # Initially empty
    stats = await graph_sync.get_stats()
    assert stats["concepts"] == 0
    assert stats["relationships"] == 0

    # Add concepts
    c1 = uuid4()
    c2 = uuid4()
    await graph_sync.sync_concept(c1, "C1", "c1", "concept")
    await graph_sync.sync_concept(c2, "C2", "c2", "concept")

    # Add relationship
    await graph_sync.sync_relationship(uuid4(), c1, c2, "RELATED_TO")

    # Check stats
    stats = await graph_sync.get_stats()
    assert stats["concepts"] == 2
    assert stats["relationships"] == 1


@pytest.mark.asyncio
async def test_context_manager():
    """Test using GraphSyncService as async context manager."""
    async with GraphSyncService() as service:
        if await service.is_available():
            concept_id = uuid4()
            await service.sync_concept(concept_id, "Test", "test", "concept")

            stats = await service.get_stats()
            assert stats["concepts"] >= 1

            await service.clear_all()

    # Driver should be closed after context
    # (cannot easily test without accessing private _driver)
