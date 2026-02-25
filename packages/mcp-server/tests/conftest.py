"""Test configuration for MCP server tests."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from dataclasses import dataclass
from datetime import datetime

from research_kb_contracts import (
    Source,
    Concept,
    ConceptRelationship,
    SourceType,
    ConceptType,
    RelationshipType,
)


@dataclass
class MockChunk:
    """Mock chunk for testing."""

    id: str
    content: str
    page_start: int = 1
    page_end: int = 1
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@pytest.fixture
def mock_embedding_client():
    """Mock embedding client for tests."""
    with patch("research_kb_api.service._embedding_client") as mock:
        client = MagicMock()
        client.embed.return_value = [0.1] * 1024
        mock.return_value = client
        yield client


@pytest.fixture
def mock_pool():
    """Mock database connection pool."""
    pool = AsyncMock()
    pool.get_size.return_value = 5
    pool.close = AsyncMock()
    pool.acquire = AsyncMock()
    return pool


@pytest.fixture
def sample_source():
    """Create a sample source for testing."""
    return Source(
        id=uuid4(),
        title="Double Machine Learning for Treatment Effects",
        source_type=SourceType.PAPER,
        authors=["Chernozhukov, V.", "Chetverikov, D.", "Demirer, M."],
        year=2018,
        domain_id="causal_inference",
        file_hash="dml_test_hash",
        metadata={"doi": "10.1214/17-EJS1341SI"},
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def sample_concept():
    """Create a sample concept for testing."""
    return Concept(
        id=uuid4(),
        name="Double Machine Learning",
        canonical_name="double_machine_learning",
        concept_type=ConceptType.METHOD,
        domain_id="causal_inference",
        definition="A method for causal inference using machine learning",
        created_at=datetime.now(),
    )


@pytest.fixture
def sample_relationship(sample_concept):
    """Create a sample relationship for testing."""
    return ConceptRelationship(
        id=uuid4(),
        source_concept_id=sample_concept.id,
        target_concept_id=uuid4(),
        relationship_type=RelationshipType.USES,
        created_at=datetime.now(),
    )


@pytest.fixture
def sample_chunk():
    """Create a sample chunk for testing."""
    return MockChunk(
        id=str(uuid4()),
        content="Double machine learning uses Neyman orthogonality to achieve robustness to regularization bias.",
        page_start=15,
        page_end=15,
        metadata={"section_header": "2.1 Method Overview"},
    )


@pytest.fixture
def mock_storage(sample_source, sample_concept, sample_relationship, sample_chunk):
    """Mock all storage operations."""
    with (
        patch("research_kb_api.service.ConceptStore") as concept_mock,
        patch("research_kb_api.service.SourceStore") as source_mock,
        patch("research_kb_api.service.ChunkStore") as chunk_mock,
        patch("research_kb_api.service.RelationshipStore") as rel_mock,
        patch("research_kb_api.service.search_hybrid") as search_mock,
        patch("research_kb_api.service.search_hybrid_v2") as search_v2_mock,
        patch("research_kb_api.service.search_with_rerank") as rerank_mock,
        patch("research_kb_api.service.search_with_expansion") as expand_mock,
        patch("research_kb_api.service.get_neighborhood") as neighborhood_mock,
        patch("research_kb_api.service.find_shortest_path") as path_mock,
        patch("research_kb_api.service.get_citing_sources") as citing_mock,
        patch("research_kb_api.service.get_cited_sources") as cited_mock,
    ):

        # Default return values
        concept_mock.count = AsyncMock(return_value=100)
        concept_mock.search = AsyncMock(return_value=[sample_concept])
        concept_mock.list_all = AsyncMock(return_value=[sample_concept])
        concept_mock.get = AsyncMock(return_value=sample_concept)

        source_mock.list_all = AsyncMock(return_value=[sample_source])
        source_mock.get = AsyncMock(return_value=sample_source)

        chunk_mock.get_by_source = AsyncMock(return_value=[sample_chunk])

        rel_mock.get_for_concept = AsyncMock(return_value=[sample_relationship])

        search_mock.return_value = []
        search_v2_mock.return_value = []
        rerank_mock.return_value = []
        expand_mock.return_value = ([], None)

        neighborhood_mock.return_value = {
            "concepts": [sample_concept],
            "relationships": [sample_relationship],
        }
        path_mock.return_value = [
            {"id": str(sample_concept.id), "name": sample_concept.name},
        ]
        citing_mock.return_value = []
        cited_mock.return_value = []

        yield {
            "concept": concept_mock,
            "source": source_mock,
            "chunk": chunk_mock,
            "relationship": rel_mock,
            "search_hybrid": search_mock,
            "search_v2": search_v2_mock,
            "rerank": rerank_mock,
            "expand": expand_mock,
            "neighborhood": neighborhood_mock,
            "path": path_mock,
            "citing": citing_mock,
            "cited": cited_mock,
        }


@pytest.fixture
def mock_db_pool():
    """Mock the database pool for stats/health checks."""
    async_conn = AsyncMock()
    async_conn.fetchval = AsyncMock(return_value=100)

    pool = AsyncMock()
    pool.acquire.return_value.__aenter__.return_value = async_conn
    pool.acquire.return_value.__aexit__.return_value = None

    return pool


@pytest.fixture
def mock_get_pool(mock_db_pool):
    """Mock get_connection_pool."""
    with patch("research_kb_storage.get_connection_pool", AsyncMock(return_value=mock_db_pool)):
        yield mock_db_pool
