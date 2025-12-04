"""Shared test fixtures for research-kb repository.

Provides:
- Database fixtures with clean test database
- Mock Ollama client for concept extraction
- Test PDF fixture paths
- Embedding server mocks
"""

from pathlib import Path


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-neo4j",
        action="store_true",
        default=False,
        help="Run tests that require Neo4j server",
    )
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from research_kb_storage import (
    DatabaseConfig,
    get_connection_pool,
    close_connection_pool,
)
from research_kb_contracts import (
    Concept,
    ConceptType,
    ConceptRelationship,
    RelationshipType,
)
from research_kb_extraction import ChunkExtraction


@pytest_asyncio.fixture(scope="function")
async def test_db() -> AsyncGenerator:
    """Provide clean test database for each test.

    This fixture:
    - Creates a fresh connection pool
    - Cleans the database before the test
    - Yields the pool to the test
    - Cleans up after the test completes

    Usage:
        async def test_my_feature(test_db):
            # test_db is the connection pool
            result = await MyStore.create(...)
    """
    # Reset global pool state
    await close_connection_pool()

    # Create new pool with test database
    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="research_kb",
        user="postgres",
        password="postgres",
    )
    pool = await get_connection_pool(config)

    # Clean database before test
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE sources CASCADE")

    yield pool

    # Clean up after test
    await close_connection_pool()


@pytest.fixture
def mock_ollama():
    """Mock Ollama client for concept extraction.

    Returns a mock that simulates concept extraction without
    requiring a running Ollama server.

    Usage:
        def test_extraction(mock_ollama):
            extractor = ConceptExtractor(client=mock_ollama)
            result = await extractor.extract_from_text(...)
    """
    client = AsyncMock()

    # Mock extract_concepts to return sample ChunkExtraction
    client.extract_concepts = AsyncMock(
        return_value=ChunkExtraction(
            concepts=[
                Concept(
                    name="Instrumental Variables",
                    canonical_name="instrumental variables",
                    concept_type=ConceptType.METHOD,
                    aliases=["IV", "IVs"],
                    confidence_score=0.92,
                    category="identification",
                )
            ],
            relationships=[
                ConceptRelationship(
                    source_canonical_name="instrumental variables",
                    target_canonical_name="endogeneity",
                    relationship_type=RelationshipType.ADDRESSES,
                    confidence_score=0.88,
                )
            ],
        )
    )

    return client


@pytest.fixture
def mock_embedding_client():
    """Mock embedding client for tests that don't need real embeddings.

    Returns a mock that generates random embedding vectors.

    Usage:
        def test_search(mock_embedding_client):
            query = SearchQuery(embedding=mock_embedding_client.embed("test"))
    """
    client = MagicMock()

    # Mock embed to return 1024-dim zero vector
    client.embed = MagicMock(return_value=[0.0] * 1024)

    # Mock batch_embed
    client.batch_embed = MagicMock(
        side_effect=lambda texts: [[0.0] * 1024 for _ in texts]
    )

    return client


@pytest.fixture
def test_pdf_path() -> Path:
    """Path to test PDF fixture.

    Returns path to a real test PDF in fixtures/papers/.

    Usage:
        async def test_ingestion(test_pdf_path):
            result = await dispatcher.ingest_pdf(test_pdf_path)
    """
    return Path(__file__).parent / "fixtures" / "papers" / "ai_iv_search_2024.pdf"


@pytest.fixture
def simple_paper_path() -> Path:
    """Path to simple test paper (single-page, fast to process).

    Returns path to a simple PDF for smoke tests.

    Usage:
        async def test_quick_ingestion(simple_paper_path):
            result = await dispatcher.ingest_pdf(simple_paper_path)
    """
    # Note: This fixture assumes a simple PDF exists
    # If not, tests using it will be skipped
    path = Path(__file__).parent / "fixtures" / "papers" / "simple_test.pdf"
    if not path.exists():
        pytest.skip(f"Simple test PDF not found: {path}")
    return path


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to fixtures directory.

    Usage:
        def test_load_fixture(fixtures_dir):
            yaml_path = fixtures_dir / "concepts" / "seed_concepts.yaml"
    """
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def seed_concepts_path(fixtures_dir) -> Path:
    """Return path to seed concepts YAML file.

    Usage:
        def test_validation(seed_concepts_path):
            with open(seed_concepts_path) as f:
                seed = yaml.safe_load(f)
    """
    return fixtures_dir / "concepts" / "seed_concepts.yaml"
