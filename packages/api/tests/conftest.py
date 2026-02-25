"""Test configuration for API tests."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncGenerator

from httpx import AsyncClient, ASGITransport

from research_kb_api.main import create_app


@pytest.fixture
def mock_embedding_client():
    """Mock embedding client for tests."""
    client = MagicMock()
    client.embed.return_value = [0.1] * 1024
    client.embed_query.return_value = [0.1] * 1024
    return client


@pytest.fixture
def mock_pool():
    """Mock database connection pool."""
    pool = AsyncMock()
    pool.get_size.return_value = 5
    pool.close = AsyncMock()
    return pool


@pytest.fixture
def mock_storage():
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
    ):

        # Default return values
        concept_mock.count = AsyncMock(return_value=100)
        concept_mock.search = AsyncMock(return_value=[])
        concept_mock.list_all = AsyncMock(return_value=[])
        concept_mock.get = AsyncMock(return_value=None)

        source_mock.list_all = AsyncMock(return_value=[])
        source_mock.count = AsyncMock(return_value=0)
        source_mock.get = AsyncMock(return_value=None)
        source_mock.get_by_id = AsyncMock(return_value=None)

        chunk_mock.get_by_source = AsyncMock(return_value=[])
        chunk_mock.list_by_source = AsyncMock(return_value=[])

        rel_mock.get_for_concept = AsyncMock(return_value=[])
        rel_mock.list_all_for_concept = AsyncMock(return_value=[])

        search_mock.return_value = []
        search_v2_mock.return_value = []
        rerank_mock.return_value = []
        expand_mock.return_value = ([], None)

        yield {
            "concept": concept_mock,
            "source": source_mock,
            "chunk": chunk_mock,
            "relationship": rel_mock,
            "search_hybrid": search_mock,
            "search_v2": search_v2_mock,
            "rerank": rerank_mock,
            "expand": expand_mock,
        }


@pytest.fixture
async def app_client(
    mock_pool, mock_storage, mock_embedding_client
) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with mocked dependencies."""
    # Create async mock for get_cached_embedding
    async_embedding_mock = AsyncMock(return_value=[0.1] * 1024)

    with (
        patch("research_kb_api.main.get_connection_pool", return_value=mock_pool),
        patch(
            "research_kb_api.service.get_embedding_client",
            return_value=mock_embedding_client,
        ),
        patch("research_kb_api.service.get_cached_embedding", async_embedding_mock),
    ):
        app = create_app()
        app.state.pool = mock_pool

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
