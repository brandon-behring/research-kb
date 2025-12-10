"""Tests for search endpoints."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4
from datetime import datetime

from research_kb_contracts import SearchResult, Source, Chunk, SourceType


def make_search_result(title: str = "Test Paper", content: str = "Test content") -> SearchResult:
    """Create a mock search result."""
    now = datetime.now()
    source = Source(
        id=uuid4(),
        title=title,
        authors=["Author One"],
        year=2023,
        source_type=SourceType.PAPER,
        file_hash="abc123",
        created_at=now,
        updated_at=now,
    )
    chunk = Chunk(
        id=uuid4(),
        source_id=source.id,
        content=content,
        content_hash="chunk_hash_123",
        page_start=1,
        page_end=2,
        metadata={"section_header": "Introduction"},
        created_at=now,
    )
    result = SearchResult(
        source=source,
        chunk=chunk,
        combined_score=0.85,
        rank=1,
    )
    # Add score components
    result.fts_score = 0.3
    result.vector_score = 0.7
    result.graph_score = 0.1
    return result


@pytest.mark.asyncio
async def test_search_basic(app_client, mock_storage, mock_embedding_client):
    """Basic search returns results."""
    mock_result = make_search_result()
    mock_storage["expand"].return_value = ([mock_result], None)

    response = await app_client.post(
        "/search",
        json={"query": "instrumental variables"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "instrumental variables"
    assert len(data["results"]) == 1
    assert data["results"][0]["source"]["title"] == "Test Paper"
    assert "scores" in data["results"][0]


@pytest.mark.asyncio
async def test_search_with_options(app_client, mock_storage, mock_embedding_client):
    """Search respects all request options."""
    mock_result = make_search_result()
    mock_storage["expand"].return_value = ([mock_result], None)

    response = await app_client.post(
        "/search",
        json={
            "query": "double machine learning",
            "limit": 5,
            "context_type": "auditing",
            "use_graph": True,
            "graph_weight": 0.3,
            "use_rerank": True,
            "use_expand": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["result_count"] == 1


@pytest.mark.asyncio
async def test_search_empty_results(app_client, mock_storage, mock_embedding_client):
    """Search handles no results gracefully."""
    mock_storage["expand"].return_value = ([], None)

    response = await app_client.post(
        "/search",
        json={"query": "nonexistent topic xyz"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["results"] == []
    assert data["metadata"]["result_count"] == 0


@pytest.mark.asyncio
async def test_search_validation(app_client):
    """Search validates request parameters."""
    # Empty query
    response = await app_client.post(
        "/search",
        json={"query": ""},
    )
    assert response.status_code == 422

    # Invalid limit
    response = await app_client.post(
        "/search",
        json={"query": "test", "limit": 0},
    )
    assert response.status_code == 422

    # Limit too high
    response = await app_client.post(
        "/search",
        json={"query": "test", "limit": 200},
    )
    assert response.status_code == 422
