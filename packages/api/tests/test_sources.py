"""Tests for sources endpoints."""

from __future__ import annotations

import pytest
from unittest.mock import patch
from uuid import uuid4
from datetime import datetime

from research_kb_contracts import Source, Chunk, SourceType


def make_source(title: str = "Test Paper") -> Source:
    """Create a mock source."""
    now = datetime.now()
    return Source(
        id=uuid4(),
        title=title,
        authors=["Author One", "Author Two"],
        year=2023,
        source_type=SourceType.PAPER,
        file_path="/path/to/paper.pdf",
        file_hash="abc123",
        metadata={"abstract": "This paper discusses important topics."},
        created_at=now,
        updated_at=now,
    )


def make_chunk(source_id, content: str = "Chunk content") -> Chunk:
    """Create a mock chunk."""
    now = datetime.now()
    return Chunk(
        id=uuid4(),
        source_id=source_id,
        content=content,
        content_hash="chunk_hash_123",
        page_start=1,
        page_end=2,
        created_at=now,
    )


@pytest.mark.asyncio
async def test_list_sources(app_client, mock_storage):
    """List sources returns paginated results."""
    sources = [make_source(f"Paper {i}") for i in range(3)]
    mock_storage["source"].list_all.return_value = sources

    response = await app_client.get("/sources")

    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) == 3
    assert data["sources"][0]["title"] == "Paper 0"
    assert "limit" in data
    assert "offset" in data


@pytest.mark.asyncio
async def test_list_sources_with_filter(app_client, mock_storage):
    """List sources respects source_type filter."""
    sources = [make_source()]
    mock_storage["source"].list_all.return_value = sources

    response = await app_client.get("/sources?source_type=PAPER&limit=10")

    assert response.status_code == 200
    mock_storage["source"].list_all.assert_called()


@pytest.mark.asyncio
async def test_get_source(app_client, mock_storage):
    """Get source by ID returns source with chunks."""
    source = make_source()
    chunks = [make_chunk(source.id, f"Chunk {i}") for i in range(2)]

    mock_storage["source"].get.return_value = source
    mock_storage["chunk"].get_by_source.return_value = chunks

    response = await app_client.get(f"/sources/{source.id}")

    assert response.status_code == 200
    data = response.json()
    assert data["source"]["title"] == "Test Paper"
    assert len(data["chunks"]) == 2
    assert data["chunk_count"] == 2


@pytest.mark.asyncio
async def test_get_source_not_found(app_client, mock_storage):
    """Get source returns 404 for unknown ID."""
    mock_storage["source"].get.return_value = None

    response = await app_client.get(f"/sources/{uuid4()}")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_source_citations(app_client, mock_storage):
    """Get source citations returns citation graph."""
    source = make_source()
    mock_storage["source"].get.return_value = source

    with patch("research_kb_api.service.get_citations_for_source") as cite_mock:
        cite_mock.return_value = {
            "source_id": str(source.id),
            "citing_sources": [
                {"id": str(uuid4()), "title": "Citing Paper", "year": 2024}
            ],
            "cited_sources": [
                {"id": str(uuid4()), "title": "Cited Paper", "year": 2020}
            ],
        }

        response = await app_client.get(f"/sources/{source.id}/citations")

        assert response.status_code == 200
        data = response.json()
        assert len(data["citing_sources"]) == 1
        assert len(data["cited_sources"]) == 1
        assert data["citation_count"] == 1
        assert data["reference_count"] == 1
