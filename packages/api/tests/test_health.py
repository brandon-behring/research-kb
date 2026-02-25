"""Tests for health endpoints."""

from __future__ import annotations

import pytest
from unittest.mock import patch

pytestmark = pytest.mark.unit


async def test_health_check(app_client):
    """Basic health check returns healthy status."""
    response = await app_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["database"] == "connected"
    assert data["embedding_model"] == "ready"


async def test_health_detailed(app_client, mock_storage):
    """Detailed health check includes component status and stats."""
    with patch("research_kb_api.service.get_stats") as stats_mock:
        stats_mock.return_value = {
            "sources": 100,
            "chunks": 5000,
            "concepts": 200,
            "relationships": 500,
            "citations": 300,
            "chunk_concepts": 1000,
        }

        response = await app_client.get("/health/detailed")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "database" in data["components"]
        assert data["stats"]["sources"] == 100
        assert data["stats"]["chunks"] == 5000


async def test_stats_endpoint(app_client):
    """Stats endpoint returns database statistics."""
    with patch("research_kb_api.service.get_stats") as stats_mock:
        stats_mock.return_value = {
            "sources": 42,
            "chunks": 1234,
            "concepts": 100,
            "relationships": 200,
            "citations": 50,
            "chunk_concepts": 500,
        }

        response = await app_client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["sources"] == 42
        assert data["chunks"] == 1234
