"""Tests for graph endpoints."""

from __future__ import annotations

import pytest
from unittest.mock import patch
from uuid import uuid4

pytestmark = pytest.mark.unit


async def test_get_neighborhood(app_client, mock_storage):
    """Get neighborhood returns graph structure."""
    concept_id = str(uuid4())

    with patch("research_kb_api.service.get_graph_neighborhood") as neighborhood_mock:
        neighborhood_mock.return_value = {
            "center": {
                "id": concept_id,
                "name": "instrumental variables",
                "type": "method",
            },
            "nodes": [
                {
                    "id": str(uuid4()),
                    "name": "two-stage least squares",
                    "type": "method",
                },
                {"id": str(uuid4()), "name": "exogeneity", "type": "assumption"},
            ],
            "edges": [
                {"source": concept_id, "target": str(uuid4()), "type": "REQUIRES"},
            ],
        }

        response = await app_client.get("/graph/neighborhood/instrumental%20variables")

        assert response.status_code == 200
        data = response.json()
        assert data["center"]["name"] == "instrumental variables"
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1


async def test_get_neighborhood_not_found(app_client, mock_storage):
    """Get neighborhood returns 404 for unknown concept."""
    with patch("research_kb_api.service.get_graph_neighborhood") as neighborhood_mock:
        neighborhood_mock.return_value = {"error": "Concept 'unknown' not found"}

        response = await app_client.get("/graph/neighborhood/unknown")

        assert response.status_code == 404


async def test_get_neighborhood_with_hops(app_client, mock_storage):
    """Get neighborhood respects hops parameter."""
    with patch("research_kb_api.service.get_graph_neighborhood") as neighborhood_mock:
        neighborhood_mock.return_value = {
            "center": {"id": str(uuid4()), "name": "iv", "type": "method"},
            "nodes": [],
            "edges": [],
        }

        response = await app_client.get("/graph/neighborhood/iv?hops=3")

        assert response.status_code == 200
        neighborhood_mock.assert_called_once()
        call_kwargs = neighborhood_mock.call_args[1]
        assert call_kwargs["hops"] == 3


async def test_get_path(app_client, mock_storage):
    """Get path returns shortest path between concepts."""
    with patch("research_kb_api.service.get_graph_path") as path_mock:
        path_mock.return_value = {
            "from": "iv",
            "to": "dml",
            "path": [
                {
                    "id": str(uuid4()),
                    "name": "instrumental variables",
                    "type": "method",
                },
                {"id": str(uuid4()), "name": "causal effect", "type": "definition"},
                {
                    "id": str(uuid4()),
                    "name": "double machine learning",
                    "type": "method",
                },
            ],
        }

        response = await app_client.get("/graph/path/iv/dml")

        assert response.status_code == 200
        data = response.json()
        assert data["from_concept"] == "iv"
        assert data["to_concept"] == "dml"
        assert len(data["path"]) == 3
        assert data["path_length"] == 2


async def test_get_path_not_found(app_client, mock_storage):
    """Get path returns 404 when no path exists."""
    with patch("research_kb_api.service.get_graph_path") as path_mock:
        path_mock.return_value = {"error": "Concept 'unknown' not found"}

        response = await app_client.get("/graph/path/unknown/other")

        assert response.status_code == 404
