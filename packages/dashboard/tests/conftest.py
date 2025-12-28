"""Pytest fixtures for dashboard tests.

Provides common fixtures for testing the dashboard package.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root / "packages/dashboard/src"))


@pytest.fixture
def mock_httpx_response():
    """Create a mock httpx Response object.

    Returns a factory function that creates mock responses.
    """

    def _create_response(json_data: dict, status_code: int = 200):
        response = MagicMock()
        response.json.return_value = json_data
        response.status_code = status_code
        response.raise_for_status = MagicMock()
        return response

    return _create_response


@pytest.fixture
def mock_httpx_client(mock_httpx_response):
    """Create a mock httpx AsyncClient.

    Returns a mock client with get/post methods as AsyncMock.
    """
    client = AsyncMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def sample_stats():
    """Sample statistics response from API."""
    return {
        "sources": 294,
        "chunks": 142962,
        "concepts": 283714,
        "relationships": 725866,
        "citations": 10758,
    }


@pytest.fixture
def sample_source():
    """Sample source data from API."""
    return {
        "id": str(uuid4()),
        "title": "Double Machine Learning",
        "authors": ["Chernozhukov, V.", "Chetverikov, D."],
        "year": 2018,
        "source_type": "paper",
        "metadata": {"doi": "10.1234/example"},
    }


@pytest.fixture
def sample_search_results(sample_source):
    """Sample search results from API."""
    return {
        "query": "instrumental variables",
        "expanded_query": "instrumental variables causal inference",
        "results": [
            {
                "source": sample_source,
                "chunk": {
                    "id": str(uuid4()),
                    "content": "IV methods provide identification...",
                    "page_start": 42,
                },
                "combined_score": 0.85,
                "scores": {
                    "fts": 0.5,
                    "vector": 0.8,
                    "graph": 0.1,
                },
            }
        ],
        "execution_time_ms": 150.0,
    }


@pytest.fixture
def sample_concepts():
    """Sample concepts from API."""
    return {
        "concepts": [
            {
                "id": str(uuid4()),
                "name": "Instrumental Variables",
                "concept_type": "METHOD",
            },
            {
                "id": str(uuid4()),
                "name": "Unconfoundedness",
                "concept_type": "ASSUMPTION",
            },
        ],
        "total": 2,
    }


@pytest.fixture
def sample_graph_neighborhood():
    """Sample graph neighborhood from API."""
    return {
        "center": {
            "id": str(uuid4()),
            "name": "Double Machine Learning",
            "type": "METHOD",
        },
        "nodes": [
            {"id": str(uuid4()), "name": "Neyman Orthogonality", "type": "THEOREM"},
            {"id": str(uuid4()), "name": "Cross-Fitting", "type": "TECHNIQUE"},
        ],
        "edges": [
            {"source": "DML", "target": "Neyman Orthogonality", "type": "USES"},
            {"source": "DML", "target": "Cross-Fitting", "type": "REQUIRES"},
        ],
    }


@pytest.fixture
def sample_citation_network(sample_source):
    """Sample citation network from API."""
    return {
        "citing_sources": [
            {
                "id": str(uuid4()),
                "title": "Downstream Paper",
                "year": 2022,
            }
        ],
        "cited_sources": [
            {
                "id": str(uuid4()),
                "title": "Foundation Paper",
                "year": 2010,
            }
        ],
    }
