"""Pytest fixtures for daemon tests."""

import pytest


@pytest.fixture
def mock_embed_response():
    """Mock embedding response."""
    return {
        "embedding": [0.1] * 1024,
        "dim": 1024,
    }


@pytest.fixture
def mock_health_response():
    """Mock embed server health response."""
    return {
        "status": "ok",
        "device": "cuda",
        "model": "BAAI/bge-large-en-v1.5",
        "dim": 1024,
    }
