"""Pytest fixtures for extraction package tests."""

import pytest
from unittest.mock import AsyncMock
from uuid import uuid4

# NOTE: --run-neo4j option is defined in root conftest.py
# Do not duplicate here to avoid pytest registration conflicts

from research_kb_extraction.models import (
    ChunkExtraction,
    ExtractedConcept,
    ExtractedRelationship,
)


@pytest.fixture
def sample_extraction():
    """Sample extraction result for testing."""
    return ChunkExtraction(
        concepts=[
            ExtractedConcept(
                name="instrumental variables",
                concept_type="method",
                definition="A method for estimating causal effects using instruments",
                aliases=["IV", "2SLS"],
                confidence=0.9,
            ),
            ExtractedConcept(
                name="relevance assumption",
                concept_type="assumption",
                definition="The instrument must be correlated with the endogenous variable",
                aliases=["relevance"],
                confidence=0.85,
            ),
            ExtractedConcept(
                name="exclusion restriction",
                concept_type="assumption",
                definition="The instrument affects outcome only through treatment",
                aliases=[],
                confidence=0.88,
            ),
        ],
        relationships=[
            ExtractedRelationship(
                source_concept="instrumental variables",
                target_concept="relevance assumption",
                relationship_type="REQUIRES",
                evidence="IV requires the relevance assumption",
                confidence=0.85,
            ),
            ExtractedRelationship(
                source_concept="instrumental variables",
                target_concept="exclusion restriction",
                relationship_type="REQUIRES",
                evidence="IV requires the exclusion restriction",
                confidence=0.82,
            ),
        ],
    )


@pytest.fixture
def sample_chunk_text():
    """Sample academic text for extraction testing."""
    return """
    Instrumental variables (IV) estimation is a widely used approach for addressing
    endogeneity in econometric analysis. The IV method relies on two key assumptions:
    the relevance condition, which requires that the instrument be correlated with
    the endogenous regressor, and the exclusion restriction, which stipulates that
    the instrument affects the outcome only through its effect on the treatment.

    Two-stage least squares (2SLS) is the most common IV estimator. In the first stage,
    the endogenous variable is regressed on the instruments and exogenous covariates.
    In the second stage, the outcome is regressed on the predicted values from the
    first stage along with the exogenous covariates.
    """


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing without actual LLM calls."""
    client = AsyncMock()
    client.is_available = AsyncMock(return_value=True)
    client.is_model_loaded = AsyncMock(return_value=True)
    return client


@pytest.fixture
def concept_ids():
    """Generate fixed UUIDs for testing."""
    return {
        "iv": uuid4(),
        "relevance": uuid4(),
        "exclusion": uuid4(),
        "endogeneity": uuid4(),
    }
