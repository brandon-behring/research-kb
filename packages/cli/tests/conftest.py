"""Fixtures for CLI testing."""

import pytest
from typer.testing import CliRunner
from unittest.mock import MagicMock
from pathlib import Path
from uuid import uuid4
import sys

# Add CLI package to path
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root / "packages/cli/src"))
sys.path.insert(0, str(repo_root / "packages/contracts/src"))
sys.path.insert(0, str(repo_root / "packages/storage/src"))
sys.path.insert(0, str(repo_root / "packages/pdf-tools/src"))
sys.path.insert(0, str(repo_root / "packages/common/src"))

# Import CLI app (after path setup)


@pytest.fixture
def cli_runner():
    """Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_embedding_client():
    """Mock embedding server for query command."""
    client = MagicMock()
    # Mock the embed method to return a 1024-dim vector
    client.embed.return_value = [0.1] * 1024
    return client


@pytest.fixture
def mock_search_results():
    """Generate fake search results for testing formatters."""
    from research_kb_contracts import Chunk, Source, SearchResult, SourceType
    from datetime import datetime

    chunk_id = uuid4()
    source_id = uuid4()

    chunk = Chunk(
        id=chunk_id,
        source_id=source_id,
        content="Test content about instrumental variables and endogeneity...",
        content_hash="test_hash_123",
        page_start=1,
        page_end=1,
        metadata={},
        created_at=datetime.now(),
    )

    source = Source(
        id=source_id,
        title="Test Paper on Causal Inference",
        authors=["Test Author", "Another Author"],
        year=2024,
        source_type=SourceType.PAPER,
        file_hash="test_file_hash",
        metadata={},
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    return [
        SearchResult(
            chunk=chunk,
            source=source,
            fts_score=0.8,
            vector_score=0.7,
            graph_score=None,
            combined_score=0.75,
            rank=1,
        )
    ]


@pytest.fixture
def mock_concepts():
    """Generate fake concepts for graph/path commands."""
    from research_kb_contracts import Concept, ConceptType
    from datetime import datetime

    return [
        Concept(
            id=uuid4(),
            name="Instrumental Variables",
            canonical_name="instrumental_variables",
            aliases=["IV", "instrument"],
            concept_type=ConceptType.METHOD,
            definition="A method for causal inference when there is unobserved confounding...",
            category="causal_inference",
            confidence_score=0.95,
            embedding=[0.1] * 1024,
            validated=True,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
        Concept(
            id=uuid4(),
            name="Exogeneity",
            canonical_name="exogeneity",
            aliases=["exogenous"],
            concept_type=ConceptType.ASSUMPTION,
            definition="The assumption that an instrumental variable is uncorrelated with the error term...",
            category="assumptions",
            confidence_score=0.90,
            embedding=[0.2] * 1024,
            validated=True,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    ]


@pytest.fixture
def mock_relationships():
    """Generate fake relationships for testing."""
    from research_kb_contracts import ConceptRelationship, RelationshipType
    from datetime import datetime

    concept1_id = uuid4()
    concept2_id = uuid4()

    return [
        ConceptRelationship(
            id=uuid4(),
            source_concept_id=concept1_id,
            target_concept_id=concept2_id,
            relationship_type=RelationshipType.REQUIRES,
            confidence_score=0.85,
            created_at=datetime.now(),
        )
    ]
