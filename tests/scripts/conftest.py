"""Fixtures for script testing."""

import pytest
import pytest_asyncio
from pathlib import Path
import sys

# Add scripts to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "scripts"))

# Add packages to path
sys.path.insert(0, str(repo_root / "packages" / "storage" / "src"))
sys.path.insert(0, str(repo_root / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(repo_root / "packages" / "contracts" / "src"))
sys.path.insert(0, str(repo_root / "packages" / "common" / "src"))


@pytest.fixture
def scripts_dir():
    """Path to scripts directory."""
    return Path(__file__).parent.parent.parent / "scripts"


@pytest.fixture
def fixtures_dir():
    """Path to fixtures directory."""
    return Path(__file__).parent.parent.parent / "fixtures"


@pytest.fixture
def embedding_available():
    """Check if embedding server is available."""
    try:
        from research_kb_pdf import EmbeddingClient

        client = EmbeddingClient()
        # Try to embed a test string
        result = client.embed("test")
        return len(result) == 1024
    except Exception:
        return False


@pytest.fixture
def ollama_available():
    """Check if Ollama server is available."""
    import subprocess

    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest_asyncio.fixture
async def corpus_ingested(test_db, embedding_available):
    """Fixture ensuring corpus is ingested (for dependent tests)."""
    if not embedding_available:
        pytest.skip("Embedding server required for corpus ingestion")

    from research_kb_storage import SourceStore

    # Check if corpus already ingested
    sources = await SourceStore.list_all(limit=1)

    if len(sources) == 0:
        # Corpus not ingested - skip test
        pytest.skip("Corpus must be ingested (run scripts/ingest_corpus.py)")

    return True


@pytest_asyncio.fixture
async def concepts_extracted(test_db):
    """Fixture ensuring concepts are extracted (for validation tests)."""
    from research_kb_storage import ConceptStore

    # Check if concepts already extracted
    concepts = await ConceptStore.list_all(limit=1)

    if len(concepts) == 0:
        pytest.skip("Concepts must be extracted (run scripts/extract_concepts.py)")

    return True
