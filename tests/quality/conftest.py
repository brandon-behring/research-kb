"""Fixtures for quality tests."""

import pytest
import pytest_asyncio
from pathlib import Path
import sys
import yaml
from dataclasses import dataclass
from typing import Optional

# Add packages to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "packages" / "storage" / "src"))
sys.path.insert(0, str(repo_root / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(repo_root / "packages" / "contracts" / "src"))
sys.path.insert(0, str(repo_root / "packages" / "common" / "src"))


@dataclass
class SeedConcept:
    """Seed concept for validation."""

    name: str
    canonical_name: str
    aliases: list[str]
    concept_type: str
    category: Optional[str] = None
    definition: Optional[str] = None


@pytest.fixture
def seed_concepts_file():
    """Path to seed concepts YAML file."""
    repo_root = Path(__file__).parent.parent.parent
    fixtures = repo_root / "fixtures" / "concepts"

    # Try multiple possible filenames
    candidates = [
        fixtures / "seed_concepts_v2.0.yaml",
        fixtures / "seed_concepts.yaml",
        fixtures / "seed_concepts_v1.yaml",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    pytest.skip("No seed concepts file found in fixtures/concepts/")


@pytest.fixture
def seed_concepts(seed_concepts_file):
    """Load seed concepts from YAML."""
    with open(seed_concepts_file, "r") as f:
        data = yaml.safe_load(f)

    concepts = []
    for item in data.get("concepts", []):
        concepts.append(
            SeedConcept(
                name=item["name"],
                canonical_name=item["canonical_name"],
                aliases=item.get("aliases", []),
                concept_type=item["concept_type"],
                category=item.get("category"),
                definition=item.get("definition"),
            )
        )

    return concepts


@pytest_asyncio.fixture
async def extracted_concepts():
    """Get all extracted concepts from database."""
    from research_kb_storage import ConceptStore, get_connection_pool, DatabaseConfig

    try:
        config = DatabaseConfig()
        await get_connection_pool(config)
        concepts = await ConceptStore.list_all(limit=10000)
    except Exception as e:
        pytest.skip(f"Database not available: {e}")

    if len(concepts) == 0:
        pytest.skip("No concepts extracted (run scripts/extract_concepts.py first)")

    return concepts


@pytest_asyncio.fixture
async def corpus_chunks():
    """Get all chunks from ingested corpus."""
    from research_kb_storage import ChunkStore, get_connection_pool, DatabaseConfig

    try:
        config = DatabaseConfig()
        await get_connection_pool(config)
        chunks = await ChunkStore.list_all(limit=10000)
    except Exception as e:
        pytest.skip(f"Database not available: {e}")

    if len(chunks) == 0:
        pytest.skip("No chunks ingested (run scripts/ingest_corpus.py first)")

    return chunks
