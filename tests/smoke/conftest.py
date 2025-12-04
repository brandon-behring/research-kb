"""Fixtures for smoke tests."""

import pytest
import pytest_asyncio
from pathlib import Path
import sys

# Add packages to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "packages" / "storage" / "src"))
sys.path.insert(0, str(repo_root / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(repo_root / "packages" / "contracts" / "src"))
sys.path.insert(0, str(repo_root / "packages" / "common" / "src"))


@pytest.fixture
def fixtures_dir():
    """Path to fixtures directory."""
    return Path(__file__).parent.parent.parent / "fixtures"


@pytest.fixture
def simple_paper_path(fixtures_dir):
    """Path to a simple test paper (smallest fixture)."""
    # Try to find the smallest paper in fixtures
    papers_dir = fixtures_dir / "papers"
    if not papers_dir.exists():
        pytest.skip("No papers directory in fixtures")

    papers = list(papers_dir.glob("*.pdf"))
    if not papers:
        pytest.skip("No PDF files in fixtures/papers/")

    # Return the first paper (for smoke testing)
    return papers[0]


@pytest.fixture
def textbook_path(fixtures_dir):
    """Path to a textbook fixture."""
    textbooks_dir = fixtures_dir / "textbooks"
    if not textbooks_dir.exists():
        pytest.skip("No textbooks directory in fixtures")

    textbooks = list(textbooks_dir.glob("*.pdf"))
    if not textbooks:
        pytest.skip("No PDF files in fixtures/textbooks/")

    return textbooks[0]


@pytest.fixture
def all_papers(fixtures_dir):
    """All paper PDFs in fixtures."""
    papers_dir = fixtures_dir / "papers"
    if not papers_dir.exists():
        return []
    return list(papers_dir.glob("*.pdf"))


@pytest_asyncio.fixture
async def ingestion_helper():
    """Helper for ingesting PDFs in tests."""
    from research_kb_storage import (
        SourceStore,
        ChunkStore,
        DatabaseConfig,
        get_connection_pool
    )
    from research_kb_pdf import (
        extract_with_headings,
        chunk_with_sections,
        EmbeddingClient
    )
    from research_kb_contracts import SourceType
    import hashlib

    class IngestionHelper:
        def __init__(self):
            self.embed_client = None

        async def ingest_pdf(self, pdf_path: Path, source_type: SourceType = SourceType.PAPER):
            """Ingest a PDF and return source + chunks."""
            # Initialize DB
            config = DatabaseConfig()
            await get_connection_pool(config)

            # Calculate file hash
            with open(pdf_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            # Create source
            source = await SourceStore.create(
                source_type=source_type,
                title=pdf_path.stem,
                authors=["Test Author"],
                year=2024,
                file_path=str(pdf_path),
                file_hash=file_hash,
                metadata={"test": True}
            )

            # Extract text and headings
            extraction, headings = extract_with_headings(str(pdf_path))

            # Chunk with sections
            chunks_data = chunk_with_sections(extraction, headings)

            # Create embeddings
            if self.embed_client is None:
                try:
                    self.embed_client = EmbeddingClient()
                except Exception:
                    # Embedding server not available
                    self.embed_client = None

            # Create chunks
            chunks = []
            for text_chunk in chunks_data:
                # Generate embedding if server available
                embedding = None
                if self.embed_client:
                    try:
                        embedding = self.embed_client.embed(text_chunk.content)
                    except Exception:
                        pass

                # Sanitize content (remove null bytes that PostgreSQL doesn't accept)
                clean_content = text_chunk.content.replace('\x00', '')

                # Calculate content hash
                content_hash = hashlib.sha256(clean_content.encode()).hexdigest()

                chunk = await ChunkStore.create(
                    source_id=source.id,
                    content=clean_content,
                    content_hash=content_hash,
                    page_start=text_chunk.start_page,
                    page_end=text_chunk.end_page,
                    embedding=embedding,
                    metadata=text_chunk.metadata or {}
                )
                chunks.append(chunk)

            return source, chunks

    return IngestionHelper()


def count_tokens(text: str) -> int:
    """Rough token count (word-based approximation)."""
    return len(text.split())
