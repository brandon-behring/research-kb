"""Tests for CitationStore against live PostgreSQL.

Phase 1.5.2: Tests for citation storage layer.
"""

import pytest
from uuid import uuid4

from research_kb_contracts import SourceType
from research_kb_storage import CitationStore, SourceStore


@pytest.fixture
async def test_source(db_pool):
    """Create a test source for citation tests."""
    source = await SourceStore.create(
        source_type=SourceType.PAPER,
        title="Test Paper for Citations",
        authors=["Test Author"],
        year=2024,
        file_hash=f"sha256:citation_test_{uuid4().hex[:8]}",
    )
    return source


class TestCitationStoreCreate:
    """Test CitationStore.create() operations."""

    async def test_create_minimal_citation(self, test_source):
        """Test creating citation with minimal required fields."""
        citation = await CitationStore.create(
            source_id=test_source.id,
            raw_string="Pearl, J. (2009). Causality.",
        )

        assert citation.id is not None
        assert citation.source_id == test_source.id
        assert citation.raw_string == "Pearl, J. (2009). Causality."
        assert citation.authors == []
        assert citation.title is None
        assert citation.year is None
        assert citation.metadata == {}

    async def test_create_full_citation(self, test_source):
        """Test creating citation with all fields."""
        citation = await CitationStore.create(
            source_id=test_source.id,
            raw_string="Pearl, J. (2009). Causality: Models, Reasoning, and Inference.",
            authors=["Judea Pearl"],
            title="Causality: Models, Reasoning, and Inference",
            year=2009,
            venue="Cambridge University Press",
            doi="10.1017/CBO9780511803161",
            arxiv_id=None,
            bibtex="@book{pearl2009causality, author={Pearl, Judea}, ...}",
            extraction_method="grobid",
            confidence_score=0.95,
            metadata={"edition": "2nd"},
        )

        assert citation.title == "Causality: Models, Reasoning, and Inference"
        assert citation.authors == ["Judea Pearl"]
        assert citation.year == 2009
        assert citation.venue == "Cambridge University Press"
        assert citation.doi == "10.1017/CBO9780511803161"
        assert citation.bibtex is not None
        assert citation.extraction_method == "grobid"
        assert abs(citation.confidence_score - 0.95) < 0.001  # REAL precision
        assert citation.metadata["edition"] == "2nd"

    async def test_create_citation_with_arxiv(self, test_source):
        """Test creating citation with arXiv ID."""
        citation = await CitationStore.create(
            source_id=test_source.id,
            raw_string="Vaswani et al. (2017). Attention Is All You Need.",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            title="Attention Is All You Need",
            year=2017,
            arxiv_id="1706.03762",
            extraction_method="grobid",
        )

        assert citation.arxiv_id == "1706.03762"
        assert citation.doi is None


class TestCitationStoreRetrieve:
    """Test CitationStore retrieval operations."""

    async def test_get_by_id_found(self, test_source):
        """Test retrieving citation by ID when it exists."""
        created = await CitationStore.create(
            source_id=test_source.id,
            raw_string="Test citation",
        )

        retrieved = await CitationStore.get_by_id(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.raw_string == created.raw_string

    async def test_get_by_id_not_found(self, db_pool):
        """Test retrieving citation by ID when it doesn't exist."""
        result = await CitationStore.get_by_id(uuid4())
        assert result is None

    async def test_list_by_source(self, test_source):
        """Test listing citations for a source."""
        # Create 5 citations for the source
        for i in range(5):
            await CitationStore.create(
                source_id=test_source.id,
                raw_string=f"Citation {i}",
                title=f"Paper {i}",
            )

        citations = await CitationStore.list_by_source(test_source.id)

        assert len(citations) == 5
        assert all(c.source_id == test_source.id for c in citations)

    async def test_list_by_source_pagination(self, test_source):
        """Test listing citations with pagination."""
        # Create 10 citations
        for i in range(10):
            await CitationStore.create(
                source_id=test_source.id,
                raw_string=f"Citation {i}",
            )

        # Get first 5
        page1 = await CitationStore.list_by_source(test_source.id, limit=5, offset=0)
        assert len(page1) == 5

        # Get next 5
        page2 = await CitationStore.list_by_source(test_source.id, limit=5, offset=5)
        assert len(page2) == 5

        # No overlap
        page1_ids = {c.id for c in page1}
        page2_ids = {c.id for c in page2}
        assert len(page1_ids & page2_ids) == 0


class TestCitationStoreFinders:
    """Test CitationStore finder operations."""

    async def test_find_by_doi(self, test_source):
        """Test finding citation by DOI."""
        doi = "10.1234/test.doi"
        created = await CitationStore.create(
            source_id=test_source.id,
            raw_string="Test DOI citation",
            doi=doi,
        )

        found = await CitationStore.find_by_doi(doi)

        assert found is not None
        assert found.id == created.id
        assert found.doi == doi

    async def test_find_by_doi_not_found(self, db_pool):
        """Test finding citation by DOI when it doesn't exist."""
        result = await CitationStore.find_by_doi("10.9999/nonexistent")
        assert result is None

    async def test_find_by_arxiv(self, test_source):
        """Test finding citation by arXiv ID."""
        arxiv_id = "2301.12345"
        created = await CitationStore.create(
            source_id=test_source.id,
            raw_string="Test arXiv citation",
            arxiv_id=arxiv_id,
        )

        found = await CitationStore.find_by_arxiv(arxiv_id)

        assert found is not None
        assert found.id == created.id
        assert found.arxiv_id == arxiv_id

    async def test_find_by_arxiv_not_found(self, db_pool):
        """Test finding citation by arXiv ID when it doesn't exist."""
        result = await CitationStore.find_by_arxiv("9999.99999")
        assert result is None


class TestCitationStoreBatch:
    """Test CitationStore batch operations."""

    async def test_batch_create(self, test_source):
        """Test batch creating multiple citations."""
        citations_data = [
            {
                "source_id": test_source.id,
                "raw_string": f"Batch citation {i}",
                "title": f"Paper {i}",
                "year": 2020 + i,
                "extraction_method": "grobid",
                "metadata": {"index": i},
            }
            for i in range(5)
        ]

        created = await CitationStore.batch_create(citations_data)

        assert len(created) == 5
        assert all(c.source_id == test_source.id for c in created)
        assert created[0].metadata["index"] == 0
        assert created[4].metadata["index"] == 4

    async def test_batch_create_empty_list(self, db_pool):
        """Test batch create with empty list returns empty list."""
        result = await CitationStore.batch_create([])
        assert result == []

    async def test_batch_create_with_bibtex(self, test_source):
        """Test batch creating citations with BibTeX entries."""
        citations_data = [
            {
                "source_id": test_source.id,
                "raw_string": "Pearl, J. (2009). Causality.",
                "title": "Causality",
                "authors": ["Judea Pearl"],
                "year": 2009,
                "bibtex": "@book{pearl2009causality,\n  author = {Pearl, Judea},\n  title = {Causality},\n  year = {2009},\n}",
                "extraction_method": "grobid",
            },
            {
                "source_id": test_source.id,
                "raw_string": "Vaswani et al. (2017). Attention.",
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani"],
                "year": 2017,
                "arxiv_id": "1706.03762",
                "bibtex": "@article{vaswani2017attention,\n  author = {Vaswani, Ashish},\n  title = {Attention Is All You Need},\n  year = {2017},\n}",
                "extraction_method": "grobid",
            },
        ]

        created = await CitationStore.batch_create(citations_data)

        assert len(created) == 2
        assert created[0].bibtex is not None
        assert "pearl2009causality" in created[0].bibtex
        assert created[1].arxiv_id == "1706.03762"


class TestCitationStoreDelete:
    """Test CitationStore delete operations."""

    async def test_delete_existing_citation(self, test_source):
        """Test deleting existing citation returns True."""
        citation = await CitationStore.create(
            source_id=test_source.id,
            raw_string="To be deleted",
        )

        deleted = await CitationStore.delete(citation.id)
        assert deleted is True

        # Verify gone
        result = await CitationStore.get_by_id(citation.id)
        assert result is None

    async def test_delete_nonexistent_citation(self, db_pool):
        """Test deleting nonexistent citation returns False."""
        deleted = await CitationStore.delete(uuid4())
        assert deleted is False


class TestCitationStoreCount:
    """Test CitationStore count operations."""

    async def test_count_by_source(self, test_source):
        """Test counting citations for a source."""
        # Initially 0
        count = await CitationStore.count_by_source(test_source.id)
        assert count == 0

        # Create 7 citations
        for i in range(7):
            await CitationStore.create(
                source_id=test_source.id,
                raw_string=f"Citation {i}",
            )

        count = await CitationStore.count_by_source(test_source.id)
        assert count == 7


class TestCitationStoreCascadeDelete:
    """Test CASCADE delete from sources to citations."""

    async def test_deleting_source_deletes_citations(self, test_source):
        """Test deleting source also deletes its citations (CASCADE)."""
        # Create 3 citations for source
        citation_ids = []
        for i in range(3):
            citation = await CitationStore.create(
                source_id=test_source.id,
                raw_string=f"Citation {i}",
            )
            citation_ids.append(citation.id)

        # Verify citations exist
        count = await CitationStore.count_by_source(test_source.id)
        assert count == 3

        # Delete source
        await SourceStore.delete(test_source.id)

        # Verify citations are gone (CASCADE)
        for citation_id in citation_ids:
            citation = await CitationStore.get_by_id(citation_id)
            assert citation is None

        count = await CitationStore.count_by_source(test_source.id)
        assert count == 0
