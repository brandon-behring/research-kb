"""Tests for SourceStore against live PostgreSQL.

These are integration tests that require the PostgreSQL container to be running:
    docker compose -f ~/Claude/research-kb/docker-compose.yml up -d postgres
"""

import pytest
from uuid import uuid4

from research_kb_common import StorageError
from research_kb_contracts import SourceType
from research_kb_storage import SourceStore

pytestmark = pytest.mark.integration


class TestSourceStoreCreate:
    """Test SourceStore.create() operations."""

    async def test_create_minimal_source(self, db_pool):
        """Test creating source with minimal required fields."""
        source = await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="Test Book",
            file_hash="sha256:test123",
        )

        assert source.id is not None
        assert source.source_type == SourceType.TEXTBOOK
        assert source.title == "Test Book"
        assert source.file_hash == "sha256:test123"
        assert source.authors == []
        assert source.metadata == {}

    async def test_create_source_with_full_metadata(self, db_pool):
        """Test creating source with all fields."""
        source = await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.PAPER,
            title="Double/debiased machine learning",
            file_hash="sha256:dml2018",
            authors=["Victor Chernozhukov", "Denis Chetverikov"],
            year=2018,
            file_path="/test/chernozhukov_2018.pdf",
            metadata={
                "doi": "10.1111/ectj.12097",
                "journal": "Econometrics Journal",
                "authority_tier": "canonical",
            },
        )

        assert source.source_type == SourceType.PAPER
        assert source.authors == ["Victor Chernozhukov", "Denis Chetverikov"]
        assert source.year == 2018
        assert source.metadata["doi"] == "10.1111/ectj.12097"
        assert source.metadata["authority_tier"] == "canonical"

    async def test_create_duplicate_file_hash_fails(self, db_pool):
        """Test creating source with duplicate file_hash raises error."""
        file_hash = "sha256:duplicate123"

        # Create first source
        await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="First Book",
            file_hash=file_hash,
        )

        # Attempt to create duplicate should fail
        with pytest.raises(StorageError) as exc_info:
            await SourceStore.create(
                domain_id="causal_inference",
                source_type=SourceType.TEXTBOOK,
                title="Second Book",
                file_hash=file_hash,  # Same hash
            )

        assert "already exists" in str(exc_info.value)


class TestSourceStoreRetrieve:
    """Test SourceStore retrieval operations."""

    async def test_get_by_id_found(self, db_pool):
        """Test retrieving source by ID when it exists."""
        created = await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="Test Book",
            file_hash="sha256:getbyid123",
        )

        retrieved = await SourceStore.get_by_id(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.title == created.title
        assert retrieved.file_hash == created.file_hash

    async def test_get_by_id_not_found(self, db_pool):
        """Test retrieving source by ID when it doesn't exist."""
        result = await SourceStore.get_by_id(uuid4())

        assert result is None

    async def test_get_by_file_hash_found(self, db_pool):
        """Test retrieving source by file hash when it exists."""
        file_hash = "sha256:getbyhash456"

        created = await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.PAPER,
            title="Test Paper",
            file_hash=file_hash,
        )

        retrieved = await SourceStore.get_by_file_hash(file_hash)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.file_hash == file_hash

    async def test_get_by_file_hash_not_found(self, db_pool):
        """Test retrieving source by file hash when it doesn't exist."""
        result = await SourceStore.get_by_file_hash("sha256:nonexistent")

        assert result is None


class TestSourceStoreUpdate:
    """Test SourceStore update operations."""

    async def test_update_metadata_merge(self, db_pool):
        """Test updating source metadata (JSONB merge)."""
        source = await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.PAPER,
            title="Test Paper",
            file_hash="sha256:update123",
            metadata={"doi": "10.1234/test", "citations_count": 100},
        )

        # Update with new metadata
        updated = await SourceStore.update_metadata(
            source_id=source.id,
            metadata={"citations_count": 150, "authority_tier": "canonical"},
        )

        # Old keys preserved, new keys added, overlapping keys updated
        assert updated.metadata["doi"] == "10.1234/test"  # Preserved
        assert updated.metadata["citations_count"] == 150  # Updated
        assert updated.metadata["authority_tier"] == "canonical"  # Added

    async def test_update_metadata_nonexistent_source(self, db_pool):
        """Test updating metadata for nonexistent source raises error."""
        with pytest.raises(StorageError) as exc_info:
            await SourceStore.update_metadata(
                source_id=uuid4(),
                metadata={"test": "value"},
            )

        assert "not found" in str(exc_info.value).lower()


class TestSourceStoreDelete:
    """Test SourceStore delete operations."""

    async def test_delete_existing_source(self, db_pool):
        """Test deleting existing source returns True."""
        source = await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="To Be Deleted",
            file_hash="sha256:delete123",
        )

        deleted = await SourceStore.delete(source.id)

        assert deleted is True

        # Verify source is gone
        result = await SourceStore.get_by_id(source.id)
        assert result is None

    async def test_delete_nonexistent_source(self, db_pool):
        """Test deleting nonexistent source returns False."""
        deleted = await SourceStore.delete(uuid4())

        assert deleted is False


class TestSourceStoreList:
    """Test SourceStore listing operations."""

    async def test_list_by_type(self, db_pool):
        """Test listing sources by type."""
        # Create 3 textbooks and 2 papers
        for i in range(3):
            await SourceStore.create(
                domain_id="causal_inference",
                source_type=SourceType.TEXTBOOK,
                title=f"Textbook {i}",
                file_hash=f"sha256:textbook{i}",
            )

        for i in range(2):
            await SourceStore.create(
                domain_id="causal_inference",
                source_type=SourceType.PAPER,
                title=f"Paper {i}",
                file_hash=f"sha256:paper{i}",
            )

        # List textbooks only
        textbooks = await SourceStore.list_by_type(SourceType.TEXTBOOK)

        assert len(textbooks) == 3
        assert all(s.source_type == SourceType.TEXTBOOK for s in textbooks)

        # List papers only
        papers = await SourceStore.list_by_type(SourceType.PAPER)

        assert len(papers) == 2
        assert all(s.source_type == SourceType.PAPER for s in papers)

    async def test_list_with_pagination(self, db_pool):
        """Test listing sources with limit/offset."""
        # Create 10 sources
        for i in range(10):
            await SourceStore.create(
                domain_id="causal_inference",
                source_type=SourceType.TEXTBOOK,
                title=f"Book {i}",
                file_hash=f"sha256:book{i}",
            )

        # Get first 5
        page1 = await SourceStore.list_by_type(SourceType.TEXTBOOK, limit=5, offset=0)
        assert len(page1) == 5

        # Get next 5
        page2 = await SourceStore.list_by_type(SourceType.TEXTBOOK, limit=5, offset=5)
        assert len(page2) == 5

        # Verify no overlap
        page1_ids = {s.id for s in page1}
        page2_ids = {s.id for s in page2}
        assert len(page1_ids & page2_ids) == 0
