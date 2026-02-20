"""Tests for ChunkStore against live PostgreSQL."""

import pytest
from uuid import uuid4

from research_kb_common import StorageError
from research_kb_contracts import SourceType
from research_kb_storage import ChunkStore, SourceStore

pytestmark = pytest.mark.integration


@pytest.fixture
async def test_source(db_pool):
    """Create a test source for chunk tests."""
    source = await SourceStore.create(
        source_type=SourceType.TEXTBOOK,
        title="Test Textbook",
        file_hash="sha256:testsource123",
    )
    return source


class TestChunkStoreCreate:
    """Test ChunkStore.create() operations."""

    async def test_create_minimal_chunk(self, test_source):
        """Test creating chunk with minimal required fields."""
        chunk = await ChunkStore.create(
            source_id=test_source.id,
            content="Test chunk content",
            content_hash="sha256:chunk123",
        )

        assert chunk.id is not None
        assert chunk.source_id == test_source.id
        assert chunk.content == "Test chunk content"
        assert chunk.content_hash == "sha256:chunk123"
        assert chunk.embedding is None
        assert chunk.metadata == {}

    async def test_create_chunk_with_embedding(self, test_source):
        """Test creating chunk with 1024-dim embedding (BGE-large-en-v1.5)."""
        embedding = [0.1] * 1024  # BGE-large-en-v1.5

        chunk = await ChunkStore.create(
            source_id=test_source.id,
            content="Chunk with embedding",
            content_hash="sha256:embedded",
            embedding=embedding,
        )

        assert chunk.embedding is not None
        assert len(chunk.embedding) == 1024
        assert all(abs(v - 0.1) < 0.001 for v in chunk.embedding)

    async def test_create_chunk_with_full_metadata(self, test_source):
        """Test creating chunk with all fields."""
        chunk = await ChunkStore.create(
            source_id=test_source.id,
            content="The backdoor criterion states...",
            content_hash="sha256:theorem",
            location="Chapter 3, Section 3.3, p. 73",
            page_start=73,
            page_end=74,
            embedding=[0.5] * 1024,
            metadata={
                "chunk_type": "theorem",
                "theorem_name": "Backdoor Criterion",
                "chapter_num": 3,
            },
        )

        assert chunk.location == "Chapter 3, Section 3.3, p. 73"
        assert chunk.page_start == 73
        assert chunk.page_end == 74
        assert chunk.metadata["chunk_type"] == "theorem"
        assert chunk.metadata["theorem_name"] == "Backdoor Criterion"


class TestChunkStoreRetrieve:
    """Test ChunkStore retrieval operations."""

    async def test_get_by_id_found(self, test_source):
        """Test retrieving chunk by ID when it exists."""
        created = await ChunkStore.create(
            source_id=test_source.id,
            content="Test content",
            content_hash="sha256:getbyid",
        )

        retrieved = await ChunkStore.get_by_id(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.content == created.content

    async def test_get_by_id_not_found(self, db_pool):
        """Test retrieving chunk by ID when it doesn't exist."""
        result = await ChunkStore.get_by_id(uuid4())
        assert result is None

    async def test_list_by_source(self, test_source):
        """Test listing chunks for a source."""
        # Create 5 chunks for the source
        for i in range(5):
            await ChunkStore.create(
                source_id=test_source.id,
                content=f"Chunk {i} content",
                content_hash=f"sha256:chunk{i}",
            )

        chunks = await ChunkStore.list_by_source(test_source.id)

        assert len(chunks) == 5
        assert all(c.source_id == test_source.id for c in chunks)

    async def test_list_by_source_pagination(self, test_source):
        """Test listing chunks with pagination."""
        # Create 10 chunks
        for i in range(10):
            await ChunkStore.create(
                source_id=test_source.id,
                content=f"Chunk {i}",
                content_hash=f"sha256:page{i}",
            )

        # Get first 5
        page1 = await ChunkStore.list_by_source(test_source.id, limit=5, offset=0)
        assert len(page1) == 5

        # Get next 5
        page2 = await ChunkStore.list_by_source(test_source.id, limit=5, offset=5)
        assert len(page2) == 5

        # No overlap
        page1_ids = {c.id for c in page1}
        page2_ids = {c.id for c in page2}
        assert len(page1_ids & page2_ids) == 0


class TestChunkStoreUpdate:
    """Test ChunkStore update operations."""

    async def test_update_embedding(self, test_source):
        """Test updating chunk embedding."""
        chunk = await ChunkStore.create(
            source_id=test_source.id,
            content="Test",
            content_hash="sha256:updateembed",
            embedding=None,  # No embedding initially
        )

        assert chunk.embedding is None

        # Update with embedding
        new_embedding = [0.9] * 1024
        updated = await ChunkStore.update_embedding(chunk.id, new_embedding)

        assert updated.embedding is not None
        assert len(updated.embedding) == 1024
        assert all(abs(v - 0.9) < 0.001 for v in updated.embedding)

    async def test_update_embedding_nonexistent_chunk(self, db_pool):
        """Test updating embedding for nonexistent chunk raises error."""
        with pytest.raises(StorageError):
            await ChunkStore.update_embedding(uuid4(), [0.1] * 1024)


class TestChunkStoreBatch:
    """Test ChunkStore batch operations."""

    async def test_batch_create(self, test_source):
        """Test batch creating multiple chunks."""
        chunks_data = [
            {
                "source_id": test_source.id,
                "content": f"Batch chunk {i}",
                "content_hash": f"sha256:batch{i}",
                "embedding": [float(i) / 10] * 1024,
                "metadata": {"index": i},
            }
            for i in range(5)
        ]

        created = await ChunkStore.batch_create(chunks_data)

        assert len(created) == 5
        assert all(c.source_id == test_source.id for c in created)
        assert created[0].metadata["index"] == 0
        assert created[4].metadata["index"] == 4

    async def test_batch_create_empty_list(self, db_pool):
        """Test batch create with empty list returns empty list."""
        result = await ChunkStore.batch_create([])
        assert result == []

    async def test_batch_create_idempotent(self, test_source):
        """Re-inserting same chunks returns existing rather than duplicating."""
        chunks_data = [
            {
                "source_id": test_source.id,
                "content": "Dedup test content",
                "content_hash": "sha256:test_dedup_batch",
                "embedding": [0.1] * 1024,
                "metadata": {"index": 0},
            }
        ]
        first = await ChunkStore.batch_create(chunks_data)
        second = await ChunkStore.batch_create(chunks_data)

        assert len(first) == len(second) == 1
        assert first[0].id == second[0].id

    async def test_create_idempotent(self, test_source):
        """Re-inserting same chunk via create() returns existing."""
        first = await ChunkStore.create(
            source_id=test_source.id,
            content="Single dedup test",
            content_hash="sha256:test_dedup_single",
        )
        second = await ChunkStore.create(
            source_id=test_source.id,
            content="Single dedup test",
            content_hash="sha256:test_dedup_single",
        )

        assert first.id == second.id


class TestChunkStoreDelete:
    """Test ChunkStore delete operations."""

    async def test_delete_existing_chunk(self, test_source):
        """Test deleting existing chunk returns True."""
        chunk = await ChunkStore.create(
            source_id=test_source.id,
            content="To be deleted",
            content_hash="sha256:delete",
        )

        deleted = await ChunkStore.delete(chunk.id)
        assert deleted is True

        # Verify gone
        result = await ChunkStore.get_by_id(chunk.id)
        assert result is None

    async def test_delete_nonexistent_chunk(self, db_pool):
        """Test deleting nonexistent chunk returns False."""
        deleted = await ChunkStore.delete(uuid4())
        assert deleted is False


class TestChunkStoreCount:
    """Test ChunkStore count operations."""

    async def test_count_by_source(self, test_source):
        """Test counting chunks for a source."""
        # Initially 0
        count = await ChunkStore.count_by_source(test_source.id)
        assert count == 0

        # Create 7 chunks
        for i in range(7):
            await ChunkStore.create(
                source_id=test_source.id,
                content=f"Chunk {i}",
                content_hash=f"sha256:count{i}",
            )

        count = await ChunkStore.count_by_source(test_source.id)
        assert count == 7


class TestChunkStoreCascadeDelete:
    """Test CASCADE delete from sources to chunks."""

    async def test_deleting_source_deletes_chunks(self, test_source):
        """Test deleting source also deletes its chunks (CASCADE)."""
        # Create 3 chunks for source
        chunk_ids = []
        for i in range(3):
            chunk = await ChunkStore.create(
                source_id=test_source.id,
                content=f"Chunk {i}",
                content_hash=f"sha256:cascade{i}",
            )
            chunk_ids.append(chunk.id)

        # Verify chunks exist
        count = await ChunkStore.count_by_source(test_source.id)
        assert count == 3

        # Delete source
        await SourceStore.delete(test_source.id)

        # Verify chunks are gone (CASCADE)
        for chunk_id in chunk_ids:
            chunk = await ChunkStore.get_by_id(chunk_id)
            assert chunk is None

        count = await ChunkStore.count_by_source(test_source.id)
        assert count == 0
