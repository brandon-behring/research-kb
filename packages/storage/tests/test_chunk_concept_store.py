"""Tests for ChunkConceptStore - CRUD operations for chunk-concept links."""

import pytest
from uuid import uuid4

from research_kb_common import StorageError
from research_kb_contracts import ConceptType
from research_kb_storage import ChunkConceptStore, ChunkStore, ConceptStore


@pytest.mark.asyncio
async def test_create_chunk_concept_link(test_db, test_source):
    """Test creating a link between chunk and concept."""
    # Create chunk
    chunk = await ChunkStore.create(
        source_id=test_source.id,
        content="Discussion of instrumental variables estimation.",
        content_hash="hash1",
    )

    # Create concept
    concept = await ConceptStore.create(
        name="Instrumental Variables",
        canonical_name="instrumental_variables",
        concept_type=ConceptType.METHOD,
    )

    # Create link
    link = await ChunkConceptStore.create(
        chunk_id=chunk.id,
        concept_id=concept.id,
        mention_type="reference",
        relevance_score=0.9,
    )

    assert link.chunk_id == chunk.id
    assert link.concept_id == concept.id
    assert link.mention_type == "reference"
    assert link.relevance_score == pytest.approx(0.9, rel=1e-5)
    assert link.created_at is not None


@pytest.mark.asyncio
async def test_create_link_duplicate_fails(test_db, test_source):
    """Test that creating duplicate links fails."""
    # Create chunk and concept
    chunk = await ChunkStore.create(
        source_id=test_source.id, content="Test content", content_hash="hash2"
    )
    concept = await ConceptStore.create(
        name="Test Concept",
        canonical_name="test_concept",
        concept_type=ConceptType.DEFINITION,
    )

    # Create first link
    await ChunkConceptStore.create(
        chunk_id=chunk.id, concept_id=concept.id, mention_type="reference"
    )

    # Try to create duplicate with same mention_type
    with pytest.raises(StorageError, match="already exists"):
        await ChunkConceptStore.create(
            chunk_id=chunk.id, concept_id=concept.id, mention_type="reference"
        )


@pytest.mark.asyncio
async def test_create_link_missing_chunk_fails(test_db):
    """Test that creating link with non-existent chunk fails."""
    # Create concept only
    concept = await ConceptStore.create(
        name="Real Concept",
        canonical_name="real_concept",
        concept_type=ConceptType.DEFINITION,
    )

    # Try to create link with fake chunk
    fake_chunk_id = uuid4()
    with pytest.raises(StorageError, match="does not exist"):
        await ChunkConceptStore.create(
            chunk_id=fake_chunk_id, concept_id=concept.id, mention_type="reference"
        )


@pytest.mark.asyncio
async def test_create_link_missing_concept_fails(test_db, test_source):
    """Test that creating link with non-existent concept fails."""
    # Create chunk only
    chunk = await ChunkStore.create(
        source_id=test_source.id, content="Test content", content_hash="hash3"
    )

    # Try to create link with fake concept
    fake_concept_id = uuid4()
    with pytest.raises(StorageError, match="does not exist"):
        await ChunkConceptStore.create(
            chunk_id=chunk.id, concept_id=fake_concept_id, mention_type="reference"
        )


@pytest.mark.asyncio
async def test_list_concepts_for_chunk(test_db, test_source):
    """Test listing all concepts linked to a chunk."""
    # Create chunk
    chunk = await ChunkStore.create(
        source_id=test_source.id,
        content="Discussion of multiple concepts",
        content_hash="hash4",
    )

    # Create concepts
    concept1 = await ConceptStore.create(
        name="Concept 1",
        canonical_name="concept_1",
        concept_type=ConceptType.DEFINITION,
    )
    concept2 = await ConceptStore.create(
        name="Concept 2",
        canonical_name="concept_2",
        concept_type=ConceptType.DEFINITION,
    )

    # Create links with different relevance scores
    await ChunkConceptStore.create(
        chunk_id=chunk.id,
        concept_id=concept1.id,
        mention_type="reference",
        relevance_score=0.9,
    )
    await ChunkConceptStore.create(
        chunk_id=chunk.id,
        concept_id=concept2.id,
        mention_type="reference",
        relevance_score=0.7,
    )

    # List concepts
    links = await ChunkConceptStore.list_concepts_for_chunk(chunk.id)

    assert len(links) == 2
    # Should be ordered by relevance DESC
    assert links[0].relevance_score == pytest.approx(0.9, rel=1e-5)
    assert links[1].relevance_score == pytest.approx(0.7, rel=1e-5)


@pytest.mark.asyncio
async def test_list_chunks_for_concept(test_db, test_source):
    """Test listing all chunks that mention a concept."""
    # Create concept
    concept = await ConceptStore.create(
        name="Popular Concept",
        canonical_name="popular_concept",
        concept_type=ConceptType.DEFINITION,
    )

    # Create chunks
    chunk1 = await ChunkStore.create(
        source_id=test_source.id, content="First mention", content_hash="hash6"
    )
    chunk2 = await ChunkStore.create(
        source_id=test_source.id, content="Second mention", content_hash="hash7"
    )

    # Create links
    await ChunkConceptStore.create(
        chunk_id=chunk1.id,
        concept_id=concept.id,
        mention_type="reference",
        relevance_score=0.8,
    )
    await ChunkConceptStore.create(
        chunk_id=chunk2.id,
        concept_id=concept.id,
        mention_type="reference",
        relevance_score=0.9,
    )

    # List chunks
    links = await ChunkConceptStore.list_chunks_for_concept(concept.id)

    assert len(links) == 2
    # Should be ordered by relevance DESC
    assert links[0].relevance_score == pytest.approx(0.9, rel=1e-5)
    assert links[1].relevance_score == pytest.approx(0.8, rel=1e-5)


@pytest.mark.asyncio
async def test_delete_chunk_concept_link(test_db, test_source):
    """Test deleting a specific chunk-concept link."""
    # Create chunk and concept
    chunk = await ChunkStore.create(
        source_id=test_source.id, content="Test content", content_hash="hash8"
    )
    concept = await ConceptStore.create(
        name="Concept to Delete",
        canonical_name="concept_to_delete",
        concept_type=ConceptType.DEFINITION,
    )

    # Create link
    await ChunkConceptStore.create(
        chunk_id=chunk.id, concept_id=concept.id, mention_type="reference"
    )

    # Delete
    deleted = await ChunkConceptStore.delete(
        chunk_id=chunk.id, concept_id=concept.id, mention_type="reference"
    )

    assert deleted is True

    # Verify deleted
    links = await ChunkConceptStore.list_concepts_for_chunk(chunk.id)
    assert len(links) == 0


@pytest.mark.asyncio
async def test_delete_all_for_chunk(test_db, test_source):
    """Test deleting all concept links for a chunk."""
    # Create chunk
    chunk = await ChunkStore.create(
        source_id=test_source.id, content="Multiple concepts", content_hash="hash10"
    )

    # Create concepts and links
    for i in range(3):
        concept = await ConceptStore.create(
            name=f"Concept {i}",
            canonical_name=f"concept_{i}",
            concept_type=ConceptType.DEFINITION,
        )
        await ChunkConceptStore.create(
            chunk_id=chunk.id, concept_id=concept.id, mention_type="reference"
        )

    # Delete all
    count = await ChunkConceptStore.delete_all_for_chunk(chunk.id)

    assert count == 3

    # Verify all deleted
    links = await ChunkConceptStore.list_concepts_for_chunk(chunk.id)
    assert len(links) == 0


@pytest.mark.asyncio
async def test_count_for_concept(test_db, test_source):
    """Test counting chunks that mention a concept."""
    # Create concept
    concept = await ConceptStore.create(
        name="Counted Concept",
        canonical_name="counted_concept",
        concept_type=ConceptType.DEFINITION,
    )

    # Initially zero
    count = await ChunkConceptStore.count_for_concept(concept.id)
    assert count == 0

    # Create chunks and links
    for i in range(4):
        chunk = await ChunkStore.create(
            source_id=test_source.id,
            content=f"Chunk {i}",
            content_hash=f"hash_count_{i}",
        )
        await ChunkConceptStore.create(
            chunk_id=chunk.id, concept_id=concept.id, mention_type="reference"
        )

    # Count
    count = await ChunkConceptStore.count_for_concept(concept.id)
    assert count == 4


@pytest.mark.asyncio
async def test_batch_create_links(test_db, test_source):
    """Test batch creating chunk-concept links."""
    # Create chunk
    chunk = await ChunkStore.create(
        source_id=test_source.id, content="Batch test", content_hash="hash11"
    )

    # Create concepts
    concepts = []
    for i in range(3):
        concept = await ConceptStore.create(
            name=f"Batch Concept {i}",
            canonical_name=f"batch_concept_{i}",
            concept_type=ConceptType.DEFINITION,
        )
        concepts.append(concept)

    # Batch create links
    links_data = [
        {
            "chunk_id": chunk.id,
            "concept_id": concepts[0].id,
            "mention_type": "defines",
            "relevance_score": 0.9,
        },
        {
            "chunk_id": chunk.id,
            "concept_id": concepts[1].id,
            "mention_type": "reference",
            "relevance_score": 0.7,
        },
        {"chunk_id": chunk.id, "concept_id": concepts[2].id, "mention_type": "example"},
    ]

    created = await ChunkConceptStore.batch_create(links_data)

    assert len(created) == 3
    assert created[0].mention_type == "defines"
    assert created[1].mention_type == "reference"
    assert created[2].mention_type == "example"


@pytest.mark.asyncio
async def test_batch_create_empty_list(test_db):
    """Test batch create with empty list returns empty list."""
    created = await ChunkConceptStore.batch_create([])
    assert created == []


@pytest.mark.asyncio
async def test_get_concept_ids_for_chunks(test_db, test_source):
    """Test getting concept IDs for multiple chunks (batch operation)."""
    # Create chunks
    chunk1 = await ChunkStore.create(
        source_id=test_source.id, content="First chunk", content_hash="hash13"
    )
    chunk2 = await ChunkStore.create(
        source_id=test_source.id, content="Second chunk", content_hash="hash14"
    )
    chunk3 = await ChunkStore.create(
        source_id=test_source.id, content="Third chunk", content_hash="hash15"
    )

    # Create concepts
    concept1 = await ConceptStore.create(
        name="Concept A",
        canonical_name="concept_a",
        concept_type=ConceptType.DEFINITION,
    )
    concept2 = await ConceptStore.create(
        name="Concept B",
        canonical_name="concept_b",
        concept_type=ConceptType.DEFINITION,
    )

    # Create links
    # chunk1 -> concept1, concept2
    await ChunkConceptStore.create(chunk_id=chunk1.id, concept_id=concept1.id)
    await ChunkConceptStore.create(
        chunk_id=chunk1.id, concept_id=concept2.id, mention_type="defines"
    )

    # chunk2 -> concept1
    await ChunkConceptStore.create(chunk_id=chunk2.id, concept_id=concept1.id)

    # chunk3 -> no concepts

    # Get concept IDs for chunks
    result = await ChunkConceptStore.get_concept_ids_for_chunks(
        [chunk1.id, chunk2.id, chunk3.id]
    )

    assert len(result) == 3
    assert len(result[chunk1.id]) == 2  # Has 2 concepts
    assert len(result[chunk2.id]) == 1  # Has 1 concept
    assert len(result[chunk3.id]) == 0  # Has 0 concepts

    assert concept1.id in result[chunk1.id]
    assert concept2.id in result[chunk1.id]
    assert concept1.id in result[chunk2.id]


@pytest.mark.asyncio
async def test_get_concept_ids_for_chunks_empty_list(test_db):
    """Test getting concept IDs for empty chunk list."""
    result = await ChunkConceptStore.get_concept_ids_for_chunks([])
    assert result == {}


@pytest.mark.asyncio
async def test_get_concept_info_for_chunks(test_db, test_source):
    """Test getting concept info with mention_type and relevance_score."""
    # Create chunks
    chunk1 = await ChunkStore.create(
        source_id=test_source.id, content="Info chunk 1", content_hash="hash_info_1"
    )
    chunk2 = await ChunkStore.create(
        source_id=test_source.id, content="Info chunk 2", content_hash="hash_info_2"
    )

    # Create concepts
    concept1 = await ConceptStore.create(
        name="Concept Info A",
        canonical_name="concept_info_a",
        concept_type=ConceptType.DEFINITION,
    )
    concept2 = await ConceptStore.create(
        name="Concept Info B",
        canonical_name="concept_info_b",
        concept_type=ConceptType.METHOD,
    )

    # Create links with different mention types and relevance scores
    await ChunkConceptStore.create(
        chunk_id=chunk1.id,
        concept_id=concept1.id,
        mention_type="defines",
        relevance_score=0.95,
    )
    await ChunkConceptStore.create(
        chunk_id=chunk1.id,
        concept_id=concept2.id,
        mention_type="reference",
        relevance_score=0.7,
    )
    await ChunkConceptStore.create(
        chunk_id=chunk2.id,
        concept_id=concept1.id,
        mention_type="example",
        # No relevance_score - should be None
    )

    # Get concept info for chunks
    result = await ChunkConceptStore.get_concept_info_for_chunks([chunk1.id, chunk2.id])

    assert len(result) == 2

    # Chunk1 has 2 concepts
    chunk1_info = result[chunk1.id]
    assert len(chunk1_info) == 2

    # Find the 'defines' link
    defines_link = next((c for c in chunk1_info if c[0] == concept1.id), None)
    assert defines_link is not None
    assert defines_link[1] == "defines"
    assert defines_link[2] == pytest.approx(0.95, rel=1e-5)  # DB uses REAL (single precision)

    # Find the 'reference' link
    ref_link = next((c for c in chunk1_info if c[0] == concept2.id), None)
    assert ref_link is not None
    assert ref_link[1] == "reference"
    assert ref_link[2] == pytest.approx(0.7, rel=1e-5)

    # Chunk2 has 1 concept with 'example' mention
    chunk2_info = result[chunk2.id]
    assert len(chunk2_info) == 1
    assert chunk2_info[0][0] == concept1.id
    assert chunk2_info[0][1] == "example"
    assert chunk2_info[0][2] is None  # No relevance score


@pytest.mark.asyncio
async def test_get_concept_info_for_chunks_empty_list(test_db):
    """Test getting concept info for empty chunk list."""
    result = await ChunkConceptStore.get_concept_info_for_chunks([])
    assert result == {}
