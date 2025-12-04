"""Tests for ConceptStore, RelationshipStore, and ChunkConceptStore."""

import pytest
from uuid import uuid4

from research_kb_contracts import ConceptType, RelationshipType, SourceType
from research_kb_storage import (
    ChunkStore,
    ChunkConceptStore,
    ConceptStore,
    RelationshipStore,
    SourceStore,
)


@pytest.fixture
async def test_source(db_pool):
    """Create test source for chunk creation."""
    source = await SourceStore.create(
        source_type=SourceType.TEXTBOOK,
        title="Test Textbook",
        file_hash=f"sha256:test_{uuid4().hex[:8]}",
        authors=["Test Author"],
    )
    return source


@pytest.fixture
async def test_chunk(test_source, db_pool):
    """Create test chunk for linking."""
    chunk = await ChunkStore.create(
        source_id=test_source.id,
        content="This is a test chunk about instrumental variables.",
        content_hash=f"sha256:chunk_{uuid4().hex[:8]}",
        location="Chapter 1, p. 1",
        embedding=[0.1] * 1024,
    )
    return chunk


class TestConceptStoreCreate:
    """Tests for ConceptStore.create()."""

    async def test_create_minimal(self, db_pool):
        """Test creating concept with minimal fields."""
        concept = await ConceptStore.create(
            name="instrumental variables",
            canonical_name=f"instrumental variables_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
        )

        assert concept.id is not None
        assert concept.name == "instrumental variables"
        assert concept.concept_type == ConceptType.METHOD
        assert concept.validated is False

    async def test_create_full(self, db_pool):
        """Test creating concept with all fields."""
        embedding = [0.5] * 1024
        concept = await ConceptStore.create(
            name="Difference-in-Differences",
            canonical_name=f"difference-in-differences_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
            aliases=["DiD", "DD", "diff-in-diff"],
            category="identification",
            definition="Compares treatment and control before and after",
            embedding=embedding,
            extraction_method="ollama:llama3.1:8b",
            confidence_score=0.92,
            validated=True,
            metadata={"source_count": 5},
        )

        assert concept.name == "Difference-in-Differences"
        assert "DiD" in concept.aliases
        assert concept.category == "identification"
        assert concept.definition is not None
        assert len(concept.embedding) == 1024
        assert concept.validated is True
        assert abs(concept.confidence_score - 0.92) < 0.001

    async def test_create_duplicate_fails(self, db_pool):
        """Test creating duplicate canonical_name fails."""
        canonical = f"unique_concept_{uuid4().hex[:8]}"

        await ConceptStore.create(
            name="Test",
            canonical_name=canonical,
            concept_type=ConceptType.METHOD,
        )

        from research_kb_common import StorageError

        with pytest.raises(StorageError) as exc_info:
            await ConceptStore.create(
                name="Test Duplicate",
                canonical_name=canonical,
                concept_type=ConceptType.ASSUMPTION,
            )

        assert "already exists" in str(exc_info.value)


class TestConceptStoreRetrieve:
    """Tests for ConceptStore retrieval methods."""

    async def test_get_by_id(self, db_pool):
        """Test retrieving concept by ID."""
        created = await ConceptStore.create(
            name="test",
            canonical_name=f"test_{uuid4().hex[:8]}",
            concept_type=ConceptType.DEFINITION,
        )

        retrieved = await ConceptStore.get_by_id(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == created.name

    async def test_get_by_id_not_found(self, db_pool):
        """Test get_by_id returns None for missing concept."""
        result = await ConceptStore.get_by_id(uuid4())
        assert result is None

    async def test_get_by_canonical_name(self, db_pool):
        """Test retrieving concept by canonical name."""
        canonical = f"canonical_test_{uuid4().hex[:8]}"
        await ConceptStore.create(
            name="Test Concept",
            canonical_name=canonical,
            concept_type=ConceptType.THEOREM,
        )

        retrieved = await ConceptStore.get_by_canonical_name(canonical)

        assert retrieved is not None
        assert retrieved.canonical_name == canonical

    async def test_list_by_type(self, db_pool):
        """Test listing concepts by type."""
        # Create some methods
        for i in range(3):
            await ConceptStore.create(
                name=f"Method {i}",
                canonical_name=f"method_{i}_{uuid4().hex[:8]}",
                concept_type=ConceptType.METHOD,
            )

        methods = await ConceptStore.list_by_type(ConceptType.METHOD, limit=10)
        assert len(methods) >= 3
        assert all(c.concept_type == ConceptType.METHOD for c in methods)


class TestConceptStoreUpdate:
    """Tests for ConceptStore.update()."""

    async def test_update_definition(self, db_pool):
        """Test updating definition."""
        concept = await ConceptStore.create(
            name="test",
            canonical_name=f"test_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
        )

        updated = await ConceptStore.update(
            concept.id,
            definition="Updated definition text",
        )

        assert updated.definition == "Updated definition text"

    async def test_update_validated(self, db_pool):
        """Test marking as validated."""
        concept = await ConceptStore.create(
            name="test",
            canonical_name=f"test_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
            validated=False,
        )

        updated = await ConceptStore.update(
            concept.id,
            validated=True,
        )

        assert updated.validated is True


class TestConceptStoreBatch:
    """Tests for batch operations."""

    async def test_batch_create(self, db_pool):
        """Test batch creating concepts."""
        concepts_data = [
            {
                "name": f"Batch Concept {i}",
                "canonical_name": f"batch_concept_{i}_{uuid4().hex[:8]}",
                "concept_type": "method",
                "confidence_score": 0.8,
            }
            for i in range(5)
        ]

        created = await ConceptStore.batch_create(concepts_data)

        assert len(created) == 5
        assert all(c.name.startswith("Batch Concept") for c in created)

    async def test_batch_create_skips_duplicates(self, db_pool):
        """Test batch create skips duplicates without failing."""
        canonical = f"dup_batch_{uuid4().hex[:8]}"

        await ConceptStore.create(
            name="Existing",
            canonical_name=canonical,
            concept_type=ConceptType.METHOD,
        )

        concepts_data = [
            {
                "name": "New",
                "canonical_name": f"new_{uuid4().hex[:8]}",
                "concept_type": "method",
            },
            {
                "name": "Duplicate",
                "canonical_name": canonical,  # This one exists
                "concept_type": "method",
            },
        ]

        created = await ConceptStore.batch_create(concepts_data)

        # Should only create the new one, skip the duplicate
        assert len(created) == 1
        assert created[0].name == "New"


class TestRelationshipStore:
    """Tests for RelationshipStore."""

    async def test_create_relationship(self, db_pool):
        """Test creating a relationship."""
        c1 = await ConceptStore.create(
            name="IV",
            canonical_name=f"iv_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
        )
        c2 = await ConceptStore.create(
            name="Relevance",
            canonical_name=f"relevance_{uuid4().hex[:8]}",
            concept_type=ConceptType.ASSUMPTION,
        )

        rel = await RelationshipStore.create(
            source_concept_id=c1.id,
            target_concept_id=c2.id,
            relationship_type=RelationshipType.REQUIRES,
            strength=0.95,
            confidence_score=0.88,
        )

        assert rel.id is not None
        assert rel.source_concept_id == c1.id
        assert rel.target_concept_id == c2.id
        assert rel.relationship_type == RelationshipType.REQUIRES
        assert abs(rel.strength - 0.95) < 0.001

    async def test_list_from_concept(self, db_pool):
        """Test listing outgoing relationships."""
        c1 = await ConceptStore.create(
            name="Source",
            canonical_name=f"source_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
        )
        c2 = await ConceptStore.create(
            name="Target1",
            canonical_name=f"target1_{uuid4().hex[:8]}",
            concept_type=ConceptType.ASSUMPTION,
        )
        c3 = await ConceptStore.create(
            name="Target2",
            canonical_name=f"target2_{uuid4().hex[:8]}",
            concept_type=ConceptType.PROBLEM,
        )

        await RelationshipStore.create(
            source_concept_id=c1.id,
            target_concept_id=c2.id,
            relationship_type=RelationshipType.REQUIRES,
        )
        await RelationshipStore.create(
            source_concept_id=c1.id,
            target_concept_id=c3.id,
            relationship_type=RelationshipType.ADDRESSES,
        )

        rels = await RelationshipStore.list_from_concept(c1.id)

        assert len(rels) == 2
        assert all(r.source_concept_id == c1.id for r in rels)

    async def test_batch_create_relationships(self, db_pool):
        """Test batch creating relationships."""
        c1 = await ConceptStore.create(
            name="Method",
            canonical_name=f"method_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
        )
        c2 = await ConceptStore.create(
            name="Assumption1",
            canonical_name=f"assumption1_{uuid4().hex[:8]}",
            concept_type=ConceptType.ASSUMPTION,
        )
        c3 = await ConceptStore.create(
            name="Assumption2",
            canonical_name=f"assumption2_{uuid4().hex[:8]}",
            concept_type=ConceptType.ASSUMPTION,
        )

        rels_data = [
            {
                "source_concept_id": c1.id,
                "target_concept_id": c2.id,
                "relationship_type": "REQUIRES",
            },
            {
                "source_concept_id": c1.id,
                "target_concept_id": c3.id,
                "relationship_type": "REQUIRES",
            },
        ]

        created = await RelationshipStore.batch_create(rels_data)

        assert len(created) == 2


class TestChunkConceptStore:
    """Tests for ChunkConceptStore."""

    async def test_create_link(self, test_chunk, db_pool):
        """Test creating chunk-concept link."""
        concept = await ConceptStore.create(
            name="IV",
            canonical_name=f"iv_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
        )

        link = await ChunkConceptStore.create(
            chunk_id=test_chunk.id,
            concept_id=concept.id,
            mention_type="defines",
            relevance_score=0.95,
        )

        assert link.chunk_id == test_chunk.id
        assert link.concept_id == concept.id
        assert link.mention_type == "defines"
        assert abs(link.relevance_score - 0.95) < 0.001

    async def test_list_concepts_for_chunk(self, test_chunk, db_pool):
        """Test listing concepts for a chunk."""
        c1 = await ConceptStore.create(
            name="Concept1",
            canonical_name=f"concept1_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
        )
        c2 = await ConceptStore.create(
            name="Concept2",
            canonical_name=f"concept2_{uuid4().hex[:8]}",
            concept_type=ConceptType.ASSUMPTION,
        )

        await ChunkConceptStore.create(test_chunk.id, c1.id, "reference")
        await ChunkConceptStore.create(test_chunk.id, c2.id, "example")

        links = await ChunkConceptStore.list_concepts_for_chunk(test_chunk.id)

        assert len(links) == 2
        concept_ids = {link.concept_id for link in links}
        assert c1.id in concept_ids
        assert c2.id in concept_ids

    async def test_batch_create_links(self, test_chunk, db_pool):
        """Test batch creating chunk-concept links."""
        concepts = []
        for i in range(3):
            c = await ConceptStore.create(
                name=f"Concept {i}",
                canonical_name=f"concept_{i}_{uuid4().hex[:8]}",
                concept_type=ConceptType.METHOD,
            )
            concepts.append(c)

        links_data = [
            {
                "chunk_id": test_chunk.id,
                "concept_id": c.id,
                "mention_type": "reference",
                "relevance_score": 0.8,
            }
            for c in concepts
        ]

        created = await ChunkConceptStore.batch_create(links_data)

        assert len(created) == 3

    async def test_get_concept_ids_for_chunks(self, test_source, db_pool):
        """Test batch getting concept IDs for multiple chunks."""
        # Create chunks
        chunks = []
        for i in range(2):
            chunk = await ChunkStore.create(
                source_id=test_source.id,
                content=f"Chunk {i}",
                content_hash=f"hash_{i}_{uuid4().hex[:8]}",
                embedding=[0.1] * 1024,
            )
            chunks.append(chunk)

        # Create concepts
        concept = await ConceptStore.create(
            name="Shared Concept",
            canonical_name=f"shared_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
        )

        # Link concept to both chunks
        await ChunkConceptStore.create(chunks[0].id, concept.id)
        await ChunkConceptStore.create(chunks[1].id, concept.id)

        # Query
        result = await ChunkConceptStore.get_concept_ids_for_chunks(
            [c.id for c in chunks]
        )

        assert len(result) == 2
        assert concept.id in result[chunks[0].id]
        assert concept.id in result[chunks[1].id]
