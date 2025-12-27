"""Tests for MethodStore - CRUD operations for methods table."""

import pytest
from uuid import uuid4

from research_kb_common import StorageError
from research_kb_contracts import ConceptType
from research_kb_storage import ConceptStore, MethodStore


class TestMethodStoreCreate:
    """Tests for MethodStore.create()."""

    async def test_create_minimal(self, test_db):
        """Create method with minimal fields."""
        # First create a concept
        concept = await ConceptStore.create(
            name="Instrumental Variables",
            canonical_name=f"iv_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
        )

        method = await MethodStore.create(concept_id=concept.id)

        assert method.id is not None
        assert method.concept_id == concept.id
        assert method.required_assumptions == []
        assert method.problem_types == []
        assert method.common_estimators == []

    async def test_create_full(self, test_db):
        """Create method with all fields."""
        concept = await ConceptStore.create(
            name="Double Machine Learning",
            canonical_name=f"dml_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
        )

        method = await MethodStore.create(
            concept_id=concept.id,
            required_assumptions=["unconfoundedness", "overlap"],
            problem_types=["ATE", "CATE"],
            common_estimators=["DML", "causal forest"],
        )

        assert method.concept_id == concept.id
        assert "unconfoundedness" in method.required_assumptions
        assert "ATE" in method.problem_types
        assert "DML" in method.common_estimators

    async def test_create_duplicate_fails(self, test_db):
        """Creating method for same concept twice fails."""
        concept = await ConceptStore.create(
            name="Matching",
            canonical_name=f"matching_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
        )

        await MethodStore.create(concept_id=concept.id)

        with pytest.raises(StorageError, match="already exists"):
            await MethodStore.create(concept_id=concept.id)

    async def test_create_nonexistent_concept_fails(self, test_db):
        """Creating method for non-existent concept fails."""
        fake_id = uuid4()

        with pytest.raises(StorageError, match="not found"):
            await MethodStore.create(concept_id=fake_id)


class TestMethodStoreRetrieve:
    """Tests for MethodStore retrieval methods."""

    async def test_get_by_id_found(self, test_db):
        """Retrieve method by ID."""
        concept = await ConceptStore.create(
            name="Test Method",
            canonical_name=f"test_method_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
        )
        created = await MethodStore.create(
            concept_id=concept.id,
            required_assumptions=["assumption1"],
        )

        retrieved = await MethodStore.get_by_id(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.required_assumptions == ["assumption1"]

    async def test_get_by_id_not_found(self, test_db):
        """Non-existent method returns None."""
        fake_id = uuid4()
        result = await MethodStore.get_by_id(fake_id)
        assert result is None

    async def test_get_by_concept_id_found(self, test_db):
        """Retrieve method by concept ID."""
        concept = await ConceptStore.create(
            name="Propensity Score",
            canonical_name=f"ps_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
        )
        created = await MethodStore.create(
            concept_id=concept.id,
            problem_types=["selection bias"],
        )

        retrieved = await MethodStore.get_by_concept_id(concept.id)

        assert retrieved is not None
        assert retrieved.concept_id == concept.id
        assert "selection bias" in retrieved.problem_types

    async def test_get_by_concept_id_not_found(self, test_db):
        """Non-existent concept returns None."""
        fake_id = uuid4()
        result = await MethodStore.get_by_concept_id(fake_id)
        assert result is None


class TestMethodStoreUpdate:
    """Tests for MethodStore.update()."""

    async def test_update_single_field(self, test_db):
        """Update single field."""
        concept = await ConceptStore.create(
            name="Update Test",
            canonical_name=f"update_test_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
        )
        method = await MethodStore.create(
            concept_id=concept.id,
            required_assumptions=["old_assumption"],
        )

        updated = await MethodStore.update(
            method_id=method.id,
            required_assumptions=["new_assumption"],
        )

        assert updated.required_assumptions == ["new_assumption"]

    async def test_update_multiple_fields(self, test_db):
        """Update multiple fields at once."""
        concept = await ConceptStore.create(
            name="Multi Update",
            canonical_name=f"multi_update_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
        )
        method = await MethodStore.create(concept_id=concept.id)

        updated = await MethodStore.update(
            method_id=method.id,
            required_assumptions=["assumption1", "assumption2"],
            problem_types=["ATE"],
            common_estimators=["OLS", "2SLS"],
        )

        assert len(updated.required_assumptions) == 2
        assert "ATE" in updated.problem_types
        assert "2SLS" in updated.common_estimators

    async def test_update_no_changes(self, test_db):
        """Update with no fields returns current record."""
        concept = await ConceptStore.create(
            name="No Change",
            canonical_name=f"no_change_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
        )
        method = await MethodStore.create(
            concept_id=concept.id,
            required_assumptions=["keep_this"],
        )

        updated = await MethodStore.update(method_id=method.id)

        assert updated.required_assumptions == ["keep_this"]


class TestMethodStoreDelete:
    """Tests for MethodStore.delete()."""

    async def test_delete_existing(self, test_db):
        """Delete existing method."""
        concept = await ConceptStore.create(
            name="To Delete",
            canonical_name=f"to_delete_{uuid4().hex[:8]}",
            concept_type=ConceptType.METHOD,
        )
        method = await MethodStore.create(concept_id=concept.id)

        deleted = await MethodStore.delete(method.id)

        assert deleted is True
        assert await MethodStore.get_by_id(method.id) is None

    async def test_delete_nonexistent(self, test_db):
        """Delete non-existent method returns False."""
        fake_id = uuid4()
        deleted = await MethodStore.delete(fake_id)
        assert deleted is False


class TestMethodStoreList:
    """Tests for MethodStore.list_all() and count()."""

    async def test_list_empty(self, test_db):
        """List with no methods returns empty list."""
        methods = await MethodStore.list_all()
        assert methods == []

    async def test_list_with_methods(self, test_db):
        """List returns all methods."""
        for i in range(3):
            concept = await ConceptStore.create(
                name=f"Method {i}",
                canonical_name=f"method_{i}_{uuid4().hex[:8]}",
                concept_type=ConceptType.METHOD,
            )
            await MethodStore.create(concept_id=concept.id)

        methods = await MethodStore.list_all()

        assert len(methods) == 3

    async def test_list_pagination(self, test_db):
        """List respects limit and offset."""
        for i in range(5):
            concept = await ConceptStore.create(
                name=f"Paginated {i}",
                canonical_name=f"paginated_{i}_{uuid4().hex[:8]}",
                concept_type=ConceptType.METHOD,
            )
            await MethodStore.create(concept_id=concept.id)

        page1 = await MethodStore.list_all(limit=2, offset=0)
        page2 = await MethodStore.list_all(limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].id != page2[0].id

    async def test_count_empty(self, test_db):
        """Count with no methods returns 0."""
        count = await MethodStore.count()
        assert count == 0

    async def test_count_with_methods(self, test_db):
        """Count returns correct number."""
        for i in range(4):
            concept = await ConceptStore.create(
                name=f"Count {i}",
                canonical_name=f"count_{i}_{uuid4().hex[:8]}",
                concept_type=ConceptType.METHOD,
            )
            await MethodStore.create(concept_id=concept.id)

        count = await MethodStore.count()

        assert count == 4
