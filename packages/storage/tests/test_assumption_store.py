"""Tests for AssumptionStore - CRUD operations for assumptions table."""

import pytest
from uuid import uuid4

from research_kb_common import StorageError
from research_kb_contracts import ConceptType
from research_kb_storage import ConceptStore, AssumptionStore

pytestmark = pytest.mark.unit


class TestAssumptionStoreCreate:
    """Tests for AssumptionStore.create()."""

    async def test_create_minimal(self, test_db):
        """Create assumption with minimal fields."""
        concept = await ConceptStore.create(
            domain_id="causal_inference",
            name="Unconfoundedness",
            canonical_name=f"unconfoundedness_{uuid4().hex[:8]}",
            concept_type=ConceptType.ASSUMPTION,
        )

        assumption = await AssumptionStore.create(concept_id=concept.id)

        assert assumption.id is not None
        assert assumption.concept_id == concept.id
        assert assumption.mathematical_statement is None
        assert assumption.is_testable is None
        assert assumption.common_tests == []
        assert assumption.violation_consequences is None

    async def test_create_full(self, test_db):
        """Create assumption with all fields."""
        concept = await ConceptStore.create(
            domain_id="causal_inference",
            name="Parallel Trends",
            canonical_name=f"parallel_trends_{uuid4().hex[:8]}",
            concept_type=ConceptType.ASSUMPTION,
        )

        assumption = await AssumptionStore.create(
            concept_id=concept.id,
            mathematical_statement="E[Y(0)_t | D=1] - E[Y(0)_t | D=0] = constant",
            is_testable=True,
            common_tests=["pre-trend test", "placebo test"],
            violation_consequences="Biased treatment effect estimates",
        )

        assert assumption.mathematical_statement is not None
        assert assumption.is_testable is True
        assert "pre-trend test" in assumption.common_tests
        assert "Biased" in assumption.violation_consequences

    async def test_create_duplicate_fails(self, test_db):
        """Creating assumption for same concept twice fails."""
        concept = await ConceptStore.create(
            domain_id="causal_inference",
            name="Overlap",
            canonical_name=f"overlap_{uuid4().hex[:8]}",
            concept_type=ConceptType.ASSUMPTION,
        )

        await AssumptionStore.create(concept_id=concept.id)

        with pytest.raises(StorageError, match="already exists"):
            await AssumptionStore.create(concept_id=concept.id)

    async def test_create_nonexistent_concept_fails(self, test_db):
        """Creating assumption for non-existent concept fails."""
        fake_id = uuid4()

        with pytest.raises(StorageError, match="not found"):
            await AssumptionStore.create(concept_id=fake_id)


class TestAssumptionStoreRetrieve:
    """Tests for AssumptionStore retrieval methods."""

    async def test_get_by_id_found(self, test_db):
        """Retrieve assumption by ID."""
        concept = await ConceptStore.create(
            domain_id="causal_inference",
            name="SUTVA",
            canonical_name=f"sutva_{uuid4().hex[:8]}",
            concept_type=ConceptType.ASSUMPTION,
        )
        created = await AssumptionStore.create(
            concept_id=concept.id,
            is_testable=False,
        )

        retrieved = await AssumptionStore.get_by_id(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.is_testable is False

    async def test_get_by_id_not_found(self, test_db):
        """Non-existent assumption returns None."""
        fake_id = uuid4()
        result = await AssumptionStore.get_by_id(fake_id)
        assert result is None

    async def test_get_by_concept_id_found(self, test_db):
        """Retrieve assumption by concept ID."""
        concept = await ConceptStore.create(
            domain_id="causal_inference",
            name="Exogeneity",
            canonical_name=f"exogeneity_{uuid4().hex[:8]}",
            concept_type=ConceptType.ASSUMPTION,
        )
        created = await AssumptionStore.create(
            concept_id=concept.id,
            mathematical_statement="E[u|X] = 0",
        )

        retrieved = await AssumptionStore.get_by_concept_id(concept.id)

        assert retrieved is not None
        assert retrieved.concept_id == concept.id
        assert "E[u|X]" in retrieved.mathematical_statement

    async def test_get_by_concept_id_not_found(self, test_db):
        """Non-existent concept returns None."""
        fake_id = uuid4()
        result = await AssumptionStore.get_by_concept_id(fake_id)
        assert result is None


class TestAssumptionStoreUpdate:
    """Tests for AssumptionStore.update()."""

    async def test_update_single_field(self, test_db):
        """Update single field."""
        concept = await ConceptStore.create(
            domain_id="causal_inference",
            name="Update Test",
            canonical_name=f"assumption_update_{uuid4().hex[:8]}",
            concept_type=ConceptType.ASSUMPTION,
        )
        assumption = await AssumptionStore.create(
            concept_id=concept.id,
            is_testable=False,
        )

        updated = await AssumptionStore.update(
            assumption_id=assumption.id,
            is_testable=True,
        )

        assert updated.is_testable is True

    async def test_update_multiple_fields(self, test_db):
        """Update multiple fields at once."""
        concept = await ConceptStore.create(
            domain_id="causal_inference",
            name="Multi Update",
            canonical_name=f"assumption_multi_{uuid4().hex[:8]}",
            concept_type=ConceptType.ASSUMPTION,
        )
        assumption = await AssumptionStore.create(concept_id=concept.id)

        updated = await AssumptionStore.update(
            assumption_id=assumption.id,
            mathematical_statement="Y(1), Y(0) independent of T | X",
            is_testable=True,
            common_tests=["balance test", "sensitivity analysis"],
            violation_consequences="Selection bias",
        )

        assert updated.mathematical_statement is not None
        assert updated.is_testable is True
        assert len(updated.common_tests) == 2
        assert "Selection bias" in updated.violation_consequences

    async def test_update_no_changes(self, test_db):
        """Update with no fields returns current record."""
        concept = await ConceptStore.create(
            domain_id="causal_inference",
            name="No Change",
            canonical_name=f"assumption_nochange_{uuid4().hex[:8]}",
            concept_type=ConceptType.ASSUMPTION,
        )
        assumption = await AssumptionStore.create(
            concept_id=concept.id,
            is_testable=True,
        )

        updated = await AssumptionStore.update(assumption_id=assumption.id)

        assert updated.is_testable is True

    async def test_update_nonexistent_fails(self, test_db):
        """Update non-existent assumption raises error."""
        fake_id = uuid4()

        with pytest.raises(StorageError, match="not found"):
            await AssumptionStore.update(
                assumption_id=fake_id,
                is_testable=True,
            )


class TestAssumptionStoreDelete:
    """Tests for AssumptionStore.delete()."""

    async def test_delete_existing(self, test_db):
        """Delete existing assumption."""
        concept = await ConceptStore.create(
            domain_id="causal_inference",
            name="To Delete",
            canonical_name=f"assumption_delete_{uuid4().hex[:8]}",
            concept_type=ConceptType.ASSUMPTION,
        )
        assumption = await AssumptionStore.create(concept_id=concept.id)

        deleted = await AssumptionStore.delete(assumption.id)

        assert deleted is True
        assert await AssumptionStore.get_by_id(assumption.id) is None

    async def test_delete_nonexistent(self, test_db):
        """Delete non-existent assumption returns False."""
        fake_id = uuid4()
        deleted = await AssumptionStore.delete(fake_id)
        assert deleted is False


class TestAssumptionStoreList:
    """Tests for AssumptionStore.list_all() and count()."""

    async def test_list_empty(self, test_db):
        """List with no assumptions returns empty list."""
        assumptions = await AssumptionStore.list_all()
        assert assumptions == []

    async def test_list_with_assumptions(self, test_db):
        """List returns all assumptions."""
        for i in range(3):
            concept = await ConceptStore.create(
                domain_id="causal_inference",
                name=f"Assumption {i}",
                canonical_name=f"assumption_{i}_{uuid4().hex[:8]}",
                concept_type=ConceptType.ASSUMPTION,
            )
            await AssumptionStore.create(concept_id=concept.id)

        assumptions = await AssumptionStore.list_all()

        assert len(assumptions) == 3

    async def test_list_pagination(self, test_db):
        """List respects limit and offset."""
        for i in range(5):
            concept = await ConceptStore.create(
                domain_id="causal_inference",
                name=f"Paginated Assumption {i}",
                canonical_name=f"assumption_page_{i}_{uuid4().hex[:8]}",
                concept_type=ConceptType.ASSUMPTION,
            )
            await AssumptionStore.create(concept_id=concept.id)

        page1 = await AssumptionStore.list_all(limit=2, offset=0)
        page2 = await AssumptionStore.list_all(limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].id != page2[0].id

    async def test_count_empty(self, test_db):
        """Count with no assumptions returns 0."""
        count = await AssumptionStore.count()
        assert count == 0

    async def test_count_with_assumptions(self, test_db):
        """Count returns correct number."""
        for i in range(4):
            concept = await ConceptStore.create(
                domain_id="causal_inference",
                name=f"Count Assumption {i}",
                canonical_name=f"assumption_count_{i}_{uuid4().hex[:8]}",
                concept_type=ConceptType.ASSUMPTION,
            )
            await AssumptionStore.create(concept_id=concept.id)

        count = await AssumptionStore.count()

        assert count == 4
