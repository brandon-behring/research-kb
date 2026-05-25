"""Tests for SourceStore role helpers + idempotent dedup lookup.

Covers the new methods added for the A4 anchor-classics flow:

- ``SourceStore.create(..., roles=[...])`` — packs roles into ``metadata.roles``
- ``SourceStore.add_roles(source_id, [...])``
- ``SourceStore.remove_roles(source_id, [...])``
- ``SourceStore.list_by_role(role)``
- ``SourceStore.find_by_title_year(title, year)`` — used for idempotent
  ``sources add-manual``

Integration tests; require the PostgreSQL container to be running.
"""

import pytest

from research_kb_contracts import SourceType
from research_kb_storage import SourceStore

pytestmark = pytest.mark.integration


class TestCreateWithRoles:
    """``SourceStore.create(..., roles=[...])`` packs into ``metadata.roles``."""

    async def test_create_with_single_role(self, db_pool):
        source = await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="Test anchor with role",
            file_hash="sha256:test_roles_single",
            roles=["anchor.rl_optimal_control"],
        )

        assert source.metadata["roles"] == ["anchor.rl_optimal_control"]

    async def test_create_with_multiple_roles_deduplicated(self, db_pool):
        source = await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="Test anchor multi role",
            file_hash="sha256:test_roles_multi",
            roles=[
                "anchor.rl_optimal_control",
                "anchor.rl_optimal_control",  # duplicate
                "quality.peer_reviewed",
            ],
        )

        assert source.metadata["roles"] == [
            "anchor.rl_optimal_control",
            "quality.peer_reviewed",
        ]

    async def test_create_without_roles_omits_key(self, db_pool):
        """When ``roles`` is not supplied, ``metadata.roles`` is not set."""
        source = await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="Test no roles",
            file_hash="sha256:test_roles_none",
        )

        assert "roles" not in source.metadata

    async def test_create_with_roles_and_other_metadata(self, db_pool):
        """Roles merge into provided metadata, not replace it."""
        source = await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="Test combined",
            file_hash="sha256:test_roles_combined",
            metadata={"isbn": "978-test-isbn", "kind": "thesis"},
            roles=["anchor.rl_optimal_control"],
        )

        assert source.metadata["isbn"] == "978-test-isbn"
        assert source.metadata["kind"] == "thesis"
        assert source.metadata["roles"] == ["anchor.rl_optimal_control"]


class TestAddRoles:
    """``SourceStore.add_roles`` appends roles with dedup."""

    async def test_add_to_source_without_prior_roles(self, db_pool):
        created = await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="Add roles to fresh source",
            file_hash="sha256:test_add_roles_fresh",
        )

        updated = await SourceStore.add_roles(created.id, ["anchor.rl_optimal_control"])

        assert updated.metadata["roles"] == ["anchor.rl_optimal_control"]

    async def test_add_preserves_existing_roles(self, db_pool):
        created = await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="Add preserves",
            file_hash="sha256:test_add_preserves",
            roles=["anchor.rl_optimal_control"],
        )

        updated = await SourceStore.add_roles(created.id, ["quality.peer_reviewed"])

        assert "anchor.rl_optimal_control" in updated.metadata["roles"]
        assert "quality.peer_reviewed" in updated.metadata["roles"]

    async def test_add_existing_role_is_idempotent(self, db_pool):
        created = await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="Add idempotent",
            file_hash="sha256:test_add_idempotent",
            roles=["anchor.rl_optimal_control"],
        )

        updated = await SourceStore.add_roles(created.id, ["anchor.rl_optimal_control"])

        assert updated.metadata["roles"] == ["anchor.rl_optimal_control"]


class TestRemoveRoles:
    """``SourceStore.remove_roles`` filters out specified roles."""

    async def test_remove_existing_role(self, db_pool):
        created = await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="Remove existing",
            file_hash="sha256:test_remove_existing",
            roles=["anchor.rl_optimal_control", "quality.peer_reviewed"],
        )

        updated = await SourceStore.remove_roles(created.id, ["anchor.rl_optimal_control"])

        assert updated.metadata["roles"] == ["quality.peer_reviewed"]

    async def test_remove_missing_role_is_noop(self, db_pool):
        """Removing a role that isn't there leaves existing roles intact."""
        created = await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="Remove missing",
            file_hash="sha256:test_remove_missing",
            roles=["anchor.rl_optimal_control"],
        )

        updated = await SourceStore.remove_roles(created.id, ["nonexistent"])

        assert updated.metadata["roles"] == ["anchor.rl_optimal_control"]

    async def test_remove_all_leaves_empty_list(self, db_pool):
        created = await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="Remove all",
            file_hash="sha256:test_remove_all",
            roles=["anchor.rl_optimal_control"],
        )

        updated = await SourceStore.remove_roles(created.id, ["anchor.rl_optimal_control"])

        # Key stays, value is empty list (downstream relies on the shape).
        assert updated.metadata["roles"] == []


class TestListByRole:
    """``SourceStore.list_by_role`` queries via JSONB containment."""

    async def test_returns_matching_sources(self, db_pool):
        tagged = await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="Tagged by role",
            file_hash="sha256:test_list_by_role_tagged",
            roles=["anchor.rl_optimal_control"],
        )

        # An untagged source must NOT appear in the result.
        await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="Untagged",
            file_hash="sha256:test_list_by_role_untagged",
        )

        results = await SourceStore.list_by_role("anchor.rl_optimal_control")
        result_ids = {s.id for s in results}

        assert tagged.id in result_ids

    async def test_returns_empty_for_unknown_role(self, db_pool):
        results = await SourceStore.list_by_role("anchor.does_not_exist_anywhere")

        assert results == []


class TestFindByTitleYear:
    """``SourceStore.find_by_title_year`` powers idempotent ``add-manual``."""

    async def test_exact_match_returns_source(self, db_pool):
        created = await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="Dynamic Programming (find-by test)",
            file_hash="sha256:test_find_exact",
            year=1957,
        )

        found = await SourceStore.find_by_title_year("Dynamic Programming (find-by test)", 1957)

        assert found is not None
        assert found.id == created.id

    async def test_normalizes_whitespace_and_case(self, db_pool):
        created = await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="Title With Spaces (find-by test)",
            file_hash="sha256:test_find_normalized",
            year=2019,
        )

        # Extra internal whitespace + different case still resolves.
        found = await SourceStore.find_by_title_year(
            "  title  with   spaces (FIND-BY TEST)  ", 2019
        )

        assert found is not None
        assert found.id == created.id

    async def test_year_mismatch_returns_none(self, db_pool):
        await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="Year mismatch test",
            file_hash="sha256:test_find_year_mismatch",
            year=2020,
        )

        found = await SourceStore.find_by_title_year("Year mismatch test", 1999)

        assert found is None

    async def test_year_none_returns_none(self, db_pool):
        """Conservatively refuse to dedup against a year-less candidate."""
        await SourceStore.create(
            domain_id="causal_inference",
            source_type=SourceType.TEXTBOOK,
            title="Year none test",
            file_hash="sha256:test_find_year_none",
            year=2020,
        )

        found = await SourceStore.find_by_title_year("Year none test", None)

        assert found is None

    async def test_no_match_returns_none(self, db_pool):
        found = await SourceStore.find_by_title_year("Title that does not exist anywhere", 2020)

        assert found is None
