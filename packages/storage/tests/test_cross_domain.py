"""Tests for CrossDomainStore — cross-domain concept link operations.

Tests cover:
- discover_links: embedding similarity search, threshold filtering, result shape
- store_link: insert with ON CONFLICT, all params forwarded
- store_discovered_links: auto-classification (EQUIVALENT/ANALOGOUS/RELATED),
  min_similarity filtering, bulk storage
- get_cross_domain_concepts: direction filtering (source/target/both)
- get_stats: total, by_type, by_domain_pair
- CrossDomainLinkType constants
- Error propagation as StorageError
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from research_kb_common import StorageError
from research_kb_storage.cross_domain import (
    CrossDomainLinkType,
    CrossDomainStore,
)

pytestmark = pytest.mark.unit


def _make_mock_pool(conn_mock):
    """Create a mock connection pool wrapping the given connection mock."""
    pool = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn_mock)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool


class TestCrossDomainLinkTypeConstants:
    """Tests for CrossDomainLinkType constants."""

    def test_constants_are_strings(self):
        """All link types are non-empty strings."""
        types = [
            CrossDomainLinkType.EQUIVALENT,
            CrossDomainLinkType.ANALOGOUS,
            CrossDomainLinkType.RELATED,
        ]
        for t in types:
            assert isinstance(t, str)
            assert len(t) > 0

    def test_constants_are_uppercase(self):
        """Link type values are UPPERCASE for consistency."""
        assert CrossDomainLinkType.EQUIVALENT == "EQUIVALENT"
        assert CrossDomainLinkType.ANALOGOUS == "ANALOGOUS"
        assert CrossDomainLinkType.RELATED == "RELATED"

    def test_constants_are_unique(self):
        """All link types are distinct."""
        types = [
            CrossDomainLinkType.EQUIVALENT,
            CrossDomainLinkType.ANALOGOUS,
            CrossDomainLinkType.RELATED,
        ]
        assert len(set(types)) == len(types)


class TestDiscoverLinks:
    """Tests for CrossDomainStore.discover_links()."""

    @patch("research_kb_storage.cross_domain.get_connection_pool", new_callable=AsyncMock)
    async def test_discover_links_returns_matches(self, mock_get_pool):
        """discover_links returns list of link dicts with expected shape."""
        src_id = uuid4()
        tgt_id = uuid4()

        source_concepts = [
            {
                "id": src_id,
                "name": "instrumental variables",
                "canonical_name": "iv",
                "concept_type": "METHOD",
                "embedding": [0.1] * 1024,
            }
        ]
        matches = [
            {
                "id": tgt_id,
                "name": "random assignment",
                "canonical_name": "random_assignment",
                "concept_type": "METHOD",
                "similarity": 0.92,
            }
        ]

        conn = AsyncMock()
        conn.fetch = AsyncMock(side_effect=[source_concepts, matches])
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await CrossDomainStore.discover_links(
            source_domain="causal_inference",
            target_domain="statistics",
            similarity_threshold=0.85,
        )

        assert len(result) == 1
        link = result[0]
        assert link["source_concept_id"] == src_id
        assert link["target_concept_id"] == tgt_id
        assert link["source_name"] == "instrumental variables"
        assert link["target_name"] == "random assignment"
        assert link["similarity"] == pytest.approx(0.92)
        assert link["source_domain"] == "causal_inference"
        assert link["target_domain"] == "statistics"

    @patch("research_kb_storage.cross_domain.get_connection_pool", new_callable=AsyncMock)
    async def test_discover_links_no_matches(self, mock_get_pool):
        """Returns empty list when no similar concepts found."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(
            side_effect=[
                [
                    {
                        "id": uuid4(),
                        "name": "x",
                        "canonical_name": "x",
                        "concept_type": "METHOD",
                        "embedding": [0.1] * 1024,
                    }
                ],
                [],  # No matches
            ]
        )
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await CrossDomainStore.discover_links("a", "b")

        assert result == []

    @patch("research_kb_storage.cross_domain.get_connection_pool", new_callable=AsyncMock)
    async def test_discover_links_no_source_concepts(self, mock_get_pool):
        """Returns empty when source domain has no embeddable concepts."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await CrossDomainStore.discover_links("empty_domain", "target")

        assert result == []

    @patch("research_kb_storage.cross_domain.get_connection_pool", new_callable=AsyncMock)
    async def test_discover_links_wraps_errors(self, mock_get_pool):
        """Database errors wrapped in StorageError."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(side_effect=RuntimeError("db error"))
        mock_get_pool.return_value = _make_mock_pool(conn)

        with pytest.raises(StorageError, match="Failed to discover"):
            await CrossDomainStore.discover_links("a", "b")


class TestStoreLink:
    """Tests for CrossDomainStore.store_link()."""

    @patch("research_kb_storage.cross_domain.get_connection_pool", new_callable=AsyncMock)
    async def test_store_link_returns_uuid(self, mock_get_pool):
        """store_link returns UUID of created/updated link."""
        expected_id = str(uuid4())
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=expected_id)
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await CrossDomainStore.store_link(
            source_concept_id=uuid4(),
            target_concept_id=uuid4(),
            link_type=CrossDomainLinkType.EQUIVALENT,
            confidence_score=0.95,
        )

        assert isinstance(result, UUID)
        assert str(result) == expected_id

    @patch("research_kb_storage.cross_domain.get_connection_pool", new_callable=AsyncMock)
    async def test_store_link_with_evidence_and_metadata(self, mock_get_pool):
        """Evidence and metadata are forwarded to SQL."""
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=str(uuid4()))
        mock_get_pool.return_value = _make_mock_pool(conn)

        await CrossDomainStore.store_link(
            source_concept_id=uuid4(),
            target_concept_id=uuid4(),
            link_type=CrossDomainLinkType.ANALOGOUS,
            confidence_score=0.88,
            evidence="Both refer to treatment effects",
            metadata={"discovery_method": "manual"},
        )

        args = conn.fetchval.call_args[0]
        assert "Both refer to treatment effects" in args
        assert {"discovery_method": "manual"} in args

    @patch("research_kb_storage.cross_domain.get_connection_pool", new_callable=AsyncMock)
    async def test_store_link_uses_upsert(self, mock_get_pool):
        """SQL uses ON CONFLICT DO UPDATE for idempotency."""
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=str(uuid4()))
        mock_get_pool.return_value = _make_mock_pool(conn)

        await CrossDomainStore.store_link(
            source_concept_id=uuid4(),
            target_concept_id=uuid4(),
            link_type=CrossDomainLinkType.RELATED,
            confidence_score=0.80,
        )

        sql = conn.fetchval.call_args[0][0]
        assert "ON CONFLICT" in sql

    @patch("research_kb_storage.cross_domain.get_connection_pool", new_callable=AsyncMock)
    async def test_store_link_wraps_errors(self, mock_get_pool):
        """Database errors wrapped in StorageError."""
        conn = AsyncMock()
        conn.fetchval = AsyncMock(side_effect=RuntimeError("db error"))
        mock_get_pool.return_value = _make_mock_pool(conn)

        with pytest.raises(StorageError, match="Failed to store cross-domain link"):
            await CrossDomainStore.store_link(
                source_concept_id=uuid4(),
                target_concept_id=uuid4(),
                link_type=CrossDomainLinkType.RELATED,
                confidence_score=0.5,
            )


class TestStoreDiscoveredLinks:
    """Tests for CrossDomainStore.store_discovered_links()."""

    @patch("research_kb_storage.cross_domain.get_connection_pool", new_callable=AsyncMock)
    async def test_classifies_equivalent_above_095(self, mock_get_pool):
        """Links with similarity >= 0.95 classified as EQUIVALENT."""
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=str(uuid4()))
        mock_get_pool.return_value = _make_mock_pool(conn)

        links = [
            {
                "source_concept_id": uuid4(),
                "target_concept_id": uuid4(),
                "similarity": 0.97,
                "source_domain": "a",
                "target_domain": "b",
                "source_name": "X",
                "target_name": "Y",
            }
        ]

        stored = await CrossDomainStore.store_discovered_links(links, min_similarity=0.90)

        assert stored == 1
        args = conn.fetchval.call_args[0]
        assert CrossDomainLinkType.EQUIVALENT in args

    @patch("research_kb_storage.cross_domain.get_connection_pool", new_callable=AsyncMock)
    async def test_classifies_analogous_090_to_095(self, mock_get_pool):
        """Links with 0.90 <= similarity < 0.95 classified as ANALOGOUS."""
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=str(uuid4()))
        mock_get_pool.return_value = _make_mock_pool(conn)

        links = [
            {
                "source_concept_id": uuid4(),
                "target_concept_id": uuid4(),
                "similarity": 0.92,
                "source_domain": "a",
                "target_domain": "b",
                "source_name": "X",
                "target_name": "Y",
            }
        ]

        stored = await CrossDomainStore.store_discovered_links(links, min_similarity=0.90)

        assert stored == 1
        args = conn.fetchval.call_args[0]
        assert CrossDomainLinkType.ANALOGOUS in args

    @patch("research_kb_storage.cross_domain.get_connection_pool", new_callable=AsyncMock)
    async def test_filters_below_min_similarity(self, mock_get_pool):
        """Links below min_similarity are skipped."""
        conn = AsyncMock()
        mock_get_pool.return_value = _make_mock_pool(conn)

        links = [
            {
                "source_concept_id": uuid4(),
                "target_concept_id": uuid4(),
                "similarity": 0.85,
                "source_domain": "a",
                "target_domain": "b",
                "source_name": "X",
                "target_name": "Y",
            }
        ]

        stored = await CrossDomainStore.store_discovered_links(links, min_similarity=0.90)

        assert stored == 0

    @patch("research_kb_storage.cross_domain.get_connection_pool", new_callable=AsyncMock)
    async def test_continues_on_individual_failures(self, mock_get_pool):
        """Individual StorageErrors don't stop batch processing."""
        call_count = 0

        async def mock_fetchval(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("db error")
            return str(uuid4())

        conn = AsyncMock()
        conn.fetchval = mock_fetchval
        mock_get_pool.return_value = _make_mock_pool(conn)

        links = [
            {
                "source_concept_id": uuid4(),
                "target_concept_id": uuid4(),
                "similarity": 0.96,
                "source_domain": "a",
                "target_domain": "b",
                "source_name": "X1",
                "target_name": "Y1",
            },
            {
                "source_concept_id": uuid4(),
                "target_concept_id": uuid4(),
                "similarity": 0.93,
                "source_domain": "a",
                "target_domain": "b",
                "source_name": "X2",
                "target_name": "Y2",
            },
        ]

        # First link fails, second succeeds
        stored = await CrossDomainStore.store_discovered_links(links, min_similarity=0.90)

        assert stored == 1


class TestGetCrossDomainConcepts:
    """Tests for CrossDomainStore.get_cross_domain_concepts()."""

    @patch("research_kb_storage.cross_domain.get_connection_pool", new_callable=AsyncMock)
    async def test_direction_both_queries_both(self, mock_get_pool):
        """direction='both' executes both source and target queries."""
        concept_id = uuid4()
        source_links = [
            {
                "link_id": uuid4(),
                "link_type": "EQUIVALENT",
                "confidence_score": 0.95,
                "evidence": "test",
                "metadata": {},
                "linked_concept_id": uuid4(),
                "linked_concept_name": "Target Concept",
                "linked_canonical": "target_concept",
                "linked_type": "METHOD",
                "linked_domain": "time_series",
                "direction": "target",
            }
        ]
        target_links = [
            {
                "link_id": uuid4(),
                "link_type": "ANALOGOUS",
                "confidence_score": 0.90,
                "evidence": "test2",
                "metadata": {},
                "linked_concept_id": uuid4(),
                "linked_concept_name": "Source Concept",
                "linked_canonical": "source_concept",
                "linked_type": "ASSUMPTION",
                "linked_domain": "statistics",
                "direction": "source",
            }
        ]

        conn = AsyncMock()
        conn.fetch = AsyncMock(side_effect=[source_links, target_links])
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await CrossDomainStore.get_cross_domain_concepts(concept_id, direction="both")

        assert len(result) == 2
        assert conn.fetch.call_count == 2

    @patch("research_kb_storage.cross_domain.get_connection_pool", new_callable=AsyncMock)
    async def test_direction_source_only(self, mock_get_pool):
        """direction='source' only queries outgoing links."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        mock_get_pool.return_value = _make_mock_pool(conn)

        await CrossDomainStore.get_cross_domain_concepts(uuid4(), direction="source")

        assert conn.fetch.call_count == 1
        sql = conn.fetch.call_args[0][0]
        assert "source_concept_id = $1" in sql

    @patch("research_kb_storage.cross_domain.get_connection_pool", new_callable=AsyncMock)
    async def test_direction_target_only(self, mock_get_pool):
        """direction='target' only queries incoming links."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        mock_get_pool.return_value = _make_mock_pool(conn)

        await CrossDomainStore.get_cross_domain_concepts(uuid4(), direction="target")

        assert conn.fetch.call_count == 1
        sql = conn.fetch.call_args[0][0]
        assert "target_concept_id = $1" in sql


class TestGetCrossDomainStats:
    """Tests for CrossDomainStore.get_stats()."""

    @patch("research_kb_storage.cross_domain.get_connection_pool", new_callable=AsyncMock)
    async def test_get_stats_returns_structure(self, mock_get_pool):
        """get_stats returns dict with total, by_type, by_domain_pair."""
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=42)
        conn.fetch = AsyncMock(
            side_effect=[
                # type_counts
                [
                    {"link_type": "EQUIVALENT", "count": 20},
                    {"link_type": "ANALOGOUS", "count": 15},
                    {"link_type": "RELATED", "count": 7},
                ],
                # domain_counts
                [
                    {
                        "source_domain": "causal_inference",
                        "target_domain": "time_series",
                        "count": 25,
                    },
                    {
                        "source_domain": "causal_inference",
                        "target_domain": "statistics",
                        "count": 17,
                    },
                ],
            ]
        )
        mock_get_pool.return_value = _make_mock_pool(conn)

        stats = await CrossDomainStore.get_stats()

        assert stats["total_links"] == 42
        assert stats["by_type"] == {"EQUIVALENT": 20, "ANALOGOUS": 15, "RELATED": 7}
        assert len(stats["by_domain_pair"]) == 2
        assert stats["by_domain_pair"][0]["source"] == "causal_inference"
        assert stats["by_domain_pair"][0]["count"] == 25

    @patch("research_kb_storage.cross_domain.get_connection_pool", new_callable=AsyncMock)
    async def test_get_stats_handles_empty(self, mock_get_pool):
        """Empty table returns zero/empty structures."""
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=None)
        conn.fetch = AsyncMock(side_effect=[[], []])
        mock_get_pool.return_value = _make_mock_pool(conn)

        stats = await CrossDomainStore.get_stats()

        assert stats["total_links"] == 0
        assert stats["by_type"] == {}
        assert stats["by_domain_pair"] == []
