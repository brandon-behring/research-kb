"""Tests for DomainStore — CRUD operations for domains table.

Tests cover:
- list_all: returns Domain models, filters by active status
- get_by_id: returns Domain or None
- create: full params, default weights, duplicate handling
- update_config: JSONB merge, not-found handling
- get_stats: joins across sources/chunks/concepts
- get_all_stats: returns list of stats for all domains
- _row_to_domain: conversion correctness
- Error propagation as StorageError
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_kb_common import StorageError
from research_kb_contracts import Domain
from research_kb_storage.domain_store import DomainStore, _row_to_domain

pytestmark = pytest.mark.unit


def _make_mock_pool(conn_mock):
    """Create a mock connection pool wrapping the given connection mock."""
    pool = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn_mock)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool


def _make_domain_row(**overrides):
    """Create a mock database row for a domain."""
    defaults = {
        "id": "causal_inference",
        "name": "Causal Inference",
        "description": "Methods for causal reasoning",
        "config": {"extraction_prompt_version": "v1"},
        "concept_types": ["method", "assumption"],
        "relationship_types": ["requires", "uses"],
        "default_fts_weight": 0.3,
        "default_vector_weight": 0.7,
        "default_graph_weight": 0.1,
        "default_citation_weight": 0.15,
        "status": "active",
        "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    row = MagicMock()
    row.__getitem__ = lambda self, k: defaults[k]
    row.keys = MagicMock(return_value=defaults.keys())
    return row


class TestRowToDomain:
    """Tests for _row_to_domain helper."""

    def test_converts_full_row(self):
        """Full row maps to Domain model with all fields."""
        row = _make_domain_row()
        domain = _row_to_domain(row)

        assert isinstance(domain, Domain)
        assert domain.id == "causal_inference"
        assert domain.name == "Causal Inference"
        assert domain.description == "Methods for causal reasoning"
        assert domain.config == {"extraction_prompt_version": "v1"}
        assert domain.concept_types == ["method", "assumption"]
        assert domain.default_fts_weight == 0.3
        assert domain.default_vector_weight == 0.7
        assert domain.status == "active"

    def test_handles_null_config(self):
        """Null config/concept_types/relationship_types default to empty."""
        row = _make_domain_row(
            config=None,
            concept_types=None,
            relationship_types=None,
        )
        domain = _row_to_domain(row)

        assert domain.config == {}
        assert domain.concept_types == []
        assert domain.relationship_types == []


class TestListAll:
    """Tests for DomainStore.list_all()."""

    @patch("research_kb_storage.domain_store.get_connection_pool", new_callable=AsyncMock)
    async def test_list_all_returns_domain_models(self, mock_get_pool):
        """list_all converts rows to Domain models."""
        rows = [
            _make_domain_row(id="causal_inference", name="Causal Inference"),
            _make_domain_row(id="time_series", name="Time Series"),
        ]
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=rows)
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await DomainStore.list_all()

        assert len(result) == 2
        assert all(isinstance(d, Domain) for d in result)
        assert result[0].id == "causal_inference"
        assert result[1].id == "time_series"

    @patch("research_kb_storage.domain_store.get_connection_pool", new_callable=AsyncMock)
    async def test_list_all_filters_active(self, mock_get_pool):
        """SQL includes WHERE status = 'active'."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        mock_get_pool.return_value = _make_mock_pool(conn)

        await DomainStore.list_all()

        sql = conn.fetch.call_args[0][0]
        assert "status = 'active'" in sql

    @patch("research_kb_storage.domain_store.get_connection_pool", new_callable=AsyncMock)
    async def test_list_all_wraps_errors(self, mock_get_pool):
        """Database errors wrapped in StorageError."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(side_effect=RuntimeError("connection refused"))
        mock_get_pool.return_value = _make_mock_pool(conn)

        with pytest.raises(StorageError, match="Failed to list domains"):
            await DomainStore.list_all()


class TestGetById:
    """Tests for DomainStore.get_by_id()."""

    @patch("research_kb_storage.domain_store.get_connection_pool", new_callable=AsyncMock)
    async def test_get_by_id_returns_domain(self, mock_get_pool):
        """Returns Domain when found."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=_make_domain_row())
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await DomainStore.get_by_id("causal_inference")

        assert isinstance(result, Domain)
        assert result.id == "causal_inference"

    @patch("research_kb_storage.domain_store.get_connection_pool", new_callable=AsyncMock)
    async def test_get_by_id_returns_none_when_missing(self, mock_get_pool):
        """Returns None when domain not found."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await DomainStore.get_by_id("nonexistent")

        assert result is None

    @patch("research_kb_storage.domain_store.get_connection_pool", new_callable=AsyncMock)
    async def test_get_by_id_passes_domain_id(self, mock_get_pool):
        """SQL query uses the domain_id parameter."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        mock_get_pool.return_value = _make_mock_pool(conn)

        await DomainStore.get_by_id("time_series")

        args = conn.fetchrow.call_args[0]
        assert "time_series" in args


class TestCreate:
    """Tests for DomainStore.create()."""

    @patch("research_kb_storage.domain_store.get_connection_pool", new_callable=AsyncMock)
    async def test_create_returns_domain(self, mock_get_pool):
        """create returns Domain model from RETURNING * row."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=_make_domain_row(id="new_domain", name="New Domain"))
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await DomainStore.create(
            domain_id="new_domain",
            name="New Domain",
            description="A new domain",
        )

        assert isinstance(result, Domain)
        assert result.id == "new_domain"
        assert result.name == "New Domain"

    @patch("research_kb_storage.domain_store.get_connection_pool", new_callable=AsyncMock)
    async def test_create_passes_default_weights(self, mock_get_pool):
        """Default search weights are passed to SQL INSERT."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=_make_domain_row())
        mock_get_pool.return_value = _make_mock_pool(conn)

        await DomainStore.create(domain_id="test", name="Test")

        args = conn.fetchrow.call_args[0]
        # Check default weights are in the SQL params
        # Positions: $7=fts, $8=vector, $9=graph, $10=citation
        assert 0.3 in args  # default_fts_weight
        assert 0.7 in args  # default_vector_weight
        assert 0.1 in args  # default_graph_weight
        assert 0.15 in args  # default_citation_weight

    @patch("research_kb_storage.domain_store.get_connection_pool", new_callable=AsyncMock)
    async def test_create_duplicate_raises_storage_error(self, mock_get_pool):
        """Duplicate domain_id raises StorageError."""
        import asyncpg

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(side_effect=asyncpg.UniqueViolationError("duplicate key"))
        mock_get_pool.return_value = _make_mock_pool(conn)

        with pytest.raises(StorageError, match="already exists"):
            await DomainStore.create(domain_id="causal_inference", name="CI")

    @patch("research_kb_storage.domain_store.get_connection_pool", new_callable=AsyncMock)
    async def test_create_defaults_config_to_empty(self, mock_get_pool):
        """None config/concept_types/relationship_types default to empty."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=_make_domain_row())
        mock_get_pool.return_value = _make_mock_pool(conn)

        await DomainStore.create(domain_id="test", name="Test")

        args = conn.fetchrow.call_args[0]
        # config ($4) defaults to {}, concept_types ($5) to [], relationship_types ($6) to []
        assert {} in args
        assert [] in args


class TestUpdateConfig:
    """Tests for DomainStore.update_config()."""

    @patch("research_kb_storage.domain_store.get_connection_pool", new_callable=AsyncMock)
    async def test_update_config_returns_updated_domain(self, mock_get_pool):
        """update_config returns updated Domain."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=_make_domain_row(config={"key": "new_value"}))
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await DomainStore.update_config(
            domain_id="causal_inference",
            config={"key": "new_value"},
        )

        assert isinstance(result, Domain)
        assert result.config == {"key": "new_value"}

    @patch("research_kb_storage.domain_store.get_connection_pool", new_callable=AsyncMock)
    async def test_update_config_not_found_raises_storage_error(self, mock_get_pool):
        """Raises StorageError when domain not found."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        mock_get_pool.return_value = _make_mock_pool(conn)

        with pytest.raises(StorageError, match="Domain not found"):
            await DomainStore.update_config("nonexistent", {"key": "val"})

    @patch("research_kb_storage.domain_store.get_connection_pool", new_callable=AsyncMock)
    async def test_update_config_uses_jsonb_merge(self, mock_get_pool):
        """SQL uses JSONB || operator for merge."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=_make_domain_row())
        mock_get_pool.return_value = _make_mock_pool(conn)

        await DomainStore.update_config("causal_inference", {"v": 2})

        sql = conn.fetchrow.call_args[0][0]
        assert "config || $1" in sql or "config = config || $1" in sql


class TestGetStats:
    """Tests for DomainStore.get_stats()."""

    @patch("research_kb_storage.domain_store.get_connection_pool", new_callable=AsyncMock)
    async def test_get_stats_returns_counts(self, mock_get_pool):
        """get_stats returns dictionary with counts."""
        stats_row = {
            "domain_id": "causal_inference",
            "name": "Causal Inference",
            "source_count": 299,
            "chunk_count": 50000,
            "concept_count": 150000,
        }
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=stats_row)
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await DomainStore.get_stats("causal_inference")

        assert result["domain_id"] == "causal_inference"
        assert result["source_count"] == 299
        assert result["chunk_count"] == 50000
        assert result["concept_count"] == 150000

    @patch("research_kb_storage.domain_store.get_connection_pool", new_callable=AsyncMock)
    async def test_get_stats_returns_none_when_not_found(self, mock_get_pool):
        """Returns None for nonexistent domain."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await DomainStore.get_stats("nonexistent")

        assert result is None


class TestGetAllStats:
    """Tests for DomainStore.get_all_stats()."""

    @patch("research_kb_storage.domain_store.get_connection_pool", new_callable=AsyncMock)
    async def test_get_all_stats_returns_list(self, mock_get_pool):
        """get_all_stats returns list of stat dicts."""
        rows = [
            {
                "domain_id": "ci",
                "name": "CI",
                "source_count": 100,
                "chunk_count": 5000,
                "concept_count": 10000,
            },
            {
                "domain_id": "ts",
                "name": "TS",
                "source_count": 50,
                "chunk_count": 2000,
                "concept_count": 5000,
            },
        ]
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=rows)
        mock_get_pool.return_value = _make_mock_pool(conn)

        result = await DomainStore.get_all_stats()

        assert len(result) == 2
        assert result[0]["domain_id"] == "ci"
        assert result[1]["source_count"] == 50

    @patch("research_kb_storage.domain_store.get_connection_pool", new_callable=AsyncMock)
    async def test_get_all_stats_wraps_errors(self, mock_get_pool):
        """Database errors wrapped in StorageError."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(side_effect=RuntimeError("db error"))
        mock_get_pool.return_value = _make_mock_pool(conn)

        with pytest.raises(StorageError, match="Failed to get domain stats"):
            await DomainStore.get_all_stats()
