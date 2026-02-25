"""Tests for daemon connection pool management.

Tests cover:
- init_pool / get_pool / close_pool lifecycle
- EmbedClient: embed_query, health_check, circuit breaker, concurrency
- get_embed_client singleton
- RerankClientWrapper: health_check
- get_rerank_client singleton
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from research_kb_daemon.pool import (
    EmbedClient,
    RerankClientWrapper,
    close_pool,
    get_embed_client,
    get_pool,
    get_rerank_client,
    init_pool,
)

pytestmark = pytest.mark.unit


class TestPoolLifecycle:
    """Tests for init_pool / get_pool / close_pool."""

    @patch("research_kb_daemon.pool.get_connection_pool")
    async def test_init_pool_creates_pool(self, mock_get_pool):
        """init_pool creates a connection pool via get_connection_pool."""
        import research_kb_daemon.pool as pool_mod

        pool_mod._pool = None  # Reset singleton
        mock_get_pool.return_value = AsyncMock()

        await init_pool()

        mock_get_pool.assert_called_once()
        assert pool_mod._pool is not None

        # Cleanup
        pool_mod._pool = None

    @patch("research_kb_daemon.pool.get_connection_pool")
    async def test_init_pool_idempotent(self, mock_get_pool):
        """Calling init_pool twice doesn't create second pool."""
        import research_kb_daemon.pool as pool_mod

        pool_mod._pool = None
        mock_get_pool.return_value = AsyncMock()

        await init_pool()
        await init_pool()

        mock_get_pool.assert_called_once()

        # Cleanup
        pool_mod._pool = None

    async def test_get_pool_raises_if_not_initialized(self):
        """get_pool raises RuntimeError before init_pool."""
        import research_kb_daemon.pool as pool_mod

        original = pool_mod._pool
        pool_mod._pool = None

        with pytest.raises(RuntimeError, match="not initialized"):
            await get_pool()

        pool_mod._pool = original

    @patch("research_kb_daemon.pool.close_connection_pool")
    async def test_close_pool_resets_singleton(self, mock_close):
        """close_pool sets _pool to None."""
        import research_kb_daemon.pool as pool_mod

        pool_mod._pool = AsyncMock()
        mock_close.return_value = None

        await close_pool()

        assert pool_mod._pool is None


class TestEmbedClient:
    """Tests for EmbedClient."""

    def test_init_sets_socket_path(self):
        """EmbedClient stores socket path."""
        client = EmbedClient("/tmp/test.sock")
        assert client.socket_path == "/tmp/test.sock"

    def test_init_creates_semaphore(self):
        """EmbedClient creates concurrency semaphore."""
        client = EmbedClient()
        assert isinstance(client._semaphore, asyncio.Semaphore)

    def test_max_concurrent_embeds(self):
        """MAX_CONCURRENT_EMBEDS is a reasonable value."""
        assert EmbedClient.MAX_CONCURRENT_EMBEDS == 3

    def test_embed_timeout(self):
        """EMBED_TIMEOUT is set for fast failure."""
        assert EmbedClient.EMBED_TIMEOUT == 5.0

    def test_sync_request_connection_error(self):
        """_sync_request raises ConnectionError for bad socket."""
        client = EmbedClient("/tmp/nonexistent.sock")

        with pytest.raises(ConnectionError):
            client._sync_request({"action": "ping"}, timeout=1.0)

    def test_circuit_breaker_initially_closed(self):
        """Circuit breaker starts closed."""
        client = EmbedClient()
        assert not client._is_circuit_open()

    def test_circuit_breaker_opens_after_3_failures(self):
        """Circuit opens after 3 consecutive failures."""
        client = EmbedClient()

        client._record_failure()
        assert not client._is_circuit_open()
        client._record_failure()
        assert not client._is_circuit_open()
        client._record_failure()
        assert client._is_circuit_open()

    def test_success_resets_failure_count(self):
        """Recording success resets the consecutive failure counter."""
        client = EmbedClient()

        client._record_failure()
        client._record_failure()
        client._record_success()
        client._record_failure()
        # Only 1 failure since reset, should still be closed
        assert not client._is_circuit_open()

    @patch.object(EmbedClient, "_sync_request")
    async def test_embed_query_returns_embedding(self, mock_request):
        """embed_query returns 1024-dim vector on success."""
        mock_request.return_value = {"embedding": [0.1] * 1024}
        client = EmbedClient()

        result = await client.embed_query("test query")

        assert len(result) == 1024
        assert result[0] == pytest.approx(0.1)

    @patch.object(EmbedClient, "_sync_request")
    async def test_embed_query_validates_dimension(self, mock_request):
        """embed_query raises ValueError for wrong dimension."""
        mock_request.return_value = {"embedding": [0.1] * 512}  # Wrong dim
        client = EmbedClient()

        with pytest.raises(ValueError, match="Invalid embedding dimension"):
            await client.embed_query("test")

    @patch.object(EmbedClient, "_sync_request")
    async def test_embed_query_handles_error_response(self, mock_request):
        """embed_query raises ValueError on server error."""
        mock_request.return_value = {"error": "model not loaded"}
        client = EmbedClient()

        with pytest.raises(ValueError, match="Embedding failed"):
            await client.embed_query("test")

    async def test_embed_query_fails_fast_with_open_circuit(self):
        """embed_query raises RuntimeError when circuit is open."""
        client = EmbedClient()
        # Force circuit open
        client._record_failure()
        client._record_failure()
        client._record_failure()

        with pytest.raises(RuntimeError, match="circuit breaker open"):
            await client.embed_query("test")

    @patch.object(EmbedClient, "_sync_request")
    async def test_health_check_healthy(self, mock_request):
        """health_check returns healthy status."""
        mock_request.return_value = {
            "status": "ok",
            "device": "cuda",
            "model": "BAAI/bge-large-en-v1.5",
            "dim": 1024,
        }
        client = EmbedClient()

        result = await client.health_check()

        assert result["status"] == "healthy"
        assert result["device"] == "cuda"
        assert result["model"] == "BAAI/bge-large-en-v1.5"
        assert result["dim"] == 1024

    @patch.object(EmbedClient, "_sync_request")
    async def test_health_check_error_response(self, mock_request):
        """health_check returns unhealthy on error response."""
        mock_request.return_value = {"error": "model loading failed"}
        client = EmbedClient()

        result = await client.health_check()

        assert result["status"] == "unhealthy"
        assert "model loading failed" in result["error"]

    @patch.object(EmbedClient, "_sync_request")
    async def test_health_check_connection_failure(self, mock_request):
        """health_check returns unhealthy on connection error."""
        mock_request.side_effect = ConnectionError("no server")
        client = EmbedClient()

        result = await client.health_check()

        assert result["status"] == "unhealthy"


class TestGetEmbedClient:
    """Tests for get_embed_client singleton."""

    def test_returns_embed_client(self):
        """get_embed_client returns EmbedClient instance."""
        import research_kb_daemon.pool as pool_mod

        pool_mod._embed_client = None  # Reset singleton

        client = get_embed_client("/tmp/test.sock")

        assert isinstance(client, EmbedClient)
        assert client.socket_path == "/tmp/test.sock"

        # Cleanup
        pool_mod._embed_client = None

    def test_returns_same_instance(self):
        """get_embed_client is a singleton."""
        import research_kb_daemon.pool as pool_mod

        pool_mod._embed_client = None

        c1 = get_embed_client()
        c2 = get_embed_client()

        assert c1 is c2

        # Cleanup
        pool_mod._embed_client = None


class TestRerankClientWrapper:
    """Tests for RerankClientWrapper."""

    def test_init_sets_socket_path(self):
        """RerankClientWrapper stores socket path."""
        client = RerankClientWrapper("/tmp/rerank_test.sock")
        assert client.socket_path == "/tmp/rerank_test.sock"

    def test_sync_ping_connection_error(self):
        """_sync_ping raises ConnectionError for bad socket."""
        client = RerankClientWrapper("/tmp/nonexistent.sock")

        with pytest.raises(ConnectionError):
            client._sync_ping()

    @patch.object(RerankClientWrapper, "_sync_ping")
    async def test_health_check_healthy(self, mock_ping):
        """health_check returns healthy for ok response."""
        mock_ping.return_value = {
            "status": "ok",
            "model": "ms-marco-MiniLM-L-6-v2",
            "device": "cuda",
        }
        client = RerankClientWrapper()

        result = await client.health_check()

        assert result["status"] == "healthy"
        assert result["model"] == "ms-marco-MiniLM-L-6-v2"

    @patch.object(RerankClientWrapper, "_sync_ping")
    async def test_health_check_unhealthy(self, mock_ping):
        """health_check returns unhealthy on connection failure."""
        mock_ping.side_effect = ConnectionError("no server")
        client = RerankClientWrapper()

        result = await client.health_check()

        assert result["status"] == "unhealthy"


class TestGetRerankClient:
    """Tests for get_rerank_client singleton."""

    def test_returns_rerank_client(self):
        """get_rerank_client returns RerankClientWrapper."""
        import research_kb_daemon.pool as pool_mod

        pool_mod._rerank_client = None

        client = get_rerank_client("/tmp/test.sock")

        assert isinstance(client, RerankClientWrapper)

        pool_mod._rerank_client = None

    def test_returns_same_instance(self):
        """get_rerank_client is a singleton."""
        import research_kb_daemon.pool as pool_mod

        pool_mod._rerank_client = None

        c1 = get_rerank_client()
        c2 = get_rerank_client()

        assert c1 is c2

        pool_mod._rerank_client = None
