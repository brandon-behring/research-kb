"""Connection pool management for daemon.

Manages:
- Database connection pool (asyncpg)
- Embedding server client (Unix socket)
"""

import asyncio
import json
import os
import socket
from typing import Any, Optional

from research_kb_common import get_logger
from research_kb_storage import (
    DatabaseConfig,
    close_connection_pool,
    get_connection_pool,
)

logger = get_logger(__name__)

# Default paths
EMBED_SOCKET_PATH = os.getenv("EMBED_SOCKET_PATH", "/tmp/research_kb_embed.sock")
RERANK_SOCKET_PATH = os.getenv("RERANK_SOCKET_PATH", "/tmp/research_kb_rerank.sock")

# Connection pool singleton
_pool = None


async def init_pool(config: Optional[DatabaseConfig] = None) -> None:
    """Initialize database connection pool.

    Args:
        config: Database configuration. If None, uses environment defaults.
    """
    global _pool
    if _pool is not None:
        return

    _pool = await get_connection_pool(config)
    logger.info("db_pool_initialized")


async def get_pool() -> Any:
    """Get the database connection pool.

    Raises:
        RuntimeError: If pool not initialized
    """
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_pool() first.")
    return _pool


async def close_pool() -> None:
    """Close the database connection pool."""
    global _pool
    if _pool is not None:
        await close_connection_pool()
        _pool = None
        logger.info("db_pool_closed")


class EmbedClient:
    """Client for embedding server via Unix socket.

    Thread-safe async client for embedding queries.
    Uses semaphore to allow controlled concurrency (not serialization).
    """

    # Allow N concurrent embedding requests (not just 1)
    MAX_CONCURRENT_EMBEDS = 3
    EMBED_TIMEOUT = 5.0  # Reduced from 30s for faster failure

    def __init__(self, socket_path: str = EMBED_SOCKET_PATH):
        """Initialize embed client.

        Args:
            socket_path: Path to embedding server Unix socket
        """
        self.socket_path = socket_path
        # Use semaphore instead of lock to allow concurrent requests
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_EMBEDS)
        self._consecutive_failures = 0
        self._circuit_open_until: Optional[float] = None

    def _sync_request(self, data: dict, timeout: float) -> dict[str, Any]:
        """Send synchronous request to embed server.

        Args:
            data: Request data
            timeout: Socket timeout in seconds

        Returns:
            Response data

        Raises:
            ConnectionError: If cannot connect to server
            ValueError: If response invalid
            TimeoutError: If request times out
        """
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(self.socket_path)

            # Send request
            request_bytes = json.dumps(data).encode("utf-8")
            sock.sendall(request_bytes)
            sock.shutdown(socket.SHUT_WR)

            # Receive response
            chunks = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)

            sock.close()

            response_bytes = b"".join(chunks)
            result: dict[str, Any] = json.loads(response_bytes.decode("utf-8"))
            return result

        except socket.timeout:
            raise TimeoutError(f"Embed server timeout after {timeout}s")
        except socket.error as e:
            raise ConnectionError(f"Cannot connect to embed server at {self.socket_path}: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid response from embed server: {e}")

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open (failing fast).

        Returns:
            True if circuit is open and we should fail fast
        """
        import time

        if self._circuit_open_until is None:
            return False
        if time.time() > self._circuit_open_until:
            # Reset circuit
            self._circuit_open_until = None
            self._consecutive_failures = 0
            logger.info("embed_circuit_closed", message="Circuit breaker reset")
            return False
        return True

    def _record_failure(self) -> None:
        """Record a failure for circuit breaker logic."""
        import time

        self._consecutive_failures += 1
        if self._consecutive_failures >= 3:
            # Open circuit for 30 seconds
            self._circuit_open_until = time.time() + 30.0
            logger.warning(
                "embed_circuit_opened",
                failures=self._consecutive_failures,
                cooldown_seconds=30,
            )

    def _record_success(self) -> None:
        """Record a success (resets failure counter)."""
        self._consecutive_failures = 0

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query string.

        Uses BGE query instruction prefix for better retrieval.
        Allows up to MAX_CONCURRENT_EMBEDS concurrent requests.

        Args:
            text: Query text

        Returns:
            1024-dimensional embedding vector

        Raises:
            ConnectionError: If embed server unavailable
            ValueError: If embedding fails
            TimeoutError: If request times out
            RuntimeError: If circuit breaker is open
        """
        # Check circuit breaker first
        if self._is_circuit_open():
            raise RuntimeError("Embed server circuit breaker open - failing fast")

        async with self._semaphore:
            try:
                # Run sync socket in thread to not block event loop
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        self._sync_request,
                        {"action": "embed_query", "text": text},
                        self.EMBED_TIMEOUT,
                    ),
                    timeout=self.EMBED_TIMEOUT + 1.0,  # Async timeout slightly longer
                )

                if "error" in response:
                    self._record_failure()
                    raise ValueError(f"Embedding failed: {response['error']}")

                embedding = response.get("embedding")
                if not embedding or not isinstance(embedding, list) or len(embedding) != 1024:
                    self._record_failure()
                    raise ValueError(
                        f"Invalid embedding dimension: {len(embedding) if embedding else 0}"
                    )

                self._record_success()
                result: list[float] = embedding
                return result

            except (TimeoutError, asyncio.TimeoutError) as e:
                self._record_failure()
                raise TimeoutError(f"Embed query timeout: {e}")
            except (ConnectionError, OSError) as e:
                self._record_failure()
                raise ConnectionError(f"Embed server connection failed: {e}")

    async def health_check(self) -> dict[str, Any]:
        """Check embedding server health.

        Returns:
            Health status dict with keys: status, device, model, dim, circuit_breaker
        """
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._sync_request,
                {"action": "ping"},
                5.0,  # 5s health check timeout
            )

            if "error" in response:
                return {"status": "unhealthy", "error": response["error"]}

            return {
                "status": "healthy",
                "device": response.get("device", "unknown"),
                "model": response.get("model", "unknown"),
                "dim": response.get("dim", 0),
                "circuit_breaker": "closed" if not self._is_circuit_open() else "open",
                "concurrent_limit": self.MAX_CONCURRENT_EMBEDS,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Singleton embed client
_embed_client: Optional[EmbedClient] = None


def get_embed_client(socket_path: str = EMBED_SOCKET_PATH) -> EmbedClient:
    """Get or create embed client singleton.

    Args:
        socket_path: Path to embedding server socket

    Returns:
        EmbedClient instance
    """
    global _embed_client
    if _embed_client is None:
        _embed_client = EmbedClient(socket_path)
    return _embed_client


class RerankClientWrapper:
    """Async wrapper for reranking server health checks.

    Wraps synchronous RerankClient for use in async daemon.
    """

    def __init__(self, socket_path: str = RERANK_SOCKET_PATH):
        """Initialize rerank client wrapper.

        Args:
            socket_path: Path to reranking server Unix socket
        """
        self.socket_path = socket_path
        self._lock = asyncio.Lock()

    def _sync_ping(self) -> dict[str, Any]:
        """Synchronous ping to rerank server.

        Returns:
            Response dict with status, model, device info

        Raises:
            ConnectionError: If cannot connect
        """
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)  # 5 second timeout for health check
            sock.connect(self.socket_path)

            # Send ping request
            request = json.dumps({"action": "ping"}).encode("utf-8")
            sock.sendall(request)
            sock.shutdown(socket.SHUT_WR)

            # Receive response
            chunks = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)

            sock.close()
            result: dict[str, Any] = json.loads(b"".join(chunks).decode("utf-8"))
            return result

        except socket.error as e:
            raise ConnectionError(f"Cannot connect to rerank server: {e}")

    async def health_check(self) -> dict[str, Any]:
        """Check reranking server health.

        Returns:
            Health status dict with keys: status, model, device
        """
        try:
            async with self._lock:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, self._sync_ping)

            if response.get("status") == "ok":
                return {
                    "status": "healthy",
                    "model": response.get("model", "unknown"),
                    "device": response.get("device", "unknown"),
                }
            return {"status": "unhealthy", "error": "unexpected response"}

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Singleton rerank client
_rerank_client: Optional[RerankClientWrapper] = None


def get_rerank_client(socket_path: str = RERANK_SOCKET_PATH) -> RerankClientWrapper:
    """Get or create rerank client singleton.

    Args:
        socket_path: Path to reranking server socket

    Returns:
        RerankClientWrapper instance
    """
    global _rerank_client
    if _rerank_client is None:
        _rerank_client = RerankClientWrapper(socket_path)
    return _rerank_client
