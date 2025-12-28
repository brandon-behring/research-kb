"""Connection pool management for daemon.

Manages:
- Database connection pool (asyncpg)
- Embedding server client (Unix socket)
"""

import asyncio
import json
import os
import socket
from typing import Optional

from research_kb_common import get_logger
from research_kb_storage import DatabaseConfig, get_connection_pool, close_connection_pool

logger = get_logger(__name__)

# Default paths
EMBED_SOCKET_PATH = os.getenv("EMBED_SOCKET_PATH", "/tmp/research_kb_embed.sock")

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


async def get_pool():
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

    Thread-safe synchronous client for embedding queries.
    Uses simple JSON protocol over Unix socket.
    """

    def __init__(self, socket_path: str = EMBED_SOCKET_PATH):
        """Initialize embed client.

        Args:
            socket_path: Path to embedding server Unix socket
        """
        self.socket_path = socket_path
        self._lock = asyncio.Lock()

    def _sync_request(self, data: dict) -> dict:
        """Send synchronous request to embed server.

        Args:
            data: Request data

        Returns:
            Response data

        Raises:
            ConnectionError: If cannot connect to server
            ValueError: If response invalid
        """
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(30.0)  # 30 second timeout for embedding
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
            return json.loads(response_bytes.decode("utf-8"))

        except socket.error as e:
            raise ConnectionError(f"Cannot connect to embed server at {self.socket_path}: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid response from embed server: {e}")

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query string.

        Uses BGE query instruction prefix for better retrieval.

        Args:
            text: Query text

        Returns:
            1024-dimensional embedding vector

        Raises:
            ConnectionError: If embed server unavailable
            ValueError: If embedding fails
        """
        async with self._lock:
            # Run sync socket in thread to not block event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, self._sync_request, {"action": "embed_query", "text": text}
            )

        if "error" in response:
            raise ValueError(f"Embedding failed: {response['error']}")

        embedding = response.get("embedding")
        if not embedding or len(embedding) != 1024:
            raise ValueError(f"Invalid embedding dimension: {len(embedding) if embedding else 0}")

        return embedding

    async def health_check(self) -> dict:
        """Check embedding server health.

        Returns:
            Health status dict with keys: status, device, model, dim
        """
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, self._sync_request, {"action": "ping"}
            )

            if "error" in response:
                return {"status": "unhealthy", "error": response["error"]}

            return {
                "status": "healthy",
                "device": response.get("device", "unknown"),
                "model": response.get("model", "unknown"),
                "dim": response.get("dim", 0),
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
