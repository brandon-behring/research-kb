"""Embedding client for research-kb PDF processing.

Provides simple API for embedding text chunks using the embed_server daemon.
Handles connection management, batching, and error recovery.
"""

import json
import socket

from research_kb_common import get_logger, retry_on_exception
from research_kb_pdf.chunker import TextChunk

logger = get_logger(__name__)

# Default socket path (matches embed_server.py)
DEFAULT_SOCKET_PATH = "/tmp/research_kb_embed.sock"
BUFFER_SIZE = 131072  # 128KB


class EmbeddingClient:
    """Client for communicating with embedding server."""

    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH):
        """Initialize embedding client.

        Args:
            socket_path: Path to Unix domain socket

        Example:
            >>> client = EmbeddingClient()
            >>> embedding = client.embed("Hello world")
            >>> len(embedding)
            1024
        """
        self.socket_path = socket_path

    @retry_on_exception(
        exception_types=(ConnectionError, TimeoutError, BrokenPipeError, OSError),
        max_attempts=3,
        min_wait_seconds=1.0,
        max_wait_seconds=5.0,
    )
    def _send_request(self, request: dict) -> dict:
        """Send request to embedding server.

        Args:
            request: Request dictionary with 'action' field

        Returns:
            Response dictionary

        Raises:
            ConnectionError: If cannot connect to server (after retries exhausted)
            ValueError: If server returns error (not retried)
        """
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(self.socket_path)
            client.sendall(json.dumps(request).encode("utf-8"))
            client.shutdown(socket.SHUT_WR)  # Signal end of request

            response_data = b""
            while True:
                chunk = client.recv(BUFFER_SIZE)
                if not chunk:
                    break
                response_data += chunk

            client.close()
            response = json.loads(response_data.decode("utf-8"))

            if "error" in response:
                raise ValueError(f"Embedding server error: {response['error']}")

            return response

        except FileNotFoundError:
            raise ConnectionError(
                "Embedding server not running. Start with: "
                "python -m research_kb_pdf.embed_server"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to embedding server: {e}")

    def ping(self) -> dict:
        """Health check for embedding server.

        Returns:
            Status dictionary with device and model info

        Example:
            >>> client = EmbeddingClient()
            >>> status = client.ping()
            >>> status['status']
            'ok'
        """
        return self._send_request({"action": "ping"})

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            1024-dimensional embedding vector

        Example:
            >>> client = EmbeddingClient()
            >>> embedding = client.embed("Introduction to causality")
            >>> len(embedding)
            1024
        """
        response = self._send_request({"action": "embed", "text": text})
        return response["embedding"]

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed multiple texts in a batch.

        Args:
            texts: List of texts to embed
            batch_size: Server-side batch size (default 32)

        Returns:
            List of 1024-dimensional embeddings

        Example:
            >>> client = EmbeddingClient()
            >>> embeddings = client.embed_batch(["Text 1", "Text 2"])
            >>> len(embeddings)
            2
        """
        response = self._send_request(
            {"action": "embed_batch", "texts": texts, "batch_size": batch_size}
        )
        return response["embeddings"]

    def embed_chunks(self, chunks: list[TextChunk]) -> list[list[float]]:
        """Embed text chunks from chunker.

        Args:
            chunks: List of TextChunk objects

        Returns:
            List of embeddings (same order as chunks)

        Example:
            >>> from research_kb_pdf import extract_pdf, chunk_document
            >>> doc = extract_pdf("paper.pdf")
            >>> chunks = chunk_document(doc)
            >>> client = EmbeddingClient()
            >>> embeddings = client.embed_chunks(chunks)
            >>> len(embeddings) == len(chunks)
            True
        """
        logger.info("embedding_chunks", count=len(chunks))
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_batch(texts)
        logger.info("chunks_embedded", count=len(embeddings))
        return embeddings

    def shutdown_server(self) -> None:
        """Request graceful shutdown of embedding server.

        Example:
            >>> client = EmbeddingClient()
            >>> client.shutdown_server()
        """
        try:
            self._send_request({"action": "shutdown"})
        except Exception:
            # Server may close connection before responding
            pass


def embed_text(text: str, socket_path: str = DEFAULT_SOCKET_PATH) -> list[float]:
    """Convenience function to embed a single text.

    Args:
        text: Text to embed
        socket_path: Path to Unix domain socket

    Returns:
        1024-dimensional embedding vector

    Example:
        >>> embedding = embed_text("Hello world")
        >>> len(embedding)
        1024
    """
    client = EmbeddingClient(socket_path)
    return client.embed(text)


def embed_texts(
    texts: list[str], socket_path: str = DEFAULT_SOCKET_PATH
) -> list[list[float]]:
    """Convenience function to embed multiple texts.

    Args:
        texts: List of texts to embed
        socket_path: Path to Unix domain socket

    Returns:
        List of 1024-dimensional embeddings

    Example:
        >>> embeddings = embed_texts(["Text 1", "Text 2"])
        >>> len(embeddings)
        2
    """
    client = EmbeddingClient(socket_path)
    return client.embed_batch(texts)
