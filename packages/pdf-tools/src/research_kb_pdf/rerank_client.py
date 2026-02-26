"""Reranking client for research-kb search enhancement.

Provides simple API for reranking search results using the rerank_server daemon.
Handles connection management, batching, and error recovery.

Usage:
    >>> from research_kb_pdf.rerank_client import RerankClient
    >>> client = RerankClient()
    >>> reranked = client.rerank("query", documents, top_k=10)
"""

import json
import socket
from dataclasses import dataclass
from typing import Any, Callable, Optional

from research_kb_common import get_logger

logger = get_logger(__name__)

# Default socket path (matches rerank_server.py)
DEFAULT_SOCKET_PATH = "/tmp/research_kb_rerank.sock"
BUFFER_SIZE = 262144  # 256KB


@dataclass
class RerankResult:
    """Result from cross-encoder reranking.

    Attributes:
        content: The text content that was reranked
        original_rank: Position before reranking (1-based)
        rerank_score: Cross-encoder relevance score
        new_rank: Position after reranking (1-based)
    """

    content: str
    original_rank: int
    rerank_score: float
    new_rank: int


class RerankClient:
    """Client for communicating with reranking server."""

    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH):
        """Initialize reranking client.

        Args:
            socket_path: Path to Unix domain socket

        Example:
            >>> client = RerankClient()
            >>> results = client.rerank("IV", ["doc1", "doc2"])
        """
        self.socket_path = socket_path

    def _send_request(self, request: dict) -> dict:
        """Send request to reranking server.

        Args:
            request: Request dictionary with 'action' field

        Returns:
            Response dictionary

        Raises:
            ConnectionError: If cannot connect to server
            ValueError: If server returns error
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
            response: dict = json.loads(response_data.decode("utf-8"))

            if "error" in response:
                raise ValueError(f"Rerank server error: {response['error']}")

            return response

        except FileNotFoundError:
            raise ConnectionError(
                "Rerank server not running. Start with: " "python -m research_kb_pdf.rerank_server"
            )
        except Exception as e:
            if isinstance(e, (ConnectionError, ValueError)):
                raise
            raise ConnectionError(f"Failed to connect to rerank server: {e}")

    def ping(self) -> dict:
        """Health check for reranking server.

        Returns:
            Status dictionary with device and model info

        Example:
            >>> client = RerankClient()
            >>> status = client.ping()
            >>> status['status']
            'ok'
        """
        return self._send_request({"action": "ping"})

    def is_available(self) -> bool:
        """Check if reranking server is available.

        Returns:
            True if server is running and responding

        Example:
            >>> client = RerankClient()
            >>> if client.is_available():
            ...     results = client.rerank(query, docs)
        """
        try:
            status = self.ping()
            return status.get("status") == "ok"
        except (ConnectionError, ValueError):
            return False

    def predict_scores(self, query: str, documents: list[str]) -> list[float]:
        """Get raw relevance scores for query-document pairs.

        Args:
            query: Search query
            documents: List of document texts

        Returns:
            List of relevance scores (higher = more relevant)

        Example:
            >>> scores = client.predict_scores("IV", ["doc1", "doc2"])
            >>> len(scores)
            2
        """
        response = self._send_request(
            {
                "action": "predict",
                "query": query,
                "documents": documents,
            }
        )
        scores: list[float] = response["scores"]
        return scores

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[RerankResult]:
        """Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of results to return

        Returns:
            Top-k RerankResult objects sorted by rerank_score

        Example:
            >>> results = client.rerank("IV assumptions", docs, top_k=5)
            >>> results[0].new_rank
            1
        """
        response = self._send_request(
            {
                "action": "rerank",
                "query": query,
                "documents": documents,
                "top_k": top_k,
            }
        )

        return [
            RerankResult(
                content=r["content"],
                original_rank=r["original_rank"],
                rerank_score=r["rerank_score"],
                new_rank=r["new_rank"],
            )
            for r in response["results"]
        ]

    def rerank_search_results(
        self,
        query: str,
        results: list,
        top_k: int = 10,
        content_extractor: Optional[Callable[..., Any]] = None,
    ) -> list:
        """Rerank SearchResult objects from storage.search.

        Args:
            query: Search query
            results: List of SearchResult objects (or any objects with chunk.content)
            top_k: Number of results to return
            content_extractor: Optional function to extract content from result
                              (default: lambda r: r.chunk.content)

        Returns:
            Reranked list of results with updated rerank_score

        Example:
            >>> from research_kb_storage.search import search_hybrid_v2
            >>> results = await search_hybrid_v2(query)
            >>> reranked = client.rerank_search_results(query.text, results)
        """
        if not results:
            return []

        # Extract content from results
        if content_extractor is None:

            def content_extractor(r):
                return r.chunk.content

        documents = [content_extractor(r) for r in results]

        # Get scores from server
        response = self._send_request(
            {
                "action": "predict",
                "query": query,
                "documents": documents,
            }
        )
        scores = response["scores"]

        # Pair results with scores
        scored = list(zip(results, scores))

        # Sort by rerank score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Update results with rerank scores and new ranks
        reranked = []
        for new_rank, (result, score) in enumerate(scored[:top_k], start=1):
            # Update the result object
            result.rerank_score = score
            result.combined_score = score  # Override combined for final ranking
            result.rank = new_rank
            reranked.append(result)

        return reranked

    def shutdown_server(self) -> None:
        """Request graceful shutdown of reranking server.

        Example:
            >>> client = RerankClient()
            >>> client.shutdown_server()
        """
        try:
            self._send_request({"action": "shutdown"})
        except Exception:
            # Server may close connection before responding
            pass


def rerank_texts(
    query: str,
    documents: list[str],
    top_k: int = 10,
    socket_path: str = DEFAULT_SOCKET_PATH,
) -> list[RerankResult]:
    """Convenience function to rerank documents.

    Args:
        query: Search query
        documents: List of document texts
        top_k: Number of results to return
        socket_path: Path to Unix domain socket

    Returns:
        Top-k RerankResult objects

    Example:
        >>> results = rerank_texts("IV", ["doc1", "doc2"])
    """
    client = RerankClient(socket_path)
    return client.rerank(query, documents, top_k)
