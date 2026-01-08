"""Unix socket client for research-kb daemon.

Provides low-latency queries with CLI fallback.

Usage
-----
>>> from research_kb_client import DaemonClient
>>> client = DaemonClient()
>>> if client.is_available():
...     results = client.search("instrumental variables")
...     for r in results.results:
...         print(f"{r.source.title}: {r.score:.2f}")
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any, Optional

from .models import (
    ConceptInfo,
    ConceptNeighborhood,
    HealthStatus,
    SearchResponse,
    SearchResult,
    SearchResultChunk,
    SearchResultSource,
    StatsResponse,
)


class ResearchKBError(Exception):
    """Error from research-kb communication.

    Raised with explicit error information rather than returning empty results.
    """
    pass


class ConnectionError(ResearchKBError):
    """Cannot connect to daemon or CLI."""
    pass


class TimeoutError(ResearchKBError):
    """Request timed out."""
    pass


def _default_socket_path() -> str:
    """Get user-specific socket path."""
    user = os.environ.get("USER", "unknown")
    return os.environ.get(
        "RESEARCH_KB_SOCKET_PATH",
        f"/tmp/research_kb_daemon_{user}.sock"
    )


def _default_cli_path() -> str:
    """Get CLI path."""
    return os.environ.get(
        "RESEARCH_KB_CLI_PATH",
        str(Path.home() / "Claude/research-kb/venv/bin/research-kb"),
    )


class DaemonClient:
    """Client for research-kb daemon via Unix socket.

    Provides fast queries with automatic CLI fallback.

    Parameters
    ----------
    socket_path : str, optional
        Unix socket path. Default: /tmp/research_kb_daemon_$USER.sock
    cli_path : str, optional
        Path to research-kb CLI for fallback
    daemon_timeout : float
        Socket timeout in seconds (default: 2.0)
    cli_timeout : float
        CLI subprocess timeout (default: 10.0)

    Examples
    --------
    >>> client = DaemonClient()
    >>> if client.is_available():
    ...     health = client.health()
    ...     print(f"Status: {health.status}")

    >>> results = client.search("IV assumptions", limit=5)
    >>> for r in results.results:
    ...     print(f"[{r.score:.2f}] {r.source.title}")
    """

    def __init__(
        self,
        socket_path: Optional[str] = None,
        cli_path: Optional[str] = None,
        daemon_timeout: float = 2.0,
        cli_timeout: float = 10.0,
    ) -> None:
        self.socket_path = socket_path or _default_socket_path()
        self.cli_path = cli_path or _default_cli_path()
        self.daemon_timeout = daemon_timeout
        self.cli_timeout = cli_timeout

    def is_available(self) -> bool:
        """Check if daemon or CLI is available."""
        return self._is_daemon_available() or self._is_cli_available()

    def _is_daemon_available(self) -> bool:
        """Check if daemon socket responds."""
        if not os.path.exists(self.socket_path):
            return False
        try:
            return self.ping()
        except Exception:
            return False

    def _is_cli_available(self) -> bool:
        """Check if CLI executable exists."""
        return os.path.isfile(self.cli_path) and os.access(self.cli_path, os.X_OK)

    _request_id = 0

    def _send_request(self, method: str, params: Optional[dict[str, Any]] = None) -> Any:
        """Send JSON-RPC 2.0 request to daemon socket.

        Args:
            method: RPC method name (search, health, stats)
            params: Method parameters

        Returns:
            Result from daemon

        Raises:
            ConnectionError: Cannot connect
            TimeoutError: Request timed out
            ResearchKBError: RPC error response
        """
        DaemonClient._request_id += 1

        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.daemon_timeout)
            sock.connect(self.socket_path)

            # JSON-RPC 2.0 request format
            request = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {},
                "id": DaemonClient._request_id,
            }

            # Send request with shutdown to signal end
            request_bytes = json.dumps(request).encode("utf-8")
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
            response = json.loads(response_bytes.decode("utf-8"))

            # JSON-RPC 2.0 response format
            if "error" in response and response["error"]:
                error = response["error"]
                msg = error.get("message", str(error))
                raise ResearchKBError(f"RPC error: {msg}")

            return response.get("result")

        except socket.timeout:
            raise TimeoutError(f"Daemon timeout after {self.daemon_timeout}s")
        except socket.error as e:
            raise ConnectionError(f"Cannot connect to daemon at {self.socket_path}: {e}")
        except json.JSONDecodeError as e:
            raise ResearchKBError(f"Invalid response from daemon: {e}")

    def ping(self) -> bool:
        """Ping daemon to check connectivity.

        Returns:
            True if daemon responds

        Note:
            Uses health method as proxy since daemon has no dedicated ping.
        """
        try:
            self._send_request("health")
            return True
        except Exception:
            return False

    def health(self) -> HealthStatus:
        """Get daemon health status.

        Returns:
            Health status with component states

        Raises:
            ResearchKBError: If health check fails
        """
        data = self._send_request("health")
        if not data:
            data = {}

        # Handle nested status dicts
        db_status = data.get("database", "unknown")
        if isinstance(db_status, dict):
            db_status = db_status.get("status", "unknown")

        embed_status = data.get("embed_server", {})
        if isinstance(embed_status, dict):
            embed_status = embed_status.get("status", "unknown")

        rerank_status = data.get("rerank_server", {})
        if isinstance(rerank_status, dict):
            rerank_status = rerank_status.get("status")

        return HealthStatus(
            status=data.get("status", "unknown"),
            database=db_status,
            embed_server=embed_status,
            rerank_server=rerank_status,
            uptime_seconds=data.get("uptime_seconds"),
        )

    def stats(self) -> StatsResponse:
        """Get database statistics.

        Returns:
            Counts of sources, chunks, concepts, etc.
        """
        data = self._send_request("stats")
        if not data:
            data = {}

        return StatsResponse(
            sources=data.get("sources", 0),
            chunks=data.get("chunks", 0),
            concepts=data.get("concepts", 0),
            relationships=data.get("relationships", 0),
            citations=data.get("citations", 0),
            chunk_concepts=data.get("chunk_concepts", 0),
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        use_graph: bool = True,
        use_citations: bool = True,
        context_type: str = "balanced",
    ) -> SearchResponse:
        """Search research-kb for relevant literature.

        Parameters
        ----------
        query : str
            Search query
        limit : int
            Maximum results (default: 10)
        use_graph : bool
            Enable knowledge graph boost (default: True)
        use_citations : bool
            Enable citation authority boost (default: True)
        context_type : str
            Weight preset: "building", "auditing", "balanced"

        Returns:
            Search results with scores

        Raises:
            ValueError: Empty query
            ResearchKBError: Search failed

        Examples:
            >>> results = client.search("double machine learning", limit=5)
            >>> for r in results.results:
            ...     print(f"{r.source.title}: {r.score:.2f}")
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Try daemon first
        if self._is_daemon_available():
            try:
                return self._search_daemon(query, limit, use_graph, use_citations, context_type)
            except Exception:
                pass  # Fall through to CLI

        # Fall back to CLI
        if self._is_cli_available():
            return self._search_cli(query, limit, use_graph)

        raise ConnectionError(
            f"research-kb unavailable. "
            f"Daemon: {self.socket_path} (exists: {os.path.exists(self.socket_path)}), "
            f"CLI: {self.cli_path} (exists: {os.path.exists(self.cli_path)})"
        )

    def _search_daemon(
        self,
        query: str,
        limit: int,
        use_graph: bool,
        use_citations: bool,
        context_type: str,
    ) -> SearchResponse:
        """Search via daemon socket."""
        params = {
            "query": query,
            "limit": limit,
            "use_graph": use_graph,
            "use_citations": use_citations,
            "context_type": context_type,
        }

        data = self._send_request("search", params)
        if not data:
            data = []

        # Daemon returns list directly, wrap in dict for parser
        return self._parse_search_response({"results": data}, query)

    def _search_cli(self, query: str, limit: int, use_graph: bool) -> SearchResponse:
        """Search via CLI subprocess."""
        cmd = [
            self.cli_path,
            "query",
            query,
            "--format", "json",
            "--limit", str(limit),
        ]
        if not use_graph:
            cmd.append("--no-graph")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.cli_timeout,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"CLI timeout after {self.cli_timeout}s")

        if result.returncode != 0:
            raise ResearchKBError(f"CLI error (code {result.returncode}): {result.stderr}")

        # Parse JSON from output (skip log lines)
        lines = result.stdout.split("\n")
        json_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_start = i
                break

        if json_start is None:
            raise ResearchKBError(f"CLI returned no JSON: {result.stdout[:200]}")

        try:
            data = json.loads("\n".join(lines[json_start:]))
            return self._parse_search_response(data, query)
        except json.JSONDecodeError as e:
            raise ResearchKBError(f"Failed to parse CLI JSON: {e}")

    def _parse_search_response(self, data: dict[str, Any], query: str) -> SearchResponse:
        """Parse raw search response into typed model."""
        results = []
        for item in data.get("results", []):
            source_data = item.get("source", {})
            chunk_data = item.get("chunk", {})

            # Handle authors (list or string)
            authors = source_data.get("authors")
            if isinstance(authors, str):
                authors = [a.strip() for a in authors.split(",")]

            source = SearchResultSource(
                id=source_data.get("id", ""),
                title=source_data.get("title", "Unknown"),
                authors=authors,
                year=source_data.get("year"),
                source_type=source_data.get("type", source_data.get("source_type", "unknown")),
            )

            chunk = SearchResultChunk(
                id=chunk_data.get("id", ""),
                content=chunk_data.get("content", "")[:500],
                page_number=chunk_data.get("page_number"),
                section_header=chunk_data.get("section_header"),
            )

            # Score field varies
            score = item.get("score", item.get("combined_score", 0.0))

            results.append(SearchResult(
                source=source,
                chunk=chunk,
                score=score,
                fts_score=item.get("fts_score"),
                vector_score=item.get("vector_score"),
                graph_score=item.get("graph_score"),
                citation_score=item.get("citation_score"),
                concepts=item.get("concepts", []),
            ))

        return SearchResponse(
            results=results,
            query=query,
            expanded_query=data.get("expanded_query"),
            total_count=data.get("total_count"),
        )


# Convenience functions

def search_or_none(query: str, limit: int = 5) -> Optional[SearchResponse]:
    """Search with graceful failure.

    Returns None if research-kb unavailable instead of raising.
    Useful for optional context enrichment.

    Parameters
    ----------
    query : str
        Search query
    limit : int
        Maximum results

    Returns:
        Search response or None if unavailable

    Examples:
        >>> results = search_or_none("walk-forward validation")
        >>> if results:
        ...     for r in results.results:
        ...         print(r.source.title)
    """
    try:
        client = DaemonClient()
        if not client.is_available():
            return None
        return client.search(query, limit=limit)
    except Exception:
        return None


def get_methodology_context(topic: str, limit: int = 3) -> str:
    """Get formatted methodology context.

    Convenience function for hooks and scripts.

    Parameters
    ----------
    topic : str
        Topic to search
    limit : int
        Maximum results

    Returns:
        Markdown-formatted context or empty string if unavailable
    """
    response = search_or_none(topic, limit=limit)
    if not response or not response.results:
        return ""

    lines = ["## Research Context (research-kb)", ""]
    for r in response.results:
        lines.append(f"**{r.source.title}**")
        if r.source.authors:
            lines.append(f"*{', '.join(r.source.authors[:3])}*")
        lines.append(f"> {r.chunk.content[:200]}...")
        lines.append(f"Score: {r.score:.2f}")
        if r.concepts:
            lines.append(f"Concepts: {', '.join(r.concepts[:5])}")
        lines.append("")

    return "\n".join(lines)
