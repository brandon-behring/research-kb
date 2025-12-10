#!/usr/bin/env python3
"""Client library for research-kb daemon.

Example usage:
    from research_kb_daemon.client import DaemonClient

    client = DaemonClient()
    if client.is_available():
        results = client.search("instrumental variables", limit=5)
    else:
        # Fall back to CLI
        pass
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any

SOCKET_PATH = os.environ.get("RESEARCH_KB_SOCKET_PATH", "/tmp/research_kb_daemon.sock")
DEFAULT_TIMEOUT = 5.0


class DaemonClient:
    """Client for research-kb Unix socket daemon."""

    def __init__(
        self,
        socket_path: str = SOCKET_PATH,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.socket_path = socket_path
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if daemon is running and responsive."""
        try:
            response = self._send_request({"action": "ping"})
            return response.get("status") == "ok"
        except Exception:
            return False

    def search(
        self,
        query: str,
        limit: int = 5,
        context_type: str = "balanced",
        use_graph: bool = True,
        use_rerank: bool = True,
        use_expand: bool = True,
    ) -> dict[str, Any]:
        """Execute a search query.

        Parameters
        ----------
        query : str
            Search query text
        limit : int
            Maximum results to return
        context_type : str
            "building", "auditing", or "balanced"
        use_graph : bool
            Enable graph-boosted search
        use_rerank : bool
            Enable cross-encoder reranking
        use_expand : bool
            Enable query expansion

        Returns
        -------
        dict
            Search results with "query", "results", "execution_time_ms"

        Raises
        ------
        DaemonError
            If daemon is unavailable or returns an error
        """
        response = self._send_request({
            "action": "search",
            "query": query,
            "limit": limit,
            "context_type": context_type,
            "use_graph": use_graph,
            "use_rerank": use_rerank,
            "use_expand": use_expand,
        })
        return self._handle_response(response)

    def concepts(self, query: str | None = None, limit: int = 10) -> dict[str, Any]:
        """Search or list concepts.

        Parameters
        ----------
        query : str, optional
            Search query for concept names
        limit : int
            Maximum results

        Returns
        -------
        dict
            Concepts with "concepts" list

        Raises
        ------
        DaemonError
            If daemon is unavailable or returns an error
        """
        response = self._send_request({
            "action": "concepts",
            "query": query,
            "limit": limit,
        })
        return self._handle_response(response)

    def graph(self, concept: str, hops: int = 2) -> dict[str, Any]:
        """Get graph neighborhood for a concept.

        Parameters
        ----------
        concept : str
            Concept name to explore
        hops : int
            Number of relationship hops (1-5)

        Returns
        -------
        dict
            Graph neighborhood with "center", "nodes", "edges"

        Raises
        ------
        DaemonError
            If daemon is unavailable or returns an error
        """
        response = self._send_request({
            "action": "graph",
            "concept": concept,
            "hops": hops,
        })
        return self._handle_response(response)

    def shutdown(self) -> None:
        """Request daemon shutdown."""
        try:
            self._send_request({"action": "shutdown"})
        except Exception:
            pass  # Daemon may close connection before responding

    def _send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send request to daemon and return response."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)

        try:
            sock.connect(self.socket_path)

            # Send request
            data = json.dumps(request) + "\n"
            sock.sendall(data.encode("utf-8"))

            # Read response
            response_data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk
                if b"\n" in response_data:
                    break

            return json.loads(response_data.decode("utf-8").strip())

        finally:
            sock.close()

    def _handle_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Handle daemon response, raising on errors."""
        if response.get("status") == "error":
            raise DaemonError(response.get("error", "Unknown error"))
        return response.get("data", {})


class DaemonError(Exception):
    """Error from daemon communication."""

    pass


# Convenience functions for shell integration
def search_or_fallback(query: str, limit: int = 3, timeout: float = 0.5) -> str | None:
    """Search via daemon, return None if unavailable.

    Designed for shell integration - returns formatted text or None to signal
    that the caller should fall back to CLI.
    """
    try:
        client = DaemonClient(timeout=timeout)
        if not client.is_available():
            return None

        data = client.search(query, limit=limit)
        results = data.get("results", [])

        if not results:
            return "No results found."

        lines = []
        for i, r in enumerate(results, 1):
            source = r.get("source", {})
            chunk = r.get("chunk", {})
            lines.append(
                f"{i}. [{source.get('title', 'Unknown')}] "
                f"({source.get('year', '?')}) - {chunk.get('content', '')[:200]}..."
            )
        return "\n".join(lines)

    except Exception:
        return None


if __name__ == "__main__":
    import sys

    # Simple CLI test
    client = DaemonClient()

    if not client.is_available():
        print("Daemon not available", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: client.py <query>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    try:
        data = client.search(query)
        print(json.dumps(data, indent=2))
    except DaemonError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
