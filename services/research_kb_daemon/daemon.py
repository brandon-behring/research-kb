#!/usr/bin/env python3
"""Research-KB Unix Socket Daemon.

Provides low-latency access to research-kb search functionality via Unix socket.
Keeps embedding model warm for <100ms response times after initial cold start.

Socket location: /tmp/research_kb_daemon.sock (configurable via RESEARCH_KB_SOCKET_PATH)

Protocol:
    JSON messages, newline-delimited.
    Request: {"action": "search|concepts|graph|ping|shutdown", ...params}
    Response: {"status": "ok|error", "data": ...}

Actions:
    search  - {"action": "search", "query": "...", "limit": 5, "context_type": "balanced"}
    concepts - {"action": "concepts", "query": "...", "limit": 10}
    graph   - {"action": "graph", "concept": "...", "hops": 2}
    ping    - {"action": "ping"}
    shutdown - {"action": "shutdown"}
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

# Add packages to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "api" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "common" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "pdf-tools" / "src"))

from research_kb_common import get_logger
from research_kb_api import service
from research_kb_api.service import SearchOptions, ContextType

logger = get_logger(__name__)

# Configuration
SOCKET_PATH = os.environ.get("RESEARCH_KB_SOCKET_PATH", "/tmp/research_kb_daemon.sock")
MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB


class DaemonServer:
    """Unix socket server for research-kb queries."""

    def __init__(self, socket_path: str = SOCKET_PATH):
        self.socket_path = socket_path
        self.server: asyncio.Server | None = None
        self.running = False
        self._warmup_done = False

    async def warmup(self) -> None:
        """Pre-warm the embedding model with a dummy query."""
        if self._warmup_done:
            return

        logger.info("daemon_warmup_start")
        try:
            # Trigger embedding model load
            service.get_cached_embedding("warmup query")
            self._warmup_done = True
            logger.info("daemon_warmup_complete")
        except Exception as e:
            logger.error("daemon_warmup_failed", error=str(e))

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a single client connection."""
        addr = writer.get_extra_info("peername")
        logger.debug("client_connected", addr=str(addr))

        try:
            while True:
                # Read line-delimited JSON
                data = await reader.readline()
                if not data:
                    break

                # Parse request
                try:
                    request = json.loads(data.decode("utf-8"))
                except json.JSONDecodeError as e:
                    response = {"status": "error", "error": f"Invalid JSON: {e}"}
                    await self._send_response(writer, response)
                    continue

                # Process request
                response = await self._process_request(request)
                await self._send_response(writer, response)

                # Check for shutdown
                if request.get("action") == "shutdown":
                    logger.info("daemon_shutdown_requested")
                    self.running = False
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("client_handler_error", error=str(e))
        finally:
            writer.close()
            await writer.wait_closed()
            logger.debug("client_disconnected", addr=str(addr))

    async def _process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process a request and return response."""
        action = request.get("action")

        if not action:
            return {"status": "error", "error": "Missing 'action' field"}

        try:
            if action == "ping":
                return {"status": "ok", "data": {"message": "pong"}}

            elif action == "shutdown":
                return {"status": "ok", "data": {"message": "shutting down"}}

            elif action == "search":
                return await self._handle_search(request)

            elif action == "concepts":
                return await self._handle_concepts(request)

            elif action == "graph":
                return await self._handle_graph(request)

            else:
                return {"status": "error", "error": f"Unknown action: {action}"}

        except Exception as e:
            logger.error("request_processing_error", action=action, error=str(e))
            return {"status": "error", "error": str(e)}

    async def _handle_search(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle search request."""
        query = request.get("query")
        if not query:
            return {"status": "error", "error": "Missing 'query' field"}

        # Parse context type
        context_type_str = request.get("context_type", "balanced")
        context_type_map = {
            "building": ContextType.building,
            "auditing": ContextType.auditing,
            "balanced": ContextType.balanced,
        }
        context_type = context_type_map.get(context_type_str, ContextType.balanced)

        options = SearchOptions(
            query=query,
            limit=request.get("limit", 5),
            context_type=context_type,
            source_filter=request.get("source_filter"),
            use_graph=request.get("use_graph", True),
            use_rerank=request.get("use_rerank", True),
            use_expand=request.get("use_expand", True),
        )

        response = await service.search(options)

        # Convert to serializable format
        results = []
        for r in response.results:
            results.append({
                "source": {
                    "id": r.source.id,
                    "title": r.source.title,
                    "authors": r.source.authors,
                    "year": r.source.year,
                },
                "chunk": {
                    "id": r.chunk.id,
                    "content": r.chunk.content[:500],  # Truncate for socket response
                    "page_start": r.chunk.page_start,
                    "section": r.chunk.section,
                },
                "scores": {
                    "fts": r.scores.fts,
                    "vector": r.scores.vector,
                    "graph": r.scores.graph,
                    "combined": r.scores.combined,
                },
            })

        return {
            "status": "ok",
            "data": {
                "query": response.query,
                "expanded_query": response.expanded_query,
                "results": results,
                "execution_time_ms": response.execution_time_ms,
            },
        }

    async def _handle_concepts(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle concepts search request."""
        query = request.get("query")
        limit = request.get("limit", 10)

        concepts = await service.get_concepts(query=query, limit=limit)

        return {
            "status": "ok",
            "data": {
                "concepts": [
                    {
                        "id": str(c.id),
                        "name": c.name,
                        "type": c.concept_type.value if c.concept_type else None,
                        "definition": c.definition[:200] if c.definition else None,
                    }
                    for c in concepts
                ],
            },
        }

    async def _handle_graph(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle graph neighborhood request."""
        concept = request.get("concept")
        if not concept:
            return {"status": "error", "error": "Missing 'concept' field"}

        hops = request.get("hops", 2)

        neighborhood = await service.get_graph_neighborhood(
            concept_name=concept,
            hops=hops,
        )

        return {"status": "ok", "data": neighborhood}

    async def _send_response(
        self, writer: asyncio.StreamWriter, response: dict[str, Any]
    ) -> None:
        """Send JSON response with newline delimiter."""
        data = json.dumps(response) + "\n"
        writer.write(data.encode("utf-8"))
        await writer.drain()

    async def start(self) -> None:
        """Start the daemon server."""
        # Remove existing socket file
        socket_path = Path(self.socket_path)
        if socket_path.exists():
            socket_path.unlink()

        # Pre-warm embedding model
        await self.warmup()

        # Create server
        self.server = await asyncio.start_unix_server(
            self.handle_client,
            path=self.socket_path,
        )

        # Set socket permissions (readable/writable by all)
        os.chmod(self.socket_path, 0o666)

        self.running = True
        logger.info("daemon_started", socket_path=self.socket_path)

        async with self.server:
            while self.running:
                await asyncio.sleep(0.1)

        # Cleanup
        socket_path.unlink(missing_ok=True)
        logger.info("daemon_stopped")

    def stop(self) -> None:
        """Signal the server to stop."""
        self.running = False
        if self.server:
            self.server.close()


async def main() -> None:
    """Main entry point."""
    daemon = DaemonServer()

    # Handle signals
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, daemon.stop)

    await daemon.start()


if __name__ == "__main__":
    asyncio.run(main())
