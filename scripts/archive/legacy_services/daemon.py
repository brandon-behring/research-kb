#!/usr/bin/env python3
"""Research-KB Unix Socket Daemon.

Provides low-latency access to research-kb search functionality via Unix socket.
Keeps embedding model warm for <100ms response times after initial cold start.

Socket location: /tmp/research_kb_daemon.sock (configurable via RESEARCH_KB_SOCKET_PATH)
Health endpoint: http://localhost:9001 (configurable via RESEARCH_KB_HEALTH_PORT)

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

Health Endpoints:
    GET /health       - Liveness probe
    GET /health/ready - Readiness probe
    GET /metrics      - Prometheus metrics
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

# Add packages to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "api" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "common" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "pdf-tools" / "src"))
# Add daemon directory for local imports when running as script
sys.path.insert(0, str(Path(__file__).parent))

from research_kb_common import get_logger
from research_kb_api import service
from research_kb_api.service import SearchOptions, ContextType

# Local imports - handle both module and script execution
try:
    from .metrics import metrics
    from .health_server import HealthServer
except ImportError:
    # Running as script directly
    from metrics import metrics
    from health_server import HealthServer

logger = get_logger(__name__)

# Configuration
USER = os.environ.get("USER", "unknown")
SOCKET_PATH = os.environ.get("RESEARCH_KB_SOCKET_PATH", f"/tmp/research_kb_daemon_{USER}.sock")
PID_FILE = os.environ.get("RESEARCH_KB_PID_FILE", f"/tmp/research_kb_daemon_{USER}.pid")
MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB


def write_pid_file() -> None:
    """Write PID file for daemon management."""
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    logger.info("pid_file_written", pid=os.getpid(), path=PID_FILE)


def remove_pid_file() -> None:
    """Clean up PID file."""
    Path(PID_FILE).unlink(missing_ok=True)
    logger.debug("pid_file_removed", path=PID_FILE)


def check_existing_daemon() -> int | None:
    """Check if daemon already running, return PID or None."""
    if not Path(PID_FILE).exists():
        return None
    try:
        pid = int(Path(PID_FILE).read_text().strip())
        os.kill(pid, 0)  # Check if process exists
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        remove_pid_file()
        return None


class DaemonServer:
    """Unix socket server for research-kb queries."""

    def __init__(self, socket_path: str = SOCKET_PATH):
        self.socket_path = socket_path
        self.server: asyncio.Server | None = None
        self.running = False
        self._warmup_done = False
        self.health_server = HealthServer(is_ready=lambda: self._warmup_done)

    async def warmup(self) -> None:
        """Pre-warm the embedding model with a dummy query."""
        if self._warmup_done:
            return

        logger.info("daemon_warmup_start")
        try:
            # Trigger embedding model load
            await service.get_cached_embedding("warmup query")
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
                    metrics.record_error("json_parse")
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
        start_time = time.perf_counter()

        if not action:
            metrics.record_error("missing_action")
            return {"status": "error", "error": "Missing 'action' field"}

        try:
            if action == "ping":
                result = {"status": "ok", "data": {"message": "pong"}}

            elif action == "shutdown":
                result = {"status": "ok", "data": {"message": "shutting down"}}

            elif action == "search":
                result = await self._handle_search(request)

            elif action == "fast_search":
                result = await self._handle_fast_search(request)

            elif action == "concepts":
                result = await self._handle_concepts(request)

            elif action == "graph":
                result = await self._handle_graph(request)

            elif action == "reload":
                result = await self._handle_reload()

            else:
                metrics.record_error("unknown_action")
                return {"status": "error", "error": f"Unknown action: {action}"}

            # Record successful request metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.record_request(action, duration_ms)
            return result

        except Exception as e:
            logger.error("request_processing_error", action=action, error=str(e))
            metrics.record_error("internal")
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

    async def _handle_fast_search(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle fast_search request - vector-only for low latency.

        Optimized for latency-sensitive contexts (ProactiveContext integration).
        Skips FTS, graph, citation, rerank, and expansion.

        Performance: ~30ms database + ~150ms embedding = ~200ms total
        (vs ~3s for full hybrid search with normalization)
        """
        import time as time_module
        from research_kb_storage import SearchQuery, search_vector_only

        query = request.get("query")
        if not query:
            return {"status": "error", "error": "Missing 'query' field"}

        limit = request.get("limit", 5)
        start_time = time_module.perf_counter()

        # Get embedding (cached after warmup)
        embedding = await service.get_cached_embedding(query)

        # Build minimal query for vector-only search
        search_query = SearchQuery(
            text=query,
            embedding=embedding,
            fts_weight=0.0,
            vector_weight=1.0,
            limit=limit,
            use_graph=False,
            graph_weight=0.0,
            use_citations=False,
            citation_weight=0.0,
        )

        # Execute vector-only search (no normalization overhead)
        raw_results = await search_vector_only(search_query)

        # Convert to same nested format as regular search
        results = []
        for r in raw_results:
            chunk_metadata = r.chunk.metadata if r.chunk else {}
            results.append({
                "source": {
                    "id": str(r.source.id) if r.source else None,
                    "title": r.source.title if r.source else None,
                    "authors": r.source.authors if r.source else [],
                    "year": r.source.year if r.source else None,
                },
                "chunk": {
                    "id": str(r.chunk.id) if r.chunk else None,
                    "content": r.chunk.content[:500] if r.chunk else "",
                    "page_start": r.chunk.page_start if r.chunk else None,
                    "section": chunk_metadata.get("section_header") if chunk_metadata else None,
                },
                "scores": {
                    "fts": r.fts_score or 0.0,
                    "vector": r.vector_score or 0.0,
                    "graph": 0.0,  # Skipped
                    "combined": r.vector_score or 0.0,  # Vector-only
                },
            })

        execution_time_ms = (time_module.perf_counter() - start_time) * 1000

        return {
            "status": "ok",
            "data": {
                "query": query,
                "results": results,
                "execution_time_ms": round(execution_time_ms, 1),
                "fast_path": True,
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

    async def _handle_reload(self) -> dict[str, Any]:
        """Handle reload request - clear caches."""
        # Clear embedding cache in service layer
        service._embedding_cache.clear()
        self._warmup_done = False
        logger.info("daemon_cache_cleared")
        return {"status": "ok", "data": {"message": "cache cleared"}}

    def reload(self) -> None:
        """Reload daemon - clear caches (called from SIGHUP handler)."""
        service._embedding_cache.clear()
        self._warmup_done = False
        logger.info("daemon_reloaded_via_signal")

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

        # Start health server for HTTP endpoints
        await self.health_server.start()
        logger.info("health_server_started", port=9001)

        # Create Unix socket server
        self.server = await asyncio.start_unix_server(
            self.handle_client,
            path=self.socket_path,
        )

        # Set socket permissions (readable/writable by all)
        os.chmod(self.socket_path, 0o666)

        self.running = True
        logger.info("daemon_started", socket_path=self.socket_path, health_port=9001)

        async with self.server:
            while self.running:
                await asyncio.sleep(0.1)

        # Cleanup
        await self.health_server.stop()
        socket_path.unlink(missing_ok=True)
        logger.info("daemon_stopped")

    def stop(self) -> None:
        """Signal the server to stop."""
        self.running = False
        if self.server:
            self.server.close()


async def main() -> None:
    """Main entry point."""
    # Check for existing daemon
    existing_pid = check_existing_daemon()
    if existing_pid:
        print(f"Daemon already running (PID {existing_pid})", file=sys.stderr)
        print(f"Socket: {SOCKET_PATH}", file=sys.stderr)
        sys.exit(1)

    # Write PID file
    write_pid_file()

    daemon = DaemonServer()

    # Handle signals
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, daemon.stop)

    # SIGHUP for reload (cache clearing)
    loop.add_signal_handler(signal.SIGHUP, daemon.reload)

    try:
        await daemon.start()
    finally:
        remove_pid_file()


if __name__ == "__main__":
    asyncio.run(main())
