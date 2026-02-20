"""Unix socket server with JSON-RPC 2.0 protocol.

Provides low-latency query interface for research-kb.

Usage:
    research-kb-daemon [--socket /tmp/research_kb_daemon_user.sock]

Protocol:
    JSON-RPC 2.0 over Unix domain socket.

    Request:
        {"jsonrpc": "2.0", "method": "search", "params": {"query": "IV"}, "id": 1}

    Response:
        {"jsonrpc": "2.0", "result": [...], "id": 1}

    Error:
        {"jsonrpc": "2.0", "error": {"code": -32600, "message": "..."}, "id": 1}
"""

import argparse
import asyncio
import json
import os
import signal
import sys
from typing import Any, Optional

from prometheus_client import start_http_server as start_prometheus_server
from research_kb_common import get_logger

from research_kb_daemon.handler import dispatch
from research_kb_daemon.metrics import ACTIVE_CONNECTIONS
from research_kb_daemon.pool import close_pool, init_pool

logger = get_logger(__name__)

# Default socket path (user-scoped)
DEFAULT_SOCKET_PATH = f"/tmp/research_kb_daemon_{os.getenv('USER', 'unknown')}.sock"

# Connection limits
MAX_CONCURRENT_CONNECTIONS = 50
CONNECTION_TIMEOUT = 10.0  # Graph queries take 1.7-5.8s; 5s was too aggressive

# Connection tracking
_connection_semaphore: Optional[asyncio.Semaphore] = None
_active_connections = 0

# JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def make_response(
    result: Any = None,
    error: Optional[dict] = None,
    request_id: Optional[int | str] = None,
) -> bytes:
    """Create JSON-RPC response.

    Args:
        result: Success result (mutually exclusive with error)
        error: Error dict with code and message
        request_id: Request ID to echo back

    Returns:
        Encoded JSON-RPC response
    """
    response = {"jsonrpc": "2.0", "id": request_id}

    if error is not None:
        response["error"] = error
    else:
        response["result"] = result

    return json.dumps(response).encode("utf-8")


def make_error(code: int, message: str, request_id: Optional[int | str] = None) -> bytes:
    """Create JSON-RPC error response.

    Args:
        code: Error code (negative integer)
        message: Error message
        request_id: Request ID

    Returns:
        Encoded error response
    """
    return make_response(error={"code": code, "message": message}, request_id=request_id)


async def handle_request(data: bytes) -> bytes:
    """Handle a single JSON-RPC request.

    Args:
        data: Raw request bytes

    Returns:
        Response bytes
    """
    request_id = None

    try:
        # Parse JSON
        try:
            request = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("parse_error", error=str(e))
            return make_error(PARSE_ERROR, f"Parse error: {e}")

        # Validate JSON-RPC structure
        if not isinstance(request, dict):
            return make_error(INVALID_REQUEST, "Request must be an object")

        request_id = request.get("id")
        jsonrpc = request.get("jsonrpc")

        if jsonrpc != "2.0":
            return make_error(INVALID_REQUEST, "Missing or invalid jsonrpc version", request_id)

        method = request.get("method")
        if not method or not isinstance(method, str):
            return make_error(INVALID_REQUEST, "Missing or invalid method", request_id)

        params = request.get("params", {})
        if not isinstance(params, (dict, list)):
            params = {}

        # Handle positional params (convert to dict for our handlers)
        if isinstance(params, list):
            # Our methods only support named params for now
            return make_error(
                INVALID_PARAMS,
                "Positional params not supported, use named params",
                request_id,
            )

        # Dispatch method
        try:
            result = await dispatch(method, params)
            return make_response(result=result, request_id=request_id)

        except ValueError as e:
            # Method not found or invalid params
            if "Method not found" in str(e):
                return make_error(METHOD_NOT_FOUND, str(e), request_id)
            return make_error(INVALID_PARAMS, str(e), request_id)

        except Exception as e:
            logger.exception("method_error", method=method, error=str(e))
            return make_error(INTERNAL_ERROR, f"Internal error: {e}", request_id)

    except Exception as e:
        logger.exception("request_error", error=str(e))
        return make_error(INTERNAL_ERROR, f"Unexpected error: {e}", request_id)


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Handle a client connection.

    Uses a semaphore to limit concurrent connections and prevent resource exhaustion.

    Args:
        reader: Stream reader
        writer: Stream writer
    """
    global _active_connections

    addr = writer.get_extra_info("peername")

    # Try to acquire semaphore (with timeout to prevent deadlock)
    try:
        await asyncio.wait_for(_connection_semaphore.acquire(), timeout=1.0)
    except asyncio.TimeoutError:
        logger.warning(
            "connection_rejected",
            reason="max_connections_reached",
            active=_active_connections,
            max=MAX_CONCURRENT_CONNECTIONS,
        )
        writer.write(make_error(INTERNAL_ERROR, "Server busy, try again later"))
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        return

    _active_connections += 1
    ACTIVE_CONNECTIONS.set(_active_connections)
    logger.debug("connection_opened", active=_active_connections)

    try:
        # Read entire request (client should close write end after sending)
        data = await asyncio.wait_for(reader.read(1024 * 1024), timeout=CONNECTION_TIMEOUT)

        if not data:
            return

        # Process request
        response = await handle_request(data)

        # Send response
        writer.write(response)
        await writer.drain()

    except asyncio.TimeoutError:
        logger.warning("client_timeout", addr=addr, timeout=CONNECTION_TIMEOUT)
        writer.write(make_error(INTERNAL_ERROR, f"Request timeout ({CONNECTION_TIMEOUT}s)"))
        await writer.drain()

    except Exception as e:
        logger.exception("client_error", addr=addr, error=str(e))

    finally:
        _active_connections -= 1
        ACTIVE_CONNECTIONS.set(_active_connections)
        _connection_semaphore.release()
        logger.debug("connection_closed", active=_active_connections)
        writer.close()
        await writer.wait_closed()


async def run_server(socket_path: str, skip_warm: bool = False) -> None:
    """Run the Unix socket server.

    Args:
        socket_path: Path to Unix domain socket
        skip_warm: If True, skip KuzuDB pre-warming on startup
    """
    global _connection_semaphore

    # Initialize connection semaphore
    _connection_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CONNECTIONS)
    logger.info(
        "connection_limits_configured",
        max_connections=MAX_CONCURRENT_CONNECTIONS,
        timeout=CONNECTION_TIMEOUT,
    )

    # Start Prometheus metrics HTTP server on port 9001
    # Scraped by Prometheus (monitoring/prometheus/prometheus.yml)
    try:
        start_prometheus_server(9001)
        logger.info("prometheus_metrics_started", port=9001)
    except OSError as e:
        # Port already in use — non-fatal, log and continue
        logger.warning("prometheus_metrics_port_busy", port=9001, error=str(e))

    # Remove existing socket
    if os.path.exists(socket_path):
        os.remove(socket_path)

    # Initialize database pool
    await init_pool()

    # Schedule KuzuDB pre-warming (background, non-blocking)
    warmup_task = None
    if not skip_warm:
        from research_kb_daemon.warmup import warm_kuzu

        warmup_task = asyncio.create_task(warm_kuzu())
        logger.info("kuzu_warmup_scheduled")
    else:
        logger.info("kuzu_warmup_skipped")

    # Create server
    server = await asyncio.start_unix_server(handle_client, path=socket_path)

    # Set socket permissions (user-only for security)
    os.chmod(socket_path, 0o600)

    logger.info("daemon_started", socket=socket_path)
    print(f"Research KB daemon listening on {socket_path}", file=sys.stderr)

    # Handle shutdown signals
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("shutdown_signal_received")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        async with server:
            # Wait for shutdown signal
            await shutdown_event.wait()

    finally:
        # Cancel warmup if still running
        if warmup_task and not warmup_task.done():
            warmup_task.cancel()
            try:
                await warmup_task
            except asyncio.CancelledError:
                pass

        # Cleanup
        logger.info("daemon_stopping")
        await close_pool()

        # Remove socket
        if os.path.exists(socket_path):
            os.remove(socket_path)

        logger.info("daemon_stopped")


def main() -> None:
    """Entry point for research-kb-daemon command."""
    parser = argparse.ArgumentParser(
        description="Research KB daemon - low-latency query service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Protocol:
    JSON-RPC 2.0 over Unix domain socket.

Methods:
    search(query, limit?, context_type?, use_graph?, use_citations?)
        Execute hybrid search.

    health()
        Check system health.

    stats()
        Get database statistics.

Example:
    echo '{"jsonrpc":"2.0","method":"health","id":1}' | nc -U /tmp/research_kb_daemon_$USER.sock
        """,
    )

    parser.add_argument(
        "--socket",
        default=DEFAULT_SOCKET_PATH,
        help=f"Unix socket path (default: {DEFAULT_SOCKET_PATH})",
    )
    parser.add_argument(
        "--no-warm",
        action="store_true",
        default=False,
        help="Skip KuzuDB pre-warming on startup",
    )

    args = parser.parse_args()

    # Run server
    try:
        asyncio.run(run_server(args.socket, skip_warm=args.no_warm))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
