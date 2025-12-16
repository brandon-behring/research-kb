"""HTTP Health Server for daemon observability.

Runs alongside Unix socket server on port 9001 to expose:
- /health - Basic liveness probe
- /health/ready - Readiness probe (warmup complete)
- /metrics - Prometheus metrics endpoint

Uses minimal asyncio HTTP implementation (no external dependencies).
"""

from __future__ import annotations

import asyncio
import json
from typing import Callable, Awaitable

from .metrics import metrics

# Configuration
HEALTH_PORT = 9001
HEALTH_HOST = "0.0.0.0"


class HealthServer:
    """Minimal async HTTP server for health endpoints."""

    def __init__(self, is_ready: Callable[[], bool]):
        """
        Initialize health server.

        Args:
            is_ready: Callable that returns True when daemon is ready
        """
        self.is_ready = is_ready
        self.server: asyncio.Server | None = None

    async def handle_request(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle HTTP request."""
        try:
            # Read request line
            request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not request_line:
                return

            # Parse request
            parts = request_line.decode().strip().split()
            if len(parts) < 2:
                await self._send_response(writer, 400, "Bad Request")
                return

            method, path = parts[0], parts[1]

            # Consume headers (we don't need them)
            while True:
                line = await reader.readline()
                if line == b"\r\n" or line == b"\n" or not line:
                    break

            # Route request
            if method != "GET":
                await self._send_response(writer, 405, "Method Not Allowed")
            elif path == "/health":
                await self._handle_health(writer)
            elif path == "/health/ready":
                await self._handle_ready(writer)
            elif path == "/metrics":
                await self._handle_metrics(writer)
            else:
                await self._send_response(writer, 404, "Not Found")

        except asyncio.TimeoutError:
            pass
        except Exception:
            await self._send_response(writer, 500, "Internal Server Error")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _handle_health(self, writer: asyncio.StreamWriter) -> None:
        """Handle /health - liveness probe."""
        body = json.dumps({
            "status": "ok",
            "uptime_seconds": metrics.uptime_seconds(),
        })
        await self._send_json(writer, 200, body)

    async def _handle_ready(self, writer: asyncio.StreamWriter) -> None:
        """Handle /health/ready - readiness probe."""
        if self.is_ready():
            body = json.dumps({
                "status": "ready",
                "uptime_seconds": metrics.uptime_seconds(),
                "requests_served": metrics.total_requests(),
            })
            await self._send_json(writer, 200, body)
        else:
            body = json.dumps({
                "status": "not_ready",
                "message": "Warming up embedding model",
            })
            await self._send_json(writer, 503, body)

    async def _handle_metrics(self, writer: asyncio.StreamWriter) -> None:
        """Handle /metrics - Prometheus metrics."""
        body = metrics.to_prometheus()
        await self._send_response(
            writer, 200, body, content_type="text/plain; version=0.0.4"
        )

    async def _send_json(
        self, writer: asyncio.StreamWriter, status: int, body: str
    ) -> None:
        """Send JSON response."""
        await self._send_response(writer, status, body, "application/json")

    async def _send_response(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        body: str,
        content_type: str = "text/plain",
    ) -> None:
        """Send HTTP response."""
        status_text = {
            200: "OK",
            400: "Bad Request",
            404: "Not Found",
            405: "Method Not Allowed",
            500: "Internal Server Error",
            503: "Service Unavailable",
        }.get(status, "Unknown")

        response = (
            f"HTTP/1.1 {status} {status_text}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
            f"{body}"
        )
        writer.write(response.encode())
        await writer.drain()

    async def start(self) -> None:
        """Start HTTP health server."""
        self.server = await asyncio.start_server(
            self.handle_request,
            host=HEALTH_HOST,
            port=HEALTH_PORT,
        )
        # Don't block - let the main daemon loop handle running

    async def stop(self) -> None:
        """Stop HTTP health server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
