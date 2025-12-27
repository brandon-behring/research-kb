#!/usr/bin/env python3
"""Tests for research-kb daemon.

Run with: pytest services/research_kb_daemon/test_daemon.py -v
"""

from __future__ import annotations

import asyncio
import json
import pytest
import socket
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "api" / "src"))

from daemon import DaemonServer
from client import DaemonClient, DaemonError


@pytest.fixture
def temp_socket():
    """Create a temporary socket path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "test.sock")


@pytest.fixture
def mock_service():
    """Mock the service module."""
    with patch("daemon.service") as mock:
        # Mock warmup (async function)
        mock.get_cached_embedding = AsyncMock(return_value=[0.1] * 1024)

        # Mock search
        search_response = MagicMock()
        search_response.query = "test query"
        search_response.expanded_query = None
        search_response.results = []
        search_response.execution_time_ms = 50.0
        mock.search = AsyncMock(return_value=search_response)

        # Mock concepts
        mock.get_concepts = AsyncMock(return_value=[])

        # Mock graph
        mock.get_graph_neighborhood = AsyncMock(
            return_value={"center": {}, "nodes": [], "edges": []}
        )

        yield mock


@pytest.mark.asyncio
async def test_daemon_ping(temp_socket, mock_service):
    """Daemon responds to ping."""
    server = DaemonServer(socket_path=temp_socket)

    # Start server in background
    server_task = asyncio.create_task(server.start())

    # Wait for server to start
    await asyncio.sleep(0.2)

    try:
        # Connect and send ping
        reader, writer = await asyncio.open_unix_connection(temp_socket)
        writer.write(b'{"action": "ping"}\n')
        await writer.drain()

        response = await reader.readline()
        data = json.loads(response)

        assert data["status"] == "ok"
        assert data["data"]["message"] == "pong"

        writer.close()
        await writer.wait_closed()

    finally:
        server.stop()
        await asyncio.sleep(0.1)
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_daemon_search(temp_socket, mock_service):
    """Daemon handles search requests."""
    # Setup mock
    result = MagicMock()
    result.source = MagicMock(id="src1", title="Test Paper", authors=[], year=2023)
    result.chunk = MagicMock(id="chunk1", content="Test content", page_start=1, section="Intro")
    result.scores = MagicMock(fts=0.3, vector=0.7, graph=0.1, combined=0.8)

    search_response = MagicMock()
    search_response.query = "test"
    search_response.expanded_query = None
    search_response.results = [result]
    search_response.execution_time_ms = 50.0
    mock_service.search = AsyncMock(return_value=search_response)

    server = DaemonServer(socket_path=temp_socket)
    server_task = asyncio.create_task(server.start())
    await asyncio.sleep(0.2)

    try:
        reader, writer = await asyncio.open_unix_connection(temp_socket)
        request = {"action": "search", "query": "test", "limit": 5}
        writer.write((json.dumps(request) + "\n").encode())
        await writer.drain()

        response = await reader.readline()
        data = json.loads(response)

        assert data["status"] == "ok"
        assert "results" in data["data"]
        assert len(data["data"]["results"]) == 1
        assert data["data"]["results"][0]["source"]["title"] == "Test Paper"

        writer.close()
        await writer.wait_closed()

    finally:
        server.stop()
        await asyncio.sleep(0.1)
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


class TestDaemonClient:
    """Tests for DaemonClient."""

    def test_is_available_false_when_no_daemon(self):
        """Client reports unavailable when daemon not running."""
        client = DaemonClient(socket_path="/nonexistent/socket.sock")
        assert client.is_available() is False

    def test_search_raises_when_unavailable(self):
        """Client raises when daemon unavailable."""
        client = DaemonClient(socket_path="/nonexistent/socket.sock")
        with pytest.raises(Exception):
            client.search("test query")
