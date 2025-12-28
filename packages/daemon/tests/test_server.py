"""Tests for daemon server JSON-RPC protocol."""

import json
import pytest
from unittest.mock import AsyncMock, patch

from research_kb_daemon.server import (
    handle_request,
    make_response,
    make_error,
    PARSE_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)


class TestMakeResponse:
    """Tests for response formatting."""

    def test_success_response(self):
        """Test success response format."""
        response = make_response(result={"data": "test"}, request_id=1)
        parsed = json.loads(response.decode())

        assert parsed["jsonrpc"] == "2.0"
        assert parsed["id"] == 1
        assert parsed["result"] == {"data": "test"}
        assert "error" not in parsed

    def test_error_response(self):
        """Test error response format."""
        response = make_response(error={"code": -32600, "message": "Test"}, request_id=2)
        parsed = json.loads(response.decode())

        assert parsed["jsonrpc"] == "2.0"
        assert parsed["id"] == 2
        assert parsed["error"]["code"] == -32600
        assert parsed["error"]["message"] == "Test"
        assert "result" not in parsed

    def test_null_id(self):
        """Test response with null ID (notification response)."""
        response = make_response(result="ok", request_id=None)
        parsed = json.loads(response.decode())

        assert parsed["id"] is None


class TestMakeError:
    """Tests for error response helper."""

    def test_make_error_format(self):
        """Test error helper produces correct format."""
        response = make_error(INVALID_REQUEST, "Bad request", request_id=5)
        parsed = json.loads(response.decode())

        assert parsed["error"]["code"] == INVALID_REQUEST
        assert parsed["error"]["message"] == "Bad request"
        assert parsed["id"] == 5


class TestHandleRequest:
    """Tests for request handling."""

    @pytest.mark.asyncio
    async def test_parse_error_invalid_json(self):
        """Test parse error for invalid JSON."""
        response = await handle_request(b"not json")
        parsed = json.loads(response.decode())

        assert parsed["error"]["code"] == PARSE_ERROR
        assert "Parse error" in parsed["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_request_not_object(self):
        """Test invalid request for non-object."""
        response = await handle_request(b'"string"')
        parsed = json.loads(response.decode())

        assert parsed["error"]["code"] == INVALID_REQUEST
        assert "object" in parsed["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_request_missing_version(self):
        """Test invalid request for missing jsonrpc version."""
        response = await handle_request(json.dumps({"method": "test", "id": 1}).encode())
        parsed = json.loads(response.decode())

        assert parsed["error"]["code"] == INVALID_REQUEST
        assert "jsonrpc" in parsed["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_request_wrong_version(self):
        """Test invalid request for wrong jsonrpc version."""
        response = await handle_request(
            json.dumps({"jsonrpc": "1.0", "method": "test", "id": 1}).encode()
        )
        parsed = json.loads(response.decode())

        assert parsed["error"]["code"] == INVALID_REQUEST

    @pytest.mark.asyncio
    async def test_invalid_request_missing_method(self):
        """Test invalid request for missing method."""
        response = await handle_request(json.dumps({"jsonrpc": "2.0", "id": 1}).encode())
        parsed = json.loads(response.decode())

        assert parsed["error"]["code"] == INVALID_REQUEST
        assert "method" in parsed["error"]["message"]

    @pytest.mark.asyncio
    async def test_method_not_found(self):
        """Test method not found error."""
        with patch("research_kb_daemon.server.dispatch", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.side_effect = ValueError("Method not found: unknown")

            response = await handle_request(
                json.dumps({"jsonrpc": "2.0", "method": "unknown", "id": 1}).encode()
            )
            parsed = json.loads(response.decode())

            assert parsed["error"]["code"] == METHOD_NOT_FOUND
            assert parsed["id"] == 1

    @pytest.mark.asyncio
    async def test_invalid_params(self):
        """Test invalid params error."""
        with patch("research_kb_daemon.server.dispatch", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.side_effect = ValueError("Missing required parameter: query")

            response = await handle_request(
                json.dumps({"jsonrpc": "2.0", "method": "search", "params": {}, "id": 1}).encode()
            )
            parsed = json.loads(response.decode())

            assert parsed["error"]["code"] == INVALID_PARAMS

    @pytest.mark.asyncio
    async def test_successful_request(self):
        """Test successful request handling."""
        with patch("research_kb_daemon.server.dispatch", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.return_value = {"status": "healthy"}

            response = await handle_request(
                json.dumps({"jsonrpc": "2.0", "method": "health", "id": 42}).encode()
            )
            parsed = json.loads(response.decode())

            assert parsed["jsonrpc"] == "2.0"
            assert parsed["id"] == 42
            assert parsed["result"]["status"] == "healthy"
            assert "error" not in parsed

    @pytest.mark.asyncio
    async def test_request_with_params(self):
        """Test request with parameters."""
        with patch("research_kb_daemon.server.dispatch", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.return_value = []

            response = await handle_request(
                json.dumps({
                    "jsonrpc": "2.0",
                    "method": "search",
                    "params": {"query": "test", "limit": 5},
                    "id": 1,
                }).encode()
            )

            # Verify dispatch was called with params
            mock_dispatch.assert_called_once_with("search", {"query": "test", "limit": 5})

    @pytest.mark.asyncio
    async def test_internal_error(self):
        """Test internal error handling."""
        with patch("research_kb_daemon.server.dispatch", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.side_effect = RuntimeError("Unexpected error")

            response = await handle_request(
                json.dumps({"jsonrpc": "2.0", "method": "stats", "id": 1}).encode()
            )
            parsed = json.loads(response.decode())

            assert parsed["error"]["code"] == INTERNAL_ERROR
            assert "error" in parsed["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_preserves_request_id(self):
        """Test request ID is preserved in response."""
        with patch("research_kb_daemon.server.dispatch", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.return_value = "ok"

            # String ID
            response = await handle_request(
                json.dumps({"jsonrpc": "2.0", "method": "health", "id": "abc-123"}).encode()
            )
            parsed = json.loads(response.decode())
            assert parsed["id"] == "abc-123"

            # Integer ID
            response = await handle_request(
                json.dumps({"jsonrpc": "2.0", "method": "health", "id": 999}).encode()
            )
            parsed = json.loads(response.decode())
            assert parsed["id"] == 999

    @pytest.mark.asyncio
    async def test_positional_params_rejected(self):
        """Test positional params are rejected."""
        response = await handle_request(
            json.dumps({
                "jsonrpc": "2.0",
                "method": "search",
                "params": ["test", 10],
                "id": 1,
            }).encode()
        )
        parsed = json.loads(response.decode())

        assert parsed["error"]["code"] == INVALID_PARAMS
        assert "positional" in parsed["error"]["message"].lower()
