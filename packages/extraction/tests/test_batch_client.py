"""Tests for Anthropic Batch API client."""

import json
import os
import sys
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from research_kb_extraction.models import ChunkExtraction

pytestmark = pytest.mark.unit


# Create a mock anthropic module for testing
@pytest.fixture(autouse=True)
def mock_anthropic_module():
    """Mock the anthropic module before any imports."""
    mock_anthropic = MagicMock()
    mock_client_instance = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client_instance

    # Insert into sys.modules before imports
    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        yield mock_anthropic, mock_client_instance


class TestDataclasses:
    """Tests for batch client dataclasses."""

    def test_batch_request_defaults(self):
        """Test BatchRequest dataclass with defaults."""
        from research_kb_extraction.batch_client import BatchRequest

        chunk_id = uuid4()
        request = BatchRequest(
            custom_id="test-id",
            chunk_id=chunk_id,
            chunk_content="Sample content",
        )

        assert request.custom_id == "test-id"
        assert request.chunk_id == chunk_id
        assert request.chunk_content == "Sample content"
        assert request.attempt == 1
        assert isinstance(request.created_at, datetime)

    def test_batch_result_success(self):
        """Test BatchResult dataclass for success case."""
        from research_kb_extraction.batch_client import BatchResult

        chunk_id = uuid4()
        extraction = ChunkExtraction()

        result = BatchResult(
            custom_id="test-id",
            chunk_id=chunk_id,
            extraction=extraction,
            success=True,
        )

        assert result.success is True
        assert result.error is None
        assert result.extraction is not None

    def test_batch_result_failure(self):
        """Test BatchResult dataclass for failure case."""
        from research_kb_extraction.batch_client import BatchResult

        chunk_id = uuid4()
        result = BatchResult(
            custom_id="test-id",
            chunk_id=chunk_id,
            extraction=None,
            error="API error",
            success=False,
        )

        assert result.success is False
        assert result.error == "API error"
        assert result.extraction is None

    def test_quarantined_chunk(self):
        """Test QuarantinedChunk dataclass."""
        from research_kb_extraction.batch_client import QuarantinedChunk

        chunk_id = uuid4()
        quarantined = QuarantinedChunk(
            chunk_id=chunk_id,
            error="Failed after max retries",
            attempts=3,
        )

        assert quarantined.chunk_id == chunk_id
        assert quarantined.error == "Failed after max retries"
        assert quarantined.attempts == 3
        assert isinstance(quarantined.quarantined_at, datetime)


class TestBatchClientInit:
    """Tests for BatchClient initialization."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_default_config(self, mock_anthropic_module):
        """Test default configuration."""
        from research_kb_extraction.batch_client import BatchClient

        client = BatchClient()

        assert client.model_name == "haiku-4.5"
        assert client.model_id == "claude-haiku-4-5-20251001"
        assert client.max_retries == 2

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_custom_config(self, mock_anthropic_module):
        """Test custom configuration."""
        from research_kb_extraction.batch_client import BatchClient

        client = BatchClient(
            model="sonnet",
            max_retries=5,
        )

        assert client.model_name == "sonnet"
        assert client.model_id == "claude-sonnet-4-20250514"
        assert client.max_retries == 5

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_model_alias_resolution(self, mock_anthropic_module):
        """Test model name to ID resolution."""
        from research_kb_extraction.batch_client import BatchClient

        # Test known aliases
        client1 = BatchClient(model="haiku")
        assert client1.model_id == "claude-3-haiku-20240307"

        client2 = BatchClient(model="haiku-3.5")
        assert client2.model_id == "claude-3-5-haiku-20241022"

        # Test unknown model (uses as-is)
        client3 = BatchClient(model="custom-model-id")
        assert client3.model_id == "custom-model-id"

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises(self, mock_anthropic_module):
        """Test missing API key raises ValueError."""
        from research_kb_extraction.batch_client import BatchClient

        # Ensure key is not in environment
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        with pytest.raises(ValueError) as exc_info:
            BatchClient()

        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-api-key"})
    def test_api_key_from_env(self, mock_anthropic_module):
        """Test API key is read from environment."""
        from research_kb_extraction.batch_client import BatchClient

        client = BatchClient()

        # Key should be used (stripped)
        assert client._api_key == "env-api-key"


class TestBuildRequest:
    """Tests for _build_request method."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_request_structure(self, mock_anthropic_module):
        """Test request has correct structure."""
        from research_kb_extraction.batch_client import BatchClient

        client = BatchClient(model="haiku-4.5")
        request = client._build_request("custom-123", "Sample chunk content", "causal_inference")

        assert request["custom_id"] == "custom-123"
        assert "params" in request
        assert request["params"]["model"] == "claude-haiku-4-5-20251001"
        assert request["params"]["max_tokens"] == 4096
        assert request["params"]["temperature"] == 0.1

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_includes_tool_choice(self, mock_anthropic_module):
        """Test request includes tool_choice."""
        from research_kb_extraction.batch_client import BatchClient

        client = BatchClient()
        request = client._build_request("id", "content", "causal_inference")

        assert "tool_choice" in request["params"]
        assert request["params"]["tool_choice"]["type"] == "tool"
        assert request["params"]["tool_choice"]["name"] == "extract_concepts"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_includes_system_prompt(self, mock_anthropic_module):
        """Test request includes system prompt."""
        from research_kb_extraction.batch_client import BatchClient

        client = BatchClient()
        request = client._build_request("id", "content", "causal_inference")

        assert "system" in request["params"]
        assert len(request["params"]["system"]) > 0


class TestSubmitBatch:
    """Tests for submit_batch method."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    async def test_empty_chunks_raises(self, mock_anthropic_module):
        """Test empty chunks list raises ValueError."""
        from research_kb_extraction.batch_client import BatchClient

        client = BatchClient()

        with pytest.raises(ValueError) as exc_info:
            await client.submit_batch([], domain_id="causal_inference")

        assert "No chunks" in str(exc_info.value)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    async def test_exceeds_max_raises(self, mock_anthropic_module):
        """Test exceeding max batch size raises ValueError."""
        from research_kb_extraction.batch_client import BatchClient

        client = BatchClient()

        # Create 100,001 chunks
        chunks = [(uuid4(), "content") for _ in range(100001)]

        with pytest.raises(ValueError) as exc_info:
            await client.submit_batch(chunks, domain_id="causal_inference")

        assert "100,000" in str(exc_info.value)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    async def test_success_returns_batch_id(self, mock_anthropic_module):
        """Test successful submission returns batch ID."""
        from research_kb_extraction.batch_client import BatchClient

        mock_anthropic, mock_client = mock_anthropic_module

        # Setup mock
        mock_batch = MagicMock()
        mock_batch.id = "batch-123"
        mock_batch.processing_status = "in_progress"
        mock_client.messages.batches.create.return_value = mock_batch

        client = BatchClient()
        chunks = [(uuid4(), "Test content")]

        batch_id = await client.submit_batch(chunks, domain_id="causal_inference")

        assert batch_id == "batch-123"


class TestGetBatchStatus:
    """Tests for get_batch_status method."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    async def test_returns_status_dict(self, mock_anthropic_module):
        """Test status returns dict with correct fields."""
        from research_kb_extraction.batch_client import BatchClient

        mock_anthropic, mock_client = mock_anthropic_module

        mock_batch = MagicMock()
        mock_batch.id = "batch-123"
        mock_batch.processing_status = "in_progress"
        mock_batch.created_at = "2024-01-01T00:00:00Z"
        mock_batch.ended_at = None
        mock_batch.request_counts = {"succeeded": 5, "errored": 0, "processing": 10}
        mock_client.messages.batches.retrieve.return_value = mock_batch

        client = BatchClient()
        status = await client.get_batch_status("batch-123")

        assert status["id"] == "batch-123"
        assert status["status"] == "in_progress"
        assert "request_counts" in status

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    async def test_handles_ended_status(self, mock_anthropic_module):
        """Test handling of ended batch status."""
        from research_kb_extraction.batch_client import BatchClient

        mock_anthropic, mock_client = mock_anthropic_module

        mock_batch = MagicMock()
        mock_batch.id = "batch-123"
        mock_batch.processing_status = "ended"
        mock_batch.created_at = "2024-01-01T00:00:00Z"
        mock_batch.ended_at = "2024-01-01T01:00:00Z"
        mock_batch.request_counts = {"succeeded": 15, "errored": 0, "processing": 0}
        mock_client.messages.batches.retrieve.return_value = mock_batch

        client = BatchClient()
        status = await client.get_batch_status("batch-123")

        assert status["status"] == "ended"
        assert status["ended_at"] == "2024-01-01T01:00:00Z"


class TestWaitForResults:
    """Tests for wait_for_results method."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    async def test_polls_until_ended(self, mock_anthropic_module):
        """Test polling continues until batch ends."""
        from research_kb_extraction.batch_client import BatchClient

        mock_anthropic, mock_client = mock_anthropic_module

        # Create mock that changes status after a few polls
        call_count = 0

        def mock_retrieve(batch_id):
            nonlocal call_count
            call_count += 1
            mock_batch = MagicMock()
            mock_batch.id = batch_id
            mock_batch.created_at = "2024-01-01"
            mock_batch.ended_at = None if call_count < 2 else "2024-01-01"
            mock_batch.processing_status = "in_progress" if call_count < 2 else "ended"
            mock_batch.request_counts = {}
            return mock_batch

        mock_client.messages.batches.retrieve = mock_retrieve
        mock_client.messages.batches.results.return_value = []

        client = BatchClient()

        # Use short poll interval for test
        with patch("asyncio.sleep", new_callable=AsyncMock):
            results = await client.wait_for_results("batch-123", poll_interval=0)

        assert call_count >= 2

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    async def test_raises_on_failed_status(self, mock_anthropic_module):
        """Test RuntimeError raised on failed batch."""
        from research_kb_extraction.batch_client import BatchClient

        mock_anthropic, mock_client = mock_anthropic_module

        mock_batch = MagicMock()
        mock_batch.id = "batch-123"
        mock_batch.processing_status = "failed"
        mock_batch.created_at = "2024-01-01"
        mock_batch.ended_at = None
        mock_batch.request_counts = {}
        mock_client.messages.batches.retrieve.return_value = mock_batch

        client = BatchClient()

        with pytest.raises(RuntimeError) as exc_info:
            await client.wait_for_results("batch-123")

        assert "failed" in str(exc_info.value)


class TestFetchResults:
    """Tests for _fetch_results method."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    async def test_parses_successful_results(self, mock_anthropic_module):
        """Test successful results are parsed correctly."""
        from research_kb_extraction.batch_client import BatchClient

        mock_anthropic, mock_client = mock_anthropic_module
        chunk_id = uuid4()

        # Create mock result with tool_use response
        mock_result = MagicMock()
        mock_result.custom_id = str(chunk_id)
        mock_result.result.type = "succeeded"

        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "extract_concepts"
        mock_tool_use.input = {
            "concepts": [{"name": "test concept", "concept_type": "method", "confidence": 0.9}],
            "relationships": [],
        }

        mock_result.result.message.content = [mock_tool_use]
        mock_client.messages.batches.results.return_value = [mock_result]

        client = BatchClient()
        results = await client._fetch_results("batch-123")

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].chunk_id == chunk_id

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    async def test_handles_failed_results(self, mock_anthropic_module):
        """Test failed results are handled correctly."""
        from research_kb_extraction.batch_client import BatchClient

        mock_anthropic, mock_client = mock_anthropic_module
        chunk_id = uuid4()

        mock_result = MagicMock()
        mock_result.custom_id = str(chunk_id)
        mock_result.result.type = "errored"
        mock_result.result.error = "Rate limit exceeded"
        mock_client.messages.batches.results.return_value = [mock_result]

        client = BatchClient()
        results = await client._fetch_results("batch-123")

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error is not None


class TestParseResponse:
    """Tests for _parse_response method."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_extracts_tool_use(self, mock_anthropic_module):
        """Test tool_use extraction from response."""
        from research_kb_extraction.batch_client import BatchClient

        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "extract_concepts"
        mock_tool_use.input = {
            "concepts": [{"name": "IV", "concept_type": "method", "confidence": 0.9}],
            "relationships": [],
        }

        mock_message = MagicMock()
        mock_message.content = [mock_tool_use]

        client = BatchClient()
        result = client._parse_response(mock_message)

        assert isinstance(result, ChunkExtraction)
        assert len(result.concepts) == 1

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_handles_missing_tool_use(self, mock_anthropic_module):
        """Test handling of response without tool_use."""
        from research_kb_extraction.batch_client import BatchClient

        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "No tool use here"

        mock_message = MagicMock()
        mock_message.content = [mock_text]

        client = BatchClient()
        result = client._parse_response(mock_message)

        # Should return empty extraction
        assert isinstance(result, ChunkExtraction)
        assert len(result.concepts) == 0

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_handles_parse_error(self, mock_anthropic_module):
        """Test handling of invalid tool input."""
        from research_kb_extraction.batch_client import BatchClient

        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "extract_concepts"
        mock_tool_use.input = {"invalid": "data"}  # Missing required fields

        mock_message = MagicMock()
        mock_message.content = [mock_tool_use]

        client = BatchClient()
        result = client._parse_response(mock_message)

        # Should return empty extraction on parse error
        assert isinstance(result, ChunkExtraction)


class TestExtractWithRetry:
    """Tests for extract_with_retry method."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    async def test_success_on_first_attempt(self, mock_anthropic_module):
        """Test successful extraction on first attempt."""
        from research_kb_extraction.batch_client import BatchClient

        mock_anthropic, mock_client = mock_anthropic_module
        chunk_id = uuid4()

        # Mock batch submission and results
        mock_batch = MagicMock()
        mock_batch.id = "batch-123"
        mock_batch.processing_status = "ended"
        mock_batch.created_at = "2024-01-01"
        mock_batch.ended_at = "2024-01-01"
        mock_batch.request_counts = {}

        mock_result = MagicMock()
        mock_result.custom_id = str(chunk_id)
        mock_result.result.type = "succeeded"

        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "extract_concepts"
        mock_tool_use.input = {"concepts": [], "relationships": []}
        mock_result.result.message.content = [mock_tool_use]

        mock_client.messages.batches.create.return_value = mock_batch
        mock_client.messages.batches.retrieve.return_value = mock_batch
        mock_client.messages.batches.results.return_value = [mock_result]

        client = BatchClient()
        chunks = [(chunk_id, "Test content")]

        results, quarantined = await client.extract_with_retry(chunks, domain_id="causal_inference")

        assert len(results) == 1
        assert len(quarantined) == 0

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    async def test_quarantines_after_max(self, mock_anthropic_module):
        """Test chunks are quarantined after max retries."""
        from research_kb_extraction.batch_client import BatchClient

        mock_anthropic, mock_client = mock_anthropic_module
        chunk_id = uuid4()

        # Mock batch that always fails
        mock_batch = MagicMock()
        mock_batch.id = "batch-123"
        mock_batch.processing_status = "ended"
        mock_batch.created_at = "2024-01-01"
        mock_batch.ended_at = "2024-01-01"
        mock_batch.request_counts = {}

        mock_result = MagicMock()
        mock_result.custom_id = str(chunk_id)
        mock_result.result.type = "errored"
        mock_result.result.error = "Always fails"

        mock_client.messages.batches.create.return_value = mock_batch
        mock_client.messages.batches.retrieve.return_value = mock_batch
        mock_client.messages.batches.results.return_value = [mock_result]

        client = BatchClient(max_retries=2)
        chunks = [(chunk_id, "Test content")]

        results, quarantined = await client.extract_with_retry(chunks, domain_id="causal_inference")

        assert len(results) == 0
        assert len(quarantined) == 1
        assert quarantined[0].chunk_id == chunk_id


class TestSaveQuarantineReport:
    """Tests for save_quarantine_report method."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_creates_file(self, mock_anthropic_module, tmp_path):
        """Test quarantine report creates file."""
        from research_kb_extraction.batch_client import BatchClient, QuarantinedChunk

        client = BatchClient(output_dir=tmp_path)
        client.quarantined.append(
            QuarantinedChunk(
                chunk_id=uuid4(),
                error="Test error",
                attempts=2,
            )
        )

        path = client.save_quarantine_report()

        assert path.exists()
        assert path.suffix == ".json"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_json_structure(self, mock_anthropic_module, tmp_path):
        """Test quarantine report has correct JSON structure."""
        from research_kb_extraction.batch_client import BatchClient, QuarantinedChunk

        chunk_id = uuid4()
        client = BatchClient(output_dir=tmp_path)
        client.quarantined.append(
            QuarantinedChunk(
                chunk_id=chunk_id,
                error="Test error",
                attempts=2,
            )
        )

        path = client.save_quarantine_report()

        with open(path) as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["chunk_id"] == str(chunk_id)
        assert data[0]["error"] == "Test error"
        assert data[0]["attempts"] == 2
        assert "quarantined_at" in data[0]

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_custom_path(self, mock_anthropic_module, tmp_path):
        """Test quarantine report with custom path."""
        from research_kb_extraction.batch_client import BatchClient, QuarantinedChunk

        custom_path = tmp_path / "custom_report.json"
        client = BatchClient(output_dir=tmp_path)
        client.quarantined.append(
            QuarantinedChunk(
                chunk_id=uuid4(),
                error="Error",
                attempts=1,
            )
        )

        path = client.save_quarantine_report(path=custom_path)

        assert path == custom_path
        assert custom_path.exists()


class TestExtractionMethod:
    """Tests for extraction_method property."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_extraction_method_format(self, mock_anthropic_module):
        """Test extraction_method returns correct format."""
        from research_kb_extraction.batch_client import BatchClient

        client = BatchClient(model="haiku-4.5")
        assert client.extraction_method == "anthropic-batch:haiku-4.5"

        client2 = BatchClient(model="sonnet")
        assert client2.extraction_method == "anthropic-batch:sonnet"
