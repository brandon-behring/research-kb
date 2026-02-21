"""Tests for PDF Dispatcher and Dead Letter Queue."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from research_kb_pdf import DLQEntry, DeadLetterQueue, PDFDispatcher, IngestResult
from research_kb_contracts import SourceType

pytestmark = pytest.mark.unit


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
TEST_PDF = FIXTURES_DIR / "test_simple.pdf"


# ============================================================================
# Dead Letter Queue Tests
# ============================================================================


class TestDLQEntry:
    """Test DLQEntry dataclass."""

    def test_dlq_entry_creation(self):
        """Test creating a DLQEntry object."""
        entry = DLQEntry(
            id="test-id-123",
            file_path="/data/paper.pdf",
            error_type="GROBIDError",
            error_message="Service unavailable",
            traceback="Traceback: ...",
            timestamp="2025-11-29T12:00:00Z",
            retry_count=0,
            metadata={"file_size": 1024},
        )

        assert entry.id == "test-id-123"
        assert entry.file_path == "/data/paper.pdf"
        assert entry.error_type == "GROBIDError"
        assert entry.error_message == "Service unavailable"
        assert entry.retry_count == 0
        assert entry.metadata["file_size"] == 1024

    def test_dlq_entry_metadata_default(self):
        """Test DLQEntry initializes metadata to empty dict."""
        entry = DLQEntry(
            id="test-id",
            file_path="/data/paper.pdf",
            error_type="ValueError",
            error_message="Invalid PDF",
            traceback="...",
            timestamp="2025-11-29T12:00:00Z",
        )

        assert entry.metadata == {}
        assert isinstance(entry.metadata, dict)


class TestDeadLetterQueue:
    """Test DeadLetterQueue operations."""

    def test_dlq_initialization(self, tmp_path):
        """Test DLQ creates directory and file."""
        dlq_path = tmp_path / "dlq" / "failed.jsonl"
        dlq = DeadLetterQueue(dlq_path)

        assert dlq.dlq_path == dlq_path
        assert dlq.dlq_path.exists()
        assert dlq.dlq_path.stat().st_size == 0

    def test_dlq_add_entry(self, tmp_path):
        """Test adding an entry to DLQ."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        try:
            raise ValueError("Test error")
        except Exception as e:
            entry = dlq.add(
                file_path="/data/test.pdf",
                error=e,
                metadata={"file_size": 1024},
            )

        assert entry.error_type == "ValueError"
        assert entry.error_message == "Test error"
        assert entry.metadata["file_size"] == 1024
        assert entry.retry_count == 0
        assert "Traceback" in entry.traceback

        # Verify written to file
        assert dlq.dlq_path.exists()
        content = dlq.dlq_path.read_text()
        assert "ValueError" in content
        assert "Test error" in content

    def test_dlq_list_all(self, tmp_path):
        """Test listing all DLQ entries."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        # Add multiple entries
        errors = [ValueError("Error 1"), TypeError("Error 2"), KeyError("Error 3")]
        for err in errors:
            dlq.add("/data/test.pdf", err)

        entries = dlq.list()
        assert len(entries) == 3
        assert all(isinstance(e, DLQEntry) for e in entries)

    def test_dlq_list_empty(self, tmp_path):
        """Test listing returns empty list for new DLQ."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")
        entries = dlq.list()
        assert entries == []

    def test_dlq_list_filtered_by_error_type(self, tmp_path):
        """Test filtering DLQ entries by error type."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        # Add different error types
        dlq.add("/data/test1.pdf", ValueError("Error 1"))
        dlq.add("/data/test2.pdf", TypeError("Error 2"))
        dlq.add("/data/test3.pdf", ValueError("Error 3"))

        # Filter by ValueError
        value_errors = dlq.list(error_type="ValueError")
        assert len(value_errors) == 2
        assert all(e.error_type == "ValueError" for e in value_errors)

        # Filter by TypeError
        type_errors = dlq.list(error_type="TypeError")
        assert len(type_errors) == 1
        assert type_errors[0].error_type == "TypeError"

    def test_dlq_get_by_id(self, tmp_path):
        """Test retrieving specific DLQ entry by ID."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        entry1 = dlq.add("/data/test1.pdf", ValueError("Error 1"))
        _entry2 = dlq.add("/data/test2.pdf", TypeError("Error 2"))  # noqa: F841

        # Get by ID
        retrieved = dlq.get(entry1.id)
        assert retrieved is not None
        assert retrieved.id == entry1.id
        assert retrieved.error_type == "ValueError"

        # Get non-existent ID
        assert dlq.get("non-existent-id") is None

    def test_dlq_remove_entry(self, tmp_path):
        """Test removing a DLQ entry."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        entry1 = dlq.add("/data/test1.pdf", ValueError("Error 1"))
        entry2 = dlq.add("/data/test2.pdf", TypeError("Error 2"))

        assert dlq.count() == 2

        # Remove entry1
        removed = dlq.remove(entry1.id)
        assert removed is True
        assert dlq.count() == 1

        # Verify entry1 is gone
        assert dlq.get(entry1.id) is None
        assert dlq.get(entry2.id) is not None

        # Try removing non-existent entry
        removed = dlq.remove("non-existent-id")
        assert removed is False

    def test_dlq_count(self, tmp_path):
        """Test counting DLQ entries."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        assert dlq.count() == 0

        dlq.add("/data/test1.pdf", ValueError("Error 1"))
        dlq.add("/data/test2.pdf", TypeError("Error 2"))

        assert dlq.count() == 2

        # Count by error type
        assert dlq.count(error_type="ValueError") == 1
        assert dlq.count(error_type="TypeError") == 1
        assert dlq.count(error_type="KeyError") == 0

    def test_dlq_clear_all(self, tmp_path):
        """Test clearing all DLQ entries."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        dlq.add("/data/test1.pdf", ValueError("Error 1"))
        dlq.add("/data/test2.pdf", TypeError("Error 2"))
        dlq.add("/data/test3.pdf", KeyError("Error 3"))

        assert dlq.count() == 3

        cleared_count = dlq.clear_all()
        assert cleared_count == 3
        assert dlq.count() == 0
        assert dlq.list() == []

    def test_dlq_retry_count_increments(self, tmp_path):
        """Test retry_count field usage."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        # Add with retry_count
        entry = dlq.add("/data/test.pdf", ValueError("Error"), retry_count=3)
        assert entry.retry_count == 3

        # Retrieve and verify
        retrieved = dlq.get(entry.id)
        assert retrieved.retry_count == 3


# ============================================================================
# PDF Dispatcher Tests
# ============================================================================


class TestPDFDispatcher:
    """Test PDFDispatcher orchestration."""

    def test_dispatcher_initialization(self, tmp_path):
        """Test dispatcher initializes with GROBID client and DLQ."""
        dispatcher = PDFDispatcher(
            grobid_url="http://localhost:8070",
            dlq_path=tmp_path / "dlq.jsonl",
        )

        assert dispatcher.grobid_client is not None
        assert dispatcher.dlq is not None
        assert isinstance(dispatcher.dlq, DeadLetterQueue)

    def test_calculate_file_hash(self, tmp_path):
        """Test file hash calculation."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        dispatcher = PDFDispatcher(dlq_path=tmp_path / "dlq.jsonl")
        file_hash = dispatcher._calculate_file_hash(TEST_PDF)

        assert isinstance(file_hash, str)
        assert len(file_hash) == 64  # SHA256 hex digest

        # Same file should produce same hash
        file_hash2 = dispatcher._calculate_file_hash(TEST_PDF)
        assert file_hash == file_hash2

    def test_calculate_file_hash_missing_file(self, tmp_path):
        """Test file hash raises FileNotFoundError for missing file."""
        dispatcher = PDFDispatcher(dlq_path=tmp_path / "dlq.jsonl")

        with pytest.raises(FileNotFoundError):
            dispatcher._calculate_file_hash("nonexistent.pdf")

    @pytest.mark.asyncio
    async def test_ingest_pdf_missing_file(self, tmp_path):
        """Test ingest_pdf raises FileNotFoundError for missing PDF."""
        dispatcher = PDFDispatcher(dlq_path=tmp_path / "dlq.jsonl")

        with pytest.raises(FileNotFoundError):
            await dispatcher.ingest_pdf(
                pdf_path="nonexistent.pdf",
                source_type=SourceType.PAPER,
                title="Test Paper",
                domain_id="causal_inference",
            )

    @pytest.mark.asyncio
    @patch("research_kb_pdf.dispatcher.ChunkStore")
    @patch("research_kb_pdf.dispatcher.SourceStore")
    async def test_ingest_pdf_already_exists(self, mock_source_store, mock_chunk_store, tmp_path):
        """Test ingest_pdf returns existing source if already ingested."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        # Mock existing source
        from datetime import datetime, timezone
        from research_kb_contracts import Source

        existing_source = Source(
            id=uuid4(),
            source_type=SourceType.PAPER,
            title="Existing Paper",
            authors=["Author"],
            year=2025,
            file_path=str(TEST_PDF),
            file_hash="abc123",
            metadata={"total_headings": 5, "extraction_method": "pymupdf"},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_source_store.get_by_file_hash = AsyncMock(return_value=existing_source)
        mock_chunk_store.count_by_source = AsyncMock(return_value=10)

        dispatcher = PDFDispatcher(dlq_path=tmp_path / "dlq.jsonl")

        result = await dispatcher.ingest_pdf(
            pdf_path=TEST_PDF,
            source_type=SourceType.PAPER,
            title="Test Paper",
            domain_id="causal_inference",
        )

        # Result is now IngestResult
        assert isinstance(result, IngestResult)
        assert result.source.id == existing_source.id
        assert result.source.title == "Existing Paper"
        assert result.chunk_count == 10

        # Verify get_by_file_hash was called
        mock_source_store.get_by_file_hash.assert_called_once()

    @pytest.mark.asyncio
    @patch("research_kb_pdf.dispatcher.ChunkStore")
    @patch("research_kb_pdf.dispatcher.SourceStore")
    async def test_ingest_pdf_force_pymupdf(self, mock_source_store, mock_chunk_store, tmp_path):
        """Test ingest_pdf with force_pymupdf skips GROBID."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        # Mock no existing source
        mock_source_store.get_by_file_hash = AsyncMock(return_value=None)

        # Mock SourceStore.create to avoid DB interaction
        from datetime import datetime, timezone
        from research_kb_contracts import Source

        created_source = Source(
            id=uuid4(),
            source_type=SourceType.TEXTBOOK,
            title="Test Book",
            authors=[],
            year=None,
            file_path=str(TEST_PDF),
            file_hash="test_hash",
            metadata={"extraction_method": "pymupdf"},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_source_store.create = AsyncMock(return_value=created_source)
        mock_chunk_store.batch_create = AsyncMock(return_value=[])

        dispatcher = PDFDispatcher(dlq_path=tmp_path / "dlq.jsonl")

        result = await dispatcher.ingest_pdf(
            pdf_path=TEST_PDF,
            source_type=SourceType.TEXTBOOK,
            title="Test Book",
            domain_id="causal_inference",
            force_pymupdf=True,
            skip_embedding=True,  # Skip embedding for test
        )

        # Result is now IngestResult
        assert isinstance(result, IngestResult)
        assert result.source.metadata["extraction_method"] == "pymupdf"
        mock_source_store.create.assert_called_once()

    @pytest.mark.asyncio
    @patch("research_kb_pdf.dispatcher.ChunkStore")
    @patch("research_kb_pdf.dispatcher.SourceStore")
    async def test_ingest_pdf_grobid_fallback(self, mock_source_store, mock_chunk_store, tmp_path):
        """Test GROBID→PyMuPDF fallback when GROBID fails."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        # Mock no existing source
        mock_source_store.get_by_file_hash = AsyncMock(return_value=None)

        # Mock SourceStore.create
        from datetime import datetime, timezone
        from research_kb_contracts import Source

        created_source = Source(
            id=uuid4(),
            source_type=SourceType.PAPER,
            title="Test Paper",
            authors=[],
            year=None,
            file_path=str(TEST_PDF),
            file_hash="test_hash",
            metadata={"extraction_method": "pymupdf"},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_source_store.create = AsyncMock(return_value=created_source)
        mock_chunk_store.batch_create = AsyncMock(return_value=[])

        dispatcher = PDFDispatcher(
            grobid_url="http://localhost:8070",
            dlq_path=tmp_path / "dlq.jsonl",
        )

        # Mock GROBID client to fail (is_alive is sync, not async)
        dispatcher.grobid_client.is_alive = MagicMock(return_value=False)

        result = await dispatcher.ingest_pdf(
            pdf_path=TEST_PDF,
            source_type=SourceType.PAPER,
            title="Test Paper",
            domain_id="causal_inference",
            skip_embedding=True,  # Skip embedding for test
        )

        # Result is now IngestResult, should fall back to PyMuPDF
        assert isinstance(result, IngestResult)
        assert result.source.metadata["extraction_method"] == "pymupdf"
        mock_source_store.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_pdf_complete_failure_adds_to_dlq(self, tmp_path):
        """Test complete failure (both GROBID and PyMuPDF) adds to DLQ."""
        # Create a corrupted/empty PDF that will fail PyMuPDF extraction
        bad_pdf = tmp_path / "corrupted.pdf"
        bad_pdf.write_bytes(b"Not a real PDF")

        dispatcher = PDFDispatcher(dlq_path=tmp_path / "dlq.jsonl")

        # Mock GROBID to fail (is_alive is sync, not async)
        dispatcher.grobid_client.is_alive = MagicMock(return_value=False)

        # Mock SourceStore to avoid DB
        with patch("research_kb_pdf.dispatcher.SourceStore", None):
            with pytest.raises(ValueError, match="PDF ingestion failed"):
                await dispatcher.ingest_pdf(
                    pdf_path=bad_pdf,
                    source_type=SourceType.PAPER,
                    title="Bad Paper",
                    domain_id="causal_inference",
                )

        # Verify added to DLQ
        entries = dispatcher.dlq.list()
        assert len(entries) == 1
        assert entries[0].file_path == str(bad_pdf)
        assert entries[0].metadata["title"] == "Bad Paper"
        assert entries[0].retry_count == 0

    @pytest.mark.asyncio
    @patch("research_kb_pdf.dispatcher.CitationStore")
    @patch("research_kb_pdf.dispatcher.ChunkStore")
    @patch("research_kb_pdf.dispatcher.SourceStore")
    async def test_retry_from_dlq_success(
        self, mock_source_store, mock_chunk_store, mock_citation_store, tmp_path
    ):
        """Test successful retry from DLQ."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        dispatcher = PDFDispatcher(dlq_path=tmp_path / "dlq.jsonl")

        # Add entry to DLQ
        entry = dispatcher.dlq.add(
            file_path=TEST_PDF,
            error=ValueError("Initial failure"),
            metadata={
                "source_type": "paper",
                "title": "Test Paper",
                "domain_id": "causal_inference",
            },
        )

        # Mock SourceStore
        mock_source_store.get_by_file_hash = AsyncMock(return_value=None)

        from datetime import datetime, timezone
        from research_kb_contracts import Source

        created_source = Source(
            id=uuid4(),
            source_type=SourceType.PAPER,
            title="Test Paper",
            authors=[],
            year=None,
            file_path=str(TEST_PDF),
            file_hash="test_hash",
            metadata={"extraction_method": "pymupdf"},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_source_store.create = AsyncMock(return_value=created_source)
        mock_chunk_store.batch_create = AsyncMock(return_value=[])
        mock_citation_store.batch_create = AsyncMock(return_value=[])

        # Retry (with skip_embedding since we're testing)
        # Note: retry_from_dlq doesn't expose skip_embedding, but ingest_pdf is called
        # which will try to embed. We mock the embed_client to avoid failure.
        dispatcher.embed_client.embed = MagicMock(return_value=[0.1] * 1024)

        result = await dispatcher.retry_from_dlq(entry.id)

        # Result is now IngestResult
        assert result is not None
        assert isinstance(result, IngestResult)
        assert result.source.id == created_source.id

        # Entry should be removed from DLQ
        assert dispatcher.dlq.get(entry.id) is None

    @pytest.mark.asyncio
    async def test_retry_from_dlq_not_found(self, tmp_path):
        """Test retry with non-existent DLQ entry."""
        dispatcher = PDFDispatcher(dlq_path=tmp_path / "dlq.jsonl")

        result = await dispatcher.retry_from_dlq("non-existent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_retry_from_dlq_failure_increments_retry_count(self, tmp_path):
        """Test failed retry increments retry_count."""
        # Create bad PDF
        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(b"Not a PDF")

        dispatcher = PDFDispatcher(dlq_path=tmp_path / "dlq.jsonl")

        # Add entry
        entry = dispatcher.dlq.add(
            file_path=bad_pdf,
            error=ValueError("Initial failure"),
            retry_count=0,
            metadata={"source_type": "paper", "title": "Bad Paper"},
        )

        # Mock SourceStore to avoid DB
        with patch("research_kb_pdf.dispatcher.SourceStore", None):
            # Retry should fail
            with pytest.raises(ValueError):
                await dispatcher.retry_from_dlq(entry.id)

        # Old entry removed
        assert dispatcher.dlq.get(entry.id) is None

        # Check that there's an entry with retry_count = 1
        # (Note: There may be multiple entries if nested failures occur,
        #  but we care that retry_count was incremented)
        new_entries = dispatcher.dlq.list()
        assert len(new_entries) >= 1

        # Find entry with retry_count = 1
        retry_entries = [e for e in new_entries if e.retry_count == 1]
        assert len(retry_entries) >= 1
        assert retry_entries[0].file_path == str(bad_pdf)
