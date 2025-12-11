"""Tests for Dead Letter Queue (DLQ) module."""

import json
from pathlib import Path

from research_kb_pdf.dlq import DLQEntry, DeadLetterQueue


class TestDLQEntry:
    """Tests for DLQEntry dataclass."""

    def test_dlq_entry_creation(self):
        """Test creating a DLQ entry with all fields."""
        entry = DLQEntry(
            id="test-uuid",
            file_path="/data/test.pdf",
            error_type="ValueError",
            error_message="Test error",
            traceback="Traceback...",
            timestamp="2025-01-01T00:00:00+00:00",
            retry_count=2,
            metadata={"size": 1024},
        )

        assert entry.id == "test-uuid"
        assert entry.file_path == "/data/test.pdf"
        assert entry.error_type == "ValueError"
        assert entry.error_message == "Test error"
        assert entry.retry_count == 2
        assert entry.metadata == {"size": 1024}

    def test_dlq_entry_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        entry = DLQEntry(
            id="test-uuid",
            file_path="/data/test.pdf",
            error_type="ValueError",
            error_message="Test error",
            traceback="Traceback...",
            timestamp="2025-01-01T00:00:00+00:00",
        )

        assert entry.metadata == {}
        assert entry.retry_count == 0


class TestDeadLetterQueue:
    """Tests for DeadLetterQueue class."""

    def test_dlq_creates_file(self, tmp_path):
        """Test DLQ creates file on initialization."""
        dlq_path = tmp_path / "test.jsonl"
        _dlq = DeadLetterQueue(dlq_path)

        assert dlq_path.exists()

    def test_dlq_creates_parent_dirs(self, tmp_path):
        """Test DLQ creates parent directories."""
        dlq_path = tmp_path / "nested" / "dir" / "test.jsonl"
        _dlq = DeadLetterQueue(dlq_path)

        assert dlq_path.exists()
        assert dlq_path.parent.exists()

    def test_add_entry(self, tmp_path):
        """Test adding an entry to DLQ."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        entry = dlq.add("/data/test.pdf", ValueError("Test error"))

        assert entry.id is not None
        assert entry.file_path == "/data/test.pdf"
        assert entry.error_type == "ValueError"
        assert entry.error_message == "Test error"
        assert entry.traceback  # Should have traceback
        assert entry.timestamp  # Should have timestamp

    def test_add_entry_with_metadata(self, tmp_path):
        """Test adding entry with metadata."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        entry = dlq.add(
            "/data/test.pdf",
            ValueError("Test error"),
            metadata={"file_size": 1024, "grobid_version": "0.8.0"},
        )

        assert entry.metadata["file_size"] == 1024
        assert entry.metadata["grobid_version"] == "0.8.0"

    def test_add_entry_with_retry_count(self, tmp_path):
        """Test adding entry with retry count."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        entry = dlq.add("/data/test.pdf", ValueError("Test error"), retry_count=3)

        assert entry.retry_count == 3

    def test_list_entries(self, tmp_path):
        """Test listing all DLQ entries."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        dlq.add("/data/test1.pdf", ValueError("Error 1"))
        dlq.add("/data/test2.pdf", TypeError("Error 2"))
        dlq.add("/data/test3.pdf", RuntimeError("Error 3"))

        entries = dlq.list()

        assert len(entries) == 3
        assert entries[0].file_path == "/data/test1.pdf"
        assert entries[1].file_path == "/data/test2.pdf"
        assert entries[2].file_path == "/data/test3.pdf"

    def test_list_empty_dlq(self, tmp_path):
        """Test listing empty DLQ returns empty list."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        entries = dlq.list()

        assert entries == []

    def test_list_filter_by_error_type(self, tmp_path):
        """Test filtering DLQ by error type."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        dlq.add("/data/test1.pdf", ValueError("Error 1"))
        dlq.add("/data/test2.pdf", TypeError("Error 2"))
        dlq.add("/data/test3.pdf", ValueError("Error 3"))

        value_errors = dlq.list(error_type="ValueError")
        type_errors = dlq.list(error_type="TypeError")

        assert len(value_errors) == 2
        assert len(type_errors) == 1
        assert all(e.error_type == "ValueError" for e in value_errors)

    def test_get_entry_by_id(self, tmp_path):
        """Test getting specific entry by ID."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        entry1 = dlq.add("/data/test1.pdf", ValueError("Error 1"))
        _entry2 = dlq.add("/data/test2.pdf", TypeError("Error 2"))

        retrieved = dlq.get(entry1.id)

        assert retrieved is not None
        assert retrieved.id == entry1.id
        assert retrieved.file_path == "/data/test1.pdf"

    def test_get_nonexistent_entry(self, tmp_path):
        """Test getting non-existent entry returns None."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        dlq.add("/data/test.pdf", ValueError("Error"))

        result = dlq.get("non-existent-id")

        assert result is None

    def test_remove_entry(self, tmp_path):
        """Test removing an entry from DLQ."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        entry1 = dlq.add("/data/test1.pdf", ValueError("Error 1"))
        entry2 = dlq.add("/data/test2.pdf", TypeError("Error 2"))

        result = dlq.remove(entry1.id)

        assert result is True
        assert dlq.count() == 1
        assert dlq.get(entry1.id) is None
        assert dlq.get(entry2.id) is not None

    def test_remove_nonexistent_entry(self, tmp_path):
        """Test removing non-existent entry returns False."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        dlq.add("/data/test.pdf", ValueError("Error"))

        result = dlq.remove("non-existent-id")

        assert result is False
        assert dlq.count() == 1

    def test_count_entries(self, tmp_path):
        """Test counting DLQ entries."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        assert dlq.count() == 0

        dlq.add("/data/test1.pdf", ValueError("Error 1"))
        assert dlq.count() == 1

        dlq.add("/data/test2.pdf", TypeError("Error 2"))
        assert dlq.count() == 2

    def test_count_by_error_type(self, tmp_path):
        """Test counting entries by error type."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        dlq.add("/data/test1.pdf", ValueError("Error 1"))
        dlq.add("/data/test2.pdf", TypeError("Error 2"))
        dlq.add("/data/test3.pdf", ValueError("Error 3"))

        assert dlq.count(error_type="ValueError") == 2
        assert dlq.count(error_type="TypeError") == 1
        assert dlq.count(error_type="RuntimeError") == 0

    def test_clear_all(self, tmp_path):
        """Test clearing all DLQ entries."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        dlq.add("/data/test1.pdf", ValueError("Error 1"))
        dlq.add("/data/test2.pdf", TypeError("Error 2"))

        count = dlq.clear_all()

        assert count == 2
        assert dlq.count() == 0

    def test_clear_empty_dlq(self, tmp_path):
        """Test clearing empty DLQ returns 0."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        count = dlq.clear_all()

        assert count == 0

    def test_entry_persistence(self, tmp_path):
        """Test entries persist across DLQ instances."""
        dlq_path = tmp_path / "test.jsonl"

        # Create first instance and add entry
        dlq1 = DeadLetterQueue(dlq_path)
        entry = dlq1.add("/data/test.pdf", ValueError("Error"))

        # Create second instance and verify entry exists
        dlq2 = DeadLetterQueue(dlq_path)
        entries = dlq2.list()

        assert len(entries) == 1
        assert entries[0].id == entry.id

    def test_jsonl_format(self, tmp_path):
        """Test DLQ file is valid JSONL."""
        dlq_path = tmp_path / "test.jsonl"
        dlq = DeadLetterQueue(dlq_path)

        dlq.add("/data/test1.pdf", ValueError("Error 1"))
        dlq.add("/data/test2.pdf", TypeError("Error 2"))

        # Manually read and parse JSONL
        with open(dlq_path) as f:
            lines = f.readlines()

        assert len(lines) == 2

        for line in lines:
            entry = json.loads(line)
            assert "id" in entry
            assert "file_path" in entry
            assert "error_type" in entry
            assert "error_message" in entry
            assert "timestamp" in entry

    def test_path_type_conversion(self, tmp_path):
        """Test file_path accepts Path objects."""
        dlq = DeadLetterQueue(tmp_path / "test.jsonl")

        entry = dlq.add(Path("/data/test.pdf"), ValueError("Error"))

        assert entry.file_path == "/data/test.pdf"
        assert isinstance(entry.file_path, str)
