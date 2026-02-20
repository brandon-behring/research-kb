"""Tests for checkpoint management for S2 enrichment jobs."""

import json
import signal
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
from uuid import UUID, uuid4

from s2_client.checkpoint import (
    EnrichmentCheckpoint,
    GracefulShutdown,
    list_checkpoints,
)

pytestmark = pytest.mark.unit


class TestEnrichmentCheckpointInit:
    """Tests for EnrichmentCheckpoint initialization."""

    def test_default_values(self):
        """Test default field values."""
        checkpoint = EnrichmentCheckpoint(job_id="test_job")

        assert checkpoint.job_id == "test_job"
        assert checkpoint.processed_ids == set()
        assert checkpoint.total_citations == 0
        assert checkpoint.matched == 0
        assert checkpoint.ambiguous == 0
        assert checkpoint.unmatched == 0
        assert checkpoint.errors == 0
        assert checkpoint.rps == 0.2
        assert checkpoint.checkpoint_interval == 100
        assert checkpoint.last_saved_at is None
        assert isinstance(checkpoint.started_at, datetime)

    def test_custom_values(self):
        """Test custom field values."""
        ids = {uuid4(), uuid4()}
        started = datetime(2025, 1, 1, tzinfo=timezone.utc)

        checkpoint = EnrichmentCheckpoint(
            job_id="custom_job",
            processed_ids=ids,
            total_citations=1000,
            started_at=started,
            matched=50,
            ambiguous=10,
            unmatched=20,
            errors=5,
            rps=0.5,
            checkpoint_interval=200,
        )

        assert checkpoint.job_id == "custom_job"
        assert checkpoint.processed_ids == ids
        assert checkpoint.total_citations == 1000
        assert checkpoint.started_at == started
        assert checkpoint.matched == 50


class TestEnrichmentCheckpointLoad:
    """Tests for checkpoint loading from disk."""

    def test_load_specific_job_id(self, tmp_path):
        """Test loading a specific checkpoint by job_id."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        test_data = {
            "job_id": "test_123",
            "processed_ids": [str(uuid4()), str(uuid4())],
            "total_citations": 500,
            "started_at": "2025-01-01T00:00:00+00:00",
            "last_saved_at": "2025-01-01T01:00:00+00:00",
            "stats": {
                "matched": 100,
                "ambiguous": 20,
                "unmatched": 30,
                "errors": 5,
            },
            "config": {
                "rps": 0.3,
                "checkpoint_interval": 150,
            },
        }

        checkpoint_file = checkpoint_dir / "test_123.json"
        checkpoint_file.write_text(json.dumps(test_data))

        with patch("s2_client.checkpoint.CHECKPOINT_DIR", checkpoint_dir):
            result = EnrichmentCheckpoint.load(job_id="test_123")

            assert result is not None
            assert result.job_id == "test_123"
            assert result.total_citations == 500
            assert result.matched == 100
            assert len(result.processed_ids) == 2

    def test_load_missing_job_returns_none(self, tmp_path):
        """Test loading non-existent checkpoint returns None."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        with patch("s2_client.checkpoint.CHECKPOINT_DIR", checkpoint_dir):
            result = EnrichmentCheckpoint.load(job_id="nonexistent")
            assert result is None

    def test_load_most_recent_without_job_id(self, tmp_path):
        """Test loading most recent checkpoint when no job_id given."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create two checkpoints with different timestamps
        for i, name in enumerate(["20250101_100000", "20250101_120000"]):
            data = {
                "job_id": name,
                "started_at": f"2025-01-01T{10 + i*2}:00:00+00:00",
                "processed_ids": [],
            }
            (checkpoint_dir / f"{name}.json").write_text(json.dumps(data))

        with patch("s2_client.checkpoint.CHECKPOINT_DIR", checkpoint_dir):
            result = EnrichmentCheckpoint.load()
            # Should load most recent (sorted reverse)
            assert result is not None

    def test_load_invalid_json_returns_none(self, tmp_path):
        """Test loading invalid JSON returns None."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        checkpoint_file = checkpoint_dir / "bad.json"
        checkpoint_file.write_text("not valid json {{{")

        with patch("s2_client.checkpoint.CHECKPOINT_DIR", checkpoint_dir):
            result = EnrichmentCheckpoint.load(job_id="bad")
            assert result is None


class TestEnrichmentCheckpointSave:
    """Tests for checkpoint saving to disk."""

    def test_save_creates_file(self, tmp_path):
        """Test save creates checkpoint file."""
        checkpoint_dir = tmp_path / "checkpoints"

        with patch("s2_client.checkpoint.CHECKPOINT_DIR", checkpoint_dir):
            checkpoint = EnrichmentCheckpoint(
                job_id="save_test",
                total_citations=100,
                matched=50,
            )

            path = checkpoint.save()

            assert path.exists()
            assert path.name == "save_test.json"

    def test_save_updates_last_saved_at(self, tmp_path):
        """Test save updates last_saved_at timestamp."""
        checkpoint_dir = tmp_path / "checkpoints"

        with patch("s2_client.checkpoint.CHECKPOINT_DIR", checkpoint_dir):
            checkpoint = EnrichmentCheckpoint(job_id="timestamp_test")
            assert checkpoint.last_saved_at is None

            checkpoint.save()

            assert checkpoint.last_saved_at is not None
            # Should be recent
            age = (datetime.now(timezone.utc) - checkpoint.last_saved_at).total_seconds()
            assert age < 5

    def test_save_preserves_processed_ids(self, tmp_path):
        """Test save correctly serializes UUIDs."""
        checkpoint_dir = tmp_path / "checkpoints"

        with patch("s2_client.checkpoint.CHECKPOINT_DIR", checkpoint_dir):
            ids = {uuid4(), uuid4(), uuid4()}
            checkpoint = EnrichmentCheckpoint(
                job_id="uuid_test",
                processed_ids=ids,
            )

            path = checkpoint.save()

            data = json.loads(path.read_text())
            saved_ids = {UUID(uid) for uid in data["processed_ids"]}
            assert saved_ids == ids


class TestEnrichmentCheckpointDelete:
    """Tests for checkpoint deletion."""

    def test_delete_removes_file(self, tmp_path):
        """Test delete removes checkpoint file."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        with patch("s2_client.checkpoint.CHECKPOINT_DIR", checkpoint_dir):
            checkpoint = EnrichmentCheckpoint(job_id="delete_test")
            path = checkpoint.save()
            assert path.exists()

            checkpoint.delete()
            assert not path.exists()

    def test_delete_nonexistent_file_no_error(self, tmp_path):
        """Test delete on non-existent file doesn't raise."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        with patch("s2_client.checkpoint.CHECKPOINT_DIR", checkpoint_dir):
            checkpoint = EnrichmentCheckpoint(job_id="never_saved")
            checkpoint.delete()  # Should not raise


class TestEnrichmentCheckpointProgress:
    """Tests for progress calculation properties."""

    def test_processed_count(self):
        """Test processed_count returns set size."""
        checkpoint = EnrichmentCheckpoint(
            job_id="test",
            processed_ids={uuid4(), uuid4(), uuid4()},
        )
        assert checkpoint.processed_count == 3

    def test_remaining(self):
        """Test remaining calculates correctly."""
        checkpoint = EnrichmentCheckpoint(
            job_id="test",
            processed_ids={uuid4(), uuid4()},
            total_citations=10,
        )
        assert checkpoint.remaining == 8

    def test_remaining_never_negative(self):
        """Test remaining never goes negative."""
        checkpoint = EnrichmentCheckpoint(
            job_id="test",
            processed_ids={uuid4() for _ in range(15)},
            total_citations=10,
        )
        assert checkpoint.remaining == 0

    def test_elapsed_seconds(self):
        """Test elapsed_seconds calculation."""
        start = datetime.now(timezone.utc) - timedelta(seconds=100)
        checkpoint = EnrichmentCheckpoint(job_id="test", started_at=start)

        elapsed = checkpoint.elapsed_seconds
        assert 99 <= elapsed <= 102  # Allow small variance

    def test_rate_per_second(self):
        """Test rate_per_second calculation."""
        start = datetime.now(timezone.utc) - timedelta(seconds=100)
        checkpoint = EnrichmentCheckpoint(
            job_id="test",
            started_at=start,
            processed_ids={uuid4() for _ in range(50)},
        )

        rate = checkpoint.rate_per_second
        assert 0.4 <= rate <= 0.6  # ~50/100 = 0.5

    def test_rate_per_second_zero_elapsed(self):
        """Test rate_per_second returns 0 with no elapsed time."""
        checkpoint = EnrichmentCheckpoint(
            job_id="test",
            started_at=datetime.now(timezone.utc),
        )
        assert checkpoint.rate_per_second == 0.0

    def test_eta_seconds(self):
        """Test ETA calculation."""
        start = datetime.now(timezone.utc) - timedelta(seconds=100)
        checkpoint = EnrichmentCheckpoint(
            job_id="test",
            started_at=start,
            processed_ids={uuid4() for _ in range(50)},
            total_citations=100,
        )

        eta = checkpoint.eta_seconds
        # 50 remaining at 0.5/sec = 100 seconds
        assert eta is not None
        assert 90 <= eta <= 110

    def test_eta_seconds_none_when_no_progress(self):
        """Test ETA is None when no progress made."""
        checkpoint = EnrichmentCheckpoint(
            job_id="test",
            total_citations=100,
        )
        assert checkpoint.eta_seconds is None

    def test_format_progress(self):
        """Test progress string formatting."""
        start = datetime.now(timezone.utc) - timedelta(hours=1)
        checkpoint = EnrichmentCheckpoint(
            job_id="test",
            started_at=start,
            processed_ids={uuid4() for _ in range(500)},
            total_citations=1000,
        )

        progress = checkpoint.format_progress()
        assert "500/1000" in progress
        assert "50.0%" in progress
        # Should have ETA
        assert "ETA:" in progress

    def test_format_stats(self):
        """Test stats string formatting."""
        checkpoint = EnrichmentCheckpoint(
            job_id="test",
            matched=100,
            ambiguous=20,
            unmatched=30,
            errors=5,
        )

        stats = checkpoint.format_stats()
        assert "Matched: 100" in stats
        assert "Ambiguous: 20" in stats
        assert "Unmatched: 30" in stats
        assert "Errors: 5" in stats


class TestGracefulShutdown:
    """Tests for graceful shutdown signal handling."""

    def test_context_manager_sets_handlers(self):
        """Test context manager installs signal handlers."""
        checkpoint = EnrichmentCheckpoint(job_id="test")

        with patch("signal.signal") as mock_signal:
            with GracefulShutdown(checkpoint) as shutdown:
                # Should have set SIGINT and SIGTERM handlers
                assert mock_signal.call_count >= 2

    def test_context_manager_restores_handlers(self):
        """Test context manager restores original handlers on exit."""
        checkpoint = EnrichmentCheckpoint(job_id="test")
        original_sigint = signal.getsignal(signal.SIGINT)

        with GracefulShutdown(checkpoint):
            pass

        # Handler should be restored
        current_sigint = signal.getsignal(signal.SIGINT)
        assert current_sigint == original_sigint

    def test_shutdown_not_requested_initially(self):
        """Test shutdown_requested is False initially."""
        checkpoint = EnrichmentCheckpoint(job_id="test")

        with GracefulShutdown(checkpoint) as shutdown:
            assert shutdown.shutdown_requested is False

    def test_handle_signal_sets_flag(self, tmp_path):
        """Test signal handler sets shutdown_requested flag."""
        checkpoint_dir = tmp_path / "checkpoints"

        with patch("s2_client.checkpoint.CHECKPOINT_DIR", checkpoint_dir):
            checkpoint = EnrichmentCheckpoint(job_id="signal_test")

            shutdown = GracefulShutdown(checkpoint)
            shutdown.__enter__()

            # Simulate signal
            shutdown._handle_signal(signal.SIGINT, None)

            assert shutdown.shutdown_requested is True
            shutdown.__exit__(None, None, None)

    def test_handle_signal_saves_checkpoint(self, tmp_path):
        """Test signal handler saves checkpoint."""
        checkpoint_dir = tmp_path / "checkpoints"

        with patch("s2_client.checkpoint.CHECKPOINT_DIR", checkpoint_dir):
            checkpoint = EnrichmentCheckpoint(
                job_id="save_on_signal",
                matched=42,
            )

            shutdown = GracefulShutdown(checkpoint)
            shutdown.__enter__()

            # Simulate signal
            shutdown._handle_signal(signal.SIGTERM, None)

            # Checkpoint should be saved
            checkpoint_file = checkpoint_dir / "save_on_signal.json"
            assert checkpoint_file.exists()

            data = json.loads(checkpoint_file.read_text())
            assert data["stats"]["matched"] == 42

            shutdown.__exit__(None, None, None)


class TestListCheckpoints:
    """Tests for listing available checkpoints."""

    def test_list_empty_directory(self, tmp_path):
        """Test listing empty checkpoint directory."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        with patch("s2_client.checkpoint.CHECKPOINT_DIR", checkpoint_dir):
            result = list_checkpoints()
            assert result == []

    def test_list_multiple_checkpoints(self, tmp_path):
        """Test listing multiple checkpoints."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        for i, name in enumerate(["job_a", "job_b", "job_c"]):
            data = {
                "job_id": name,
                "started_at": f"2025-01-0{i+1}T00:00:00+00:00",
                "processed_ids": [str(uuid4())],
                "total_citations": (i + 1) * 100,
                "stats": {"matched": i * 10},
            }
            (checkpoint_dir / f"{name}.json").write_text(json.dumps(data))

        with patch("s2_client.checkpoint.CHECKPOINT_DIR", checkpoint_dir):
            result = list_checkpoints()

            assert len(result) == 3
            # Check fields are extracted
            job_ids = {cp["job_id"] for cp in result}
            assert job_ids == {"job_a", "job_b", "job_c"}

    def test_list_skips_invalid_files(self, tmp_path):
        """Test listing skips invalid JSON files."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Valid checkpoint
        (checkpoint_dir / "valid.json").write_text(
            json.dumps(
                {
                    "job_id": "valid",
                    "started_at": "2025-01-01T00:00:00+00:00",
                    "processed_ids": [],
                }
            )
        )

        # Invalid JSON
        (checkpoint_dir / "invalid.json").write_text("not json")

        with patch("s2_client.checkpoint.CHECKPOINT_DIR", checkpoint_dir):
            result = list_checkpoints()

            assert len(result) == 1
            assert result[0]["job_id"] == "valid"
