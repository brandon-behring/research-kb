"""Unit tests for regenerate_golden_candidates.py.

Tests the pure-logic components: find_latest_golden_file, summary
computation, and entry processing.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from regenerate_golden_candidates import find_latest_golden_file, regenerate

pytestmark = pytest.mark.unit


# ── find_latest_golden_file ─────────────────────────────────────────────────


class TestFindLatestGoldenFile:
    """Test golden file discovery."""

    def test_finds_latest_by_date(self, tmp_path):
        """Returns lexicographically last file (most recent date)."""
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        (eval_dir / "golden_candidates_2026-01-01.json").write_text("[]")
        (eval_dir / "golden_candidates_2026-02-20.json").write_text("[]")

        with patch("regenerate_golden_candidates.FIXTURES_DIR", eval_dir):
            result = find_latest_golden_file()
            assert result.name == "golden_candidates_2026-02-20.json"

    def test_raises_when_empty(self, tmp_path):
        """FileNotFoundError when no golden files exist."""
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()

        with patch("regenerate_golden_candidates.FIXTURES_DIR", eval_dir):
            with pytest.raises(FileNotFoundError, match="No golden_candidates"):
                find_latest_golden_file()


# ── regenerate ──────────────────────────────────────────────────────────────


class TestRegenerate:
    """Test the regenerate function with mocked DB and embeddings."""

    @pytest.mark.asyncio
    @patch("regenerate_golden_candidates.EmbeddingClient")
    @patch("regenerate_golden_candidates.get_connection_pool")
    @patch("regenerate_golden_candidates.find_best_chunks_for_query")
    async def test_perfect_match_summary(
        self, mock_find_chunks, mock_pool, mock_embed_cls, tmp_path
    ):
        """All queries matched → match_rate == 1.0."""
        mock_pool.return_value = MagicMock()
        mock_embed_cls.return_value = MagicMock()

        chunk_id = str(uuid4())
        mock_find_chunks.return_value = [
            {"chunk_id": chunk_id, "similarity": 0.92, "content_preview": "...", "source_title": "Test"}
        ]

        # Write test input
        input_path = tmp_path / "golden_candidates_test.json"
        entries = [
            {
                "query": "what is DML?",
                "target_chunk_ids": [str(uuid4())],
                "domain": "causal_inference",
                "source_title": "A Great Paper",
                "difficulty": "easy",
            },
            {
                "query": "cross-fitting procedure",
                "target_chunk_ids": [str(uuid4())],
                "domain": "causal_inference",
                "source_title": "Another Paper",
                "difficulty": "medium",
            },
        ]
        input_path.write_text(json.dumps(entries))

        # Patch FIXTURES_DIR so output goes to tmp
        with patch("regenerate_golden_candidates.FIXTURES_DIR", tmp_path):
            summary = await regenerate(input_path, top_k=2, verify_only=True)

        assert summary["total"] == 2
        assert summary["matched"] == 2
        assert summary["failed"] == 0
        assert summary["match_rate"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    @patch("regenerate_golden_candidates.EmbeddingClient")
    @patch("regenerate_golden_candidates.get_connection_pool")
    @patch("regenerate_golden_candidates.find_best_chunks_for_query")
    async def test_partial_match_summary(
        self, mock_find_chunks, mock_pool, mock_embed_cls, tmp_path
    ):
        """Some queries fail → match_rate < 1.0."""
        mock_pool.return_value = MagicMock()
        mock_embed_cls.return_value = MagicMock()

        # First call succeeds, second returns no matches
        mock_find_chunks.side_effect = [
            [{"chunk_id": str(uuid4()), "similarity": 0.85, "content_preview": "...", "source_title": "A"}],
            [],  # No match
        ]

        input_path = tmp_path / "golden_candidates_test.json"
        entries = [
            {"query": "q1", "target_chunk_ids": [str(uuid4())], "domain": "d1", "source_title": "S1", "difficulty": "easy"},
            {"query": "q2", "target_chunk_ids": [str(uuid4())], "domain": "d2", "source_title": "S2", "difficulty": "hard"},
        ]
        input_path.write_text(json.dumps(entries))

        with patch("regenerate_golden_candidates.FIXTURES_DIR", tmp_path):
            summary = await regenerate(input_path, top_k=2, verify_only=True)

        assert summary["matched"] == 1
        assert summary["failed"] == 1
        assert summary["match_rate"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    @patch("regenerate_golden_candidates.EmbeddingClient")
    @patch("regenerate_golden_candidates.get_connection_pool")
    @patch("regenerate_golden_candidates.find_best_chunks_for_query")
    async def test_low_similarity_counted(
        self, mock_find_chunks, mock_pool, mock_embed_cls, tmp_path
    ):
        """Low-similarity matches are counted separately."""
        mock_pool.return_value = MagicMock()
        mock_embed_cls.return_value = MagicMock()

        mock_find_chunks.return_value = [
            {"chunk_id": str(uuid4()), "similarity": 0.3, "content_preview": "...", "source_title": "S"}
        ]

        input_path = tmp_path / "golden_candidates_test.json"
        entries = [
            {"query": "q1", "target_chunk_ids": [str(uuid4())], "domain": "d1", "source_title": "S1", "difficulty": "easy"},
        ]
        input_path.write_text(json.dumps(entries))

        with patch("regenerate_golden_candidates.FIXTURES_DIR", tmp_path):
            summary = await regenerate(input_path, top_k=1, verify_only=True)

        assert summary["low_similarity"] == 1
        assert summary["matched"] == 1

    @pytest.mark.asyncio
    @patch("regenerate_golden_candidates.EmbeddingClient")
    @patch("regenerate_golden_candidates.get_connection_pool")
    @patch("regenerate_golden_candidates.find_best_chunks_for_query")
    async def test_writes_output_file(
        self, mock_find_chunks, mock_pool, mock_embed_cls, tmp_path
    ):
        """Non-verify mode writes a new golden candidates file."""
        mock_pool.return_value = MagicMock()
        mock_embed_cls.return_value = MagicMock()

        new_chunk_id = str(uuid4())
        mock_find_chunks.return_value = [
            {"chunk_id": new_chunk_id, "similarity": 0.9, "content_preview": "...", "source_title": "S"}
        ]

        input_path = tmp_path / "golden_candidates_test.json"
        entries = [
            {"query": "test", "target_chunk_ids": [str(uuid4())], "domain": "d1", "source_title": "S1", "difficulty": "easy"},
        ]
        input_path.write_text(json.dumps(entries))

        with patch("regenerate_golden_candidates.FIXTURES_DIR", tmp_path):
            summary = await regenerate(input_path, top_k=1, verify_only=False)

        assert "output_path" in summary
        output = Path(summary["output_path"])
        assert output.exists()

        # Verify content
        with open(output) as f:
            written = json.load(f)
        assert len(written) == 1
        assert written[0]["target_chunk_ids"] == [new_chunk_id]
        # No internal _status fields in output
        assert "_status" not in written[0]

    @pytest.mark.asyncio
    @patch("regenerate_golden_candidates.EmbeddingClient")
    @patch("regenerate_golden_candidates.get_connection_pool")
    @patch("regenerate_golden_candidates.find_best_chunks_for_query")
    async def test_exception_counted_as_failure(
        self, mock_find_chunks, mock_pool, mock_embed_cls, tmp_path
    ):
        """Exceptions during chunk search count as failures."""
        mock_pool.return_value = MagicMock()
        mock_embed_cls.return_value = MagicMock()

        mock_find_chunks.side_effect = RuntimeError("DB connection lost")

        input_path = tmp_path / "golden_candidates_test.json"
        entries = [
            {"query": "q1", "target_chunk_ids": [str(uuid4())], "domain": "d1", "source_title": "S1", "difficulty": "easy"},
        ]
        input_path.write_text(json.dumps(entries))

        with patch("regenerate_golden_candidates.FIXTURES_DIR", tmp_path):
            summary = await regenerate(input_path, top_k=1, verify_only=True)

        assert summary["failed"] == 1
        assert summary["matched"] == 0
