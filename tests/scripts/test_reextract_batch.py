"""Tests for the batch/checkpoint helpers in scripts/reextract_with_mineru.py.

Covers ``load_work_list``, ``load_checkpoint``, and ``append_checkpoint``.
These helpers enable resumable Tier 1 / Tier 2 campaigns per issue #9.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from uuid import uuid4

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "reextract_with_mineru.py"
_spec = importlib.util.spec_from_file_location("reextract_with_mineru", _SCRIPT_PATH)
reextract = importlib.util.module_from_spec(_spec)
sys.modules["reextract_with_mineru"] = reextract
_spec.loader.exec_module(reextract)  # type: ignore[union-attr]

pytestmark = pytest.mark.unit


class TestLoadWorkList:
    """Tests for ``load_work_list``."""

    def test_accepts_audit_formula_gaps_output(self, tmp_path: Path):
        """The output shape of scripts/audit_formula_gaps.py --format json."""
        a, b = str(uuid4()), str(uuid4())
        path = tmp_path / "wl.json"
        path.write_text(
            json.dumps(
                {
                    "total_sources": 2,
                    "total_gap_chunks": 100,
                    "rows": [
                        {"source_id": a, "title": "A", "gap_chunks": 60},
                        {"source_id": b, "title": "B", "gap_chunks": 40},
                    ],
                }
            )
        )
        ids = reextract.load_work_list(path)
        assert ids == [a, b]

    def test_accepts_flat_list_of_strings(self, tmp_path: Path):
        a, b = str(uuid4()), str(uuid4())
        path = tmp_path / "wl.json"
        path.write_text(json.dumps([a, b]))
        assert reextract.load_work_list(path) == [a, b]

    def test_accepts_flat_list_of_objects(self, tmp_path: Path):
        a, b = str(uuid4()), str(uuid4())
        path = tmp_path / "wl.json"
        path.write_text(
            json.dumps([{"source_id": a, "note": "foo"}, {"source_id": b, "note": "bar"}])
        )
        assert reextract.load_work_list(path) == [a, b]

    def test_rejects_unknown_shape(self, tmp_path: Path):
        path = tmp_path / "wl.json"
        path.write_text(json.dumps({"unexpected": "format"}))
        with pytest.raises(ValueError, match="unrecognized work list shape"):
            reextract.load_work_list(path)

    def test_rejects_missing_source_id(self, tmp_path: Path):
        path = tmp_path / "wl.json"
        path.write_text(json.dumps([{"title": "no id here"}]))
        with pytest.raises(ValueError, match="no 'source_id'"):
            reextract.load_work_list(path)


class TestCheckpoint:
    """Tests for ``load_checkpoint`` + ``append_checkpoint``."""

    def _result(self, source_id: str, status: str = "ok") -> dict:
        return {
            "source_id": source_id,
            "title": "fixture",
            "old_count": 100,
            "new_count": 200,
            "old_gaps": 50,
            "new_gaps": 0,
            "status": status,
            "reason": "",
            "elapsed_s": 90.1,
        }

    def test_missing_file_returns_empty_map(self, tmp_path: Path):
        assert reextract.load_checkpoint(tmp_path / "nope.jsonl") == {}

    def test_round_trip_single_result(self, tmp_path: Path):
        path = tmp_path / "ckpt.jsonl"
        sid = str(uuid4())
        reextract.append_checkpoint(path, self._result(sid))
        back = reextract.load_checkpoint(path)
        assert list(back.keys()) == [sid]
        assert back[sid]["status"] == "ok"
        assert "checkpointed_at" in back[sid]  # timestamp injected

    def test_round_trip_multiple_append(self, tmp_path: Path):
        path = tmp_path / "ckpt.jsonl"
        a, b, c = str(uuid4()), str(uuid4()), str(uuid4())
        reextract.append_checkpoint(path, self._result(a))
        reextract.append_checkpoint(path, self._result(b))
        reextract.append_checkpoint(path, self._result(c, status="failed"))
        back = reextract.load_checkpoint(path)
        assert set(back.keys()) == {a, b, c}
        assert back[c]["status"] == "failed"
        # Each appended line should be valid JSON
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            json.loads(line)

    def test_tolerates_malformed_lines(self, tmp_path: Path):
        path = tmp_path / "ckpt.jsonl"
        a, b = str(uuid4()), str(uuid4())
        reextract.append_checkpoint(path, self._result(a))
        # Inject a broken line between two valid entries.
        with path.open("a") as fh:
            fh.write("{not valid json\n")
        reextract.append_checkpoint(path, self._result(b))
        back = reextract.load_checkpoint(path)
        assert set(back.keys()) == {a, b}

    def test_last_entry_wins_on_duplicate_source_id(self, tmp_path: Path):
        """If the same source_id appears twice (e.g., re-run over the same
        source for some reason), the later entry wins — this matches how an
        operator reasons about the tail of the file being most recent."""
        path = tmp_path / "ckpt.jsonl"
        sid = str(uuid4())
        reextract.append_checkpoint(path, self._result(sid, status="failed"))
        reextract.append_checkpoint(path, self._result(sid, status="ok"))
        back = reextract.load_checkpoint(path)
        assert back[sid]["status"] == "ok"

    def test_creates_parent_directory(self, tmp_path: Path):
        path = tmp_path / "nested" / "subdir" / "ckpt.jsonl"
        reextract.append_checkpoint(path, self._result(str(uuid4())))
        assert path.exists()
        assert path.parent.is_dir()
