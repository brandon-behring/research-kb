"""Unit tests for optimize_weights.py.

Tests the pure-logic components: WeightConfig normalization,
find_latest_golden_file, and the EvalCase dataclass.
"""

import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest

# Script uses sys.path manipulation; import directly after pathing
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from optimize_weights import (
    CURRENT_WEIGHTS,
    EvalCase,
    WeightConfig,
    find_latest_golden_file,
)

pytestmark = pytest.mark.unit


# ── WeightConfig ────────────────────────────────────────────────────────────


class TestWeightConfig:
    """Test WeightConfig normalization and serialization."""

    def test_normalization_sums_to_one(self):
        """Weights are normalized to sum to 1.0 on init."""
        w = WeightConfig(fts=0.3, vector=0.7, graph=0.2, citation=0.15)
        total = w.fts + w.vector + w.graph + w.citation
        assert total == pytest.approx(1.0, rel=1e-6)

    def test_uniform_weights(self):
        """Equal inputs produce equal outputs."""
        w = WeightConfig(fts=1.0, vector=1.0, graph=1.0, citation=1.0)
        assert w.fts == pytest.approx(0.25, rel=1e-6)
        assert w.vector == pytest.approx(0.25, rel=1e-6)
        assert w.graph == pytest.approx(0.25, rel=1e-6)
        assert w.citation == pytest.approx(0.25, rel=1e-6)

    def test_zero_total_no_crash(self):
        """All-zero weights don't cause division by zero."""
        w = WeightConfig(fts=0.0, vector=0.0, graph=0.0, citation=0.0)
        # Should remain zero (total == 0 guard)
        assert w.fts == 0.0
        assert w.vector == 0.0

    def test_single_dominant_weight(self):
        """A single large weight dominates."""
        w = WeightConfig(fts=0.0, vector=10.0, graph=0.0, citation=0.0)
        assert w.vector == pytest.approx(1.0, rel=1e-6)
        assert w.fts == pytest.approx(0.0, rel=1e-6)

    def test_to_dict_rounded(self):
        """to_dict() rounds to 4 decimal places."""
        w = WeightConfig(fts=0.3, vector=0.7, graph=0.2, citation=0.15)
        d = w.to_dict()
        assert "fts_weight" in d
        assert "vector_weight" in d
        assert "graph_weight" in d
        assert "citation_weight" in d
        # Sum should be ~1.0
        total = d["fts_weight"] + d["vector_weight"] + d["graph_weight"] + d["citation_weight"]
        assert total == pytest.approx(1.0, abs=0.01)

    def test_repr_format(self):
        """__repr__ is readable."""
        w = WeightConfig(fts=0.3, vector=0.7, graph=0.0, citation=0.0)
        r = repr(w)
        assert "fts=" in r
        assert "vec=" in r
        assert "graph=" in r
        assert "cite=" in r

    def test_to_dict_json_serializable(self):
        """to_dict output is JSON-serializable."""
        w = WeightConfig(fts=0.3, vector=0.7, graph=0.2, citation=0.15)
        serialized = json.dumps(w.to_dict())
        parsed = json.loads(serialized)
        assert parsed["fts_weight"] > 0


# ── CURRENT_WEIGHTS ─────────────────────────────────────────────────────────


class TestCurrentWeights:
    """Test that CURRENT_WEIGHTS are reasonable defaults."""

    def test_current_weights_sum_to_one(self):
        """Default weights sum to 1.0."""
        total = (
            CURRENT_WEIGHTS.fts
            + CURRENT_WEIGHTS.vector
            + CURRENT_WEIGHTS.graph
            + CURRENT_WEIGHTS.citation
        )
        assert total == pytest.approx(1.0, rel=1e-6)

    def test_vector_dominates(self):
        """Vector weight is the largest component (expected for semantic search)."""
        assert CURRENT_WEIGHTS.vector > CURRENT_WEIGHTS.fts
        assert CURRENT_WEIGHTS.vector > CURRENT_WEIGHTS.graph
        assert CURRENT_WEIGHTS.vector > CURRENT_WEIGHTS.citation


# ── EvalCase ────────────────────────────────────────────────────────────────


class TestEvalCase:
    """Test EvalCase dataclass."""

    def test_construction(self):
        """EvalCase can be constructed with all fields."""
        case = EvalCase(
            query="test query",
            target_chunk_ids={uuid4(), uuid4()},
            domain="causal_inference",
            source_title="Test Source",
            difficulty="medium",
            embedding=[0.1] * 1024,
        )
        assert case.query == "test query"
        assert len(case.target_chunk_ids) == 2
        assert len(case.embedding) == 1024


# ── find_latest_golden_file ─────────────────────────────────────────────────


class TestFindLatestGoldenFile:
    """Test golden file discovery."""

    def test_finds_latest_by_sort(self, tmp_path):
        """Returns the lexicographically last file (= most recent date)."""
        eval_dir = tmp_path / "fixtures" / "eval"
        eval_dir.mkdir(parents=True)

        # Create multiple files with different dates
        (eval_dir / "golden_candidates_2026-01-01.json").write_text("[]")
        (eval_dir / "golden_candidates_2026-02-20.json").write_text("[]")
        (eval_dir / "golden_candidates_2026-01-15.json").write_text("[]")

        with patch("optimize_weights.FIXTURES_DIR", eval_dir):
            result = find_latest_golden_file()
            assert result.name == "golden_candidates_2026-02-20.json"

    def test_raises_when_no_files(self, tmp_path):
        """FileNotFoundError when no golden files exist."""
        eval_dir = tmp_path / "fixtures" / "eval"
        eval_dir.mkdir(parents=True)

        with patch("optimize_weights.FIXTURES_DIR", eval_dir):
            with pytest.raises(FileNotFoundError, match="No golden_candidates"):
                find_latest_golden_file()

    def test_single_file(self, tmp_path):
        """Works with exactly one file."""
        eval_dir = tmp_path / "fixtures" / "eval"
        eval_dir.mkdir(parents=True)
        (eval_dir / "golden_candidates_2026-03-01.json").write_text("[]")

        with patch("optimize_weights.FIXTURES_DIR", eval_dir):
            result = find_latest_golden_file()
            assert result.name == "golden_candidates_2026-03-01.json"


# ── Softmax parameterization ───────────────────────────────────────────────


class TestSoftmaxParameterization:
    """Test the softmax constraint used in optimization.

    The optimizer uses softmax to map unconstrained params to the simplex.
    This tests the math matches what optimize_global expects.
    """

    def test_softmax_sums_to_one(self):
        """Softmax output sums to 1.0 for any input."""
        params = np.array([0.5, 1.2, -0.3, 0.1])
        exp_params = np.exp(params - np.max(params))
        w = exp_params / exp_params.sum()
        assert w.sum() == pytest.approx(1.0, rel=1e-6)

    def test_softmax_preserves_ordering(self):
        """Larger input → larger output weight."""
        params = np.array([0.1, 2.0, 0.5, 0.3])
        exp_params = np.exp(params - np.max(params))
        w = exp_params / exp_params.sum()
        assert w[1] == max(w)  # Index 1 (value=2.0) should be largest

    def test_log_current_weights_roundtrip(self):
        """log(current_weights) → softmax → ~current_weights."""
        cw = CURRENT_WEIGHTS
        x0 = np.log([cw.fts, cw.vector, cw.graph, cw.citation])
        exp_params = np.exp(x0 - np.max(x0))
        w = exp_params / exp_params.sum()

        recovered = WeightConfig(
            fts=float(w[0]),
            vector=float(w[1]),
            graph=float(w[2]),
            citation=float(w[3]),
        )
        # Should approximately match (normalization applied twice)
        assert recovered.fts == pytest.approx(cw.fts, abs=0.01)
        assert recovered.vector == pytest.approx(cw.vector, abs=0.01)
