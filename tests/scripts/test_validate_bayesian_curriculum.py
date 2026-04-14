"""Tests for the Bayesian curriculum validator."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.scripts
def test_validate_bayesian_curriculum_script_exists(scripts_dir):
    script_path = scripts_dir / "validate_bayesian_curriculum.py"
    assert script_path.exists(), "validate_bayesian_curriculum.py should exist"


@pytest.mark.scripts
def test_validate_bayesian_curriculum_runs(repo_root):
    script_path = repo_root / "scripts" / "validate_bayesian_curriculum.py"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "validation passed" in result.stdout.lower()
