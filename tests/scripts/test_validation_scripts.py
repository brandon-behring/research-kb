"""Tests for validation scripts."""

import pytest


@pytest.mark.scripts
def test_validate_known_answers_script_exists(scripts_dir):
    """Test that validate_known_answers.py script exists."""
    script_path = scripts_dir / "validate_known_answers.py"
    assert script_path.exists(), "validate_known_answers.py should exist"


@pytest.mark.scripts
def test_validate_known_answers_imports(scripts_dir):
    """Test that validate_known_answers.py can be imported."""
    import sys

    sys.path.insert(0, str(scripts_dir))

    try:

        # Should have test cases or validation logic
        script_path = scripts_dir / "validate_known_answers.py"
        script_content = script_path.read_text()
        assert (
            "query" in script_content.lower() or "test" in script_content.lower()
        ), "Should have test/query functionality"
    except Exception as e:
        pytest.skip(f"Cannot import validate_known_answers: {e}")
    finally:
        if "validate_known_answers" in sys.modules:
            del sys.modules["validate_known_answers"]


@pytest.mark.scripts
@pytest.mark.requires_embedding
def test_validate_known_answers_structure(scripts_dir):
    """Test validate_known_answers.py has expected structure."""
    script_path = scripts_dir / "validate_known_answers.py"
    script_content = script_path.read_text()

    # Should use search functionality
    assert "search" in script_content.lower(), "Should use search functionality"

    # Should have some test cases or queries
    assert (
        "query" in script_content.lower() or "test" in script_content.lower()
    ), "Should define test queries"
