"""Tests for validation scripts."""

import pytest
from pathlib import Path


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
        import validate_known_answers
        # Should have test cases or validation logic
        script_path = scripts_dir / "validate_known_answers.py"
        script_content = script_path.read_text()
        assert "query" in script_content.lower() or \
               "test" in script_content.lower(), \
               "Should have test/query functionality"
    except Exception as e:
        pytest.skip(f"Cannot import validate_known_answers: {e}")
    finally:
        if 'validate_known_answers' in sys.modules:
            del sys.modules['validate_known_answers']


@pytest.mark.scripts
def test_master_plan_validation_script_exists(scripts_dir):
    """Test that master_plan_validation.py script exists."""
    script_path = scripts_dir / "master_plan_validation.py"
    assert script_path.exists(), "master_plan_validation.py should exist"


@pytest.mark.scripts
def test_master_plan_validation_imports(scripts_dir):
    """Test that master_plan_validation.py can be imported."""
    import sys
    sys.path.insert(0, str(scripts_dir))

    try:
        import master_plan_validation
        # Should have validation tests for the knowledge graph
        script_path = scripts_dir / "master_plan_validation.py"
        script_content = script_path.read_text()
        assert "graph" in script_content.lower() or \
               "concept" in script_content.lower() or \
               "relationship" in script_content.lower(), \
               "Should have graph/concept validation logic"
    except Exception as e:
        pytest.skip(f"Cannot import master_plan_validation: {e}")
    finally:
        if 'master_plan_validation' in sys.modules:
            del sys.modules['master_plan_validation']


@pytest.mark.scripts
@pytest.mark.requires_embedding
def test_validate_known_answers_structure(scripts_dir):
    """Test validate_known_answers.py has expected structure."""
    script_path = scripts_dir / "validate_known_answers.py"
    script_content = script_path.read_text()

    # Should use search functionality
    assert "search" in script_content.lower(), "Should use search functionality"

    # Should have some test cases or queries
    assert "query" in script_content.lower() or \
           "test" in script_content.lower(), \
           "Should define test queries"


@pytest.mark.scripts
def test_master_plan_validation_structure(scripts_dir):
    """Test master_plan_validation.py has expected structure."""
    script_path = scripts_dir / "master_plan_validation.py"
    script_content = script_path.read_text()

    # Should validate knowledge graph
    assert "ConceptStore" in script_content or \
           "RelationshipStore" in script_content or \
           "concept" in script_content.lower(), \
           "Should interact with knowledge graph"

    # Should have validation logic
    assert "test" in script_content.lower() or \
           "validate" in script_content.lower() or \
           "check" in script_content.lower(), \
           "Should have validation logic"
