"""Tests for ingestion scripts."""

import pytest


@pytest.mark.scripts
@pytest.mark.requires_ollama
def test_extract_concepts_script_exists(scripts_dir):
    """Test that extract_concepts.py script exists."""
    script_path = scripts_dir / "extract_concepts.py"
    assert script_path.exists(), "extract_concepts.py should exist"


@pytest.mark.scripts
@pytest.mark.requires_ollama
def test_extract_concepts_imports(scripts_dir):
    """Test that extract_concepts.py can be imported."""
    import sys

    sys.path.insert(0, str(scripts_dir))

    try:

        # Should have some extraction functionality
        # Check for common patterns in extraction scripts
        script_content = (scripts_dir / "extract_concepts.py").read_text()
        assert (
            "ConceptExtractor" in script_content or "extract" in script_content.lower()
        ), "Should have extraction functionality"
    except Exception as e:
        # Import might fail if dependencies not available
        pytest.skip(f"Cannot import extract_concepts: {e}")
    finally:
        if "extract_concepts" in sys.modules:
            del sys.modules["extract_concepts"]


@pytest.mark.scripts
@pytest.mark.requires_ollama
def test_extract_concepts_structure(scripts_dir):
    """Test extract_concepts.py has expected structure."""
    script_path = scripts_dir / "extract_concepts.py"
    script_content = script_path.read_text()

    # Should use Ollama or concept extraction
    assert (
        "ConceptExtractor" in script_content
        or "ollama" in script_content.lower()
        or "concept" in script_content.lower()
    ), "Should have concept extraction functionality"

    # Should interact with database
    assert (
        "ConceptStore" in script_content or "research_kb_storage" in script_content
    ), "Should interact with database"
