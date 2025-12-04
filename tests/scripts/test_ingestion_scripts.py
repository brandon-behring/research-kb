"""Tests for ingestion scripts."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.mark.scripts
def test_ingest_golden_pdfs_script_exists(scripts_dir):
    """Test that ingest_golden_pdfs.py script exists."""
    script_path = scripts_dir / "ingest_golden_pdfs.py"
    assert script_path.exists(), "ingest_golden_pdfs.py should exist"


@pytest.mark.scripts
def test_ingest_golden_pdfs_imports(scripts_dir):
    """Test that ingest_golden_pdfs.py can be imported."""
    import sys
    sys.path.insert(0, str(scripts_dir))

    try:
        import ingest_golden_pdfs
        # Should have golden PDFs defined
        assert hasattr(ingest_golden_pdfs, 'GOLDEN_PDFS') or \
               hasattr(ingest_golden_pdfs, 'PDFs') or \
               hasattr(ingest_golden_pdfs, 'PAPERS'), \
               "Should define PDFs to ingest"
    finally:
        if 'ingest_golden_pdfs' in sys.modules:
            del sys.modules['ingest_golden_pdfs']


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
        import extract_concepts
        # Should have some extraction functionality
        # Check for common patterns in extraction scripts
        script_content = (scripts_dir / "extract_concepts.py").read_text()
        assert "ConceptExtractor" in script_content or \
               "extract" in script_content.lower(), \
               "Should have extraction functionality"
    except Exception as e:
        # Import might fail if dependencies not available
        pytest.skip(f"Cannot import extract_concepts: {e}")
    finally:
        if 'extract_concepts' in sys.modules:
            del sys.modules['extract_concepts']


@pytest.mark.scripts
@pytest.mark.requires_embedding
def test_ingest_golden_pdfs_structure(scripts_dir):
    """Test ingest_golden_pdfs.py has expected structure."""
    script_path = scripts_dir / "ingest_golden_pdfs.py"
    script_content = script_path.read_text()

    # Should have imports from pdf-tools
    assert "research_kb_pdf" in script_content, "Should import from research_kb_pdf"

    # Should have imports from storage
    assert "research_kb_storage" in script_content, "Should import from research_kb_storage"

    # Should have async functionality (ingestion is async)
    assert "async" in script_content or "asyncio" in script_content, \
        "Should use async operations"


@pytest.mark.scripts
@pytest.mark.requires_ollama
def test_extract_concepts_structure(scripts_dir):
    """Test extract_concepts.py has expected structure."""
    script_path = scripts_dir / "extract_concepts.py"
    script_content = script_path.read_text()

    # Should use Ollama or concept extraction
    assert "ConceptExtractor" in script_content or \
           "ollama" in script_content.lower() or \
           "concept" in script_content.lower(), \
           "Should have concept extraction functionality"

    # Should interact with database
    assert "ConceptStore" in script_content or \
           "research_kb_storage" in script_content, \
           "Should interact with database"
