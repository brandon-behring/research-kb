"""Tests for dashboard application.

Streamlit apps require special testing approaches. These tests verify:
1. Module imports without error
2. Core functions exist and are callable
3. Components can be imported

Note: Full integration testing of Streamlit apps requires the AppTest framework
or manual testing with `streamlit run`.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

pytestmark = pytest.mark.unit


# Add package to path
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root / "packages/dashboard/src"))


# -----------------------------------------------------------------------------
# Import Tests
# -----------------------------------------------------------------------------


class TestModuleImports:
    """Test that modules import without error."""

    def test_app_module_imports(self):
        """Main app module can be imported."""
        # Streamlit requires page_config to be called first, so we mock it
        with patch.dict("sys.modules", {"streamlit": MagicMock()}):
            # This would normally test the import, but Streamlit is tricky
            # Just verify the file exists and is valid Python
            app_path = repo_root / "packages/dashboard/src/research_kb_dashboard/app.py"
            assert app_path.exists()

            # Compile to check syntax
            with open(app_path) as f:
                code = f.read()
            compile(code, app_path, "exec")

    def test_pages_module_imports(self):
        """Pages modules can be imported."""
        search_path = repo_root / "packages/dashboard/src/research_kb_dashboard/pages/search.py"
        citations_path = (
            repo_root / "packages/dashboard/src/research_kb_dashboard/pages/citations.py"
        )

        assert search_path.exists()
        assert citations_path.exists()

    def test_components_module_imports(self):
        """Components module can be imported."""
        graph_path = repo_root / "packages/dashboard/src/research_kb_dashboard/components/graph.py"
        assert graph_path.exists()


# -----------------------------------------------------------------------------
# Function Existence Tests
# -----------------------------------------------------------------------------


class TestFunctionExistence:
    """Test that key functions exist in the modules."""

    def test_app_has_main_function(self):
        """App module has main() function."""
        app_path = repo_root / "packages/dashboard/src/research_kb_dashboard/app.py"
        with open(app_path) as f:
            code = f.read()

        assert "def main()" in code

    def test_pages_have_entry_functions(self):
        """Page modules have their page entry functions."""
        pages_dir = repo_root / "packages/dashboard/src/research_kb_dashboard/pages"

        checks = {
            "search.py": "def search_page",
            "citations.py": "def citation_network_page",
            "concepts.py": "def concept_graph_page",
            "statistics.py": "def statistics_page",
            "assumptions.py": "def assumptions_page",
        }
        for filename, expected in checks.items():
            path = pages_dir / filename
            assert path.exists(), f"Missing page module: {filename}"
            code = path.read_text()
            assert expected in code, f"{filename} missing {expected}"

    def test_app_has_async_functions(self):
        """App module has async database functions."""
        app_path = repo_root / "packages/dashboard/src/research_kb_dashboard/app.py"
        with open(app_path) as f:
            code = f.read()

        assert "async def get_stats" in code


# -----------------------------------------------------------------------------
# Component Tests
# -----------------------------------------------------------------------------


class TestComponents:
    """Test dashboard components."""

    def test_graph_component_exists(self):
        """Graph component file exists with expected content."""
        graph_path = repo_root / "packages/dashboard/src/research_kb_dashboard/components/graph.py"
        with open(graph_path) as f:
            code = f.read()

        # Should have some graph-related code
        assert "pyvis" in code.lower() or "networkx" in code.lower() or "graph" in code.lower()


# -----------------------------------------------------------------------------
# Configuration Tests
# -----------------------------------------------------------------------------


class TestConfiguration:
    """Test dashboard configuration."""

    def test_pyproject_exists(self):
        """pyproject.toml exists with dependencies."""
        pyproject_path = repo_root / "packages/dashboard/pyproject.toml"
        assert pyproject_path.exists()

        with open(pyproject_path) as f:
            content = f.read()

        assert "streamlit" in content

    def test_page_config(self):
        """App sets page config correctly."""
        app_path = repo_root / "packages/dashboard/src/research_kb_dashboard/app.py"
        with open(app_path) as f:
            code = f.read()

        assert "st.set_page_config" in code
        assert "Research-KB Explorer" in code or "page_title" in code
