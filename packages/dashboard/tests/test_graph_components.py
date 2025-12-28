"""Tests for graph visualization components.

Tests the pure functions in graph.py that don't require Streamlit.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root / "packages/dashboard/src"))

from research_kb_dashboard.components.graph import (
    get_node_color,
    get_node_size,
    truncate_title,
    create_network,
)


class TestGetNodeColor:
    """Tests for get_node_color function."""

    def test_paper_color(self):
        """Papers get blue color."""
        assert get_node_color("paper") == "#4299e1"

    def test_paper_color_case_insensitive(self):
        """Color lookup is case-insensitive."""
        assert get_node_color("PAPER") == "#4299e1"
        assert get_node_color("Paper") == "#4299e1"

    def test_textbook_color(self):
        """Textbooks get green color."""
        assert get_node_color("textbook") == "#48bb78"

    def test_code_repo_color(self):
        """Code repos get orange color."""
        assert get_node_color("code_repo") == "#ed8936"

    def test_unknown_type_color(self):
        """Unknown types get gray color."""
        assert get_node_color("unknown") == "#a0aec0"

    def test_unrecognized_type_defaults_to_unknown(self):
        """Unrecognized types default to gray."""
        assert get_node_color("nonexistent") == "#a0aec0"
        assert get_node_color("") == "#a0aec0"


class TestGetNodeSize:
    """Tests for get_node_size function."""

    def test_min_authority_gives_min_size(self):
        """Zero authority gives minimum size."""
        result = get_node_size(0.0, min_size=10, max_size=50)
        assert result == 10

    def test_max_authority_gives_max_size(self):
        """Full authority gives maximum size."""
        result = get_node_size(1.0, min_size=10, max_size=50)
        assert result == 50

    def test_half_authority_gives_middle_size(self):
        """Half authority gives middle size."""
        result = get_node_size(0.5, min_size=10, max_size=50)
        assert result == 30  # 10 + 0.5 * (50 - 10) = 30

    def test_custom_size_range(self):
        """Custom min/max sizes are respected."""
        result = get_node_size(0.25, min_size=20, max_size=100)
        assert result == 40  # 20 + 0.25 * (100 - 20) = 40

    def test_returns_integer(self):
        """Result is always an integer."""
        result = get_node_size(0.333, min_size=10, max_size=50)
        assert isinstance(result, int)

    def test_default_parameters(self):
        """Default parameters work correctly."""
        # With defaults: min_size=10, max_size=50
        assert get_node_size(0.0) == 10
        assert get_node_size(1.0) == 50


class TestTruncateTitle:
    """Tests for truncate_title function."""

    def test_short_title_unchanged(self):
        """Short titles pass through unchanged."""
        title = "Short Paper"
        assert truncate_title(title) == title

    def test_exact_length_unchanged(self):
        """Title at exact limit unchanged."""
        title = "x" * 40
        assert truncate_title(title, max_length=40) == title

    def test_long_title_truncated(self):
        """Long titles are truncated with ellipsis."""
        title = "A Very Long Paper Title That Exceeds The Maximum Length"
        result = truncate_title(title, max_length=40)

        assert len(result) == 40
        assert result.endswith("...")

    def test_custom_max_length(self):
        """Custom max_length is respected."""
        title = "A Moderately Long Title"
        result = truncate_title(title, max_length=15)

        assert len(result) == 15
        assert result.endswith("...")

    def test_empty_string(self):
        """Empty string handled gracefully."""
        assert truncate_title("") == ""

    def test_ellipsis_included_in_length(self):
        """Ellipsis is included in the max_length."""
        title = "123456789012345"  # 15 chars
        result = truncate_title(title, max_length=10)

        assert len(result) == 10
        assert result == "1234567..."


class TestCreateNetwork:
    """Tests for create_network function."""

    def test_returns_network_instance(self):
        """create_network returns a Network instance."""
        with patch("research_kb_dashboard.components.graph.Network") as mock_network:
            mock_instance = MagicMock()
            mock_network.return_value = mock_instance

            result = create_network()

            assert result == mock_instance
            mock_network.assert_called_once()

    def test_default_parameters(self):
        """Network created with default parameters."""
        with patch("research_kb_dashboard.components.graph.Network") as mock_network:
            mock_instance = MagicMock()
            mock_network.return_value = mock_instance

            create_network()

            call_kwargs = mock_network.call_args[1]
            assert call_kwargs["height"] == "600px"
            assert call_kwargs["width"] == "100%"
            assert call_kwargs["bgcolor"] == "#ffffff"
            assert call_kwargs["font_color"] == "#000000"
            assert call_kwargs["directed"] is True
            assert call_kwargs["notebook"] is False

    def test_custom_parameters(self):
        """Network created with custom parameters."""
        with patch("research_kb_dashboard.components.graph.Network") as mock_network:
            mock_instance = MagicMock()
            mock_network.return_value = mock_instance

            create_network(
                height="800px",
                width="80%",
                bgcolor="#333333",
                font_color="#ffffff",
                directed=False,
            )

            call_kwargs = mock_network.call_args[1]
            assert call_kwargs["height"] == "800px"
            assert call_kwargs["width"] == "80%"
            assert call_kwargs["bgcolor"] == "#333333"
            assert call_kwargs["font_color"] == "#ffffff"
            assert call_kwargs["directed"] is False

    def test_physics_options_set(self):
        """Network physics options are configured."""
        with patch("research_kb_dashboard.components.graph.Network") as mock_network:
            mock_instance = MagicMock()
            mock_network.return_value = mock_instance

            create_network()

            # Verify set_options was called with physics config
            mock_instance.set_options.assert_called_once()
            options_arg = mock_instance.set_options.call_args[0][0]
            assert "physics" in options_arg
            assert "barnesHut" in options_arg
            assert "interaction" in options_arg


class TestGraphModuleImports:
    """Tests for module imports and existence."""

    def test_module_imports(self):
        """Module can be imported without error."""
        from research_kb_dashboard.components import graph
        assert hasattr(graph, "create_network")
        assert hasattr(graph, "render_network")
        assert hasattr(graph, "get_node_color")
        assert hasattr(graph, "get_node_size")
        assert hasattr(graph, "truncate_title")

    def test_color_palette_completeness(self):
        """All documented source types have colors."""
        expected_types = ["paper", "textbook", "code_repo", "unknown"]
        for source_type in expected_types:
            color = get_node_color(source_type)
            assert color.startswith("#")
            assert len(color) == 7  # #RRGGBB format
