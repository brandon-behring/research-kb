"""Tests for `research-kb graph export` CLI subcommand.

Covers the selector-resolution framework (D13b in the planning doc):
- Selector validation (exactly one selector enforced)
- `_load_lines` newline file parsing
- Output format validation
- Integration smoke test via mocked storage layer
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from research_kb_cli.commands.graph import _load_lines
from research_kb_cli.main import app

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _load_lines (pure helper)
# ---------------------------------------------------------------------------


class TestLoadLines:
    """Tests for `_load_lines` helper that parses newline-delimited files."""

    def test_loads_simple_lines(self, tmp_path: Path):
        f = tmp_path / "ids.txt"
        f.write_text("1707.06347\n1502.05477\n2106.01345\n")
        assert _load_lines(f) == ["1707.06347", "1502.05477", "2106.01345"]

    def test_strips_whitespace(self, tmp_path: Path):
        f = tmp_path / "ids.txt"
        f.write_text("  1707.06347  \n\t1502.05477\n")
        assert _load_lines(f) == ["1707.06347", "1502.05477"]

    def test_skips_blank_lines(self, tmp_path: Path):
        f = tmp_path / "ids.txt"
        f.write_text("1707.06347\n\n\n1502.05477\n\n")
        assert _load_lines(f) == ["1707.06347", "1502.05477"]

    def test_strips_hash_comments(self, tmp_path: Path):
        f = tmp_path / "ids.txt"
        f.write_text("# RL foundations\n" "1707.06347  # PPO\n" "1502.05477\n" "# end\n")
        assert _load_lines(f) == ["1707.06347", "1502.05477"]

    def test_empty_file_returns_empty_list(self, tmp_path: Path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        assert _load_lines(f) == []

    def test_only_comments_returns_empty_list(self, tmp_path: Path):
        f = tmp_path / "comments.txt"
        f.write_text("# just\n# comments\n\n# nothing else\n")
        assert _load_lines(f) == []


# ---------------------------------------------------------------------------
# Selector validation (CLI-level — at least one selector required; selectors union)
# ---------------------------------------------------------------------------


class TestSelectorValidation:
    """Tests for the selector-resolution framework's input validation."""

    def test_no_selector_fails_with_exit_code_2(self, cli_runner, tmp_path: Path):
        """No selector flag → typer.Exit(2) with descriptive error."""
        out = tmp_path / "out.json"
        result = cli_runner.invoke(app, ["graph", "export", "--output", str(out)])
        assert result.exit_code == 2
        assert "must specify at least one selector" in result.output

    def test_unsupported_format_fails_with_exit_code_2(self, cli_runner, tmp_path: Path):
        """Unsupported --format value → typer.Exit(2)."""
        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("1707.06347\n")
        out = tmp_path / "out.json"
        result = cli_runner.invoke(
            app,
            [
                "graph",
                "export",
                "--arxiv-ids",
                str(ids_file),
                "--output",
                str(out),
                "--format",
                "cytoscape-native",
            ],
        )
        assert result.exit_code == 2
        assert "not supported" in result.output

    def test_arxiv_ids_invokes_format_helper(self, cli_runner, tmp_path: Path, monkeypatch):
        """When --arxiv-ids is the only selector, the format helper is called
        and its dict result is written to --output."""
        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("1707.06347\n1502.05477\n")
        out = tmp_path / "subdir" / "out.json"

        fake_dict = {
            "schema_version": "1.0",
            "graph_type": "citation",
            "metadata": {
                "topic": "RL",
                "source": "research-kb",
                "generated_at": "2026-05-24T00:00:00+00:00",
                "node_count": 2,
                "edge_count": 0,
                "ingested_count": 0,
                "not_ingested_count": 2,
                "node_types": ["source"],
                "edge_types": [],
            },
            "nodes": [
                {"id": "source:arxiv:1707.06347", "type": "source", "label": "x", "data": {}},
                {"id": "source:arxiv:1502.05477", "type": "source", "label": "y", "data": {}},
            ],
            "edges": [],
        }

        # Patch the format helper at the import site used by the CLI command.
        async_mock = AsyncMock(return_value=fake_dict)
        with patch("research_kb_cli.commands.graph.format_citation_graph_export", async_mock):
            # Patch get_connection_pool so we don't try to open a real DB.
            with patch(
                "research_kb_cli.commands.graph.get_connection_pool", AsyncMock(return_value=None)
            ):
                result = cli_runner.invoke(
                    app,
                    [
                        "graph",
                        "export",
                        "--arxiv-ids",
                        str(ids_file),
                        "--output",
                        str(out),
                        "--topic-label",
                        "RL",
                    ],
                )

        assert result.exit_code == 0, result.output
        async_mock.assert_awaited_once()
        # Verify it was called with the loaded arxiv-ids list
        call_kwargs = async_mock.call_args.kwargs
        assert call_kwargs["arxiv_ids"] == ["1707.06347", "1502.05477"]
        assert call_kwargs["topic_label"] == "RL"

        # Verify the output file was written and contains the dict
        assert out.exists()
        import json

        written = json.loads(out.read_text())
        assert written == fake_dict
        # Subdir was created
        assert out.parent.is_dir()


# ---------------------------------------------------------------------------
# --role selector (new in A4 anchor-classics work)
# ---------------------------------------------------------------------------


class TestRoleSelector:
    """``--role`` selector + combined ``--arxiv-ids + --role`` (union)."""

    def test_role_only_invokes_format_helper(self, cli_runner, tmp_path: Path):
        """``--role`` alone is a valid selector; format helper receives it."""
        out = tmp_path / "role_only.json"
        fake_dict = {
            "schema_version": "1.0",
            "graph_type": "citation",
            "metadata": {
                "topic": "",
                "source": "research-kb",
                "generated_at": "2026-05-25T00:00:00+00:00",
                "node_count": 0,
                "edge_count": 0,
                "ingested_count": 0,
                "not_ingested_count": 0,
                "node_types": ["source"],
                "edge_types": [],
                "selectors": {
                    "arxiv_ids_count": 0,
                    "role": "anchor.rl_optimal_control",
                },
            },
            "nodes": [],
            "edges": [],
        }

        async_mock = AsyncMock(return_value=fake_dict)
        with patch("research_kb_cli.commands.graph.format_citation_graph_export", async_mock):
            with patch(
                "research_kb_cli.commands.graph.get_connection_pool",
                AsyncMock(return_value=None),
            ):
                result = cli_runner.invoke(
                    app,
                    [
                        "graph",
                        "export",
                        "--role",
                        "anchor.rl_optimal_control",
                        "--output",
                        str(out),
                    ],
                )

        assert result.exit_code == 0, result.output
        async_mock.assert_awaited_once()
        call_kwargs = async_mock.call_args.kwargs
        assert call_kwargs["arxiv_ids"] == []
        assert call_kwargs["role"] == "anchor.rl_optimal_control"

    def test_combined_arxiv_ids_and_role(self, cli_runner, tmp_path: Path):
        """``--arxiv-ids`` and ``--role`` together → union of selections."""
        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("1707.06347\n1502.05477\n")
        out = tmp_path / "combined.json"

        fake_dict = {
            "schema_version": "1.0",
            "graph_type": "citation",
            "metadata": {
                "topic": "",
                "source": "research-kb",
                "generated_at": "2026-05-25T00:00:00+00:00",
                "node_count": 0,
                "edge_count": 0,
                "ingested_count": 0,
                "not_ingested_count": 0,
                "node_types": ["source"],
                "edge_types": [],
            },
            "nodes": [],
            "edges": [],
        }

        async_mock = AsyncMock(return_value=fake_dict)
        with patch("research_kb_cli.commands.graph.format_citation_graph_export", async_mock):
            with patch(
                "research_kb_cli.commands.graph.get_connection_pool",
                AsyncMock(return_value=None),
            ):
                result = cli_runner.invoke(
                    app,
                    [
                        "graph",
                        "export",
                        "--arxiv-ids",
                        str(ids_file),
                        "--role",
                        "anchor.rl_optimal_control",
                        "--output",
                        str(out),
                    ],
                )

        assert result.exit_code == 0, result.output
        async_mock.assert_awaited_once()
        call_kwargs = async_mock.call_args.kwargs
        assert call_kwargs["arxiv_ids"] == ["1707.06347", "1502.05477"]
        assert call_kwargs["role"] == "anchor.rl_optimal_control"
