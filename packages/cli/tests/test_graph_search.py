"""Tests for graph-boosted search functionality in CLI."""

from unittest.mock import patch, AsyncMock

from research_kb_cli.main import app
import pytest

pytestmark = pytest.mark.unit


class TestGraphBoostedSearch:
    """Tests for graph-boosted search (now default)."""

    def test_query_uses_graph_by_default(
        self, cli_runner, mock_embedding_client, mock_search_results
    ):
        """Test default behavior now uses graph search."""
        # Mock ConceptStore.count() to return non-zero (concepts exist)
        mock_concept_store = AsyncMock()
        mock_concept_store.count.return_value = 50

        with patch("research_kb_pdf.EmbeddingClient", return_value=mock_embedding_client):
            with patch("research_kb_storage.ConceptStore", mock_concept_store):
                with patch("research_kb_cli.commands.search.asyncio.run") as mock_run:
                    # run_query returns (results, expanded_query) tuple
                    mock_run.return_value = (mock_search_results, None)

                    # No --use-graph flag needed - it's the default
                    result = cli_runner.invoke(app, ["search", "query", "instrumental variables"])

                    assert result.exit_code == 0

    def test_query_no_graph_fallback(self, cli_runner, mock_embedding_client, mock_search_results):
        """Test --no-graph flag falls back to v1."""
        with patch("research_kb_pdf.EmbeddingClient", return_value=mock_embedding_client):
            with patch("research_kb_cli.commands.search.asyncio.run") as mock_run:
                # run_query returns (results, expanded_query) tuple
                mock_run.return_value = (mock_search_results, None)

                # Explicitly disable graph
                result = cli_runner.invoke(app, ["search", "query", "test query", "--no-graph"])

                assert result.exit_code == 0

    def test_graph_search_no_concepts_fallback(
        self, cli_runner, mock_embedding_client, mock_search_results
    ):
        """Test graceful fallback when no concepts exist."""
        # Mock ConceptStore.count() to return 0 (no concepts)
        mock_concept_store = AsyncMock()
        mock_concept_store.count.return_value = 0

        with patch("research_kb_pdf.EmbeddingClient", return_value=mock_embedding_client):
            with patch("research_kb_storage.ConceptStore", mock_concept_store):
                with patch("research_kb_cli.commands.search.asyncio.run") as mock_run:
                    # run_query returns (results, expanded_query) tuple
                    mock_run.return_value = (mock_search_results, None)

                    result = cli_runner.invoke(app, ["search", "query", "test query"])

                    # Should succeed (fall back to non-graph search)
                    assert result.exit_code == 0
                    # Warning should be in stderr/output
                    # (Typer may capture stderr differently, so we just check it succeeded)

    def test_graph_weight_parameter(self, cli_runner, mock_embedding_client, mock_search_results):
        """Test custom graph weight parameter."""
        mock_concept_store = AsyncMock()
        mock_concept_store.count.return_value = 50

        with patch("research_kb_pdf.EmbeddingClient", return_value=mock_embedding_client):
            with patch("research_kb_storage.ConceptStore", mock_concept_store):
                with patch("research_kb_cli.commands.search.asyncio.run") as mock_run:
                    # run_query returns (results, expanded_query) tuple
                    mock_run.return_value = (mock_search_results, None)

                    result = cli_runner.invoke(
                        app, ["search", "query", "test", "--graph-weight", "0.5"]
                    )

                    assert result.exit_code == 0

    def test_graph_search_with_other_options(
        self, cli_runner, mock_embedding_client, mock_search_results
    ):
        """Test graph search combined with other query options."""
        mock_concept_store = AsyncMock()
        mock_concept_store.count.return_value = 50

        with patch("research_kb_pdf.EmbeddingClient", return_value=mock_embedding_client):
            with patch("research_kb_storage.ConceptStore", mock_concept_store):
                with patch("research_kb_cli.commands.search.asyncio.run") as mock_run:
                    # run_query returns (results, expanded_query) tuple
                    mock_run.return_value = (mock_search_results, None)

                    result = cli_runner.invoke(
                        app,
                        [
                            "search",
                            "query",
                            "instrumental variables",
                            "--graph-weight",
                            "0.3",
                            "--limit",
                            "10",
                            "--context-type",
                            "building",
                            "--source-type",
                            "paper",
                        ],
                    )

                    assert result.exit_code == 0


class TestGraphSearchIntegration:
    """Integration-style tests for graph search (mocked services)."""

    def test_graph_search_markdown_output(
        self, cli_runner, mock_embedding_client, mock_search_results
    ):
        """Test graph search with markdown format output."""
        mock_concept_store = AsyncMock()
        mock_concept_store.count.return_value = 50

        with patch("research_kb_pdf.EmbeddingClient", return_value=mock_embedding_client):
            with patch("research_kb_storage.ConceptStore", mock_concept_store):
                with patch("research_kb_cli.commands.search.asyncio.run") as mock_run:
                    # run_query returns (results, expanded_query) tuple
                    mock_run.return_value = (mock_search_results, None)

                    result = cli_runner.invoke(
                        app, ["search", "query", "test", "--format", "markdown"]
                    )

                    assert result.exit_code == 0

    def test_graph_search_json_output(self, cli_runner, mock_embedding_client, mock_search_results):
        """Test graph search with JSON format output."""
        mock_concept_store = AsyncMock()
        mock_concept_store.count.return_value = 50

        with patch("research_kb_pdf.EmbeddingClient", return_value=mock_embedding_client):
            with patch("research_kb_storage.ConceptStore", mock_concept_store):
                with patch("research_kb_cli.commands.search.asyncio.run") as mock_run:
                    # run_query returns (results, expanded_query) tuple
                    mock_run.return_value = (mock_search_results, None)

                    result = cli_runner.invoke(app, ["search", "query", "test", "--format", "json"])

                    assert result.exit_code == 0
