"""Tests for CLI commands."""

from unittest.mock import patch
from uuid import uuid4


# Import the CLI app
from research_kb_cli.main import app
import pytest

pytestmark = pytest.mark.unit


# ============================================================================
# Tier 1: Simple Commands
# ============================================================================


class TestSourcesCommand:
    """Tests for the sources command."""

    def test_sources_empty_database(self, cli_runner):
        """Test sources command with no data."""
        with patch("research_kb_cli.main.asyncio.run") as mock_run:
            mock_run.return_value = []  # Empty list

            result = cli_runner.invoke(app, ["sources"])

            assert result.exit_code == 0
            assert "No sources" in result.stdout

    def test_sources_with_data(self, cli_runner):
        """Test sources command lists all sources."""
        from research_kb_contracts import Source, SourceType
        from datetime import datetime

        # Create mock sources
        mock_sources = [
            Source(
                id=uuid4(),
                title="Test Paper 1",
                authors=["Author A"],
                year=2023,
                source_type=SourceType.PAPER,
                file_hash="hash1",
                metadata={},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
            Source(
                id=uuid4(),
                title="Test Paper 2",
                authors=["Author B", "Author C"],
                year=2024,
                source_type=SourceType.TEXTBOOK,
                file_hash="hash2",
                metadata={},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ]

        with patch("research_kb_cli.main.asyncio.run") as mock_run:
            mock_run.return_value = mock_sources

            result = cli_runner.invoke(app, ["sources"])

            assert result.exit_code == 0
            assert "Test Paper 1" in result.stdout
            assert "Test Paper 2" in result.stdout
            assert "Found 2 sources" in result.stdout

    def test_sources_database_connection_error(self, cli_runner):
        """Test error handling when DB unavailable."""
        with patch("research_kb_cli.main.asyncio.run", side_effect=ConnectionError("DB down")):
            result = cli_runner.invoke(app, ["sources"])

            assert result.exit_code == 1
            # Typer writes errors to stdout or output, check both
            assert "Error" in result.output or "Error" in result.stdout


class TestStatsCommand:
    """Tests for the stats command."""

    def test_stats_empty_database(self, cli_runner):
        """Test stats with zero data."""
        with patch("research_kb_cli.main.asyncio.run") as mock_run:
            mock_run.return_value = (0, 0, [])  # source_count, chunk_count, by_type

            result = cli_runner.invoke(app, ["stats"])

            assert result.exit_code == 0
            assert "Total sources: 0" in result.stdout
            assert "Total chunks:  0" in result.stdout

    def test_stats_with_data(self, cli_runner):
        """Test stats with ingested data."""
        with patch("research_kb_cli.main.asyncio.run") as mock_run:
            by_type = [
                {"source_type": "paper", "count": 5},
                {"source_type": "textbook", "count": 2},
            ]
            mock_run.return_value = (7, 150, by_type)

            result = cli_runner.invoke(app, ["stats"])

            assert result.exit_code == 0
            assert "Total sources: 7" in result.stdout
            assert "Total chunks:  150" in result.stdout
            assert "paper" in result.stdout
            assert "textbook" in result.stdout


class TestExtractionStatusCommand:
    """Tests for the extraction-status command."""

    def test_extraction_status_no_extractions(self, cli_runner):
        """Test extraction-status with no concepts."""
        with patch("research_kb_cli.main.asyncio.run") as mock_run:
            mock_run.return_value = {
                "concept_count": 0,
                "concepts_by_type": [],
                "relationship_count": 0,
                "relationships_by_type": [],
                "validated_count": 0,
                "avg_confidence": None,
                "confidence_dist": [],
                "chunks_with_concepts": 0,
                "total_chunks": 100,
            }

            result = cli_runner.invoke(app, ["extraction-status"])

            assert result.exit_code == 0
            assert "Total concepts extracted: 0" in result.stdout

    def test_extraction_status_with_extractions(self, cli_runner):
        """Test extraction-status shows aggregations."""
        with patch("research_kb_cli.main.asyncio.run") as mock_run:
            mock_run.return_value = {
                "concept_count": 50,
                "concepts_by_type": [
                    {"concept_type": "method", "count": 30},
                    {"concept_type": "assumption", "count": 20},
                ],
                "relationship_count": 75,
                "relationships_by_type": [
                    {"relationship_type": "REQUIRES", "count": 40},
                    {"relationship_type": "USES", "count": 35},
                ],
                "validated_count": 45,
                "avg_confidence": 0.85,
                "confidence_dist": [
                    {"confidence_range": "High (>=0.9)", "count": 20},
                    {"confidence_range": "Medium (0.7-0.9)", "count": 25},
                ],
                "chunks_with_concepts": 80,
                "total_chunks": 100,
            }

            result = cli_runner.invoke(app, ["extraction-status"])

            assert result.exit_code == 0
            assert "Total concepts extracted: 50" in result.stdout
            assert "method" in result.stdout
            assert "Average confidence: 0.85" in result.stdout


# ============================================================================
# Tier 2: Graph Commands
# ============================================================================


class TestConceptsCommand:
    """Tests for the concepts command."""

    def test_concepts_exact_match(self, cli_runner, mock_concepts):
        """Test exact concept name matching."""
        with patch("research_kb_cli.main.asyncio.run") as mock_run:
            mock_run.return_value = ([mock_concepts[0]], [])

            result = cli_runner.invoke(app, ["concepts", "Instrumental Variables"])

            # asyncio.run is called, but we need to check if it returns the right value
            # The actual result depends on the mock
            assert result.exit_code == 0

    def test_concepts_fuzzy_search(self, cli_runner, mock_concepts):
        """Test fuzzy matching (e.g., 'iv' matches 'instrumental variables')."""

        async def mock_search():
            return [mock_concepts[0]], []

        with patch("research_kb_cli.main.asyncio.run") as mock_run:
            mock_run.return_value = ([mock_concepts[0]], [])

            result = cli_runner.invoke(app, ["concepts", "iv"])

            assert result.exit_code == 0

    def test_concepts_not_found(self, cli_runner):
        """Test concept not found error."""
        with patch("research_kb_cli.main.asyncio.run") as mock_run:
            mock_run.return_value = ([], [])

            result = cli_runner.invoke(app, ["concepts", "NonexistentConcept"])

            assert result.exit_code == 0
            assert "No concepts found" in result.stdout

    def test_concepts_with_relationships(self, cli_runner, mock_concepts, mock_relationships):
        """Test concept with relationships displayed."""
        with patch("research_kb_cli.main.asyncio.run") as mock_run:
            mock_run.return_value = ([mock_concepts[0]], mock_relationships)

            result = cli_runner.invoke(app, ["concepts", "Instrumental Variables"])

            assert result.exit_code == 0


class TestGraphCommand:
    """Tests for the graph command."""

    def test_graph_1_hop(self, cli_runner, mock_concepts):
        """Test 1-hop neighborhood retrieval."""
        neighborhood = {
            "concepts": mock_concepts,
            "relationships": [],
        }

        with patch("research_kb_cli.main.asyncio.run") as mock_run:
            mock_run.return_value = (mock_concepts[0], neighborhood, None)

            result = cli_runner.invoke(app, ["graph", "Instrumental Variables", "--hops", "1"])

            assert result.exit_code == 0

    def test_graph_concept_not_found(self, cli_runner):
        """Test graph command when concept doesn't exist."""
        with patch("research_kb_cli.main.asyncio.run") as mock_run:
            mock_run.return_value = (None, None, None)

            result = cli_runner.invoke(app, ["graph", "NonexistentConcept"])

            assert result.exit_code == 0
            assert "not found" in result.stdout

    def test_graph_with_relationship_filter(self, cli_runner, mock_concepts):
        """Test filtering by relationship type."""
        from research_kb_contracts import RelationshipType

        neighborhood = {
            "concepts": mock_concepts,
            "relationships": [],
        }

        with patch("research_kb_cli.main.asyncio.run") as mock_run:
            mock_run.return_value = (
                mock_concepts[0],
                neighborhood,
                RelationshipType.REQUIRES,
            )

            result = cli_runner.invoke(app, ["graph", "IV", "--hops", "2", "--type", "REQUIRES"])

            assert result.exit_code == 0


class TestPathCommand:
    """Tests for the path command."""

    def test_path_direct_connection(self, cli_runner, mock_concepts, mock_relationships):
        """Test shortest path between directly connected concepts."""
        # Create path with proper tuple structure (concept, relationship)
        path = [
            (mock_concepts[0], None),
            (mock_concepts[1], mock_relationships[0]),
        ]

        with patch("research_kb_cli.main.asyncio.run") as mock_run:
            mock_run.return_value = (mock_concepts[0], mock_concepts[1], path)

            result = cli_runner.invoke(app, ["path", "Instrumental Variables", "Exogeneity"])

            assert result.exit_code == 0
            assert "Path length: 1 hop" in result.stdout or "Path length: 1 hop" in result.output

    def test_path_no_connection(self, cli_runner, mock_concepts):
        """Test no path exists between concepts."""
        with patch("research_kb_cli.main.asyncio.run") as mock_run:
            mock_run.return_value = (mock_concepts[0], mock_concepts[1], None)

            result = cli_runner.invoke(app, ["path", "ConceptA", "ConceptB"])

            assert result.exit_code == 0
            assert "No path found" in result.stdout

    def test_path_start_concept_not_found(self, cli_runner):
        """Test path command when start concept doesn't exist."""
        with patch("research_kb_cli.main.asyncio.run") as mock_run:
            mock_run.return_value = (None, None, None)

            result = cli_runner.invoke(app, ["path", "NonexistentStart", "SomeEnd"])

            assert result.exit_code == 0
            assert "not found" in result.stdout


# ============================================================================
# Tier 3: Complex Query Command
# ============================================================================


class TestQueryCommand:
    """Tests for the query command."""

    def test_query_basic_markdown(self, cli_runner, mock_embedding_client, mock_search_results):
        """Test basic query with markdown output."""
        with patch("research_kb_pdf.EmbeddingClient", return_value=mock_embedding_client):
            with patch("research_kb_cli.main.asyncio.run") as mock_run:
                # run_query returns (results, expanded_query) tuple
                mock_run.return_value = (mock_search_results, None)

                result = cli_runner.invoke(
                    app, ["query", "instrumental variables", "--format", "markdown"]
                )

                # The command should execute successfully
                # Actual formatting depends on formatters which we test separately
                assert result.exit_code == 0

    def test_query_json_format(self, cli_runner, mock_embedding_client, mock_search_results):
        """Test JSON output format."""
        with patch("research_kb_pdf.EmbeddingClient", return_value=mock_embedding_client):
            with patch("research_kb_cli.main.asyncio.run") as mock_run:
                # run_query returns (results, expanded_query) tuple
                mock_run.return_value = (mock_search_results, None)

                result = cli_runner.invoke(app, ["query", "test query", "--format", "json"])

                assert result.exit_code == 0
                # Output should be valid JSON (tested in formatter tests)

    def test_query_agent_format(self, cli_runner, mock_embedding_client, mock_search_results):
        """Test agent-optimized output format."""
        with patch("research_kb_pdf.EmbeddingClient", return_value=mock_embedding_client):
            with patch("research_kb_cli.main.asyncio.run") as mock_run:
                # run_query returns (results, expanded_query) tuple
                mock_run.return_value = (mock_search_results, None)

                result = cli_runner.invoke(app, ["query", "test", "--format", "agent"])

                assert result.exit_code == 0

    def test_query_with_limit(self, cli_runner, mock_embedding_client, mock_search_results):
        """Test result limit parameter."""
        with patch("research_kb_pdf.EmbeddingClient", return_value=mock_embedding_client):
            with patch("research_kb_cli.main.asyncio.run") as mock_run:
                # run_query returns (results, expanded_query) tuple
                mock_run.return_value = (mock_search_results[:3], None)

                result = cli_runner.invoke(app, ["query", "test", "--limit", "3"])

                assert result.exit_code == 0

    def test_query_embedding_server_down(self, cli_runner):
        """Test error when embedding server unavailable."""
        with patch(
            "research_kb_cli.main.asyncio.run",
            side_effect=ConnectionError("Embed server down"),
        ):
            result = cli_runner.invoke(app, ["query", "test"])

            assert result.exit_code == 1
            assert "Error" in result.output or "Error" in result.stdout

    def test_query_context_type_building(
        self, cli_runner, mock_embedding_client, mock_search_results
    ):
        """Test context type affects search weights."""
        with patch("research_kb_pdf.EmbeddingClient", return_value=mock_embedding_client):
            with patch("research_kb_cli.main.asyncio.run") as mock_run:
                # run_query returns (results, expanded_query) tuple
                mock_run.return_value = (mock_search_results, None)

                result = cli_runner.invoke(app, ["query", "test", "--context-type", "building"])

                assert result.exit_code == 0

    def test_query_source_filter(self, cli_runner, mock_embedding_client, mock_search_results):
        """Test filtering by source type."""
        with patch("research_kb_pdf.EmbeddingClient", return_value=mock_embedding_client):
            with patch("research_kb_cli.main.asyncio.run") as mock_run:
                # run_query returns (results, expanded_query) tuple
                mock_run.return_value = (mock_search_results, None)

                result = cli_runner.invoke(app, ["query", "test", "--source-type", "paper"])

                assert result.exit_code == 0


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling across commands."""

    def test_database_connection_failure_stats(self, cli_runner):
        """Test stats command handles DB connection errors gracefully."""
        with patch(
            "research_kb_cli.main.asyncio.run",
            side_effect=ConnectionError("DB unavailable"),
        ):
            result = cli_runner.invoke(app, ["stats"])

            assert result.exit_code == 1
            assert "Error" in result.output or "Error" in result.stdout

    def test_invalid_format_argument(self, cli_runner):
        """Test commands reject invalid format argument."""
        # Typer should catch this before execution
        result = cli_runner.invoke(app, ["query", "test", "--format", "invalid"])

        # Should fail with argument error
        assert result.exit_code != 0

    def test_missing_required_arguments(self, cli_runner):
        """Test commands fail with missing required args."""
        result = cli_runner.invoke(app, ["query"])  # Missing query text

        assert result.exit_code != 0
