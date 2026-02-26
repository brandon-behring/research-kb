"""Tests for CLI commands.

Tests the sub-app structure: search, graph, citations, sources.
All commands now live under grouped sub-apps (Phase 4 restructure).
"""

from unittest.mock import patch
from uuid import uuid4

from research_kb_cli.main import app
import pytest

pytestmark = pytest.mark.unit


# ============================================================================
# Tier 1: Simple Commands (sources sub-app)
# ============================================================================


class TestSourcesListCommand:
    """Tests for the sources list command."""

    def test_sources_empty_database(self, cli_runner):
        """Test sources list command with no data."""
        with patch("research_kb_cli.commands.sources.asyncio.run") as mock_run:
            mock_run.return_value = []

            result = cli_runner.invoke(app, ["sources", "list"])

            assert result.exit_code == 0
            assert "No sources" in result.stdout

    def test_sources_with_data(self, cli_runner):
        """Test sources list command lists all sources."""
        from research_kb_contracts import Source, SourceType
        from datetime import datetime

        mock_sources = [
            Source(
                id=uuid4(),
                title="Test Paper 1",
                authors=["Author A"],
                year=2023,
                source_type=SourceType.PAPER,
                domain_id="causal_inference",
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
                domain_id="causal_inference",
                file_hash="hash2",
                metadata={},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ]

        with patch("research_kb_cli.commands.sources.asyncio.run") as mock_run:
            mock_run.return_value = mock_sources

            result = cli_runner.invoke(app, ["sources", "list"])

            assert result.exit_code == 0
            assert "Test Paper 1" in result.stdout
            assert "Test Paper 2" in result.stdout
            assert "Found 2 sources" in result.stdout

    def test_sources_database_connection_error(self, cli_runner):
        """Test error handling when DB unavailable."""
        with patch(
            "research_kb_cli.commands.sources.asyncio.run",
            side_effect=ConnectionError("DB down"),
        ):
            result = cli_runner.invoke(app, ["sources", "list"])

            assert result.exit_code == 1
            assert "Error" in result.output or "Error" in result.stdout


class TestStatsCommand:
    """Tests for the sources stats command."""

    def test_stats_empty_database(self, cli_runner):
        """Test stats with zero data."""
        with patch("research_kb_cli.commands.sources.asyncio.run") as mock_run:
            mock_run.return_value = (0, 0, [])

            result = cli_runner.invoke(app, ["sources", "stats"])

            assert result.exit_code == 0
            assert "Total sources: 0" in result.stdout
            assert "Total chunks:  0" in result.stdout

    def test_stats_with_data(self, cli_runner):
        """Test stats with ingested data."""
        with patch("research_kb_cli.commands.sources.asyncio.run") as mock_run:
            by_type = [
                {"source_type": "paper", "count": 5},
                {"source_type": "textbook", "count": 2},
            ]
            mock_run.return_value = (7, 150, by_type)

            result = cli_runner.invoke(app, ["sources", "stats"])

            assert result.exit_code == 0
            assert "Total sources: 7" in result.stdout
            assert "Total chunks:  150" in result.stdout
            assert "paper" in result.stdout
            assert "textbook" in result.stdout


class TestExtractionStatusCommand:
    """Tests for the sources extraction-status command."""

    def test_extraction_status_no_extractions(self, cli_runner):
        """Test extraction-status with no concepts."""
        with patch("research_kb_cli.commands.sources.asyncio.run") as mock_run:
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

            result = cli_runner.invoke(app, ["sources", "extraction-status"])

            assert result.exit_code == 0
            assert "Total concepts extracted: 0" in result.stdout

    def test_extraction_status_with_extractions(self, cli_runner):
        """Test extraction-status shows aggregations."""
        with patch("research_kb_cli.commands.sources.asyncio.run") as mock_run:
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

            result = cli_runner.invoke(app, ["sources", "extraction-status"])

            assert result.exit_code == 0
            assert "Total concepts extracted: 50" in result.stdout
            assert "method" in result.stdout
            assert "Average confidence: 0.85" in result.stdout


# ============================================================================
# Tier 2: Graph Commands (graph sub-app)
# ============================================================================


class TestConceptsCommand:
    """Tests for the graph concepts command."""

    def test_concepts_exact_match(self, cli_runner, mock_concepts):
        """Test exact concept name matching."""
        with patch("research_kb_cli.commands.graph.asyncio.run") as mock_run:
            mock_run.return_value = ([mock_concepts[0]], [])

            result = cli_runner.invoke(app, ["graph", "concepts", "Instrumental Variables"])

            assert result.exit_code == 0

    def test_concepts_fuzzy_search(self, cli_runner, mock_concepts):
        """Test fuzzy matching (e.g., 'iv' matches 'instrumental variables')."""
        with patch("research_kb_cli.commands.graph.asyncio.run") as mock_run:
            mock_run.return_value = ([mock_concepts[0]], [])

            result = cli_runner.invoke(app, ["graph", "concepts", "iv"])

            assert result.exit_code == 0

    def test_concepts_not_found(self, cli_runner):
        """Test concept not found error."""
        with patch("research_kb_cli.commands.graph.asyncio.run") as mock_run:
            mock_run.return_value = ([], [])

            result = cli_runner.invoke(app, ["graph", "concepts", "NonexistentConcept"])

            assert result.exit_code == 0
            assert "No concepts found" in result.stdout

    def test_concepts_with_relationships(self, cli_runner, mock_concepts, mock_relationships):
        """Test concept with relationships displayed."""
        with patch("research_kb_cli.commands.graph.asyncio.run") as mock_run:
            mock_run.return_value = ([mock_concepts[0]], mock_relationships)

            result = cli_runner.invoke(app, ["graph", "concepts", "Instrumental Variables"])

            assert result.exit_code == 0


class TestNeighborhoodCommand:
    """Tests for the graph neighborhood command (formerly 'graph')."""

    def test_neighborhood_1_hop(self, cli_runner, mock_concepts):
        """Test 1-hop neighborhood retrieval."""
        neighborhood = {
            "concepts": mock_concepts,
            "relationships": [],
        }

        with patch("research_kb_cli.commands.graph.asyncio.run") as mock_run:
            mock_run.return_value = (mock_concepts[0], neighborhood, None)

            result = cli_runner.invoke(
                app, ["graph", "neighborhood", "Instrumental Variables", "--hops", "1"]
            )

            assert result.exit_code == 0

    def test_neighborhood_concept_not_found(self, cli_runner):
        """Test neighborhood command when concept doesn't exist."""
        with patch("research_kb_cli.commands.graph.asyncio.run") as mock_run:
            mock_run.return_value = (None, None, None)

            result = cli_runner.invoke(app, ["graph", "neighborhood", "NonexistentConcept"])

            assert result.exit_code == 0
            assert "not found" in result.stdout

    def test_neighborhood_with_relationship_filter(self, cli_runner, mock_concepts):
        """Test filtering by relationship type."""
        from research_kb_contracts import RelationshipType

        neighborhood = {
            "concepts": mock_concepts,
            "relationships": [],
        }

        with patch("research_kb_cli.commands.graph.asyncio.run") as mock_run:
            mock_run.return_value = (
                mock_concepts[0],
                neighborhood,
                RelationshipType.REQUIRES,
            )

            result = cli_runner.invoke(
                app, ["graph", "neighborhood", "IV", "--hops", "2", "--type", "REQUIRES"]
            )

            assert result.exit_code == 0


class TestPathCommand:
    """Tests for the graph path command."""

    def test_path_direct_connection(self, cli_runner, mock_concepts, mock_relationships):
        """Test shortest path between directly connected concepts."""
        path = [
            (mock_concepts[0], None),
            (mock_concepts[1], mock_relationships[0]),
        ]

        with patch("research_kb_cli.commands.graph.asyncio.run") as mock_run:
            mock_run.return_value = (mock_concepts[0], mock_concepts[1], path)

            result = cli_runner.invoke(
                app, ["graph", "path", "Instrumental Variables", "Exogeneity"]
            )

            assert result.exit_code == 0
            assert "Path length: 1 hop" in result.stdout or "Path length: 1 hop" in result.output

    def test_path_no_connection(self, cli_runner, mock_concepts):
        """Test no path exists between concepts."""
        with patch("research_kb_cli.commands.graph.asyncio.run") as mock_run:
            mock_run.return_value = (mock_concepts[0], mock_concepts[1], None)

            result = cli_runner.invoke(app, ["graph", "path", "ConceptA", "ConceptB"])

            assert result.exit_code == 0
            assert "No path found" in result.stdout

    def test_path_start_concept_not_found(self, cli_runner):
        """Test path command when start concept doesn't exist."""
        with patch("research_kb_cli.commands.graph.asyncio.run") as mock_run:
            mock_run.return_value = (None, None, None)

            result = cli_runner.invoke(app, ["graph", "path", "NonexistentStart", "SomeEnd"])

            assert result.exit_code == 0
            assert "not found" in result.stdout


# ============================================================================
# Tier 3: Search Commands (search sub-app)
# ============================================================================


class TestQueryCommand:
    """Tests for the search query command."""

    def test_query_basic_markdown(self, cli_runner, mock_embedding_client, mock_search_results):
        """Test basic query with markdown output."""
        with patch("research_kb_pdf.EmbeddingClient", return_value=mock_embedding_client):
            with patch("research_kb_cli.commands.search.asyncio.run") as mock_run:
                mock_run.return_value = (mock_search_results, None)

                result = cli_runner.invoke(
                    app, ["search", "query", "instrumental variables", "--format", "markdown"]
                )

                assert result.exit_code == 0

    def test_query_json_format(self, cli_runner, mock_embedding_client, mock_search_results):
        """Test JSON output format."""
        with patch("research_kb_pdf.EmbeddingClient", return_value=mock_embedding_client):
            with patch("research_kb_cli.commands.search.asyncio.run") as mock_run:
                mock_run.return_value = (mock_search_results, None)

                result = cli_runner.invoke(
                    app, ["search", "query", "test query", "--format", "json"]
                )

                assert result.exit_code == 0

    def test_query_agent_format(self, cli_runner, mock_embedding_client, mock_search_results):
        """Test agent-optimized output format."""
        with patch("research_kb_pdf.EmbeddingClient", return_value=mock_embedding_client):
            with patch("research_kb_cli.commands.search.asyncio.run") as mock_run:
                mock_run.return_value = (mock_search_results, None)

                result = cli_runner.invoke(app, ["search", "query", "test", "--format", "agent"])

                assert result.exit_code == 0

    def test_query_with_limit(self, cli_runner, mock_embedding_client, mock_search_results):
        """Test result limit parameter."""
        with patch("research_kb_pdf.EmbeddingClient", return_value=mock_embedding_client):
            with patch("research_kb_cli.commands.search.asyncio.run") as mock_run:
                mock_run.return_value = (mock_search_results[:3], None)

                result = cli_runner.invoke(app, ["search", "query", "test", "--limit", "3"])

                assert result.exit_code == 0

    def test_query_embedding_server_down(self, cli_runner):
        """Test error when embedding server unavailable."""
        with patch(
            "research_kb_cli.commands.search.asyncio.run",
            side_effect=ConnectionError("Embed server down"),
        ):
            result = cli_runner.invoke(app, ["search", "query", "test"])

            assert result.exit_code == 1
            assert "Error" in result.output or "Error" in result.stdout

    def test_query_context_type_building(
        self, cli_runner, mock_embedding_client, mock_search_results
    ):
        """Test context type affects search weights."""
        with patch("research_kb_pdf.EmbeddingClient", return_value=mock_embedding_client):
            with patch("research_kb_cli.commands.search.asyncio.run") as mock_run:
                mock_run.return_value = (mock_search_results, None)

                result = cli_runner.invoke(
                    app, ["search", "query", "test", "--context-type", "building"]
                )

                assert result.exit_code == 0

    def test_query_source_filter(self, cli_runner, mock_embedding_client, mock_search_results):
        """Test filtering by source type."""
        with patch("research_kb_pdf.EmbeddingClient", return_value=mock_embedding_client):
            with patch("research_kb_cli.commands.search.asyncio.run") as mock_run:
                mock_run.return_value = (mock_search_results, None)

                result = cli_runner.invoke(
                    app, ["search", "query", "test", "--source-type", "paper"]
                )

                assert result.exit_code == 0


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling across commands."""

    def test_database_connection_failure_stats(self, cli_runner):
        """Test stats command handles DB connection errors gracefully."""
        with patch(
            "research_kb_cli.commands.sources.asyncio.run",
            side_effect=ConnectionError("DB unavailable"),
        ):
            result = cli_runner.invoke(app, ["sources", "stats"])

            assert result.exit_code == 1
            assert "Error" in result.output or "Error" in result.stdout

    def test_invalid_format_argument(self, cli_runner):
        """Test commands reject invalid format argument."""
        result = cli_runner.invoke(app, ["search", "query", "test", "--format", "invalid"])

        assert result.exit_code != 0

    def test_missing_required_arguments(self, cli_runner):
        """Test commands fail with missing required args."""
        result = cli_runner.invoke(app, ["search", "query"])  # Missing query text

        assert result.exit_code != 0


# ============================================================================
# Tier 4: Citations Commands (citations sub-app)
# ============================================================================


class TestCitationsListCommand:
    """Tests for the citations list command."""

    def test_citations_list_with_results(self, cli_runner):
        """Test listing citations for a known source."""
        from research_kb_contracts import Source, SourceType
        from datetime import datetime

        mock_source = Source(
            id="00000000-0000-0000-0000-000000000001",
            title="Pearl Causality 2009",
            authors=["Judea Pearl"],
            year=2009,
            source_type=SourceType.TEXTBOOK,
            domain_id="causal_inference",
            file_hash="hash_pearl",
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_citations = [
            {
                "id": "00000000-0000-0000-0000-000000000010",
                "title": "Causal Inference in Statistics",
                "authors": ["Pearl", "Glymour"],
                "year": 2016,
                "venue": "Cambridge Press",
                "doi": "10.1017/example",
                "arxiv_id": None,
                "raw_string": None,
            },
        ]

        with patch("research_kb_cli.commands.citations.asyncio.run") as mock_run:
            mock_run.return_value = (mock_source, mock_citations)

            result = cli_runner.invoke(app, ["citations", "list", "Pearl 2009"])

            assert result.exit_code == 0
            assert "Citations in: Pearl Causality 2009" in result.stdout
            assert "Found 1 citations" in result.stdout
            assert "Causal Inference in Statistics" in result.stdout

    def test_citations_list_source_not_found(self, cli_runner):
        """Test list when source doesn't match."""
        with patch("research_kb_cli.commands.citations.asyncio.run") as mock_run:
            mock_run.return_value = (None, [])

            result = cli_runner.invoke(app, ["citations", "list", "NonexistentSource"])

            assert result.exit_code == 0
            assert "No source found" in result.stdout

    def test_citations_list_no_citations(self, cli_runner):
        """Test list when source exists but has no citations."""
        from research_kb_contracts import Source, SourceType
        from datetime import datetime

        mock_source = Source(
            id="00000000-0000-0000-0000-000000000001",
            title="Empty Source",
            authors=["Author"],
            year=2024,
            source_type=SourceType.PAPER,
            domain_id="causal_inference",
            file_hash="hash_empty",
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with patch("research_kb_cli.commands.citations.asyncio.run") as mock_run:
            mock_run.return_value = (mock_source, [])

            result = cli_runner.invoke(app, ["citations", "list", "Empty Source"])

            assert result.exit_code == 0
            assert "No citations extracted" in result.stdout

    def test_citations_list_error_handling(self, cli_runner):
        """Test error handling in citations list."""
        with patch(
            "research_kb_cli.commands.citations.asyncio.run",
            side_effect=ConnectionError("DB down"),
        ):
            result = cli_runner.invoke(app, ["citations", "list", "test"])

            assert result.exit_code == 1
            assert "Error" in result.output or "Error" in result.stdout


class TestCitationsCitedByCommand:
    """Tests for the citations cited-by command."""

    def test_cited_by_found(self, cli_runner):
        """Test finding sources that cite a given source."""
        from research_kb_contracts import Source, SourceType
        from datetime import datetime

        mock_source = Source(
            id="00000000-0000-0000-0000-000000000001",
            title="Pearl Causality 2009",
            authors=["Judea Pearl"],
            year=2009,
            source_type=SourceType.TEXTBOOK,
            domain_id="causal_inference",
            file_hash="hash_pearl",
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_citing = [
            Source(
                id="00000000-0000-0000-0000-000000000002",
                title="DML Paper 2018",
                authors=["Chernozhukov"],
                year=2018,
                source_type=SourceType.PAPER,
                domain_id="causal_inference",
                file_hash="hash_dml",
                metadata={},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ]
        mock_stats = {
            "authority_score": 0.0234,
            "cited_by_papers": 5,
            "cited_by_textbooks": 2,
        }

        with patch("research_kb_cli.commands.citations.asyncio.run") as mock_run:
            mock_run.return_value = (mock_source, mock_citing, mock_stats)

            result = cli_runner.invoke(app, ["citations", "cited-by", "Pearl 2009"])

            assert result.exit_code == 0
            assert "Who cites: Pearl Causality 2009" in result.stdout
            assert "Citation Authority Score:" in result.stdout
            assert "Citing sources" in result.stdout

    def test_cited_by_no_citations(self, cli_runner):
        """Test when no corpus sources cite this work."""
        from research_kb_contracts import Source, SourceType
        from datetime import datetime

        mock_source = Source(
            id="00000000-0000-0000-0000-000000000001",
            title="Obscure Paper",
            authors=["Unknown"],
            year=2024,
            source_type=SourceType.PAPER,
            domain_id="causal_inference",
            file_hash="hash_obscure",
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with patch("research_kb_cli.commands.citations.asyncio.run") as mock_run:
            mock_run.return_value = (
                mock_source,
                [],
                {"authority_score": 0.0, "cited_by_papers": 0, "cited_by_textbooks": 0},
            )

            result = cli_runner.invoke(app, ["citations", "cited-by", "Obscure Paper"])

            assert result.exit_code == 0
            assert "No corpus sources cite this work" in result.stdout

    def test_cited_by_source_not_found(self, cli_runner):
        """Test cited-by when source doesn't exist."""
        with patch("research_kb_cli.commands.citations.asyncio.run") as mock_run:
            mock_run.return_value = (None, [], {})

            result = cli_runner.invoke(app, ["citations", "cited-by", "Nonexistent"])

            assert result.exit_code == 0
            assert "No source found" in result.stdout


class TestCitationsCitesCommand:
    """Tests for the citations cites command."""

    def test_cites_found(self, cli_runner):
        """Test finding what a source cites."""
        from research_kb_contracts import Source, SourceType
        from datetime import datetime

        mock_source = Source(
            id="00000000-0000-0000-0000-000000000001",
            title="DML Paper 2018",
            authors=["Chernozhukov"],
            year=2018,
            source_type=SourceType.PAPER,
            domain_id="causal_inference",
            file_hash="hash_dml",
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_cited = [
            Source(
                id="00000000-0000-0000-0000-000000000002",
                title="Pearl Causality 2009",
                authors=["Pearl"],
                year=2009,
                source_type=SourceType.TEXTBOOK,
                domain_id="causal_inference",
                file_hash="hash_pearl",
                metadata={},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ]
        mock_stats = {
            "cites_papers": 10,
            "cites_textbooks": 3,
        }

        with patch("research_kb_cli.commands.citations.asyncio.run") as mock_run:
            mock_run.return_value = (mock_source, mock_cited, mock_stats)

            result = cli_runner.invoke(app, ["citations", "cites", "DML"])

            assert result.exit_code == 0
            assert "What does it cite: DML Paper 2018" in result.stdout
            assert "Cited corpus sources" in result.stdout

    def test_cites_no_references(self, cli_runner):
        """Test when source cites nothing in corpus."""
        from research_kb_contracts import Source, SourceType
        from datetime import datetime

        mock_source = Source(
            id="00000000-0000-0000-0000-000000000001",
            title="Standalone Paper",
            authors=["Author"],
            year=2024,
            source_type=SourceType.PAPER,
            domain_id="causal_inference",
            file_hash="hash_standalone",
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with patch("research_kb_cli.commands.citations.asyncio.run") as mock_run:
            mock_run.return_value = (
                mock_source,
                [],
                {"cites_papers": 0, "cites_textbooks": 0},
            )

            result = cli_runner.invoke(app, ["citations", "cites", "Standalone"])

            assert result.exit_code == 0
            assert "No corpus sources found" in result.stdout

    def test_cites_source_not_found(self, cli_runner):
        """Test cites when source doesn't exist."""
        with patch("research_kb_cli.commands.citations.asyncio.run") as mock_run:
            mock_run.return_value = (None, [], {})

            result = cli_runner.invoke(app, ["citations", "cites", "Nonexistent"])

            assert result.exit_code == 0
            assert "No source found" in result.stdout


class TestCitationsStatsCommand:
    """Tests for the citations stats command."""

    def test_stats_with_data(self, cli_runner):
        """Test corpus-wide citation stats display."""
        mock_summary = {
            "total_citations": 15166,
            "total_edges": 1234,
            "internal_edges": 800,
            "external_edges": 434,
            "paper_to_paper": 500,
            "paper_to_textbook": 100,
            "textbook_to_paper": 150,
            "textbook_to_textbook": 50,
        }
        mock_most_cited = [
            {
                "title": "Pearl Causality",
                "source_type": "textbook",
                "cited_by_count": 42,
                "citation_authority": 0.0234,
            },
        ]

        with patch("research_kb_cli.commands.citations.asyncio.run") as mock_run:
            mock_run.return_value = (mock_summary, mock_most_cited)

            result = cli_runner.invoke(app, ["citations", "stats"])

            assert result.exit_code == 0
            assert "Citation Graph Statistics" in result.stdout
            assert "15,166" in result.stdout
            assert "Most cited sources" in result.stdout
            assert "Pearl Causality" in result.stdout

    def test_stats_empty_corpus(self, cli_runner):
        """Test stats with no citation data."""
        mock_summary = {
            "total_citations": 0,
            "total_edges": 0,
            "internal_edges": 0,
            "external_edges": 0,
            "paper_to_paper": 0,
            "paper_to_textbook": 0,
            "textbook_to_paper": 0,
            "textbook_to_textbook": 0,
        }

        with patch("research_kb_cli.commands.citations.asyncio.run") as mock_run:
            mock_run.return_value = (mock_summary, [])

            result = cli_runner.invoke(app, ["citations", "stats"])

            assert result.exit_code == 0
            assert "No citation graph data" in result.stdout

    def test_stats_error_handling(self, cli_runner):
        """Test error handling in stats command."""
        with patch(
            "research_kb_cli.commands.citations.asyncio.run",
            side_effect=ConnectionError("DB unavailable"),
        ):
            result = cli_runner.invoke(app, ["citations", "stats"])

            assert result.exit_code == 1
            assert "Error" in result.output or "Error" in result.stdout


class TestCitationsSimilarCommand:
    """Tests for the citations similar command."""

    def test_similar_found(self, cli_runner):
        """Test finding similar sources via bibliographic coupling."""
        from research_kb_contracts import Source, SourceType
        from datetime import datetime

        mock_source = Source(
            id="00000000-0000-0000-0000-000000000001",
            title="DML Paper 2018",
            authors=["Chernozhukov"],
            year=2018,
            source_type=SourceType.PAPER,
            domain_id="causal_inference",
            file_hash="hash_dml",
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_similar = [
            {
                "title": "Causal Forest Paper",
                "source_type": "paper",
                "authors": ["Wager", "Athey"],
                "year": 2018,
                "coupling_strength": 0.45,
                "shared_references": 12,
            },
        ]

        with patch("research_kb_cli.commands.citations.asyncio.run") as mock_run:
            mock_run.return_value = (mock_source, mock_similar)

            result = cli_runner.invoke(app, ["citations", "similar", "DML"])

            assert result.exit_code == 0
            assert "Bibliographic Coupling" in result.stdout
            assert "Causal Forest Paper" in result.stdout
            assert "Coupling:" in result.stdout

    def test_similar_none_found(self, cli_runner):
        """Test when no similar sources found."""
        from research_kb_contracts import Source, SourceType
        from datetime import datetime

        mock_source = Source(
            id="00000000-0000-0000-0000-000000000001",
            title="Isolated Paper",
            authors=["Author"],
            year=2024,
            source_type=SourceType.PAPER,
            domain_id="causal_inference",
            file_hash="hash_iso",
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with patch("research_kb_cli.commands.citations.asyncio.run") as mock_run:
            mock_run.return_value = (mock_source, [])

            result = cli_runner.invoke(app, ["citations", "similar", "Isolated"])

            assert result.exit_code == 0
            assert "No similar sources found" in result.stdout

    def test_similar_source_not_found(self, cli_runner):
        """Test similar when source doesn't exist."""
        with patch("research_kb_cli.commands.citations.asyncio.run") as mock_run:
            mock_run.return_value = (None, [])

            result = cli_runner.invoke(app, ["citations", "similar", "Nonexistent"])

            assert result.exit_code == 1
            # Error written to stderr (err=True), check combined output
            assert "No source found" in result.output or "No source found" in result.stdout
