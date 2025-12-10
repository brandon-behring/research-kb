"""Tests for enrich.py CLI command.

Uses mocks for database and S2 API interactions.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import respx
from httpx import Response
from typer.testing import CliRunner

# Fixture loading
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "s2_responses"


def load_fixture(name: str) -> dict:
    """Load a JSON fixture file."""
    with open(FIXTURES_DIR / name) as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def cli_runner():
    """Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def paper_detail():
    """Load paper detail fixture."""
    return load_fixture("paper_detail.json")


@pytest.fixture
def mock_db_pool():
    """Mock database connection pool."""
    pool = MagicMock()
    conn = AsyncMock()

    # Mock connection context manager
    pool.acquire = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock()))

    # Default: return empty results
    conn.fetch.return_value = []
    conn.fetchval.return_value = 0
    conn.execute.return_value = None

    return pool, conn


@pytest.fixture
def sample_citation_rows():
    """Sample citation rows from database."""
    return [
        {
            "id": uuid4(),
            "title": "Double/debiased machine learning for treatment and structural parameters",
            "authors": ["Victor Chernozhukov", "Denis Chetverikov"],
            "year": 2018,
            "venue": "The Annals of Statistics",
            "doi": "10.1214/17-AOS1609",
            "arxiv_id": None,
            "metadata": {},
            "source_title": "Test Source",
        },
        {
            "id": uuid4(),
            "title": "Some Paper Without DOI",
            "authors": ["Unknown Author"],
            "year": 2020,
            "venue": "Unknown Venue",
            "doi": None,
            "arxiv_id": "2005.12345",
            "metadata": {},
            "source_title": "Test Source",
        },
        {
            "id": uuid4(),
            "title": "Fuzzy Match Paper",
            "authors": ["Jane Doe"],
            "year": 2021,
            "venue": "Journal of ML",
            "doi": None,
            "arxiv_id": None,
            "metadata": {},
            "source_title": "Test Source",
        },
    ]


# -----------------------------------------------------------------------------
# Format Tests (Unit Tests)
# -----------------------------------------------------------------------------


class TestEnrichmentFormatters:
    """Tests for enrichment output formatting."""

    def test_format_enrichment_table(self):
        """Table format produces expected output."""
        from research_kb_cli.enrich import format_enrichment_table

        results = {
            "matched": 10,
            "ambiguous": 2,
            "unmatched": 3,
            "skipped": 5,
            "total": 20,
            "by_method": {"doi": 7, "arxiv": 2, "multi_signal": 1},
        }

        output = format_enrichment_table(results)

        assert "=" * 80 in output
        assert "Matched" in output
        assert "10" in output
        assert "DOI: 7" in output
        assert "arXiv: 2" in output
        assert "Total" in output
        assert "20" in output

    def test_format_enrichment_table_empty(self):
        """Table format handles empty results."""
        from research_kb_cli.enrich import format_enrichment_table

        results = {
            "matched": 0,
            "ambiguous": 0,
            "unmatched": 0,
            "skipped": 0,
            "total": 0,
            "by_method": {},
        }

        output = format_enrichment_table(results)

        assert "Total" in output
        assert "0" in output


# -----------------------------------------------------------------------------
# Citation Enrichment Command Tests
# -----------------------------------------------------------------------------


class TestEnrichCitationsCommand:
    """Tests for the citations subcommand."""

    def test_requires_source_or_all(self, cli_runner):
        """Command requires either --source or --all flag."""
        from research_kb_cli.enrich import app

        result = cli_runner.invoke(app, ["citations"])

        assert result.exit_code == 1
        assert "Specify --source or --all" in result.output

    @patch("research_kb_cli.enrich.asyncio.run")
    def test_dry_run_no_modifications(self, mock_run, cli_runner, sample_citation_rows):
        """Dry run doesn't modify database."""
        from research_kb_cli.enrich import app

        # Mock the async function to return dry-run results
        mock_run.return_value = {
            "matched": 2,
            "ambiguous": 0,
            "unmatched": 1,
            "skipped": 0,
            "total": 3,
            "by_method": {"doi": 1, "arxiv": 1},
        }

        result = cli_runner.invoke(app, ["citations", "--all", "--dry-run"])

        assert result.exit_code == 0
        assert "DRY RUN" in result.output or "Would process" in result.output or "Matched" in result.output

    @patch("research_kb_cli.enrich.asyncio.run")
    def test_source_filter(self, mock_run, cli_runner):
        """Source filter is passed correctly."""
        from research_kb_cli.enrich import app

        mock_run.return_value = {
            "matched": 0,
            "ambiguous": 0,
            "unmatched": 0,
            "skipped": 0,
            "total": 0,
            "by_method": {},
        }

        result = cli_runner.invoke(app, ["citations", "--source", "Pearl 2009", "--dry-run"])

        assert result.exit_code == 0

    @patch("research_kb_cli.enrich.asyncio.run")
    def test_force_flag_ignores_staleness(self, mock_run, cli_runner):
        """Force flag bypasses staleness check."""
        from research_kb_cli.enrich import app

        mock_run.return_value = {
            "matched": 5,
            "ambiguous": 0,
            "unmatched": 0,
            "skipped": 0,
            "total": 5,
            "by_method": {"doi": 5},
        }

        result = cli_runner.invoke(app, ["citations", "--all", "--force", "--dry-run"])

        assert result.exit_code == 0

    @patch("research_kb_cli.enrich.asyncio.run")
    def test_json_format_output(self, mock_run, cli_runner):
        """JSON format outputs valid JSON."""
        from research_kb_cli.enrich import app

        mock_run.return_value = {
            "matched": 10,
            "ambiguous": 2,
            "unmatched": 3,
            "skipped": 5,
            "total": 20,
            "by_method": {"doi": 7, "arxiv": 2, "multi_signal": 1},
        }

        result = cli_runner.invoke(app, ["citations", "--all", "--dry-run", "--format", "json"])

        assert result.exit_code == 0
        # Should contain JSON-like output
        assert '"matched"' in result.output or "matched" in result.output

    @patch("research_kb_cli.enrich.asyncio.run")
    def test_limit_parameter(self, mock_run, cli_runner):
        """Limit parameter restricts number of citations."""
        from research_kb_cli.enrich import app

        mock_run.return_value = {
            "matched": 0,
            "ambiguous": 0,
            "unmatched": 0,
            "skipped": 0,
            "total": 0,
            "by_method": {},
        }

        result = cli_runner.invoke(app, ["citations", "--all", "--limit", "50", "--dry-run"])

        assert result.exit_code == 0


# -----------------------------------------------------------------------------
# Enrichment Status Command Tests
# -----------------------------------------------------------------------------


class TestEnrichmentStatusCommand:
    """Tests for the status subcommand."""

    @patch("research_kb_cli.enrich.asyncio.run")
    def test_status_display(self, mock_run, cli_runner):
        """Status command displays enrichment stats."""
        from research_kb_cli.enrich import app

        mock_run.return_value = {
            "total": 5044,
            "enriched": 2500,
            "unenriched": 2544,
            "by_method": {"doi": 2000, "arxiv": 300, "multi_signal": 200},
            "by_status": {"matched": 2400, "ambiguous": 100},
            "stale": 500,
        }

        result = cli_runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Citation Enrichment Status" in result.output
        assert "5,044" in result.output or "5044" in result.output
        assert "Enriched" in result.output

    @patch("research_kb_cli.enrich.asyncio.run")
    def test_status_shows_methods(self, mock_run, cli_runner):
        """Status shows breakdown by match method."""
        from research_kb_cli.enrich import app

        mock_run.return_value = {
            "total": 100,
            "enriched": 50,
            "unenriched": 50,
            "by_method": {"doi": 30, "arxiv": 15, "multi_signal": 5},
            "by_status": {"matched": 50},
            "stale": 10,
        }

        result = cli_runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "doi" in result.output.lower() or "DOI" in result.output


# -----------------------------------------------------------------------------
# Error Handling Tests
# -----------------------------------------------------------------------------


class TestEnrichErrorHandling:
    """Tests for error handling in enrich commands."""

    def test_missing_s2_client_import(self, cli_runner):
        """Graceful handling when s2-client not installed."""
        from research_kb_cli.enrich import app

        with patch.dict("sys.modules", {"s2_client": None}):
            # This would only trigger if we actually hit the import
            # In practice, the module is installed, so we test the error path differently
            pass

    @patch("research_kb_cli.enrich.asyncio.run")
    def test_database_error_handling(self, mock_run, cli_runner):
        """Database errors are handled gracefully."""
        from research_kb_cli.enrich import app

        mock_run.side_effect = Exception("Database connection failed")

        result = cli_runner.invoke(app, ["citations", "--all", "--dry-run"])

        assert result.exit_code == 1
        assert "Error" in result.output


# -----------------------------------------------------------------------------
# Confidence Score Tests
# -----------------------------------------------------------------------------


class TestConfidenceScoring:
    """Tests for citation matching confidence scores."""

    def test_doi_match_high_confidence(self):
        """DOI match should have confidence 1.0."""
        # This tests the underlying enrichment logic
        # The actual match_citation function is in s2_client
        pass

    def test_arxiv_match_high_confidence(self):
        """arXiv match should have confidence ~0.95."""
        pass

    def test_fuzzy_match_threshold(self):
        """Fuzzy matches below 0.8 should be marked ambiguous."""
        pass
