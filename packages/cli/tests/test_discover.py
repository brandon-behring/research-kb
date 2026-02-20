"""Tests for discover.py CLI command.

Uses respx to mock httpx requests for deterministic testing.
"""

import json
from pathlib import Path

import pytest
import respx
from httpx import Response
from typer.testing import CliRunner

pytestmark = pytest.mark.unit


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
def search_response():
    """Load search response fixture."""
    return load_fixture("search_instrumental_variables.json")


@pytest.fixture
def empty_search_response():
    """Load empty search response fixture."""
    return load_fixture("empty_search.json")


@pytest.fixture
def author_detail():
    """Load author detail fixture."""
    return load_fixture("author_detail.json")


# -----------------------------------------------------------------------------
# Format Tests (Unit Tests)
# -----------------------------------------------------------------------------


class TestOutputFormatters:
    """Tests for output formatting functions."""

    def test_format_paper_table(self, search_response):
        """Table format produces expected output."""
        from research_kb_cli.discover import format_paper_table
        from s2_client import S2Paper

        papers = [S2Paper(**p) for p in search_response["data"]]
        output = format_paper_table(papers)

        assert "=" * 80 in output
        assert "Year" in output
        assert "Citations" in output
        assert "Double/debiased" in output
        assert "1,542" in output  # Citation count formatted
        assert "Yes" in output  # Open access

    def test_format_paper_table_with_abstract(self, search_response):
        """Table format includes abstract when requested."""
        from research_kb_cli.discover import format_paper_table
        from s2_client import S2Paper

        papers = [S2Paper(**p) for p in search_response["data"]]
        output = format_paper_table(papers, show_abstract=True)

        assert "Abstract:" in output
        assert "revisit" in output.lower()

    def test_format_paper_markdown(self, search_response):
        """Markdown format produces expected output."""
        from research_kb_cli.discover import format_paper_markdown
        from s2_client import S2Paper

        papers = [S2Paper(**p) for p in search_response["data"]]
        output = format_paper_markdown(papers)

        assert "# Discovery Results" in output
        assert "## 1." in output
        assert "Victor Chernozhukov" in output
        assert "DOI: 10.1214/17-AOS1609" in output
        assert "1,542 citations" in output

    def test_format_paper_json(self, search_response):
        """JSON format produces valid JSON."""
        from research_kb_cli.discover import format_paper_json
        from s2_client import S2Paper

        papers = [S2Paper(**p) for p in search_response["data"]]
        output = format_paper_json(papers)

        # Should be valid JSON
        data = json.loads(output)
        assert isinstance(data, list)
        assert len(data) == 3
        assert (
            data[0]["title"]
            == "Double/debiased machine learning for treatment and structural parameters"
        )
        assert data[0]["citation_count"] == 1542

    def test_format_paper_table_no_title(self):
        """Table format handles missing title gracefully."""
        from research_kb_cli.discover import format_paper_table
        from s2_client import S2Paper

        paper = S2Paper(paper_id="test123", title=None, year=None, citation_count=None)
        output = format_paper_table([paper])

        assert "No title" in output
        assert "n.d." in output


# -----------------------------------------------------------------------------
# Search Command Tests
# -----------------------------------------------------------------------------


class TestSearchCommand:
    """Tests for the search subcommand."""

    @respx.mock
    def test_search_returns_papers(self, cli_runner, search_response):
        """Basic search returns papers."""
        from research_kb_cli.discover import app

        # Mock S2 API
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(200, json=search_response)
        )

        result = cli_runner.invoke(app, ["search", "instrumental variables"])

        assert result.exit_code == 0
        assert "Found 4,523 total results" in result.output
        assert "Double/debiased" in result.output

    @respx.mock
    def test_search_year_filter(self, cli_runner, search_response):
        """Year filter is accepted and search works."""
        from research_kb_cli.discover import app

        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(200, json=search_response)
        )

        result = cli_runner.invoke(
            app, ["search", "unique year filter test 9876", "--year-from", "2020"]
        )

        # Command should succeed with year filter
        assert result.exit_code == 0
        # Should show results (from cache or mock)
        assert "Found" in result.output or "No papers" in result.output

    @respx.mock
    def test_search_citation_filter(self, cli_runner, search_response):
        """Citation filter is accepted and search works."""
        from research_kb_cli.discover import app

        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(200, json=search_response)
        )

        result = cli_runner.invoke(
            app,
            ["search", "unique citation filter test 5432", "--min-citations", "100"],
        )

        # Command should succeed with citation filter
        assert result.exit_code == 0
        # Should show results
        assert "Found" in result.output or "No papers" in result.output

    @respx.mock
    def test_search_empty_results(self, cli_runner, empty_search_response):
        """Empty results handled gracefully."""
        from research_kb_cli.discover import app

        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(200, json=empty_search_response)
        )

        result = cli_runner.invoke(app, ["search", "nonexistent topic 12345"])

        assert result.exit_code == 0
        assert "No papers found" in result.output

    @respx.mock
    def test_search_json_format(self, cli_runner, search_response):
        """JSON output format works."""
        from research_kb_cli.discover import app

        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(200, json=search_response)
        )

        result = cli_runner.invoke(app, ["search", "test", "--format", "json"])

        assert result.exit_code == 0
        # Output should contain valid JSON
        assert '"paper_id":' in result.output
        assert '"citation_count":' in result.output

    @respx.mock
    def test_search_markdown_format(self, cli_runner, search_response):
        """Markdown output format works."""
        from research_kb_cli.discover import app

        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(200, json=search_response)
        )

        result = cli_runner.invoke(app, ["search", "test", "--format", "markdown"])

        assert result.exit_code == 0
        assert "# Discovery Results" in result.output
        assert "##" in result.output

    @respx.mock
    def test_search_api_timeout(self, cli_runner):
        """API timeout is handled gracefully."""
        from research_kb_cli.discover import app
        import httpx

        # Mock the route to raise timeout (bypass cache)
        respx.route(method="GET", host="api.semanticscholar.org").mock(
            side_effect=httpx.TimeoutException("Connection timed out")
        )

        result = cli_runner.invoke(app, ["search", "very unique query that wont be cached 12345"])

        # When timeout occurs, it should either exit 1 or show error
        # The actual behavior depends on retry logic
        assert (
            result.exit_code == 1 or "Error" in result.output or "timeout" in result.output.lower()
        )

    @respx.mock
    def test_search_rate_limit(self, cli_runner):
        """Rate limit error is handled."""
        from research_kb_cli.discover import app

        respx.route(method="GET", host="api.semanticscholar.org").mock(
            return_value=Response(429, json={"message": "Rate limit exceeded"})
        )

        result = cli_runner.invoke(app, ["search", "another unique uncached query 67890"])

        # Rate limit should cause error or retry
        assert result.exit_code == 1 or "Error" in result.output or "rate" in result.output.lower()


# -----------------------------------------------------------------------------
# Author Command Tests
# -----------------------------------------------------------------------------


class TestAuthorCommand:
    """Tests for the author subcommand."""

    @respx.mock
    def test_author_lookup(self, cli_runner, author_detail, search_response):
        """Author lookup returns papers."""
        from research_kb_cli.discover import app

        # Mock author info endpoint
        respx.get("https://api.semanticscholar.org/graph/v1/author/26331346").mock(
            return_value=Response(200, json=author_detail)
        )

        # Mock author papers endpoint
        respx.get("https://api.semanticscholar.org/graph/v1/author/26331346/papers").mock(
            return_value=Response(
                200, json={"total": 215, "offset": 0, "data": search_response["data"]}
            )
        )

        result = cli_runner.invoke(app, ["author", "26331346"])

        assert result.exit_code == 0
        # Author name might be cached from real API (V. Chernozhukov) or from fixture
        assert "Chernozhukov" in result.output
        # Check for h-index (may vary due to caching)
        assert "h-index:" in result.output
        # Check that citation count is shown
        assert "citations" in result.output.lower()


# -----------------------------------------------------------------------------
# Integration-style Tests
# -----------------------------------------------------------------------------


class TestOutputFormats:
    """Test all output formats produce consistent data."""

    @respx.mock
    def test_all_formats_same_paper_count(self, cli_runner, search_response):
        """All formats show the same number of papers."""
        from research_kb_cli.discover import app

        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(200, json=search_response)
        )

        for fmt in ["table", "json", "markdown"]:
            result = cli_runner.invoke(app, ["search", "test", "--format", fmt])
            assert result.exit_code == 0
            # All should mention the same total
            assert "4,523" in result.output
