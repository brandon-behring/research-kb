"""Unit tests for healthcare reference cache ingestion helpers.

Tests parse_frontmatter() — the one piece of new logic in the script.
chunk_markdown() is reused from ingest_annuity_kb.py and already proven.
"""

import sys
from pathlib import Path

import pytest

# Add script to importable path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Import the functions under test (script has no __all__, import directly)
# We need to handle the imports that would fail without packages on path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))

from ingest_healthcare_reference_cache import parse_frontmatter


@pytest.mark.unit
class TestParseFrontmatter:
    """Tests for YAML frontmatter parsing."""

    def test_standard_frontmatter(self):
        """Parse a typical reference cache markdown file."""
        text = '---\nbib_key: "R01:cms-multipayer"\ntitle: "Multi-Payer Alignment"\nurl: "https://www.cms.gov/priorities/innovation"\nfetched_date: "2026-03-26"\ncontent_type: "webpage"\n---\n\n# Multi-Payer Alignment\n\nContent here.'
        meta, body = parse_frontmatter(text)

        assert meta["bib_key"] == "R01:cms-multipayer"
        assert meta["title"] == "Multi-Payer Alignment"
        assert meta["url"] == "https://www.cms.gov/priorities/innovation"
        assert meta["fetched_date"] == "2026-03-26"
        assert meta["content_type"] == "webpage"
        assert body.strip().startswith("# Multi-Payer Alignment")

    def test_no_frontmatter(self):
        """Text without --- delimiters returns empty dict and full text."""
        text = "# Just a heading\n\nSome content."
        meta, body = parse_frontmatter(text)

        assert meta == {}
        assert body == text

    def test_empty_frontmatter(self):
        """Empty frontmatter block (just ---\\n---) returns empty dict."""
        text = "---\n\n---\nBody text."
        meta, body = parse_frontmatter(text)

        assert meta == {}
        assert "Body text." in body

    def test_url_with_colons(self):
        """URLs contain multiple colons — only split on first."""
        text = "---\nurl: https://www.cms.gov/path?key=value\n---\nBody."
        meta, body = parse_frontmatter(text)

        assert meta["url"] == "https://www.cms.gov/path?key=value"

    def test_url_with_colons_quoted(self):
        """Quoted URLs with colons are handled correctly."""
        text = '---\nurl: "https://pmc.ncbi.nlm.nih.gov/articles/PMC10387941/"\n---\nBody.'
        meta, body = parse_frontmatter(text)

        assert meta["url"] == "https://pmc.ncbi.nlm.nih.gov/articles/PMC10387941/"

    def test_body_preserved_after_frontmatter(self):
        """Body text starts immediately after closing ---."""
        text = "---\ntitle: Test\n---\nFirst line of body.\nSecond line."
        meta, body = parse_frontmatter(text)

        assert meta["title"] == "Test"
        assert body == "First line of body.\nSecond line."

    def test_quoted_and_unquoted_values(self):
        """Both quoted and unquoted values are stripped correctly."""
        text = '---\nquoted: "hello world"\nunquoted: hello world\n---\nBody.'
        meta, body = parse_frontmatter(text)

        assert meta["quoted"] == "hello world"
        assert meta["unquoted"] == "hello world"

    def test_acquisition_method_field(self):
        """Less common fields like acquisition_method are parsed."""
        text = '---\nbib_key: "R01"\nacquisition_method: "playwright"\n---\nBody.'
        meta, body = parse_frontmatter(text)

        assert meta["acquisition_method"] == "playwright"

    def test_single_dash_line_not_frontmatter(self):
        """A single --- without closing --- is not treated as frontmatter."""
        text = "---\nSome text without closing delimiter."
        meta, body = parse_frontmatter(text)

        assert meta == {}
        assert body == text

    def test_real_pmc_frontmatter(self):
        """Parse frontmatter matching actual PMC fulltext files."""
        text = (
            "---\n"
            'bib_key: "R10:dodda-pie-best-practices"\n'
            'title: "Best Practices in Designing PIE for US HCDMs"\n'
            'url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC10387941/"\n'
            'fetched_date: "2026-03-26"\n'
            'content_type: "pmc_fulltext"\n'
            "---\n"
            "\n"
            "## Abstract\n"
            "\n"
            "Background: ...\n"
        )
        meta, body = parse_frontmatter(text)

        assert meta["bib_key"] == "R10:dodda-pie-best-practices"
        assert meta["content_type"] == "pmc_fulltext"
        assert "## Abstract" in body
        # Frontmatter should NOT appear in body
        assert "bib_key" not in body
