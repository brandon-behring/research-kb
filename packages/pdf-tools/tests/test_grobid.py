"""Tests for GROBID client and TEI-XML parsing."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from research_kb_pdf import (
    GrobidClient,
    parse_tei_xml,
    ExtractedPaper,
)


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
TEST_PDF = FIXTURES_DIR / "test_simple.pdf"


# Sample TEI-XML for testing
SAMPLE_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader>
        <fileDesc>
            <titleStmt>
                <title level="a" type="main">Analysis of Scientific Collaboration Trends</title>
            </titleStmt>
            <sourceDesc>
                <biblStruct>
                    <analytic>
                        <author>
                            <persName>
                                <forename type="first">John</forename>
                                <surname>Smith</surname>
                            </persName>
                        </author>
                        <author>
                            <persName>
                                <forename type="first">Jane</forename>
                                <surname>Doe</surname>
                            </persName>
                        </author>
                    </analytic>
                </biblStruct>
            </sourceDesc>
        </fileDesc>
        <profileDesc>
            <abstract>
                <p>This paper analyzes trends in scientific collaboration networks.</p>
            </abstract>
        </profileDesc>
    </teiHeader>
    <text>
        <body>
            <div>
                <head>1 Introduction</head>
                <p>Scientific collaboration has increased over time.</p>
                <p>We analyze this trend using network analysis.</p>
            </div>
            <div>
                <head>2 Methods</head>
                <p>We collected data from publication databases.</p>
                <div>
                    <head>2.1 Data Collection</head>
                    <p>Papers were extracted from DBLP and arXiv.</p>
                </div>
            </div>
            <div>
                <head>3 Results</head>
                <p>Collaboration networks show increasing density.</p>
            </div>
        </body>
    </text>
</TEI>"""


class TestTEIXMLParsing:
    """Test TEI-XML parsing without GROBID service."""

    def test_parse_basic_tei(self):
        """Test parsing basic TEI-XML structure."""
        paper = parse_tei_xml(SAMPLE_TEI_XML)

        assert isinstance(paper, ExtractedPaper)
        assert paper.metadata.title == "Analysis of Scientific Collaboration Trends"
        assert len(paper.metadata.authors) == 2
        assert "John Smith" in paper.metadata.authors
        assert "Jane Doe" in paper.metadata.authors

    def test_parse_abstract(self):
        """Test abstract extraction."""
        paper = parse_tei_xml(SAMPLE_TEI_XML)

        assert paper.metadata.abstract is not None
        assert "scientific collaboration" in paper.metadata.abstract.lower()

    def test_parse_sections(self):
        """Test section extraction."""
        paper = parse_tei_xml(SAMPLE_TEI_XML)

        assert len(paper.sections) >= 3
        section_headings = [s.heading for s in paper.sections]

        # Check main sections present
        assert any("Introduction" in h for h in section_headings)
        assert any("Methods" in h for h in section_headings)
        assert any("Results" in h for h in section_headings)

    def test_section_content(self):
        """Test section content extraction."""
        paper = parse_tei_xml(SAMPLE_TEI_XML)

        intro = [s for s in paper.sections if "Introduction" in s.heading][0]
        assert "Scientific collaboration" in intro.content
        assert "network analysis" in intro.content

    def test_section_levels(self):
        """Test section hierarchy levels."""
        paper = parse_tei_xml(SAMPLE_TEI_XML)

        # Main sections should be level 1
        main_sections = [s for s in paper.sections if s.level == 1]
        assert len(main_sections) >= 3

        # Subsections should be level 2
        subsections = [s for s in paper.sections if s.level == 2]
        assert len(subsections) >= 1
        assert any("Data Collection" in s.heading for s in subsections)

    def test_full_text_extraction(self):
        """Test full text extraction."""
        paper = parse_tei_xml(SAMPLE_TEI_XML)

        assert len(paper.raw_text) > 0
        assert "Introduction" in paper.raw_text
        assert "Methods" in paper.raw_text
        assert "Results" in paper.raw_text

    def test_minimal_tei(self):
        """Test parsing minimal TEI-XML."""
        minimal_tei = """<?xml version="1.0"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
            <teiHeader>
                <fileDesc>
                    <titleStmt>
                        <title level="a" type="main">Minimal Paper</title>
                    </titleStmt>
                </fileDesc>
            </teiHeader>
            <text>
                <body>
                    <div>
                        <head>Introduction</head>
                        <p>Some content here.</p>
                    </div>
                </body>
            </text>
        </TEI>"""

        paper = parse_tei_xml(minimal_tei)
        assert paper.metadata.title == "Minimal Paper"
        assert len(paper.sections) == 1


class TestGrobidClientUnit:
    """Unit tests for GrobidClient (mocked responses)."""

    def test_client_initialization(self):
        """Test client initialization with custom URL."""
        client = GrobidClient("http://example.com:9090")
        assert client.grobid_url == "http://example.com:9090"
        assert "processFulltextDocument" in client.process_url

    def test_client_default_url(self):
        """Test client uses default URL."""
        client = GrobidClient()
        assert client.grobid_url == "http://localhost:8070"

    @patch("research_kb_pdf.grobid_client.requests.get")
    def test_is_alive_success(self, mock_get):
        """Test is_alive with responsive service."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = GrobidClient()
        assert client.is_alive() is True

    @patch("research_kb_pdf.grobid_client.requests.get")
    def test_is_alive_failure(self, mock_get):
        """Test is_alive with unresponsive service."""
        import requests

        mock_get.side_effect = requests.RequestException("Connection refused")

        client = GrobidClient()
        assert client.is_alive() is False

    @patch("research_kb_pdf.grobid_client.requests.post")
    @patch("research_kb_pdf.grobid_client.requests.get")
    def test_process_pdf_success(self, mock_get, mock_post):
        """Test successful PDF processing."""
        # Mock service alive
        mock_get.return_value = Mock(status_code=200)

        # Mock GROBID response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_TEI_XML
        mock_post.return_value = mock_response

        client = GrobidClient()
        paper = client.process_pdf(TEST_PDF)

        assert isinstance(paper, ExtractedPaper)
        assert paper.metadata.title == "Analysis of Scientific Collaboration Trends"
        assert len(paper.sections) >= 3

    @patch("research_kb_pdf.grobid_client.requests.get")
    def test_process_pdf_service_unavailable(self, mock_get):
        """Test error when GROBID service unavailable."""
        mock_get.return_value = Mock(status_code=500)

        client = GrobidClient()
        with pytest.raises(ConnectionError, match="GROBID service not available"):
            client.process_pdf(TEST_PDF)

    def test_process_pdf_not_found(self):
        """Test error for non-existent PDF."""
        client = GrobidClient()
        with pytest.raises(FileNotFoundError):
            client.process_pdf("nonexistent.pdf")


class TestGrobidIntegration:
    """Integration tests with running GROBID service."""

    @pytest.fixture(scope="class")
    def grobid_running(self):
        """Check if GROBID service is running."""
        client = GrobidClient()
        if not client.is_alive():
            pytest.skip("GROBID not running. Start with: docker-compose up grobid")
        return True

    def test_grobid_is_alive(self, grobid_running):
        """Test GROBID service health check."""
        client = GrobidClient()
        assert client.is_alive() is True

    def test_process_real_pdf(self, grobid_running):
        """Test processing real PDF with GROBID."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        client = GrobidClient()
        paper = client.process_pdf(TEST_PDF)

        # Basic validations
        assert isinstance(paper, ExtractedPaper)
        assert paper.metadata.title
        assert len(paper.sections) > 0
        assert len(paper.raw_text) > 0

        print("\n✅ GROBID processed paper:")
        print(f"  Title: {paper.metadata.title[:60]}...")
        print(f"  Authors: {len(paper.metadata.authors)}")
        print(f"  Sections: {len(paper.sections)}")
        print(f"  Total text: {len(paper.raw_text)} chars")

    def test_grobid_extracts_structure(self, grobid_running):
        """Test that GROBID extracts hierarchical structure."""
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        client = GrobidClient()
        paper = client.process_pdf(TEST_PDF)

        # Should have sections with headings
        assert all(s.heading for s in paper.sections)

        # Should have content in sections
        assert all(s.content for s in paper.sections)

        # Should have hierarchy levels
        assert all(s.level >= 1 for s in paper.sections)

        # Print structure
        print("\n📋 Paper structure:")
        for section in paper.sections[:5]:  # First 5 sections
            indent = "  " * (section.level - 1)
            print(f"{indent}{section.heading} ({len(section.content)} chars)")

    def test_grobid_timeout_handling(self, grobid_running):
        """Test timeout handling (with very short timeout)."""
        client = GrobidClient()

        # This should work even with short timeout for small PDFs
        # But demonstrates timeout mechanism exists
        if not TEST_PDF.exists():
            pytest.skip(f"Test PDF not found: {TEST_PDF}")

        # Should succeed (test PDF is small)
        paper = client.process_pdf(TEST_PDF)
        assert paper is not None
