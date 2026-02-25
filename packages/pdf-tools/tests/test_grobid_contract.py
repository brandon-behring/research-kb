"""Contract tests for GrobidClient — validates request/response shapes.

Tests cover:
- is_alive: GET /api/isalive contract
- process_pdf: POST with PDF binary, TEI-XML response parsing
- parse_tei_xml: metadata extraction, section extraction, citation extraction
- Error scenarios: file not found, GROBID unavailable, timeout, parse failure

Uses requests mocking (not real GROBID).
"""

from unittest.mock import MagicMock, patch

import pytest

from research_kb_pdf.grobid_client import (
    ExtractedPaper,
    GrobidClient,
    PaperMetadata,
    PaperSection,
    parse_tei_xml,
)

pytestmark = pytest.mark.unit

# Minimal valid TEI-XML for testing
MINIMAL_TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title type="main">Double Machine Learning</title>
      </titleStmt>
      <publicationStmt>
        <date type="published" when="2018"/>
      </publicationStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author>
              <persName>
                <forename>Victor</forename>
                <surname>Chernozhukov</surname>
              </persName>
            </author>
            <author>
              <persName>
                <forename>Denis</forename>
                <surname>Chetverikov</surname>
              </persName>
            </author>
          </analytic>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>We propose double/debiased machine learning methods.</p>
      </abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>Introduction</head>
        <p>This paper introduces DML.</p>
        <p>It builds on Neyman orthogonality.</p>
      </div>
      <div>
        <head>Methods</head>
        <p>We use cross-fitting and Neyman-orthogonal scores.</p>
      </div>
    </body>
    <back>
      <listBibl>
        <biblStruct xml:id="b0">
          <analytic>
            <title>Estimation of Treatment Effects</title>
            <author>
              <persName>
                <forename>James</forename>
                <surname>Robins</surname>
              </persName>
            </author>
          </analytic>
          <monogr>
            <title level="j">Econometrica</title>
            <imprint>
              <date when="1994"/>
            </imprint>
          </monogr>
        </biblStruct>
      </listBibl>
    </back>
  </text>
</TEI>"""


class TestParseTeiXml:
    """Tests for parse_tei_xml — the core parsing contract."""

    def test_extracts_title(self):
        """Title extracted from titleStmt."""
        paper = parse_tei_xml(MINIMAL_TEI)
        assert paper.metadata.title == "Double Machine Learning"

    def test_extracts_authors(self):
        """Authors extracted as 'Forename Surname' strings."""
        paper = parse_tei_xml(MINIMAL_TEI)
        assert paper.metadata.authors == ["Victor Chernozhukov", "Denis Chetverikov"]

    def test_extracts_abstract(self):
        """Abstract extracted from profileDesc."""
        paper = parse_tei_xml(MINIMAL_TEI)
        assert "double/debiased machine learning" in paper.metadata.abstract

    def test_extracts_year(self):
        """Year extracted from publicationStmt date[@when]."""
        paper = parse_tei_xml(MINIMAL_TEI)
        assert paper.metadata.year == 2018

    def test_extracts_sections(self):
        """Body sections extracted with headings and content."""
        paper = parse_tei_xml(MINIMAL_TEI)
        assert len(paper.sections) >= 2

        headings = [s.heading for s in paper.sections]
        assert "Introduction" in headings
        assert "Methods" in headings

    def test_section_content_joins_paragraphs(self):
        """Multiple paragraphs in a section are joined."""
        paper = parse_tei_xml(MINIMAL_TEI)
        intro = next(s for s in paper.sections if s.heading == "Introduction")
        assert "DML" in intro.content
        assert "Neyman" in intro.content

    def test_extracts_citations(self):
        """Citations extracted from listBibl."""
        paper = parse_tei_xml(MINIMAL_TEI)
        assert len(paper.citations) >= 1

        citation = paper.citations[0]
        assert citation.title == "Estimation of Treatment Effects"
        assert "Robins" in citation.authors[0]
        assert citation.year == 1994
        assert citation.venue == "Econometrica"

    def test_preserves_raw_text(self):
        """raw_text contains full text content."""
        paper = parse_tei_xml(MINIMAL_TEI)
        assert len(paper.raw_text) > 0
        assert "DML" in paper.raw_text

    def test_preserves_tei_xml(self):
        """tei_xml stores original XML."""
        paper = parse_tei_xml(MINIMAL_TEI)
        assert paper.tei_xml == MINIMAL_TEI

    def test_returns_extracted_paper(self):
        """Return type is ExtractedPaper."""
        paper = parse_tei_xml(MINIMAL_TEI)
        assert isinstance(paper, ExtractedPaper)
        assert isinstance(paper.metadata, PaperMetadata)
        assert all(isinstance(s, PaperSection) for s in paper.sections)

    def test_section_level_assignment(self):
        """Sections get correct level (1 for top-level)."""
        paper = parse_tei_xml(MINIMAL_TEI)
        for section in paper.sections:
            assert section.level >= 1


class TestParseTeiXmlEdgeCases:
    """Edge cases for TEI-XML parsing."""

    def test_missing_title_defaults_to_untitled(self):
        """Missing title element defaults to 'Untitled'."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader><fileDesc><titleStmt></titleStmt>
          <sourceDesc><biblStruct/></sourceDesc></fileDesc></teiHeader>
          <text><body/></text>
        </TEI>"""
        paper = parse_tei_xml(xml)
        assert paper.metadata.title == "Untitled"

    def test_missing_abstract(self):
        """Missing abstract returns None."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader><fileDesc><titleStmt>
            <title type="main">Test</title>
          </titleStmt>
          <sourceDesc><biblStruct/></sourceDesc></fileDesc></teiHeader>
          <text><body/></text>
        </TEI>"""
        paper = parse_tei_xml(xml)
        assert paper.metadata.abstract is None

    def test_missing_year(self):
        """Missing publication date returns None year."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader><fileDesc><titleStmt>
            <title type="main">Test</title>
          </titleStmt>
          <sourceDesc><biblStruct/></sourceDesc></fileDesc></teiHeader>
          <text><body/></text>
        </TEI>"""
        paper = parse_tei_xml(xml)
        assert paper.metadata.year is None

    def test_empty_sections(self):
        """Body with no div elements returns empty sections list."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader><fileDesc><titleStmt>
            <title type="main">Test</title>
          </titleStmt>
          <sourceDesc><biblStruct/></sourceDesc></fileDesc></teiHeader>
          <text><body><p>Just a paragraph.</p></body></text>
        </TEI>"""
        paper = parse_tei_xml(xml)
        assert paper.sections == []

    def test_citation_without_title_skipped(self):
        """biblStruct entries without title are skipped."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader><fileDesc><titleStmt>
            <title type="main">Test</title>
          </titleStmt>
          <sourceDesc><biblStruct/></sourceDesc></fileDesc></teiHeader>
          <text><body/><back>
            <listBibl>
              <biblStruct xml:id="b0">
                <analytic>
                  <author><persName><surname>Smith</surname></persName></author>
                </analytic>
              </biblStruct>
            </listBibl>
          </back></text>
        </TEI>"""
        paper = parse_tei_xml(xml)
        assert len(paper.citations) == 0


class TestGrobidClientIsAlive:
    """Tests for GrobidClient.is_alive() contract."""

    @patch("research_kb_pdf.grobid_client.requests.get")
    def test_is_alive_returns_true_on_200(self, mock_get):
        """200 response means GROBID is alive."""
        mock_get.return_value = MagicMock(status_code=200)

        client = GrobidClient("http://localhost:8070")
        assert client.is_alive() is True

        mock_get.assert_called_once_with("http://localhost:8070/api/isalive", timeout=5)

    @patch("research_kb_pdf.grobid_client.requests.get")
    def test_is_alive_returns_false_on_error(self, mock_get):
        """Connection error means GROBID is not alive."""
        import requests

        mock_get.side_effect = requests.ConnectionError()

        client = GrobidClient()
        assert client.is_alive() is False

    @patch("research_kb_pdf.grobid_client.requests.get")
    def test_is_alive_returns_false_on_500(self, mock_get):
        """Non-200 status means GROBID is not alive."""
        mock_get.return_value = MagicMock(status_code=500)

        client = GrobidClient()
        assert client.is_alive() is False


class TestGrobidClientProcessPdf:
    """Tests for GrobidClient.process_pdf() contract."""

    def test_file_not_found_raises(self, tmp_path):
        """Missing PDF raises FileNotFoundError."""
        client = GrobidClient()

        with pytest.raises(FileNotFoundError, match="PDF not found"):
            client.process_pdf(tmp_path / "nonexistent.pdf")

    @patch("research_kb_pdf.grobid_client.requests.post")
    @patch("research_kb_pdf.grobid_client.requests.get")
    def test_grobid_unavailable_raises_connection_error(self, mock_get, mock_post, tmp_path):
        """GROBID down raises ConnectionError."""
        # Create a dummy PDF
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

        mock_get.return_value = MagicMock(status_code=500)  # GROBID not alive

        client = GrobidClient()

        with pytest.raises(ConnectionError, match="not available"):
            client.process_pdf(pdf_path)

    @patch("research_kb_pdf.grobid_client.requests.post")
    @patch("research_kb_pdf.grobid_client.requests.get")
    def test_successful_processing(self, mock_get, mock_post, tmp_path):
        """Successful PDF processing returns ExtractedPaper."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

        mock_get.return_value = MagicMock(status_code=200)  # GROBID alive
        mock_response = MagicMock()
        mock_response.text = MINIMAL_TEI
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        client = GrobidClient()
        paper = client.process_pdf(pdf_path)

        assert isinstance(paper, ExtractedPaper)
        assert paper.metadata.title == "Double Machine Learning"

    @patch("research_kb_pdf.grobid_client.requests.post")
    @patch("research_kb_pdf.grobid_client.requests.get")
    def test_timeout_raises_value_error(self, mock_get, mock_post, tmp_path):
        """GROBID timeout raises ValueError."""
        import requests

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

        mock_get.return_value = MagicMock(status_code=200)
        mock_post.side_effect = requests.Timeout()

        client = GrobidClient()

        with pytest.raises(ValueError, match="timeout"):
            client.process_pdf(pdf_path)

    def test_url_construction(self):
        """Process URL is correctly constructed from base URL."""
        client = GrobidClient("http://grobid:8070/")
        assert client.process_url == "http://grobid:8070/api/processFulltextDocument"

        client2 = GrobidClient("http://grobid:8070")
        assert client2.process_url == "http://grobid:8070/api/processFulltextDocument"
