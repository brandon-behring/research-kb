"""Tests for S2 Pydantic response models.

Tests model validation, serialization, and property methods for
Semantic Scholar API response schemas.
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from s2_client.models import (
    OpenAccessPdf,
    S2Author,
    S2AuthorPapersResult,
    S2Paper,
    S2SearchResult,
)


# -----------------------------------------------------------------------------
# OpenAccessPdf Tests
# -----------------------------------------------------------------------------


class TestOpenAccessPdf:
    """Tests for OpenAccessPdf model."""

    def test_parse_full_response(self):
        """Parse complete PDF response."""
        data = {
            "url": "https://arxiv.org/pdf/1608.00060.pdf",
            "status": "GREEN",
        }
        pdf = OpenAccessPdf(**data)

        assert pdf.url == "https://arxiv.org/pdf/1608.00060.pdf"
        assert pdf.status == "GREEN"

    def test_parse_minimal_response(self):
        """Parse response with missing fields."""
        pdf = OpenAccessPdf()

        assert pdf.url is None
        assert pdf.status is None

    def test_parse_partial_response(self):
        """Parse response with only URL."""
        pdf = OpenAccessPdf(url="https://example.com/paper.pdf")

        assert pdf.url == "https://example.com/paper.pdf"
        assert pdf.status is None

    def test_extra_fields_ignored(self):
        """Extra fields should be ignored."""
        data = {
            "url": "https://example.com/paper.pdf",
            "status": "GOLD",
            "extra_field": "ignored",
        }
        pdf = OpenAccessPdf(**data)

        assert pdf.url == "https://example.com/paper.pdf"
        assert not hasattr(pdf, "extra_field")

    def test_status_values(self):
        """Test various OA status values."""
        for status in ["GREEN", "BRONZE", "GOLD", "HYBRID", None]:
            pdf = OpenAccessPdf(status=status)
            assert pdf.status == status


# -----------------------------------------------------------------------------
# S2Author Tests
# -----------------------------------------------------------------------------


class TestS2Author:
    """Tests for S2Author model."""

    def test_parse_full_author(self):
        """Parse complete author response."""
        data = {
            "authorId": "26331346",
            "externalIds": {"DBLP": ["c/VictorChernozhukov"]},
            "name": "Victor Chernozhukov",
            "url": "https://www.semanticscholar.org/author/26331346",
            "affiliations": ["MIT"],
            "paperCount": 250,
            "citationCount": 45000,
            "hIndex": 75,
        }
        author = S2Author(**data)

        assert author.author_id == "26331346"
        assert author.name == "Victor Chernozhukov"
        assert author.affiliations == ["MIT"]
        assert author.paper_count == 250
        assert author.citation_count == 45000
        assert author.h_index == 75

    def test_parse_minimal_author(self):
        """Parse author with only name."""
        author = S2Author(name="John Doe")

        assert author.name == "John Doe"
        assert author.author_id is None
        assert author.affiliations is None
        assert author.paper_count is None

    def test_alias_mapping(self):
        """Test camelCase to snake_case alias mapping."""
        data = {
            "authorId": "123",
            "paperCount": 100,
            "citationCount": 5000,
            "hIndex": 30,
        }
        author = S2Author(**data)

        assert author.author_id == "123"
        assert author.paper_count == 100
        assert author.citation_count == 5000
        assert author.h_index == 30

    def test_external_ids_structure(self):
        """Test external IDs are preserved as dict."""
        data = {
            "authorId": "123",
            "name": "Test Author",
            "externalIds": {
                "DBLP": ["d/AuthorName"],
                "ORCID": "0000-0002-1234-5678",
            },
        }
        author = S2Author(**data)

        assert author.external_ids["DBLP"] == ["d/AuthorName"]
        assert author.external_ids["ORCID"] == "0000-0002-1234-5678"

    def test_multiple_affiliations(self):
        """Test multiple affiliations."""
        data = {
            "name": "Test",
            "affiliations": ["MIT", "Stanford", "Harvard"],
        }
        author = S2Author(**data)

        assert len(author.affiliations) == 3
        assert "MIT" in author.affiliations


# -----------------------------------------------------------------------------
# S2Paper Core Tests
# -----------------------------------------------------------------------------


class TestS2PaperInit:
    """Tests for S2Paper initialization and parsing."""

    def test_parse_full_paper(self):
        """Parse complete paper response."""
        data = {
            "paperId": "649def34f8be52c8b66281af98ae884c09aef38b",
            "corpusId": 14457330,
            "externalIds": {
                "DOI": "10.1214/17-AOS1609",
                "ArXiv": "1608.00060",
            },
            "title": "Double/debiased machine learning for treatment and structural parameters",
            "abstract": "We revisit the classic problem...",
            "venue": "The Annals of Statistics",
            "publicationVenue": {
                "id": "abc123",
                "name": "The Annals of Statistics",
            },
            "year": 2018,
            "publicationDate": "2018-04-01",
            "authors": [
                {"authorId": "26331346", "name": "Victor Chernozhukov"},
                {"authorId": "2149494", "name": "Denis Chetverikov"},
            ],
            "referenceCount": 78,
            "citationCount": 1542,
            "influentialCitationCount": 234,
            "s2FieldsOfStudy": [
                {"category": "Economics", "source": "s2-fos-model"},
                {"category": "Computer Science", "source": "s2-fos-model"},
            ],
            "publicationTypes": ["JournalArticle"],
            "isOpenAccess": True,
            "openAccessPdf": {
                "url": "https://arxiv.org/pdf/1608.00060.pdf",
                "status": "GREEN",
            },
            "url": "https://www.semanticscholar.org/paper/649def34",
        }
        paper = S2Paper(**data)

        assert paper.paper_id == "649def34f8be52c8b66281af98ae884c09aef38b"
        assert paper.corpus_id == 14457330
        assert paper.title == "Double/debiased machine learning for treatment and structural parameters"
        assert paper.year == 2018
        assert paper.citation_count == 1542
        assert paper.is_open_access is True
        assert len(paper.authors) == 2
        assert paper.authors[0].name == "Victor Chernozhukov"

    def test_parse_minimal_paper(self):
        """Parse paper with minimal fields."""
        paper = S2Paper(paperId="abc123", title="Test Paper")

        assert paper.paper_id == "abc123"
        assert paper.title == "Test Paper"
        assert paper.year is None
        assert paper.citation_count is None
        assert paper.authors is None

    def test_parse_empty_paper(self):
        """Parse completely empty paper."""
        paper = S2Paper()

        assert paper.paper_id is None
        assert paper.title is None
        assert paper.authors is None

    def test_alias_mapping(self):
        """Test camelCase to snake_case alias mapping."""
        data = {
            "paperId": "123",
            "corpusId": 456,
            "referenceCount": 10,
            "citationCount": 100,
            "influentialCitationCount": 20,
            "isOpenAccess": True,
            "publicationDate": "2020-01-15",
        }
        paper = S2Paper(**data)

        assert paper.paper_id == "123"
        assert paper.corpus_id == 456
        assert paper.reference_count == 10
        assert paper.citation_count == 100
        assert paper.influential_citation_count == 20
        assert paper.is_open_access is True
        assert paper.publication_date == "2020-01-15"

    def test_extra_fields_ignored(self):
        """Extra fields should be ignored."""
        data = {
            "paperId": "123",
            "title": "Test",
            "unknown_field": "value",
            "another_extra": 42,
        }
        paper = S2Paper(**data)

        assert paper.paper_id == "123"
        assert not hasattr(paper, "unknown_field")


# -----------------------------------------------------------------------------
# S2Paper Property Tests
# -----------------------------------------------------------------------------


class TestS2PaperProperties:
    """Tests for S2Paper computed properties."""

    def test_doi_property(self):
        """DOI should be extracted from external IDs."""
        paper = S2Paper(
            paperId="123",
            externalIds={"DOI": "10.1214/17-AOS1609", "ArXiv": "1608.00060"},
        )
        assert paper.doi == "10.1214/17-AOS1609"

    def test_doi_missing(self):
        """DOI should be None when not present."""
        paper = S2Paper(paperId="123", externalIds={"ArXiv": "1608.00060"})
        assert paper.doi is None

        paper_no_ids = S2Paper(paperId="123")
        assert paper_no_ids.doi is None

    def test_arxiv_id_property(self):
        """arXiv ID should be extracted from external IDs."""
        paper = S2Paper(paperId="123", externalIds={"ArXiv": "1608.00060"})
        assert paper.arxiv_id == "1608.00060"

    def test_arxiv_id_missing(self):
        """arXiv ID should be None when not present."""
        paper = S2Paper(paperId="123", externalIds={"DOI": "10.1234/test"})
        assert paper.arxiv_id is None

    def test_first_author_name(self):
        """First author name should be accessible."""
        paper = S2Paper(
            paperId="123",
            authors=[
                S2Author(authorId="1", name="Alice Smith"),
                S2Author(authorId="2", name="Bob Jones"),
            ],
        )
        assert paper.first_author_name == "Alice Smith"

    def test_first_author_name_empty_list(self):
        """First author name should be None for empty list."""
        paper = S2Paper(paperId="123", authors=[])
        assert paper.first_author_name is None

    def test_first_author_name_none(self):
        """First author name should be None when authors is None."""
        paper = S2Paper(paperId="123")
        assert paper.first_author_name is None


# -----------------------------------------------------------------------------
# S2Paper Metadata Dict Tests
# -----------------------------------------------------------------------------


class TestS2PaperToMetadataDict:
    """Tests for to_metadata_dict method."""

    def test_full_metadata_dict(self):
        """Metadata dict should have expected keys."""
        paper = S2Paper(
            paperId="abc123",
            corpusId=456,
            externalIds={"DOI": "10.1234/test", "ArXiv": "2001.12345"},
            citationCount=100,
            influentialCitationCount=20,
            isOpenAccess=True,
            s2FieldsOfStudy=[
                {"category": "Economics", "source": "s2-fos-model"},
                {"category": "Computer Science", "source": "s2-fos-model"},
            ],
        )

        metadata = paper.to_metadata_dict()

        assert metadata["s2_paper_id"] == "abc123"
        assert metadata["s2_corpus_id"] == 456
        assert metadata["doi"] == "10.1234/test"
        assert metadata["arxiv_id"] == "2001.12345"
        assert metadata["citation_count"] == 100
        assert metadata["influential_citation_count"] == 20
        assert metadata["is_open_access"] is True
        assert metadata["fields_of_study"] == ["Economics", "Computer Science"]
        assert "s2_enriched_at" in metadata

    def test_metadata_dict_with_missing_fields(self):
        """Metadata dict should handle missing fields."""
        paper = S2Paper(paperId="abc123")

        metadata = paper.to_metadata_dict()

        assert metadata["s2_paper_id"] == "abc123"
        assert metadata["doi"] is None
        assert metadata["arxiv_id"] is None
        assert metadata["citation_count"] is None
        assert metadata["fields_of_study"] == []

    def test_metadata_dict_enriched_at_format(self):
        """Enriched at timestamp should be ISO format."""
        paper = S2Paper(paperId="abc123")

        metadata = paper.to_metadata_dict()

        # Should be valid ISO format with timezone
        enriched_at = metadata["s2_enriched_at"]
        parsed = datetime.fromisoformat(enriched_at)
        assert parsed.tzinfo == timezone.utc


# -----------------------------------------------------------------------------
# S2Paper Nested Objects Tests
# -----------------------------------------------------------------------------


class TestS2PaperNestedObjects:
    """Tests for nested object parsing."""

    def test_authors_parsed_as_s2author(self):
        """Authors should be parsed as S2Author objects."""
        data = {
            "paperId": "123",
            "authors": [
                {"authorId": "1", "name": "Alice"},
                {"authorId": "2", "name": "Bob"},
            ],
        }
        paper = S2Paper(**data)

        assert len(paper.authors) == 2
        assert all(isinstance(a, S2Author) for a in paper.authors)
        assert paper.authors[0].name == "Alice"

    def test_open_access_pdf_parsed(self):
        """openAccessPdf should be parsed as OpenAccessPdf."""
        data = {
            "paperId": "123",
            "openAccessPdf": {
                "url": "https://example.com/paper.pdf",
                "status": "GREEN",
            },
        }
        paper = S2Paper(**data)

        assert isinstance(paper.open_access_pdf, OpenAccessPdf)
        assert paper.open_access_pdf.url == "https://example.com/paper.pdf"
        assert paper.open_access_pdf.status == "GREEN"

    def test_s2_fields_of_study_preserved(self):
        """Fields of study should be preserved as list of dicts."""
        data = {
            "paperId": "123",
            "s2FieldsOfStudy": [
                {"category": "Economics", "source": "s2-fos-model"},
                {"category": "Computer Science", "source": "external"},
            ],
        }
        paper = S2Paper(**data)

        assert len(paper.s2_fields_of_study) == 2
        assert paper.s2_fields_of_study[0]["category"] == "Economics"

    def test_publication_venue_preserved(self):
        """Publication venue should be preserved as dict."""
        data = {
            "paperId": "123",
            "publicationVenue": {
                "id": "venue123",
                "name": "Nature",
                "type": "journal",
                "issn": "0028-0836",
            },
        }
        paper = S2Paper(**data)

        assert paper.publication_venue["name"] == "Nature"
        assert paper.publication_venue["type"] == "journal"


# -----------------------------------------------------------------------------
# S2SearchResult Tests
# -----------------------------------------------------------------------------


class TestS2SearchResult:
    """Tests for S2SearchResult model."""

    def test_parse_full_search_result(self):
        """Parse complete search result."""
        data = {
            "total": 1542,
            "offset": 0,
            "next": 10,
            "data": [
                {"paperId": "1", "title": "Paper 1"},
                {"paperId": "2", "title": "Paper 2"},
            ],
        }
        result = S2SearchResult(**data)

        assert result.total == 1542
        assert result.offset == 0
        assert result.next_offset == 10
        assert len(result.data) == 2

    def test_parse_empty_search_result(self):
        """Parse empty search result."""
        data = {
            "total": 0,
            "offset": 0,
            "data": [],
        }
        result = S2SearchResult(**data)

        assert result.total == 0
        assert result.data == []
        assert result.next_offset is None

    def test_default_values(self):
        """Test default values."""
        result = S2SearchResult()

        assert result.total == 0
        assert result.offset == 0
        assert result.next_offset is None
        assert result.data == []

    def test_papers_parsed_as_s2paper(self):
        """Papers should be parsed as S2Paper objects."""
        data = {
            "total": 2,
            "offset": 0,
            "data": [
                {
                    "paperId": "1",
                    "title": "Paper 1",
                    "authors": [{"name": "Alice"}],
                    "citationCount": 100,
                },
                {
                    "paperId": "2",
                    "title": "Paper 2",
                    "isOpenAccess": True,
                },
            ],
        }
        result = S2SearchResult(**data)

        assert all(isinstance(p, S2Paper) for p in result.data)
        assert result.data[0].citation_count == 100
        assert result.data[1].is_open_access is True

    def test_next_alias(self):
        """Test 'next' field is mapped to next_offset."""
        data = {
            "total": 100,
            "offset": 0,
            "next": 50,
            "data": [],
        }
        result = S2SearchResult(**data)

        assert result.next_offset == 50


# -----------------------------------------------------------------------------
# S2AuthorPapersResult Tests
# -----------------------------------------------------------------------------


class TestS2AuthorPapersResult:
    """Tests for S2AuthorPapersResult model."""

    def test_parse_full_result(self):
        """Parse complete author papers result."""
        data = {
            "total": 250,
            "offset": 0,
            "next": 100,
            "data": [
                {"paperId": "1", "title": "Paper 1"},
                {"paperId": "2", "title": "Paper 2"},
            ],
        }
        result = S2AuthorPapersResult(**data)

        assert result.total == 250
        assert result.offset == 0
        assert result.next_offset == 100
        assert len(result.data) == 2

    def test_default_values(self):
        """Test default values."""
        result = S2AuthorPapersResult()

        assert result.total == 0
        assert result.offset == 0
        assert result.next_offset is None
        assert result.data == []

    def test_papers_parsed_as_s2paper(self):
        """Papers should be parsed as S2Paper objects."""
        data = {
            "total": 1,
            "data": [
                {"paperId": "1", "title": "Paper", "year": 2020},
            ],
        }
        result = S2AuthorPapersResult(**data)

        assert len(result.data) == 1
        assert isinstance(result.data[0], S2Paper)
        assert result.data[0].year == 2020


# -----------------------------------------------------------------------------
# Serialization Tests
# -----------------------------------------------------------------------------


class TestModelSerialization:
    """Tests for model serialization."""

    def test_paper_to_dict(self):
        """Paper should serialize to dict."""
        paper = S2Paper(
            paperId="123",
            title="Test",
            year=2020,
            citationCount=100,
        )

        data = paper.model_dump()

        assert data["paper_id"] == "123"
        assert data["title"] == "Test"
        assert data["year"] == 2020
        assert data["citation_count"] == 100

    def test_paper_to_dict_by_alias(self):
        """Paper should serialize with original aliases."""
        paper = S2Paper(
            paperId="123",
            title="Test",
            citationCount=100,
        )

        data = paper.model_dump(by_alias=True)

        assert data["paperId"] == "123"
        assert data["citationCount"] == 100

    def test_paper_to_json(self):
        """Paper should serialize to JSON string."""
        paper = S2Paper(paperId="123", title="Test")

        json_str = paper.model_dump_json()

        assert '"paper_id":"123"' in json_str or '"paper_id": "123"' in json_str

    def test_search_result_to_dict(self):
        """Search result should serialize with nested papers."""
        result = S2SearchResult(
            total=1,
            offset=0,
            data=[S2Paper(paperId="1", title="Paper")],
        )

        data = result.model_dump()

        assert data["total"] == 1
        assert len(data["data"]) == 1
        assert data["data"][0]["paper_id"] == "1"


# -----------------------------------------------------------------------------
# Edge Cases Tests
# -----------------------------------------------------------------------------


class TestModelEdgeCases:
    """Tests for edge cases and special values."""

    def test_paper_with_zero_citations(self):
        """Paper with zero citations should parse correctly."""
        paper = S2Paper(paperId="123", citationCount=0, influentialCitationCount=0)

        assert paper.citation_count == 0
        assert paper.influential_citation_count == 0

    def test_paper_with_very_large_citations(self):
        """Paper with very large citation count should parse."""
        paper = S2Paper(paperId="123", citationCount=1_000_000)

        assert paper.citation_count == 1_000_000

    def test_paper_with_unicode_title(self):
        """Paper with Unicode title should parse."""
        paper = S2Paper(
            paperId="123",
            title="Economic Analysis: A 中文 Title with Umlauts",
        )

        assert "中文" in paper.title

    def test_author_with_special_characters(self):
        """Author with special characters in name."""
        author = S2Author(
            name="Jean-Pierre Andrassy-Habsburg",
            affiliations=["Ecole Polytechnique Federale de Lausanne"],
        )

        assert "Jean-Pierre" in author.name

    def test_empty_authors_list(self):
        """Empty authors list should be valid."""
        paper = S2Paper(paperId="123", authors=[])

        assert paper.authors == []
        assert paper.first_author_name is None

    def test_paper_year_boundaries(self):
        """Paper year edge values."""
        old_paper = S2Paper(paperId="1", year=1900)
        future_paper = S2Paper(paperId="2", year=2030)

        assert old_paper.year == 1900
        assert future_paper.year == 2030
