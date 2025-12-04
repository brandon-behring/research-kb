"""Tests for BibTeX generation functionality.

Phase 1 Gap 4: Validate BibTeX escaping, key generation, and entry formatting.
"""

import uuid
from datetime import datetime, timezone

import pytest

from research_kb_contracts import Citation, Source, SourceType
from research_kb_pdf.bibtex_generator import (
    citation_to_bibtex,
    escape_bibtex,
    generate_bibtex_key,
    source_to_bibtex,
)


class TestEscapeBibtex:
    """Test special character escaping for BibTeX safety."""

    def test_escapes_ampersand(self):
        """Ampersand should be escaped."""
        assert escape_bibtex("Smith & Jones") == "Smith \\& Jones"

    def test_escapes_percent(self):
        """Percent should be escaped."""
        assert escape_bibtex("100% accurate") == "100\\% accurate"

    def test_escapes_hash(self):
        """Hash should be escaped."""
        assert escape_bibtex("Issue #42") == "Issue \\#42"

    def test_escapes_underscore(self):
        """Underscore should be escaped."""
        assert escape_bibtex("multi_word_name") == "multi\\_word\\_name"

    def test_escapes_braces(self):
        """Braces should be escaped."""
        assert escape_bibtex("{group}") == "\\{group\\}"

    def test_escapes_multiple_special_chars(self):
        """Multiple special chars in one string."""
        result = escape_bibtex("O'Connor & Jones: 100% #1")
        assert "\\&" in result
        assert "\\%" in result
        assert "\\#" in result

    def test_preserves_plain_text(self):
        """Plain text should pass through unchanged."""
        plain = "Normal academic title"
        assert escape_bibtex(plain) == plain


class TestGenerateBibtexKey:
    """Test citation key generation."""

    def test_generates_standard_key(self):
        """Standard format: lastnameyearfirstword."""
        key = generate_bibtex_key(
            "Judea Pearl", 2009, "Causality: Models, Reasoning and Inference"
        )
        assert key == "pearl2009causality"

    def test_handles_multi_word_name(self):
        """Extracts last name from multi-word author name."""
        key = generate_bibtex_key(
            "Joshua D. Angrist", 2009, "Mostly Harmless Econometrics"
        )
        assert key == "angrist2009mostly"

    def test_skips_stopwords_in_title(self):
        """Skips common stopwords (the, a, an, of)."""
        key = generate_bibtex_key("John Smith", 2020, "The Art of Causal Inference")
        assert key == "smith2020art"

    def test_handles_missing_year(self):
        """Uses 0000 for missing year."""
        key = generate_bibtex_key("Jane Doe", None, "Some Paper")
        assert "0000" in key

    def test_handles_empty_author(self):
        """Uses 'unknown' for empty author."""
        key = generate_bibtex_key("", 2020, "Anonymous Work")
        assert key.startswith("unknown")

    def test_removes_non_alphanumeric(self):
        """Removes special chars from key components."""
        key = generate_bibtex_key("O'Brien, Jr.", 2020, "Analysis: A Study")
        assert "'" not in key
        assert "," not in key


class TestCitationToBibtex:
    """Test Citation to BibTeX conversion."""

    def test_generates_valid_entry_format(self):
        """Generated entry should have valid BibTeX structure."""
        cit = Citation(
            authors=["Judea Pearl"],
            title="Causality",
            year=2009,
            venue="Cambridge University Press",
            raw_string="Pearl, J. (2009). Causality. CUP.",
        )
        result = citation_to_bibtex(cit)

        assert result.startswith("@article{")
        assert "author = {" in result
        assert "title = {Causality}" in result
        assert "year = {2009}" in result
        assert result.endswith("}")

    def test_formats_multiple_authors(self):
        """Authors should be joined with 'and'."""
        cit = Citation(
            authors=["John Smith", "Jane Doe"],
            title="Joint Work",
            year=2020,
            raw_string="Smith & Doe (2020)",
        )
        result = citation_to_bibtex(cit)

        assert "Smith, John and Doe, Jane" in result

    def test_includes_arxiv_fields(self):
        """arXiv papers should have eprint and archiveprefix."""
        cit = Citation(
            authors=["Victor Chernozhukov"],
            title="Double/Debiased ML",
            year=2018,
            arxiv_id="1608.00060",
            raw_string="Chernozhukov et al. (2018)",
        )
        result = citation_to_bibtex(cit)

        assert "eprint = {1608.00060}" in result
        assert "archiveprefix = {arXiv}" in result

    def test_includes_doi(self):
        """DOI should be included when present."""
        cit = Citation(
            authors=["Some Author"],
            title="Published Work",
            year=2021,
            doi="10.1234/example.2021",
            raw_string="Author (2021)",
        )
        result = citation_to_bibtex(cit)

        assert "doi = {10.1234/example.2021}" in result

    def test_uses_misc_without_venue(self):
        """Entry type should be misc when no venue."""
        cit = Citation(
            authors=["Author"],
            title="Working Paper",
            year=2020,
            raw_string="Author (2020)",
        )
        result = citation_to_bibtex(cit)

        assert result.startswith("@misc{")


class TestSourceToBibtex:
    """Test Source to BibTeX conversion."""

    @pytest.fixture
    def now(self):
        """Current timestamp for Source creation."""
        return datetime.now(timezone.utc)

    def test_book_for_textbook(self, now):
        """Textbook sources should use @book entry type."""
        source = Source(
            id=uuid.uuid4(),
            source_type=SourceType.TEXTBOOK,
            title="Causality: Models, Reasoning and Inference",
            authors=["Pearl, Judea"],
            year=2009,
            file_path="/path/to/pearl.pdf",
            file_hash="abc123",
            metadata={"publisher": "Cambridge University Press"},
            created_at=now,
            updated_at=now,
        )
        result = source_to_bibtex(source)

        assert result.startswith("@book{")
        assert "publisher = {Cambridge University Press}" in result

    def test_article_for_paper(self, now):
        """Paper sources should use @article entry type."""
        source = Source(
            id=uuid.uuid4(),
            source_type=SourceType.PAPER,
            title="Double/Debiased Machine Learning",
            authors=["Chernozhukov, Victor"],
            year=2018,
            file_path="/path/to/dml.pdf",
            file_hash="def456",
            metadata={"arxiv_id": "1608.00060"},
            created_at=now,
            updated_at=now,
        )
        result = source_to_bibtex(source)

        assert result.startswith("@article{")
        assert "eprint = {1608.00060}" in result
        assert "archiveprefix = {arXiv}" in result

    def test_includes_all_authors(self, now):
        """All authors should be included."""
        source = Source(
            id=uuid.uuid4(),
            source_type=SourceType.TEXTBOOK,
            title="Mostly Harmless Econometrics",
            authors=["Angrist, Joshua D.", "Pischke, Jörn-Steffen"],
            year=2009,
            file_path="/path/to/mhe.pdf",
            file_hash="ghi789",
            metadata={},
            created_at=now,
            updated_at=now,
        )
        result = source_to_bibtex(source)

        assert "Angrist" in result
        assert "Pischke" in result

    def test_escapes_special_chars_in_title(self, now):
        """Special chars in title should be escaped."""
        source = Source(
            id=uuid.uuid4(),
            source_type=SourceType.PAPER,
            title="Why 100% of Papers Use & Analysis",
            authors=["Author, Test"],
            year=2020,
            file_path="/path/to/paper.pdf",
            file_hash="jkl012",
            metadata={},
            created_at=now,
            updated_at=now,
        )
        result = source_to_bibtex(source)

        assert "\\%" in result
        assert "\\&" in result
