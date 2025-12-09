"""Tests for paper acquisition module."""

from pathlib import Path

import pytest

from s2_client.acquire import (
    AcquisitionResult,
    PaperAcquisition,
    compute_file_hash,
    generate_filename,
    sanitize_filename,
)
from s2_client.models import S2Paper, OpenAccessPdf, S2Author


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_paper() -> S2Paper:
    """Sample paper for testing."""
    return S2Paper(
        paperId="abc123",
        title="Double Machine Learning for Causal Inference",
        year=2020,
        authors=[
            S2Author(authorId="1", name="Victor Chernozhukov"),
            S2Author(authorId="2", name="Denis Chetverikov"),
        ],
        citationCount=1542,
        isOpenAccess=True,
        openAccessPdf=OpenAccessPdf(url="https://arxiv.org/pdf/1608.00060.pdf"),
        externalIds={
            "DOI": "10.1214/17-AOS1609",
            "ArXiv": "1608.00060",
        },
    )


@pytest.fixture
def paywalled_paper() -> S2Paper:
    """Sample paywalled paper."""
    return S2Paper(
        paperId="def456",
        title="Some Paywalled Paper",
        year=2022,
        authors=[S2Author(authorId="3", name="John Doe")],
        isOpenAccess=False,
    )


@pytest.fixture
def tmp_fixtures_dir(tmp_path: Path) -> Path:
    """Temporary fixtures directory."""
    fixtures = tmp_path / "fixtures" / "papers"
    fixtures.mkdir(parents=True)
    return fixtures


# -----------------------------------------------------------------------------
# Utility Tests
# -----------------------------------------------------------------------------


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_removes_special_chars(self):
        """Special characters should be removed."""
        result = sanitize_filename('Test: A "File" <Name>')
        assert ":" not in result
        assert '"' not in result
        assert "<" not in result
        assert ">" not in result

    def test_replaces_spaces(self):
        """Spaces should be replaced with underscores."""
        result = sanitize_filename("hello world test")
        assert " " not in result
        assert "_" in result

    def test_truncates_long_names(self):
        """Long names should be truncated."""
        long_name = "a" * 100
        result = sanitize_filename(long_name, max_length=20)
        assert len(result) == 20

    def test_lowercase(self):
        """Result should be lowercase."""
        result = sanitize_filename("HELLO WORLD")
        assert result == result.lower()


class TestGenerateFilename:
    """Tests for filename generation."""

    def test_generates_correct_format(self, sample_paper: S2Paper):
        """Filename should follow author_title_year.pdf format."""
        filename = generate_filename(sample_paper)
        assert filename.startswith("chernozhukov_")
        assert "_2020.pdf" in filename

    def test_handles_missing_author(self):
        """Should handle papers without authors."""
        paper = S2Paper(paperId="abc", title="Test Paper", year=2020)
        filename = generate_filename(paper)
        assert filename.startswith("unknown_")

    def test_handles_missing_year(self):
        """Should handle papers without year."""
        paper = S2Paper(
            paperId="abc",
            title="Test Paper",
            authors=[S2Author(authorId="1", name="John Doe")],
        )
        filename = generate_filename(paper)
        assert "_nd.pdf" in filename


class TestComputeFileHash:
    """Tests for file hashing."""

    def test_consistent_hash(self):
        """Same content should produce same hash."""
        content = b"test content"
        hash1 = compute_file_hash(content)
        hash2 = compute_file_hash(content)
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Different content should produce different hash."""
        hash1 = compute_file_hash(b"content 1")
        hash2 = compute_file_hash(b"content 2")
        assert hash1 != hash2

    def test_returns_hex_string(self):
        """Hash should be hex string."""
        hash_val = compute_file_hash(b"test")
        assert all(c in "0123456789abcdef" for c in hash_val)


# -----------------------------------------------------------------------------
# PaperAcquisition Tests
# -----------------------------------------------------------------------------


class TestPaperAcquisition:
    """Tests for PaperAcquisition class."""

    def test_is_duplicate_by_s2_id(self, sample_paper: S2Paper):
        """Should detect duplicate by S2 paper ID."""
        acq = PaperAcquisition(existing_s2_ids={"abc123"})
        assert acq.is_duplicate(sample_paper) is True

    def test_is_duplicate_by_doi(self, sample_paper: S2Paper):
        """Should detect duplicate by DOI."""
        acq = PaperAcquisition(existing_dois={"10.1214/17-AOS1609"})
        assert acq.is_duplicate(sample_paper) is True

    def test_is_duplicate_by_arxiv(self, sample_paper: S2Paper):
        """Should detect duplicate by arXiv ID."""
        acq = PaperAcquisition(existing_arxiv_ids={"1608.00060"})
        assert acq.is_duplicate(sample_paper) is True

    def test_not_duplicate_new_paper(self, sample_paper: S2Paper):
        """New paper should not be flagged as duplicate."""
        acq = PaperAcquisition()
        assert acq.is_duplicate(sample_paper) is False

    def test_get_pdf_url_open_access(self, sample_paper: S2Paper):
        """Should return S2 open access URL."""
        acq = PaperAcquisition()
        url = acq.get_pdf_url(sample_paper)
        assert url == "https://arxiv.org/pdf/1608.00060.pdf"

    def test_get_pdf_url_arxiv_fallback(self):
        """Should fall back to arXiv URL."""
        paper = S2Paper(
            paperId="abc",
            externalIds={"ArXiv": "2301.12345"},
            isOpenAccess=True,
        )
        acq = PaperAcquisition()
        url = acq.get_pdf_url(paper)
        assert url == "https://arxiv.org/pdf/2301.12345.pdf"

    def test_get_pdf_url_none_for_paywall(self, paywalled_paper: S2Paper):
        """Should return None for paywalled papers without arXiv."""
        acq = PaperAcquisition()
        url = acq.get_pdf_url(paywalled_paper)
        assert url is None


# -----------------------------------------------------------------------------
# AcquisitionResult Tests
# -----------------------------------------------------------------------------


class TestAcquisitionResult:
    """Tests for AcquisitionResult."""

    def test_empty_result(self):
        """Empty result should have zero counts."""
        result = AcquisitionResult()
        summary = result.to_summary_dict()
        assert summary["acquired"] == 0
        assert summary["total_processed"] == 0

    def test_summary_counts(self, sample_paper: S2Paper, paywalled_paper: S2Paper):
        """Summary should count all categories."""
        result = AcquisitionResult(
            acquired=[(sample_paper, Path("/tmp/test.pdf"))],
            skipped_paywall=[paywalled_paper],
        )
        summary = result.to_summary_dict()
        assert summary["acquired"] == 1
        assert summary["skipped_paywall"] == 1
        assert summary["total_processed"] == 2


# -----------------------------------------------------------------------------
# Sidecar Metadata Tests
# -----------------------------------------------------------------------------


class TestMetadataSidecar:
    """Tests for S2 metadata sidecar generation."""

    def test_sidecar_generation(self, sample_paper: S2Paper, tmp_fixtures_dir: Path):
        """Sidecar should be saved alongside PDF."""
        import json

        acq = PaperAcquisition(fixtures_dir=tmp_fixtures_dir)

        # Simulate the sidecar save (directly call private method)
        pdf_path = tmp_fixtures_dir / "test_paper.pdf"
        sidecar_path = pdf_path.with_suffix(".s2.json")
        file_hash = "abc123def456"

        acq._save_metadata_sidecar(sample_paper, sidecar_path, file_hash)

        # Verify sidecar exists
        assert sidecar_path.exists()

        # Verify contents
        with open(sidecar_path) as f:
            data = json.load(f)

        assert data["title"] == sample_paper.title
        assert data["year"] == sample_paper.year
        assert data["s2_paper_id"] == sample_paper.paper_id
        assert data["doi"] == sample_paper.doi
        assert data["arxiv_id"] == sample_paper.arxiv_id
        assert data["file_hash"] == file_hash
        assert data["sidecar_version"] == "1.0"
        assert "acquired_at" in data

    def test_sidecar_authors(self, sample_paper: S2Paper, tmp_fixtures_dir: Path):
        """Sidecar should contain author names."""
        import json

        acq = PaperAcquisition(fixtures_dir=tmp_fixtures_dir)
        sidecar_path = tmp_fixtures_dir / "test.s2.json"

        acq._save_metadata_sidecar(sample_paper, sidecar_path, "hash123")

        with open(sidecar_path) as f:
            data = json.load(f)

        assert data["authors"] == ["Victor Chernozhukov", "Denis Chetverikov"]

    def test_sidecar_handles_missing_fields(self, tmp_fixtures_dir: Path):
        """Sidecar should handle papers with missing optional fields."""
        import json

        paper = S2Paper(paperId="minimal", title="Minimal Paper")
        acq = PaperAcquisition(fixtures_dir=tmp_fixtures_dir)
        sidecar_path = tmp_fixtures_dir / "minimal.s2.json"

        acq._save_metadata_sidecar(paper, sidecar_path, "hash456")

        with open(sidecar_path) as f:
            data = json.load(f)

        assert data["title"] == "Minimal Paper"
        assert data["authors"] == []
        assert data["year"] is None
        assert data["doi"] is None
