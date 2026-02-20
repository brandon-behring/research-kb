"""Tests for citation enrichment module."""

import pytest

from s2_client.enrichment import (
    Citation,
    MatchResult,
    citation_to_enrichment_metadata,
    compute_author_overlap,
    compute_venue_match,
    fuzzy_match_score,
    normalize_citation_rank,
    normalize_string,
    score_candidates,
)
from s2_client.models import S2Author, S2Paper

pytestmark = pytest.mark.unit


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_citation() -> Citation:
    """Sample citation for testing."""
    return Citation(
        id="cit-001",
        title="Double/Debiased Machine Learning for Treatment and Structural Parameters",
        authors=["Victor Chernozhukov", "Denis Chetverikov", "Mert Demirer"],
        year=2018,
        venue="Econometrica",
        doi="10.3982/ECTA12723",
    )


@pytest.fixture
def matching_paper() -> S2Paper:
    """Paper that matches the sample citation."""
    return S2Paper(
        paperId="abc123",
        title="Double/Debiased Machine Learning for Treatment and Structural Parameters",
        year=2018,
        authors=[
            S2Author(authorId="1", name="Victor Chernozhukov"),
            S2Author(authorId="2", name="Denis Chetverikov"),
            S2Author(authorId="3", name="Mert Demirer"),
        ],
        citationCount=2500,
        venue="Econometrica",
        isOpenAccess=True,
    )


@pytest.fixture
def similar_paper() -> S2Paper:
    """Paper with similar title but different details."""
    return S2Paper(
        paperId="def456",
        title="Double Machine Learning: A Practical Guide",
        year=2020,
        authors=[
            S2Author(authorId="4", name="John Smith"),
            S2Author(authorId="5", name="Jane Doe"),
        ],
        citationCount=50,
        venue="Journal of Econometrics",
        isOpenAccess=True,
    )


# -----------------------------------------------------------------------------
# String Normalization Tests
# -----------------------------------------------------------------------------


class TestNormalizeString:
    """Tests for string normalization."""

    def test_lowercase(self):
        """Should lowercase strings."""
        assert normalize_string("HELLO WORLD") == "hello world"

    def test_strip_whitespace(self):
        """Should strip leading/trailing whitespace."""
        assert normalize_string("  test  ") == "test"

    def test_none_handling(self):
        """Should handle None input."""
        assert normalize_string(None) == ""


class TestFuzzyMatchScore:
    """Tests for fuzzy string matching."""

    def test_exact_match(self):
        """Exact strings should have score 1.0."""
        assert fuzzy_match_score("test", "test") == 1.0

    def test_similar_strings(self):
        """Similar strings should have high score."""
        score = fuzzy_match_score("Victor Chernozhukov", "V. Chernozhukov")
        assert score > 0.7

    def test_different_strings(self):
        """Different strings should have low score."""
        score = fuzzy_match_score("apple", "banana")
        assert score < 0.5

    def test_empty_strings(self):
        """Empty strings should return 0."""
        assert fuzzy_match_score("", "test") == 0.0
        assert fuzzy_match_score(None, "test") == 0.0


# -----------------------------------------------------------------------------
# Author Overlap Tests
# -----------------------------------------------------------------------------


class TestComputeAuthorOverlap:
    """Tests for author overlap computation."""

    def test_exact_match(self, matching_paper: S2Paper):
        """Exact author match should have high score."""
        citation_authors = ["Victor Chernozhukov", "Denis Chetverikov"]
        score = compute_author_overlap(citation_authors, matching_paper.authors)
        assert score >= 0.9

    def test_partial_name_match(self, matching_paper: S2Paper):
        """Abbreviated names should still match."""
        citation_authors = ["V. Chernozhukov", "D. Chetverikov"]
        score = compute_author_overlap(citation_authors, matching_paper.authors)
        assert score >= 0.5  # Some matches expected

    def test_no_overlap(self, similar_paper: S2Paper):
        """No matching authors should have low score."""
        citation_authors = ["Victor Chernozhukov", "Denis Chetverikov"]
        score = compute_author_overlap(citation_authors, similar_paper.authors)
        assert score < 0.3

    def test_empty_authors(self):
        """Empty author lists should return 0."""
        assert compute_author_overlap([], []) == 0.0
        assert compute_author_overlap(None, None) == 0.0


# -----------------------------------------------------------------------------
# Venue Match Tests
# -----------------------------------------------------------------------------


class TestComputeVenueMatch:
    """Tests for venue matching."""

    def test_exact_match(self):
        """Exact venue match should return 1.0."""
        score = compute_venue_match("Econometrica", "Econometrica")
        assert score == 1.0

    def test_case_insensitive(self):
        """Match should be case-insensitive."""
        score = compute_venue_match("ECONOMETRICA", "econometrica")
        assert score == 1.0

    def test_abbreviation_match(self):
        """Common abbreviations should match full names."""
        score = compute_venue_match("NeurIPS", "Neural Information Processing Systems")
        assert score >= 0.8

    def test_no_match(self):
        """Different venues should have low score."""
        score = compute_venue_match("Nature", "Science")
        assert score < 0.5

    def test_empty_venues(self):
        """Empty venues should return 0."""
        assert compute_venue_match("", "test") == 0.0
        assert compute_venue_match(None, None) == 0.0


# -----------------------------------------------------------------------------
# Citation Rank Tests
# -----------------------------------------------------------------------------


class TestNormalizeCitationRank:
    """Tests for citation rank normalization."""

    def test_highest_cited_paper(self, matching_paper: S2Paper, similar_paper: S2Paper):
        """Highest cited paper should have score 1.0."""
        candidates = [matching_paper, similar_paper]
        score = normalize_citation_rank(matching_paper, candidates)
        assert score == 1.0

    def test_lower_cited_paper(self, matching_paper: S2Paper, similar_paper: S2Paper):
        """Lower cited paper should have proportional score."""
        candidates = [matching_paper, similar_paper]
        score = normalize_citation_rank(similar_paper, candidates)
        assert score == 50 / 2500  # 50 / 2500

    def test_all_zero_citations(self):
        """All zero citations should return 0.5."""
        papers = [
            S2Paper(paperId="a", citationCount=0),
            S2Paper(paperId="b", citationCount=0),
        ]
        score = normalize_citation_rank(papers[0], papers)
        assert score == 0.5


# -----------------------------------------------------------------------------
# Score Candidates Tests
# -----------------------------------------------------------------------------


class TestScoreCandidates:
    """Tests for multi-signal candidate scoring."""

    def test_best_match_wins(
        self,
        sample_citation: Citation,
        matching_paper: S2Paper,
        similar_paper: S2Paper,
    ):
        """Best matching paper should be selected."""
        candidates = [similar_paper, matching_paper]  # Matching paper is second
        best_paper, score = score_candidates(sample_citation, candidates)

        assert best_paper.paper_id == matching_paper.paper_id
        assert score > 0.8  # Should exceed threshold

    def test_weights_applied(self, sample_citation: Citation, matching_paper: S2Paper):
        """Score should reflect weighted components."""
        best_paper, score = score_candidates(sample_citation, [matching_paper])

        # With perfect author (0.4), venue (0.3), year (0.2), and citation (0.1)
        # score should be close to 1.0
        assert score >= 0.9

    def test_empty_candidates_raises(self, sample_citation: Citation):
        """Empty candidates should raise ValueError."""
        with pytest.raises(ValueError, match="No candidates"):
            score_candidates(sample_citation, [])


# -----------------------------------------------------------------------------
# MatchResult Tests
# -----------------------------------------------------------------------------


class TestMatchResult:
    """Tests for MatchResult dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        result = MatchResult(status="unmatched")
        assert result.s2_paper is None
        assert result.confidence == 0.0
        assert result.match_method == ""

    def test_matched_result(self, matching_paper: S2Paper):
        """Matched result should store paper."""
        result = MatchResult(
            status="matched",
            s2_paper=matching_paper,
            confidence=0.95,
            match_method="doi",
        )
        assert result.s2_paper == matching_paper
        assert result.confidence == 0.95


# -----------------------------------------------------------------------------
# Metadata Conversion Tests
# -----------------------------------------------------------------------------


class TestCitationToEnrichmentMetadata:
    """Tests for converting match results to metadata."""

    def test_matched_result_metadata(self, matching_paper: S2Paper):
        """Matched result should include S2 paper fields."""
        result = MatchResult(
            status="matched",
            s2_paper=matching_paper,
            confidence=0.95,
            match_method="doi",
        )
        metadata = citation_to_enrichment_metadata(result)

        assert metadata["s2_paper_id"] == matching_paper.paper_id
        assert metadata["s2_citation_count"] == matching_paper.citation_count
        assert metadata["s2_match_confidence"] == 0.95
        assert metadata["s2_match_method"] == "doi"
        assert "s2_enriched_at" in metadata

    def test_unmatched_result_metadata(self):
        """Unmatched result should only have status fields."""
        result = MatchResult(status="unmatched", confidence=0.0, match_method="no_match")
        metadata = citation_to_enrichment_metadata(result)

        assert metadata["s2_match_status"] == "unmatched"
        assert metadata["s2_match_confidence"] == 0.0
        assert "s2_paper_id" not in metadata
        assert "s2_enriched_at" in metadata

    def test_ambiguous_result_metadata(self):
        """Ambiguous result should record status."""
        result = MatchResult(
            status="ambiguous",
            confidence=0.6,
            match_method="below_threshold",
        )
        metadata = citation_to_enrichment_metadata(result)

        assert metadata["s2_match_status"] == "ambiguous"
        assert metadata["s2_match_confidence"] == 0.6
