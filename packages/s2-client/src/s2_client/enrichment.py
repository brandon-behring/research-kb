"""Citation enrichment module with multi-signal scoring.

Matches research-kb citations to Semantic Scholar papers using a
hierarchical ID resolution strategy with fuzzy disambiguation.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Literal

from research_kb_common import get_logger

from s2_client.models import S2Paper

if TYPE_CHECKING:
    from s2_client.client import S2Client

logger = get_logger(__name__)


@dataclass
class MatchResult:
    """Result of matching a citation to an S2 paper.

    Attributes:
        status: Match outcome (matched, ambiguous, unmatched)
        s2_paper: Matched S2Paper if found
        confidence: Confidence score (0.0 to 1.0)
        match_method: Method used for matching (doi, arxiv, title_unique, multi_signal)
    """

    status: Literal["matched", "ambiguous", "unmatched"]
    s2_paper: S2Paper | None = None
    confidence: float = 0.0
    match_method: str = ""


@dataclass
class Citation:
    """Citation record from research-kb database.

    Represents a citation extracted from a source document,
    typically via GROBID extraction.
    """

    id: str
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None


def normalize_string(s: str | None) -> str:
    """Normalize string for comparison."""
    if not s:
        return ""
    return s.lower().strip()


def fuzzy_match_score(s1: str | None, s2: str | None) -> float:
    """Compute fuzzy string match score using SequenceMatcher.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score from 0.0 to 1.0
    """
    n1 = normalize_string(s1)
    n2 = normalize_string(s2)

    if not n1 or not n2:
        return 0.0

    return SequenceMatcher(None, n1, n2).ratio()


def compute_author_overlap(
    citation_authors: list[str] | None, paper_authors: list | None
) -> float:
    """Compute author overlap score using fuzzy matching.

    Handles name variations (e.g., "V. Chernozhukov" vs "Victor Chernozhukov").

    Args:
        citation_authors: Author names from citation
        paper_authors: S2Author objects from paper

    Returns:
        Overlap score from 0.0 to 1.0
    """
    if not citation_authors or not paper_authors:
        return 0.0

    # Extract names from S2Author objects
    paper_names = [a.name for a in paper_authors if hasattr(a, "name") and a.name]
    if not paper_names:
        return 0.0

    # Normalize all names
    citation_normalized = [normalize_string(a) for a in citation_authors]
    paper_normalized = [normalize_string(n) for n in paper_names]

    # Find best matches for each citation author
    matches = 0
    for cit_author in citation_normalized:
        best_match = max(
            (fuzzy_match_score(cit_author, paper_author) for paper_author in paper_normalized),
            default=0.0,
        )
        if best_match >= 0.7:  # Threshold for considering a match
            matches += 1

    # Score is proportion of citation authors matched
    return matches / len(citation_authors)


def compute_venue_match(citation_venue: str | None, paper_venue: str | None) -> float:
    """Compute venue match score.

    Args:
        citation_venue: Venue from citation
        paper_venue: Venue from S2 paper

    Returns:
        Match score (1.0 for exact match, 0.7+ for fuzzy match, 0.0 for no match)
    """
    if not citation_venue or not paper_venue:
        return 0.0

    # Exact match (case-insensitive)
    if normalize_string(citation_venue) == normalize_string(paper_venue):
        return 1.0

    # Fuzzy match
    score = fuzzy_match_score(citation_venue, paper_venue)

    # Common abbreviations boost
    abbrev_pairs = [
        ("neurips", "neural information processing systems"),
        ("icml", "international conference on machine learning"),
        ("iclr", "international conference on learning representations"),
        ("acl", "association for computational linguistics"),
        ("emnlp", "empirical methods in natural language processing"),
        ("jmlr", "journal of machine learning research"),
        ("aer", "american economic review"),
        ("ecta", "econometrica"),
    ]

    cv_lower = normalize_string(citation_venue)
    pv_lower = normalize_string(paper_venue)

    for abbrev, full in abbrev_pairs:
        if (abbrev in cv_lower and full in pv_lower) or (full in cv_lower and abbrev in pv_lower):
            return 0.9

    return score


def normalize_citation_rank(paper: S2Paper, candidates: list[S2Paper]) -> float:
    """Normalize citation count to rank score.

    Higher-cited papers among candidates get higher scores.

    Args:
        paper: Paper to score
        candidates: All candidate papers

    Returns:
        Normalized rank score from 0.0 to 1.0
    """
    citations = [p.citation_count or 0 for p in candidates]
    if not citations:
        return 0.0

    max_cites = max(citations)
    if max_cites == 0:
        return 0.5  # All have zero citations

    paper_cites = paper.citation_count or 0
    return paper_cites / max_cites


def score_candidates(
    citation: Citation, candidates: list[S2Paper]
) -> tuple[S2Paper, float]:
    """Score candidates using weighted multi-signal approach.

    Weights (per /iterate decision):
        0.4 × author_overlap (fuzzy string match)
        0.3 × venue_match (exact or fuzzy)
        0.2 × year_exact (1.0 if match, 0.0 if not)
        0.1 × citation_rank (higher citations = more likely correct)

    Args:
        citation: Citation to match
        candidates: Candidate S2 papers

    Returns:
        Tuple of (best_paper, score)
    """
    if not candidates:
        raise ValueError("No candidates to score")

    scores = []
    for paper in candidates:
        author_score = compute_author_overlap(citation.authors, paper.authors)
        venue_score = compute_venue_match(citation.venue, paper.venue)
        year_score = 1.0 if citation.year == paper.year else 0.0
        citation_score = normalize_citation_rank(paper, candidates)

        total = 0.4 * author_score + 0.3 * venue_score + 0.2 * year_score + 0.1 * citation_score

        logger.debug(
            "candidate_score",
            paper_id=paper.paper_id,
            title=paper.title[:40] if paper.title else "Unknown",
            author_score=f"{author_score:.2f}",
            venue_score=f"{venue_score:.2f}",
            year_score=f"{year_score:.2f}",
            citation_score=f"{citation_score:.2f}",
            total=f"{total:.2f}",
        )

        scores.append((paper, total))

    return max(scores, key=lambda x: x[1])


async def match_citation(citation: Citation, client: "S2Client") -> MatchResult:
    """Match citation to S2 paper using ID hierarchy + multi-signal scoring.

    Resolution strategy:
    1. DOI (confidence 1.0) - exact match
    2. arXiv ID (confidence 0.95) - exact match
    3. Title + Year search with multi-signal disambiguation

    Args:
        citation: Citation to match
        client: S2Client instance

    Returns:
        MatchResult with status, paper, confidence, and method
    """
    # 1. DOI (highest confidence - exact match)
    if citation.doi:
        try:
            paper = await client.get_paper(f"DOI:{citation.doi}")
            if paper:
                logger.info(
                    "match_by_doi",
                    citation_title=citation.title[:40] if citation.title else "Unknown",
                    doi=citation.doi,
                )
                return MatchResult("matched", paper, 1.0, "doi")
        except Exception as e:
            logger.debug("doi_lookup_failed", doi=citation.doi, error=str(e))

    # 2. arXiv ID (high confidence - exact match)
    if citation.arxiv_id:
        try:
            paper = await client.get_paper(f"arXiv:{citation.arxiv_id}")
            if paper:
                logger.info(
                    "match_by_arxiv",
                    citation_title=citation.title[:40] if citation.title else "Unknown",
                    arxiv_id=citation.arxiv_id,
                )
                return MatchResult("matched", paper, 0.95, "arxiv")
        except Exception as e:
            logger.debug("arxiv_lookup_failed", arxiv_id=citation.arxiv_id, error=str(e))

    # 3. Title + Year search with multi-signal disambiguation
    if citation.title and citation.year:
        try:
            results = await client.search_papers(
                f'"{citation.title}"',
                year=str(citation.year),
                limit=5,
            )

            if len(results.data) == 1:
                # Unique match
                logger.info(
                    "match_by_title_unique",
                    citation_title=citation.title[:40] if citation.title else "Unknown",
                )
                return MatchResult("matched", results.data[0], 0.8, "title_unique")

            elif len(results.data) > 1:
                # Multi-signal scoring for disambiguation
                best_paper, score = score_candidates(citation, results.data)

                if score >= 0.8:
                    logger.info(
                        "match_by_multi_signal",
                        citation_title=citation.title[:40] if citation.title else "Unknown",
                        score=f"{score:.2f}",
                    )
                    return MatchResult("matched", best_paper, score, "multi_signal")
                else:
                    logger.info(
                        "match_ambiguous",
                        citation_title=citation.title[:40] if citation.title else "Unknown",
                        candidates=len(results.data),
                        best_score=f"{score:.2f}",
                    )
                    return MatchResult("ambiguous", None, score, "below_threshold")

        except Exception as e:
            logger.debug("title_search_failed", title=citation.title[:40], error=str(e))

    logger.info(
        "match_unmatched",
        citation_title=citation.title[:40] if citation.title else "Unknown",
    )
    return MatchResult("unmatched", None, 0.0, "no_match")


def citation_to_enrichment_metadata(result: MatchResult) -> dict:
    """Convert match result to metadata dict for citations.metadata JSONB.

    Args:
        result: MatchResult from matching

    Returns:
        Dict suitable for storing in citations.metadata
    """
    from datetime import datetime, timezone

    if result.status != "matched" or not result.s2_paper:
        return {
            "s2_match_status": result.status,
            "s2_match_confidence": result.confidence,
            "s2_match_method": result.match_method,
            "s2_enriched_at": datetime.now(timezone.utc).isoformat(),
        }

    paper = result.s2_paper
    return {
        "s2_paper_id": paper.paper_id,
        "s2_citation_count": paper.citation_count,
        "s2_influential_count": paper.influential_citation_count,
        "s2_match_confidence": result.confidence,
        "s2_match_method": result.match_method,
        "s2_enriched_at": datetime.now(timezone.utc).isoformat(),
        "s2_fields_of_study": [
            f.get("category") for f in (paper.s2_fields_of_study or [])
        ],
        "s2_venue": paper.venue,
        "s2_year": paper.year,
        # Enhanced fields (Phase 2.2)
        "s2_abstract": (paper.abstract or "")[:2000],  # Truncated for storage
        "s2_reference_count": paper.reference_count,
        "s2_publication_types": paper.publication_types or [],
        "s2_is_open_access": paper.is_open_access,
        "s2_open_access_pdf_url": (
            paper.open_access_pdf.url if paper.open_access_pdf else None
        ),
    }
