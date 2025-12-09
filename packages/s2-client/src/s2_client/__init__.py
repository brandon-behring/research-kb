"""Semantic Scholar API Client.

Version: 1.0.0

An async Python client for the Semantic Scholar Academic Graph API with:
- Rate limiting (configurable RPS)
- Response caching (SQLite, 7-day TTL)
- Pydantic models for type safety
- Retry on transient errors

Usage:
    >>> from s2_client import S2Client, S2Paper
    >>> async with S2Client() as client:
    ...     papers = await client.search_papers("causal forest", limit=10)
    ...     for paper in papers:
    ...         print(f"{paper.title} ({paper.year}) - {paper.citation_count} citations")
"""

from s2_client.acquire import (
    AcquisitionResult,
    PaperAcquisition,
    load_existing_identifiers,
)
from s2_client.cache import S2Cache
from s2_client.client import S2Client
from s2_client.enrichment import (
    Citation,
    MatchResult,
    citation_to_enrichment_metadata,
    match_citation,
    score_candidates,
)
from s2_client.errors import (
    S2APIError,
    S2Error,
    S2NotFoundError,
    S2RateLimitError,
)
from s2_client.models import (
    OpenAccessPdf,
    S2Author,
    S2Paper,
    S2SearchResult,
)
from s2_client.rate_limiter import RateLimiter
from s2_client.search import DiscoveryTopic, SearchFilters, TopicDiscovery

__version__ = "1.0.0"

__all__ = [
    # Client
    "S2Client",
    # Models
    "S2Paper",
    "S2Author",
    "S2SearchResult",
    "OpenAccessPdf",
    # Search
    "SearchFilters",
    "TopicDiscovery",
    "DiscoveryTopic",
    # Acquisition
    "PaperAcquisition",
    "AcquisitionResult",
    "load_existing_identifiers",
    # Enrichment
    "Citation",
    "MatchResult",
    "match_citation",
    "score_candidates",
    "citation_to_enrichment_metadata",
    # Rate limiting & caching
    "RateLimiter",
    "S2Cache",
    # Errors
    "S2Error",
    "S2APIError",
    "S2RateLimitError",
    "S2NotFoundError",
]
