"""Tests for S2 client.

Uses respx to mock httpx requests for deterministic testing.
"""

from pathlib import Path

import pytest
import respx
from httpx import Response

from s2_client import (
    S2Client,
    S2Paper,
    S2RateLimitError,
    S2NotFoundError,
)
from s2_client.rate_limiter import RateLimiter
from s2_client.cache import S2Cache
from s2_client.search import SearchFilters, TopicDiscovery, DiscoveryTopic

pytestmark = pytest.mark.unit


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_paper_response() -> dict:
    """Sample paper response from S2 API."""
    return {
        "paperId": "649def34f8be52c8b66281af98ae884c09aef38b",
        "corpusId": 14457330,
        "externalIds": {
            "DOI": "10.1214/17-AOS1609",
            "ArXiv": "1608.00060",
        },
        "title": "Double/debiased machine learning for treatment and structural parameters",
        "abstract": "We revisit the classic problem of estimation...",
        "venue": "The Annals of Statistics",
        "year": 2018,
        "publicationDate": "2018-04-01",
        "authors": [
            {"authorId": "26331346", "name": "Victor Chernozhukov"},
            {"authorId": "2149494", "name": "Denis Chetverikov"},
        ],
        "referenceCount": 78,
        "citationCount": 1542,
        "influentialCitationCount": 234,
        "isOpenAccess": True,
        "openAccessPdf": {
            "url": "https://arxiv.org/pdf/1608.00060.pdf",
            "status": "GREEN",
        },
        "s2FieldsOfStudy": [
            {"category": "Economics", "source": "s2-fos-model"},
            {"category": "Computer Science", "source": "s2-fos-model"},
        ],
        "publicationTypes": ["JournalArticle"],
    }


@pytest.fixture
def sample_search_response(sample_paper_response: dict) -> dict:
    """Sample search response from S2 API."""
    return {
        "total": 1542,
        "offset": 0,
        "next": 10,
        "data": [sample_paper_response],
    }


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Temporary directory for cache tests."""
    cache_dir = tmp_path / "s2_cache"
    cache_dir.mkdir()
    return cache_dir


# -----------------------------------------------------------------------------
# Model Tests
# -----------------------------------------------------------------------------


class TestS2Paper:
    """Tests for S2Paper model."""

    def test_parse_paper(self, sample_paper_response: dict):
        """Paper should parse from API response."""
        paper = S2Paper(**sample_paper_response)

        assert paper.paper_id == "649def34f8be52c8b66281af98ae884c09aef38b"
        assert (
            paper.title
            == "Double/debiased machine learning for treatment and structural parameters"
        )
        assert paper.year == 2018
        assert paper.citation_count == 1542
        assert paper.is_open_access is True

    def test_doi_property(self, sample_paper_response: dict):
        """DOI should be extracted from external IDs."""
        paper = S2Paper(**sample_paper_response)
        assert paper.doi == "10.1214/17-AOS1609"

    def test_arxiv_id_property(self, sample_paper_response: dict):
        """arXiv ID should be extracted from external IDs."""
        paper = S2Paper(**sample_paper_response)
        assert paper.arxiv_id == "1608.00060"

    def test_first_author_name(self, sample_paper_response: dict):
        """First author name should be accessible."""
        paper = S2Paper(**sample_paper_response)
        assert paper.first_author_name == "Victor Chernozhukov"

    def test_to_metadata_dict(self, sample_paper_response: dict):
        """Metadata dict should have expected keys."""
        paper = S2Paper(**sample_paper_response)
        metadata = paper.to_metadata_dict()

        assert metadata["s2_paper_id"] == paper.paper_id
        assert metadata["doi"] == "10.1214/17-AOS1609"
        assert metadata["arxiv_id"] == "1608.00060"
        assert metadata["citation_count"] == 1542
        assert metadata["is_open_access"] is True
        assert "s2_enriched_at" in metadata

    def test_parse_minimal_paper(self):
        """Paper should parse with minimal fields."""
        paper = S2Paper(paperId="abc123", title="Test Paper")
        assert paper.paper_id == "abc123"
        assert paper.title == "Test Paper"
        assert paper.citation_count is None


# -----------------------------------------------------------------------------
# Rate Limiter Tests
# -----------------------------------------------------------------------------


class TestRateLimiter:
    """Tests for RateLimiter."""

    async def test_acquire_immediate(self):
        """First acquire should be immediate with full bucket."""
        limiter = RateLimiter(requests_per_second=10)
        await limiter.acquire()  # Should not block
        assert limiter.available_tokens >= 9

    async def test_context_manager(self):
        """Context manager should acquire token."""
        limiter = RateLimiter(requests_per_second=10)
        initial = limiter.available_tokens

        async with limiter:
            pass

        assert limiter.available_tokens < initial


# -----------------------------------------------------------------------------
# Cache Tests
# -----------------------------------------------------------------------------


class TestS2Cache:
    """Tests for S2Cache."""

    async def test_cache_roundtrip(self, tmp_cache_dir: Path):
        """Data should survive cache roundtrip."""
        cache = S2Cache(cache_dir=tmp_cache_dir)
        await cache.initialize()

        test_data = {"title": "Test Paper", "year": 2024}
        await cache.set("paper/123", {"fields": "title"}, test_data)

        cached = await cache.get("paper/123", {"fields": "title"})
        assert cached == test_data

        await cache.close()

    async def test_cache_miss(self, tmp_cache_dir: Path):
        """Non-existent key should return None."""
        cache = S2Cache(cache_dir=tmp_cache_dir)
        await cache.initialize()

        result = await cache.get("nonexistent", None)
        assert result is None

        await cache.close()

    async def test_cache_stats(self, tmp_cache_dir: Path):
        """Stats should reflect cache state."""
        cache = S2Cache(cache_dir=tmp_cache_dir)
        await cache.initialize()

        await cache.set("key1", None, {"data": 1})
        await cache.set("key2", None, {"data": 2})

        stats = await cache.stats()
        assert stats["valid_entries"] == 2

        await cache.close()

    async def test_cache_context_manager(self, tmp_cache_dir: Path):
        """Context manager should initialize and close."""
        async with S2Cache(cache_dir=tmp_cache_dir) as cache:
            await cache.set("test", None, {"value": 42})
            result = await cache.get("test", None)
            assert result == {"value": 42}


# -----------------------------------------------------------------------------
# Client Tests (with mocked HTTP)
# -----------------------------------------------------------------------------


class TestS2Client:
    """Tests for S2Client with mocked HTTP."""

    @respx.mock
    async def test_search_papers(self, sample_search_response: dict, tmp_cache_dir: Path):
        """Search should parse results correctly."""
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(200, json=sample_search_response)
        )

        async with S2Client(use_cache=False) as client:
            result = await client.search_papers("double machine learning", limit=10)

        assert result.total == 1542
        assert len(result.data) == 1
        assert result.data[0].title.startswith("Double/debiased")

    @respx.mock
    async def test_get_paper(self, sample_paper_response: dict):
        """Get paper by ID should return parsed paper."""
        respx.get("https://api.semanticscholar.org/graph/v1/paper/DOI:10.1214/17-AOS1609").mock(
            return_value=Response(200, json=sample_paper_response)
        )

        async with S2Client(use_cache=False) as client:
            paper = await client.get_paper("DOI:10.1214/17-AOS1609")

        assert paper.paper_id == "649def34f8be52c8b66281af98ae884c09aef38b"
        assert paper.doi == "10.1214/17-AOS1609"

    @respx.mock
    async def test_rate_limit_error(self):
        """429 response should raise S2RateLimitError."""
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(429, headers={"Retry-After": "60"})
        )

        async with S2Client(use_cache=False) as client:
            with pytest.raises(S2RateLimitError) as exc_info:
                await client.search_papers("test")

        assert exc_info.value.retry_after == 60.0

    @respx.mock
    async def test_not_found_error(self):
        """404 response should raise S2NotFoundError."""
        respx.get("https://api.semanticscholar.org/graph/v1/paper/invalid").mock(
            return_value=Response(404)
        )

        async with S2Client(use_cache=False) as client:
            with pytest.raises(S2NotFoundError):
                await client.get_paper("invalid")


# -----------------------------------------------------------------------------
# Search Filters Tests
# -----------------------------------------------------------------------------


class TestSearchFilters:
    """Tests for SearchFilters."""

    def test_year_range_params(self):
        """Year range should convert to S2 format."""
        filters = SearchFilters(year_from=2020, year_to=2024)
        params = filters.to_s2_params()
        assert params["year"] == "2020-2024"

    def test_year_from_only(self):
        """Year from only should produce open range."""
        filters = SearchFilters(year_from=2020)
        params = filters.to_s2_params()
        assert params["year"] == "2020-"

    def test_filter_by_citations(self, sample_paper_response: dict):
        """Filter should exclude low-citation papers."""
        paper = S2Paper(**sample_paper_response)

        # Paper has 1542 citations
        high_filter = SearchFilters(min_citations=2000)
        assert high_filter.filter_results([paper]) == []

        low_filter = SearchFilters(min_citations=1000)
        assert len(low_filter.filter_results([paper])) == 1

    def test_filter_excludes_paper_ids(self, sample_paper_response: dict):
        """Filter should exclude specified paper IDs."""
        paper = S2Paper(**sample_paper_response)

        filters = SearchFilters(exclude_paper_ids={paper.paper_id})
        assert filters.filter_results([paper]) == []

        filters2 = SearchFilters(exclude_paper_ids={"other_id"})
        assert len(filters2.filter_results([paper])) == 1


# -----------------------------------------------------------------------------
# Topic Discovery Tests
# -----------------------------------------------------------------------------


class TestTopicDiscovery:
    """Tests for TopicDiscovery."""

    @respx.mock
    async def test_discover_deduplicates(self, sample_paper_response: dict):
        """Discovery should deduplicate papers across topics."""
        # Return same paper for both queries
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(
                200,
                json={
                    "total": 100,
                    "offset": 0,
                    "data": [sample_paper_response],
                },
            )
        )

        async with S2Client(use_cache=False) as client:
            discovery = TopicDiscovery(client)
            result = await discovery.discover(
                topics=["query1", "query2"],
                limit_per_topic=10,
            )

        # Only one unique paper despite two queries returning same paper
        assert len(result.papers) == 1
        assert result.duplicates_removed >= 1

    def test_discovery_topic_enum_values(self):
        """All discovery topics should have string values."""
        for topic in DiscoveryTopic:
            assert isinstance(topic.value, str)
            assert len(topic.value) > 5
