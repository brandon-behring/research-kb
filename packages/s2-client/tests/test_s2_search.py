"""Tests for S2 search and topic discovery module.

Tests search filtering, topic discovery, and result aggregation.
Uses respx to mock HTTP responses from Semantic Scholar API.
"""

from datetime import datetime, timezone

import pytest
import respx
from httpx import Response

from s2_client import S2Client
from s2_client.models import S2Author, S2Paper
from s2_client.search import (
    DiscoveryResult,
    DiscoveryTopic,
    SearchFilters,
    TopicDiscovery,
)

pytestmark = pytest.mark.unit


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_paper_1() -> S2Paper:
    """Sample paper with high citations."""
    return S2Paper(
        paperId="paper1",
        corpusId=1001,
        title="Double Machine Learning for Treatment Effects",
        year=2018,
        venue="Econometrica",
        authors=[
            S2Author(authorId="a1", name="Victor Chernozhukov"),
            S2Author(authorId="a2", name="Denis Chetverikov"),
        ],
        citationCount=2500,
        influentialCitationCount=300,
        isOpenAccess=True,
        s2FieldsOfStudy=[
            {"category": "Economics", "source": "s2-fos-model"},
            {"category": "Computer Science", "source": "s2-fos-model"},
        ],
    )


@pytest.fixture
def sample_paper_2() -> S2Paper:
    """Sample paper with lower citations."""
    return S2Paper(
        paperId="paper2",
        corpusId=1002,
        title="Causal Forests",
        year=2019,
        venue="Statistical Science",
        authors=[
            S2Author(authorId="a3", name="Susan Athey"),
            S2Author(authorId="a4", name="Stefan Wager"),
        ],
        citationCount=500,
        influentialCitationCount=50,
        isOpenAccess=False,
        s2FieldsOfStudy=[
            {"category": "Computer Science", "source": "s2-fos-model"},
        ],
    )


@pytest.fixture
def sample_paper_3() -> S2Paper:
    """Sample paper without citations."""
    return S2Paper(
        paperId="paper3",
        corpusId=1003,
        title="New Causal Methods",
        year=2024,
        venue="ArXiv",
        authors=[
            S2Author(authorId="a5", name="John Doe"),
        ],
        citationCount=0,
        influentialCitationCount=0,
        isOpenAccess=True,
        s2FieldsOfStudy=[
            {"category": "Mathematics", "source": "s2-fos-model"},
        ],
    )


@pytest.fixture
def sample_search_response(sample_paper_1, sample_paper_2) -> dict:
    """Sample search API response with multiple papers."""
    return {
        "total": 1542,
        "offset": 0,
        "next": 10,
        "data": [
            {
                "paperId": sample_paper_1.paper_id,
                "corpusId": sample_paper_1.corpus_id,
                "title": sample_paper_1.title,
                "year": sample_paper_1.year,
                "venue": sample_paper_1.venue,
                "authors": [
                    {"authorId": a.author_id, "name": a.name} for a in sample_paper_1.authors
                ],
                "citationCount": sample_paper_1.citation_count,
                "influentialCitationCount": sample_paper_1.influential_citation_count,
                "isOpenAccess": sample_paper_1.is_open_access,
                "s2FieldsOfStudy": sample_paper_1.s2_fields_of_study,
            },
            {
                "paperId": sample_paper_2.paper_id,
                "corpusId": sample_paper_2.corpus_id,
                "title": sample_paper_2.title,
                "year": sample_paper_2.year,
                "venue": sample_paper_2.venue,
                "authors": [
                    {"authorId": a.author_id, "name": a.name} for a in sample_paper_2.authors
                ],
                "citationCount": sample_paper_2.citation_count,
                "influentialCitationCount": sample_paper_2.influential_citation_count,
                "isOpenAccess": sample_paper_2.is_open_access,
                "s2FieldsOfStudy": sample_paper_2.s2_fields_of_study,
            },
        ],
    }


# -----------------------------------------------------------------------------
# DiscoveryTopic Tests
# -----------------------------------------------------------------------------


class TestDiscoveryTopic:
    """Tests for DiscoveryTopic enum."""

    def test_all_topics_are_strings(self):
        """All discovery topics should have string values."""
        for topic in DiscoveryTopic:
            assert isinstance(topic.value, str)
            assert len(topic.value) > 5

    def test_causal_inference_topics(self):
        """Causal inference topics should exist."""
        assert DiscoveryTopic.DOUBLE_ML.value == "double machine learning causal inference"
        assert DiscoveryTopic.CAUSAL_FOREST.value == "causal forest heterogeneous treatment effect"
        assert DiscoveryTopic.IV_METHODS.value == "instrumental variables causal identification"
        assert DiscoveryTopic.DIFF_IN_DIFF.value == "difference-in-differences causal"
        assert DiscoveryTopic.SYNTHETIC_CONTROL.value == "synthetic control econometrics"

    def test_rag_topics(self):
        """RAG and LLM topics should exist."""
        assert "retrieval augmented generation" in DiscoveryTopic.RAG.value
        assert "context engineering" in DiscoveryTopic.CONTEXT_ENGINEERING.value
        assert "tool use" in DiscoveryTopic.TOOL_USE_LLM.value
        assert "agentic" in DiscoveryTopic.AGENTIC_AI.value.lower()

    def test_world_model_topics(self):
        """World model topics should exist."""
        assert "world model" in DiscoveryTopic.WORLD_MODEL.value
        assert "model-based" in DiscoveryTopic.MODEL_BASED_RL.value
        assert "dreamer" in DiscoveryTopic.DREAMER.value

    def test_long_context_topics(self):
        """Long context topics should exist."""
        assert "long context" in DiscoveryTopic.LONG_CONTEXT.value
        assert "state space" in DiscoveryTopic.STATE_SPACE.value

    def test_topic_enum_membership(self):
        """Topics should be enum members."""
        assert DiscoveryTopic.DOUBLE_ML in DiscoveryTopic
        assert "double machine learning" not in DiscoveryTopic  # string is not a member


# -----------------------------------------------------------------------------
# SearchFilters Tests
# -----------------------------------------------------------------------------


class TestSearchFiltersInit:
    """Tests for SearchFilters initialization."""

    def test_default_values(self):
        """Test default filter values."""
        filters = SearchFilters()

        assert filters.year_from is None
        assert filters.year_to is None
        assert filters.min_citations is None
        assert filters.min_influential_citations is None
        assert filters.open_access_only is False
        assert filters.fields_of_study is None
        assert filters.exclude_paper_ids == set()

    def test_custom_values(self):
        """Test custom filter values."""
        filters = SearchFilters(
            year_from=2020,
            year_to=2024,
            min_citations=100,
            min_influential_citations=10,
            open_access_only=True,
            fields_of_study=["Computer Science", "Economics"],
            exclude_paper_ids={"paper1", "paper2"},
        )

        assert filters.year_from == 2020
        assert filters.year_to == 2024
        assert filters.min_citations == 100
        assert filters.min_influential_citations == 10
        assert filters.open_access_only is True
        assert filters.fields_of_study == ["Computer Science", "Economics"]
        assert filters.exclude_paper_ids == {"paper1", "paper2"}


class TestSearchFiltersToS2Params:
    """Tests for converting filters to S2 API parameters."""

    def test_year_range_both(self):
        """Year range with both bounds."""
        filters = SearchFilters(year_from=2020, year_to=2024)
        params = filters.to_s2_params()

        assert params["year"] == "2020-2024"

    def test_year_from_only(self):
        """Year range with only from."""
        filters = SearchFilters(year_from=2020)
        params = filters.to_s2_params()

        assert params["year"] == "2020-"

    def test_year_to_only(self):
        """Year range with only to."""
        filters = SearchFilters(year_to=2024)
        params = filters.to_s2_params()

        assert params["year"] == "-2024"

    def test_no_year_filters(self):
        """No year filters should not include year param."""
        filters = SearchFilters()
        params = filters.to_s2_params()

        assert "year" not in params

    def test_empty_filters_return_empty_dict(self):
        """Empty filters should return empty dict."""
        filters = SearchFilters()
        params = filters.to_s2_params()

        assert params == {}


class TestSearchFiltersFilterResults:
    """Tests for post-query result filtering."""

    def test_filter_by_min_citations(self, sample_paper_1, sample_paper_2, sample_paper_3):
        """Filter should exclude papers below min citations."""
        papers = [sample_paper_1, sample_paper_2, sample_paper_3]

        # High threshold - only paper1
        high_filter = SearchFilters(min_citations=1000)
        result = high_filter.filter_results(papers)
        assert len(result) == 1
        assert result[0].paper_id == "paper1"

        # Medium threshold - paper1 and paper2
        med_filter = SearchFilters(min_citations=100)
        result = med_filter.filter_results(papers)
        assert len(result) == 2

        # Low threshold - all papers
        low_filter = SearchFilters(min_citations=0)
        result = low_filter.filter_results(papers)
        assert len(result) == 3

    def test_filter_by_influential_citations(self, sample_paper_1, sample_paper_2, sample_paper_3):
        """Filter should exclude papers below min influential citations."""
        papers = [sample_paper_1, sample_paper_2, sample_paper_3]

        filters = SearchFilters(min_influential_citations=100)
        result = filters.filter_results(papers)

        # Only paper1 has >= 100 influential citations
        assert len(result) == 1
        assert result[0].paper_id == "paper1"

    def test_filter_open_access_only(self, sample_paper_1, sample_paper_2, sample_paper_3):
        """Filter should exclude non-open-access papers."""
        papers = [sample_paper_1, sample_paper_2, sample_paper_3]

        filters = SearchFilters(open_access_only=True)
        result = filters.filter_results(papers)

        # paper2 is not open access
        assert len(result) == 2
        paper_ids = {p.paper_id for p in result}
        assert "paper1" in paper_ids
        assert "paper3" in paper_ids
        assert "paper2" not in paper_ids

    def test_filter_by_fields_of_study(self, sample_paper_1, sample_paper_2, sample_paper_3):
        """Filter should include papers matching specified fields."""
        papers = [sample_paper_1, sample_paper_2, sample_paper_3]

        # Economics - only paper1
        econ_filter = SearchFilters(fields_of_study=["Economics"])
        result = econ_filter.filter_results(papers)
        assert len(result) == 1
        assert result[0].paper_id == "paper1"

        # Computer Science - paper1 and paper2
        cs_filter = SearchFilters(fields_of_study=["Computer Science"])
        result = cs_filter.filter_results(papers)
        assert len(result) == 2

        # Mathematics - only paper3
        math_filter = SearchFilters(fields_of_study=["Mathematics"])
        result = math_filter.filter_results(papers)
        assert len(result) == 1
        assert result[0].paper_id == "paper3"

    def test_filter_excludes_paper_ids(self, sample_paper_1, sample_paper_2):
        """Filter should exclude specified paper IDs."""
        papers = [sample_paper_1, sample_paper_2]

        filters = SearchFilters(exclude_paper_ids={sample_paper_1.paper_id})
        result = filters.filter_results(papers)

        assert len(result) == 1
        assert result[0].paper_id == "paper2"

    def test_filter_handles_none_paper_id(self):
        """Filter should handle papers without paper_id."""
        paper = S2Paper(title="No ID Paper")
        filters = SearchFilters(exclude_paper_ids={"other_id"})

        result = filters.filter_results([paper])
        # Paper without ID should pass through
        assert len(result) == 1

    def test_multiple_filters_combined(self, sample_paper_1, sample_paper_2, sample_paper_3):
        """Multiple filters should be applied together."""
        papers = [sample_paper_1, sample_paper_2, sample_paper_3]

        filters = SearchFilters(
            min_citations=100,
            open_access_only=True,
            fields_of_study=["Computer Science", "Economics"],
        )
        result = filters.filter_results(papers)

        # paper1: 2500 citations, open access, has CS and Econ -> PASS
        # paper2: 500 citations, NOT open access -> FAIL
        # paper3: 0 citations -> FAIL
        assert len(result) == 1
        assert result[0].paper_id == "paper1"

    def test_filter_empty_list(self):
        """Filter should handle empty list."""
        filters = SearchFilters(min_citations=100)
        result = filters.filter_results([])
        assert result == []

    def test_filter_handles_none_values(self):
        """Filter should handle papers with None citation counts."""
        paper = S2Paper(paperId="test", title="Test", citationCount=None)

        # None is treated as 0
        filters = SearchFilters(min_citations=1)
        result = filters.filter_results([paper])
        assert len(result) == 0

        filters_zero = SearchFilters(min_citations=0)
        result = filters_zero.filter_results([paper])
        # 0 < 0 is False, so None (treated as 0) should fail
        # Actually (paper.citation_count or 0) gives 0, and 0 < 0 is False
        assert len(result) == 1


# -----------------------------------------------------------------------------
# DiscoveryResult Tests
# -----------------------------------------------------------------------------


class TestDiscoveryResult:
    """Tests for DiscoveryResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = DiscoveryResult()

        assert result.papers == []
        assert result.queries_run == []
        assert result.total_found == 0
        assert result.total_after_filters == 0
        assert result.duplicates_removed == 0
        assert isinstance(result.discovery_time, datetime)

    def test_discovery_time_is_utc(self):
        """Discovery time should be in UTC."""
        result = DiscoveryResult()
        assert result.discovery_time.tzinfo == timezone.utc

    def test_to_summary_dict(self, sample_paper_1, sample_paper_2):
        """Summary dict should contain expected keys."""
        result = DiscoveryResult(
            papers=[sample_paper_1, sample_paper_2],
            queries_run=["query1", "query2"],
            total_found=100,
            total_after_filters=50,
            duplicates_removed=10,
        )

        summary = result.to_summary_dict()

        assert summary["queries_run"] == 2
        assert summary["total_found"] == 100
        assert summary["total_after_filters"] == 50
        assert summary["duplicates_removed"] == 10
        assert summary["unique_papers"] == 2
        assert "discovery_time" in summary

    def test_summary_dict_iso_format(self):
        """Discovery time in summary should be ISO format."""
        result = DiscoveryResult()
        summary = result.to_summary_dict()

        # Should be valid ISO format
        parsed = datetime.fromisoformat(summary["discovery_time"])
        assert parsed.tzinfo == timezone.utc


# -----------------------------------------------------------------------------
# TopicDiscovery Tests
# -----------------------------------------------------------------------------


class TestTopicDiscoveryInit:
    """Tests for TopicDiscovery initialization."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_init_with_client(self):
        """TopicDiscovery should store client reference."""
        async with S2Client(use_cache=False) as client:
            discovery = TopicDiscovery(client)
            assert discovery.client is client

    @pytest.mark.asyncio
    @respx.mock
    async def test_init_clears_seen_ids(self):
        """TopicDiscovery should start with empty seen IDs."""
        async with S2Client(use_cache=False) as client:
            discovery = TopicDiscovery(client)
            assert discovery._seen_ids == set()


class TestTopicDiscoveryDiscover:
    """Tests for topic discovery functionality."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_discover_single_topic(self, sample_search_response):
        """Discover should search for single topic."""
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(200, json=sample_search_response)
        )

        async with S2Client(use_cache=False) as client:
            discovery = TopicDiscovery(client)
            result = await discovery.discover(
                topics=["causal inference"],
                limit_per_topic=10,
            )

        assert len(result.queries_run) == 1
        assert result.queries_run[0] == "causal inference"
        assert len(result.papers) == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_discover_enum_topic(self, sample_search_response):
        """Discover should work with DiscoveryTopic enum."""
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(200, json=sample_search_response)
        )

        async with S2Client(use_cache=False) as client:
            discovery = TopicDiscovery(client)
            result = await discovery.discover(
                topics=[DiscoveryTopic.DOUBLE_ML],
                limit_per_topic=10,
            )

        assert DiscoveryTopic.DOUBLE_ML.value in result.queries_run

    @pytest.mark.asyncio
    @respx.mock
    async def test_discover_deduplicates_papers(self, sample_search_response):
        """Discover should deduplicate papers across topics."""
        # Return same papers for both queries
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(200, json=sample_search_response)
        )

        async with S2Client(use_cache=False) as client:
            discovery = TopicDiscovery(client)
            result = await discovery.discover(
                topics=["query1", "query2"],
                limit_per_topic=10,
            )

        # Only unique papers should be in result
        assert len(result.papers) == 2  # Same 2 papers from both queries
        assert result.duplicates_removed >= 2  # Second query papers were dupes

    @pytest.mark.asyncio
    @respx.mock
    async def test_discover_applies_filters(self, sample_search_response):
        """Discover should apply search filters."""
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(200, json=sample_search_response)
        )

        async with S2Client(use_cache=False) as client:
            discovery = TopicDiscovery(client)
            filters = SearchFilters(min_citations=1000)

            result = await discovery.discover(
                topics=["causal inference"],
                filters=filters,
                limit_per_topic=10,
            )

        # Only paper1 has >= 1000 citations
        assert len(result.papers) == 1
        assert result.papers[0].paper_id == "paper1"

    @pytest.mark.asyncio
    @respx.mock
    async def test_discover_tracks_totals(self, sample_search_response):
        """Discover should track total counts."""
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(200, json=sample_search_response)
        )

        async with S2Client(use_cache=False) as client:
            discovery = TopicDiscovery(client)
            result = await discovery.discover(
                topics=["query1"],
                limit_per_topic=10,
            )

        assert result.total_found == 1542  # From sample response
        assert result.total_after_filters == 2  # 2 papers in response

    @pytest.mark.asyncio
    @respx.mock
    async def test_discover_continues_on_error(self, sample_search_response):
        """Discover should continue with other topics on error."""
        route = respx.get("https://api.semanticscholar.org/graph/v1/paper/search")
        route.side_effect = [
            Response(500, json={"error": "Server error"}),
            Response(200, json=sample_search_response),
        ]

        async with S2Client(use_cache=False) as client:
            discovery = TopicDiscovery(client)
            result = await discovery.discover(
                topics=["failing_query", "working_query"],
                limit_per_topic=10,
            )

        # Should have results from second query
        assert len(result.papers) == 2
        assert len(result.queries_run) == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_discover_mixed_topics(self, sample_search_response):
        """Discover should handle mix of enum and string topics."""
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(200, json=sample_search_response)
        )

        async with S2Client(use_cache=False) as client:
            discovery = TopicDiscovery(client)
            result = await discovery.discover(
                topics=[DiscoveryTopic.DOUBLE_ML, "custom query"],
                limit_per_topic=10,
            )

        assert len(result.queries_run) == 2
        assert DiscoveryTopic.DOUBLE_ML.value in result.queries_run
        assert "custom query" in result.queries_run


class TestTopicDiscoveryAllTopics:
    """Tests for discover_all_topics method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_discover_all_queries_all_topics(self, sample_search_response):
        """discover_all_topics should query all enum topics."""
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(200, json=sample_search_response)
        )

        async with S2Client(use_cache=False) as client:
            discovery = TopicDiscovery(client)
            result = await discovery.discover_all_topics(limit_per_topic=5)

        # Should have queried all topics
        expected_count = len(list(DiscoveryTopic))
        assert len(result.queries_run) == expected_count

    @pytest.mark.asyncio
    @respx.mock
    async def test_discover_all_applies_filters(self, sample_search_response):
        """discover_all_topics should apply filters."""
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(200, json=sample_search_response)
        )

        async with S2Client(use_cache=False) as client:
            discovery = TopicDiscovery(client)
            filters = SearchFilters(min_citations=10000)  # Very high
            result = await discovery.discover_all_topics(
                filters=filters,
                limit_per_topic=5,
            )

        # No papers should pass the filter
        assert len(result.papers) == 0


class TestTopicDiscoveryResetSeen:
    """Tests for reset_seen method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_reset_clears_seen_ids(self, sample_search_response):
        """reset_seen should clear the seen IDs set."""
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(200, json=sample_search_response)
        )

        async with S2Client(use_cache=False) as client:
            discovery = TopicDiscovery(client)

            # First discovery
            await discovery.discover(topics=["query1"], limit_per_topic=10)
            assert len(discovery._seen_ids) == 2

            # Reset
            discovery.reset_seen()
            assert len(discovery._seen_ids) == 0

    @pytest.mark.asyncio
    @respx.mock
    async def test_reset_allows_rediscovery(self, sample_search_response):
        """After reset, same papers can be discovered again."""
        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            return_value=Response(200, json=sample_search_response)
        )

        async with S2Client(use_cache=False) as client:
            discovery = TopicDiscovery(client)

            # First discovery
            result1 = await discovery.discover(topics=["query"], limit_per_topic=10)
            assert len(result1.papers) == 2

            # Second discovery without reset - duplicates filtered
            result2 = await discovery.discover(topics=["query"], limit_per_topic=10)
            assert len(result2.papers) == 0
            assert result2.duplicates_removed >= 2

            # Reset and discover again
            discovery.reset_seen()
            result3 = await discovery.discover(topics=["query"], limit_per_topic=10)
            assert len(result3.papers) == 2


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestSearchIntegration:
    """Integration tests for search functionality."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_full_discovery_workflow(self):
        """Test complete discovery workflow."""
        # Different responses for different queries
        dml_response = {
            "total": 100,
            "offset": 0,
            "data": [
                {
                    "paperId": "dml1",
                    "title": "Double ML Paper",
                    "year": 2020,
                    "citationCount": 500,
                    "isOpenAccess": True,
                    "s2FieldsOfStudy": [{"category": "Economics"}],
                }
            ],
        }

        cf_response = {
            "total": 50,
            "offset": 0,
            "data": [
                {
                    "paperId": "cf1",
                    "title": "Causal Forest Paper",
                    "year": 2021,
                    "citationCount": 200,
                    "isOpenAccess": False,
                    "s2FieldsOfStudy": [{"category": "Computer Science"}],
                }
            ],
        }

        call_count = [0]

        def response_callback(request):
            call_count[0] += 1
            if call_count[0] == 1:
                return Response(200, json=dml_response)
            return Response(200, json=cf_response)

        respx.get("https://api.semanticscholar.org/graph/v1/paper/search").mock(
            side_effect=response_callback
        )

        async with S2Client(use_cache=False) as client:
            discovery = TopicDiscovery(client)

            filters = SearchFilters(
                year_from=2019,
                open_access_only=True,
            )

            result = await discovery.discover(
                topics=[DiscoveryTopic.DOUBLE_ML, DiscoveryTopic.CAUSAL_FOREST],
                filters=filters,
                limit_per_topic=10,
            )

        # Only dml1 is open access
        assert len(result.papers) == 1
        assert result.papers[0].paper_id == "dml1"
        assert result.total_found == 150  # 100 + 50
