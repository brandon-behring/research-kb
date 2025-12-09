"""Topic search and discovery utilities.

Provides higher-level abstractions for paper discovery:
- Pre-configured research topics
- Filtered search with deduplication
- Discovery result aggregation
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from research_kb_common import get_logger

from s2_client.models import S2Paper

if TYPE_CHECKING:
    from s2_client.client import S2Client

logger = get_logger(__name__)


class DiscoveryTopic(str, Enum):
    """Pre-configured discovery topics for research areas."""

    # Causal Inference
    DOUBLE_ML = "double machine learning causal inference"
    CAUSAL_FOREST = "causal forest heterogeneous treatment effect"
    SYNTHETIC_CONTROL = "synthetic control econometrics"
    IV_METHODS = "instrumental variables causal identification"
    DIFF_IN_DIFF = "difference-in-differences causal"

    # RAG & LLMs
    RAG = "retrieval augmented generation"
    CONTEXT_ENGINEERING = "context engineering language model"
    TOOL_USE_LLM = "tool use large language model"
    AGENTIC_AI = "agentic AI autonomous agent"

    # World Models
    WORLD_MODEL = "world model reinforcement learning"
    MODEL_BASED_RL = "model-based reinforcement learning"
    DREAMER = "dreamer latent world model"

    # Long Context
    LONG_CONTEXT = "long context transformer attention"
    STATE_SPACE = "state space model sequence"


@dataclass
class SearchFilters:
    """Filters for paper discovery.

    Attributes:
        year_from: Minimum publication year (inclusive)
        year_to: Maximum publication year (inclusive)
        min_citations: Minimum citation count
        min_influential_citations: Minimum influential citation count
        open_access_only: Only include open access papers
        fields_of_study: Filter by research fields
        exclude_paper_ids: Paper IDs to exclude (for deduplication)
    """

    year_from: int | None = None
    year_to: int | None = None
    min_citations: int | None = None
    min_influential_citations: int | None = None
    open_access_only: bool = False
    fields_of_study: list[str] | None = None
    exclude_paper_ids: set[str] = field(default_factory=set)

    def to_s2_params(self) -> dict[str, str]:
        """Convert to S2 API search parameters.

        Returns:
            Dict of parameters for search_papers()
        """
        params: dict[str, str] = {}

        # Year range
        if self.year_from and self.year_to:
            params["year"] = f"{self.year_from}-{self.year_to}"
        elif self.year_from:
            params["year"] = f"{self.year_from}-"
        elif self.year_to:
            params["year"] = f"-{self.year_to}"

        return params

    def filter_results(self, papers: list[S2Paper]) -> list[S2Paper]:
        """Apply post-query filters to papers.

        Some filters (min_citations, influential citations) need post-filtering
        as S2 search API has limited filter support.

        Args:
            papers: Papers from S2 search

        Returns:
            Filtered list of papers
        """
        filtered = []

        for paper in papers:
            # Skip excluded papers
            if paper.paper_id and paper.paper_id in self.exclude_paper_ids:
                continue

            # Citation filters
            if self.min_citations is not None:
                if (paper.citation_count or 0) < self.min_citations:
                    continue

            if self.min_influential_citations is not None:
                if (paper.influential_citation_count or 0) < self.min_influential_citations:
                    continue

            # Open access filter (post-filter for precision)
            if self.open_access_only and not paper.is_open_access:
                continue

            # Fields of study filter
            if self.fields_of_study:
                paper_fields = {
                    f.get("category")
                    for f in (paper.s2_fields_of_study or [])
                    if f.get("category")
                }
                if not paper_fields.intersection(set(self.fields_of_study)):
                    continue

            filtered.append(paper)

        return filtered


@dataclass
class DiscoveryResult:
    """Result from a discovery run.

    Aggregates papers found across multiple queries with deduplication.
    """

    papers: list[S2Paper] = field(default_factory=list)
    queries_run: list[str] = field(default_factory=list)
    total_found: int = 0
    total_after_filters: int = 0
    duplicates_removed: int = 0
    discovery_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_summary_dict(self) -> dict:
        """Convert to summary dict for logging/reporting."""
        return {
            "queries_run": len(self.queries_run),
            "total_found": self.total_found,
            "total_after_filters": self.total_after_filters,
            "duplicates_removed": self.duplicates_removed,
            "unique_papers": len(self.papers),
            "discovery_time": self.discovery_time.isoformat(),
        }


class TopicDiscovery:
    """High-level paper discovery by topics.

    Combines multiple search queries, applies filters, and deduplicates results.

    Example:
        >>> async with S2Client() as client:
        ...     discovery = TopicDiscovery(client)
        ...     result = await discovery.discover(
        ...         topics=[DiscoveryTopic.DOUBLE_ML, DiscoveryTopic.CAUSAL_FOREST],
        ...         filters=SearchFilters(year_from=2020, min_citations=50),
        ...         limit_per_topic=20,
        ...     )
        ...     print(f"Found {len(result.papers)} unique papers")

    Attributes:
        client: S2Client instance
    """

    def __init__(self, client: "S2Client") -> None:
        """Initialize discovery.

        Args:
            client: Initialized S2Client
        """
        self.client = client
        self._seen_ids: set[str] = set()

    async def discover(
        self,
        topics: list[DiscoveryTopic | str],
        filters: SearchFilters | None = None,
        limit_per_topic: int = 50,
    ) -> DiscoveryResult:
        """Discover papers across multiple topics.

        Args:
            topics: List of DiscoveryTopic enum values or custom query strings
            filters: Optional filters to apply
            limit_per_topic: Maximum papers per topic query

        Returns:
            DiscoveryResult with deduplicated papers
        """
        filters = filters or SearchFilters()
        result = DiscoveryResult()

        for topic in topics:
            query = topic.value if isinstance(topic, DiscoveryTopic) else topic
            result.queries_run.append(query)

            logger.info("Discovering papers", query=query, limit=limit_per_topic)

            try:
                # Build S2 params
                s2_params = filters.to_s2_params()

                search_result = await self.client.search_papers(
                    query=query,
                    limit=limit_per_topic,
                    year=s2_params.get("year"),
                    open_access_only=filters.open_access_only,
                    min_citation_count=filters.min_citations,
                )

                result.total_found += search_result.total

                # Apply additional filters
                filtered_papers = filters.filter_results(search_result.data)
                result.total_after_filters += len(filtered_papers)

                # Deduplicate
                for paper in filtered_papers:
                    if paper.paper_id and paper.paper_id not in self._seen_ids:
                        self._seen_ids.add(paper.paper_id)
                        result.papers.append(paper)
                    else:
                        result.duplicates_removed += 1

                logger.info(
                    "Topic discovery complete",
                    query=query,
                    found=len(search_result.data),
                    after_filters=len(filtered_papers),
                    unique_added=len(filtered_papers) - result.duplicates_removed,
                )

            except Exception as e:
                logger.error("Discovery failed for topic", query=query, error=str(e))
                # Continue with other topics

        return result

    async def discover_all_topics(
        self,
        filters: SearchFilters | None = None,
        limit_per_topic: int = 20,
    ) -> DiscoveryResult:
        """Discover papers for all pre-configured topics.

        Convenience method that searches all DiscoveryTopic values.

        Args:
            filters: Optional filters
            limit_per_topic: Papers per topic

        Returns:
            DiscoveryResult with papers from all topics
        """
        return await self.discover(
            topics=list(DiscoveryTopic),
            filters=filters,
            limit_per_topic=limit_per_topic,
        )

    def reset_seen(self) -> None:
        """Reset seen paper IDs for fresh deduplication."""
        self._seen_ids.clear()
