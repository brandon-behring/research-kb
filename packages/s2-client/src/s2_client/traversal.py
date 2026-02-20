"""Citation-based paper discovery via graph traversal.

Given seed papers from the corpus, discover related papers via:
1. Papers they cite (references) - "intellectual ancestors"
2. Papers that cite them (citations) - "intellectual descendants"
3. Both directions - comprehensive exploration

Uses BFS traversal with configurable depth and filtering.

Example:
    >>> async with S2Client() as client:
    ...     traversal = CitationTraversal(client)
    ...     result = await traversal.discover(
    ...         seed_paper_ids=["649def34..."],  # DML paper
    ...         depth=1,
    ...         direction="both",
    ...         min_citations=30,
    ...         exclude_ids=existing_s2_ids,
    ...     )
    ...     print(f"Found {len(result.papers)} related papers")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from research_kb_common import get_logger

if TYPE_CHECKING:
    from s2_client.client import S2Client
    from s2_client.models import S2Paper

logger = get_logger(__name__)


@dataclass
class TraversalResult:
    """Result from citation graph traversal.

    Attributes:
        papers: Discovered papers meeting filter criteria
        seed_papers: IDs used to start traversal
        depth: Traversal depth used
        direction: Direction traversed ("citations", "references", "both")
        total_traversed: Total papers examined during traversal
        duplicates_removed: Papers skipped as already seen
        filtered_out: Papers that failed filter criteria
    """

    papers: list["S2Paper"] = field(default_factory=list)
    seed_papers: list[str] = field(default_factory=list)
    depth: int = 1
    direction: str = "both"
    total_traversed: int = 0
    duplicates_removed: int = 0
    filtered_out: int = 0


class CitationTraversal:
    """BFS traversal of citation graph.

    Discovers papers by following citation links from seed papers.
    Supports filtering by citation count, open access status, and
    exclusion of papers already in the corpus.

    Example:
        >>> async with S2Client() as client:
        ...     traversal = CitationTraversal(client)
        ...     result = await traversal.discover(
        ...         seed_paper_ids=["649def34f8be52c8b66281af98ae884c09aef38b"],
        ...         depth=1,
        ...         direction="both",
        ...         min_citations=30,
        ...         exclude_ids=existing_corpus_s2_ids,
        ...     )
        ...     print(f"Found {len(result.papers)} related papers")
        ...     print(f"Traversed {result.total_traversed}, filtered {result.filtered_out}")
    """

    def __init__(self, client: "S2Client") -> None:
        """Initialize traversal with S2 client.

        Args:
            client: Initialized S2Client instance (use within async context)
        """
        self.client = client
        self._seen_ids: set[str] = set()

    async def discover(
        self,
        seed_paper_ids: list[str],
        depth: int = 1,
        direction: str = "both",
        min_citations: int = 20,
        open_access_only: bool = True,
        exclude_ids: set[str] | None = None,
        limit_per_paper: int = 50,
    ) -> TraversalResult:
        """Discover papers via BFS citation traversal.

        Performs breadth-first search through the citation graph, starting
        from seed papers and expanding outward by the specified depth.

        Args:
            seed_paper_ids: Starting paper IDs (from existing corpus)
            depth: Number of hops from seeds (1 = direct citations/refs only)
            direction: Traversal direction:
                - "citations": Papers that cite seeds (descendants)
                - "references": Papers seeds cite (ancestors)
                - "both": Both directions (comprehensive)
            min_citations: Minimum citation count filter (default: 20)
            open_access_only: Only include open access papers (default: True)
            exclude_ids: Paper IDs to exclude (e.g., already in corpus)
            limit_per_paper: Max papers to fetch per citation/reference call

        Returns:
            TraversalResult with discovered papers and statistics

        Note:
            Higher depth values result in exponentially more API calls.
            Depth of 1-2 is recommended for most use cases.
        """
        result = TraversalResult(
            seed_papers=list(seed_paper_ids),
            depth=depth,
            direction=direction,
        )

        # Initialize seen set with exclusions
        self._seen_ids = set(exclude_ids or set())

        # Also exclude seed papers from results (we already have them)
        for seed_id in seed_paper_ids:
            self._seen_ids.add(seed_id)

        # BFS traversal
        current_level: set[str] = set(seed_paper_ids)

        for level in range(depth):
            next_level: set[str] = set()
            level_papers_added = 0

            logger.info(
                "traversal_level_start",
                level=level + 1,
                depth=depth,
                papers_to_process=len(current_level),
            )

            for paper_id in current_level:
                # Get citations (papers that cite this one)
                if direction in ("citations", "both"):
                    try:
                        citations_result = await self.client.get_paper_citations(
                            paper_id, limit=limit_per_paper
                        )
                        for paper in citations_result.papers:
                            if self._process_paper(
                                paper,
                                result,
                                next_level,
                                min_citations,
                                open_access_only,
                            ):
                                level_papers_added += 1
                    except Exception as e:
                        logger.warning(
                            "traversal_citations_failed",
                            paper_id=paper_id,
                            error=str(e),
                        )

                # Get references (papers this one cites)
                if direction in ("references", "both"):
                    try:
                        refs_result = await self.client.get_paper_references(
                            paper_id, limit=limit_per_paper
                        )
                        for paper in refs_result.papers:
                            if self._process_paper(
                                paper,
                                result,
                                next_level,
                                min_citations,
                                open_access_only,
                            ):
                                level_papers_added += 1
                    except Exception as e:
                        logger.warning(
                            "traversal_references_failed",
                            paper_id=paper_id,
                            error=str(e),
                        )

                result.total_traversed += 1

            logger.info(
                "traversal_level_complete",
                level=level + 1,
                papers_at_level=len(next_level),
                papers_added=level_papers_added,
                total_discovered=len(result.papers),
            )

            # Move to next BFS level
            current_level = next_level

        logger.info(
            "traversal_complete",
            total_papers=len(result.papers),
            total_traversed=result.total_traversed,
            duplicates_removed=result.duplicates_removed,
            filtered_out=result.filtered_out,
        )

        return result

    def _process_paper(
        self,
        paper: "S2Paper",
        result: TraversalResult,
        next_level: set[str],
        min_citations: int,
        open_access_only: bool,
    ) -> bool:
        """Process a discovered paper.

        Args:
            paper: Paper to process
            result: TraversalResult to update
            next_level: Set of paper IDs for next BFS level
            min_citations: Minimum citation count
            open_access_only: Whether to require open access

        Returns:
            True if paper was added to results, False otherwise
        """
        if not paper.paper_id:
            return False

        # Check if already seen
        if paper.paper_id in self._seen_ids:
            result.duplicates_removed += 1
            return False

        # Mark as seen regardless of filter outcome
        self._seen_ids.add(paper.paper_id)

        # Apply citation count filter
        citation_count = paper.citation_count or 0
        if citation_count < min_citations:
            result.filtered_out += 1
            return False

        # Apply open access filter
        if open_access_only and not paper.is_open_access:
            result.filtered_out += 1
            return False

        # Paper passes all filters
        result.papers.append(paper)
        next_level.add(paper.paper_id)
        return True
