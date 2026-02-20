"""Async Semantic Scholar API client.

Main client class that combines:
- Rate limiting
- Response caching
- Retry on transient errors
- Pydantic model parsing

Base URL: https://api.semanticscholar.org
Docs: https://api.semanticscholar.org/api-docs/graph
"""

import asyncio
import os
from typing import Any

import httpx
from research_kb_common import get_logger, retry_on_exception

from s2_client.cache import S2Cache
from s2_client.errors import S2APIError, S2NotFoundError, S2RateLimitError
from s2_client.models import (
    S2Author,
    S2AuthorPapersResult,
    S2CitationsResult,
    S2Paper,
    S2ReferencesResult,
    S2SearchResult,
)
from s2_client.rate_limiter import RateLimiter

logger = get_logger(__name__)

# API configuration from environment
S2_BASE_URL = os.environ.get("S2_BASE_URL", "https://api.semanticscholar.org")
S2_API_KEY = os.environ.get("S2_API_KEY")
S2_RPS_LIMIT = float(os.environ.get("S2_RPS_LIMIT", "10"))
S2_TIMEOUT_SECONDS = int(os.environ.get("S2_TIMEOUT_SECONDS", "30"))

# Default fields to request (balance between completeness and response size)
DEFAULT_PAPER_FIELDS = ",".join(
    [
        "paperId",
        "corpusId",
        "externalIds",
        "title",
        "abstract",
        "venue",
        "year",
        "publicationDate",
        "authors",
        "referenceCount",
        "citationCount",
        "influentialCitationCount",
        "isOpenAccess",
        "openAccessPdf",
        "s2FieldsOfStudy",
        "publicationTypes",
    ]
)

DEFAULT_AUTHOR_FIELDS = ",".join(
    [
        "authorId",
        "externalIds",
        "name",
        "url",
        "affiliations",
        "paperCount",
        "citationCount",
        "hIndex",
    ]
)


class S2Client:
    """Async Semantic Scholar API client.

    Provides methods for:
    - Paper search and lookup
    - Author lookup and papers
    - Batch operations
    - Recommendations

    Example:
        >>> async with S2Client() as client:
        ...     # Search for papers
        ...     result = await client.search_papers("causal forest", limit=10)
        ...     for paper in result.data:
        ...         print(f"{paper.title} ({paper.year})")
        ...
        ...     # Get specific paper by DOI
        ...     paper = await client.get_paper("DOI:10.1214/17-AOS1609")
        ...     print(paper.citation_count)

    Attributes:
        base_url: API base URL (default: https://api.semanticscholar.org)
        api_key: Optional API key for higher rate limits
        requests_per_second: Rate limit (default: 10)
        use_cache: Whether to use response cache (default: True)
    """

    def __init__(
        self,
        base_url: str = S2_BASE_URL,
        api_key: str | None = S2_API_KEY,
        requests_per_second: float = S2_RPS_LIMIT,
        use_cache: bool = True,
        timeout_seconds: int = S2_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize client.

        Args:
            base_url: API base URL
            api_key: Optional API key
            requests_per_second: Rate limit
            use_cache: Enable response caching
            timeout_seconds: Request timeout
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.use_cache = use_cache
        self.timeout_seconds = timeout_seconds

        # Initialize components
        self._rate_limiter = RateLimiter(requests_per_second=requests_per_second)
        self._cache = S2Cache() if use_cache else None
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "S2Client":
        """Async context manager entry."""
        headers = {"User-Agent": "s2-client/1.0.0 (research-kb)"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(self.timeout_seconds),
        )

        if self._cache:
            await self._cache.initialize()

        logger.info(
            "S2Client initialized",
            base_url=self.base_url,
            has_api_key=bool(self.api_key),
            cache_enabled=self.use_cache,
        )

        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
        if self._cache:
            await self._cache.close()

    # -------------------------------------------------------------------------
    # Paper Methods
    # -------------------------------------------------------------------------

    async def search_papers(
        self,
        query: str,
        limit: int = 100,
        offset: int = 0,
        year: str | None = None,
        fields_of_study: list[str] | None = None,
        open_access_only: bool = False,
        min_citation_count: int | None = None,
        fields: str = DEFAULT_PAPER_FIELDS,
    ) -> S2SearchResult:
        """Search for papers by query.

        Args:
            query: Search query (e.g., "causal forest heterogeneous treatment")
            limit: Maximum results to return (max 100)
            offset: Pagination offset
            year: Year filter (e.g., "2020-", "2018-2022", "2020")
            fields_of_study: Filter by fields (e.g., ["Computer Science", "Economics"])
            open_access_only: Only return open access papers
            min_citation_count: Minimum citation count filter
            fields: Comma-separated list of fields to return

        Returns:
            S2SearchResult with papers and pagination info

        Example:
            >>> result = await client.search_papers(
            ...     "double machine learning",
            ...     year="2018-",
            ...     min_citation_count=50,
            ...     limit=20
            ... )
        """
        endpoint = "/graph/v1/paper/search"
        params: dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),  # API max is 100
            "offset": offset,
            "fields": fields,
        }

        if year:
            params["year"] = year
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        if open_access_only:
            params["openAccessPdf"] = ""
        if min_citation_count is not None:
            params["minCitationCount"] = min_citation_count

        response = await self._request("GET", endpoint, params=params)
        return S2SearchResult(**response)

    async def get_paper(
        self,
        paper_id: str,
        fields: str = DEFAULT_PAPER_FIELDS,
    ) -> S2Paper:
        """Get paper by ID.

        Args:
            paper_id: S2 paper ID, DOI, arXiv ID, etc.
                - S2 ID: "649def34f8be52c8b66281af98ae884c09aef38b"
                - DOI: "DOI:10.1214/17-AOS1609"
                - arXiv: "arXiv:1608.00060"
                - PMID: "PMID:19872477"
            fields: Fields to return

        Returns:
            S2Paper with requested fields

        Raises:
            S2NotFoundError: If paper not found
        """
        endpoint = f"/graph/v1/paper/{paper_id}"
        params = {"fields": fields}

        response = await self._request("GET", endpoint, params=params)
        return S2Paper(**response)

    async def get_papers_batch(
        self,
        paper_ids: list[str],
        fields: str = DEFAULT_PAPER_FIELDS,
    ) -> list[S2Paper]:
        """Get multiple papers by ID (batch lookup).

        More efficient than individual lookups. Max 500 IDs per request.

        Args:
            paper_ids: List of paper IDs (S2, DOI, arXiv, etc.)
            fields: Fields to return

        Returns:
            List of S2Paper objects (maintains order, None for not found)
        """
        endpoint = "/graph/v1/paper/batch"
        params = {"fields": fields}

        # API max is 500 per batch
        results: list[S2Paper] = []
        for i in range(0, len(paper_ids), 500):
            batch = paper_ids[i : i + 500]
            response = await self._request(
                "POST",
                endpoint,
                params=params,
                json={"ids": batch},
            )

            for paper_data in response:
                if paper_data:
                    results.append(S2Paper(**paper_data))

        return results

    async def get_paper_citations(
        self,
        paper_id: str,
        limit: int = 100,
        offset: int = 0,
        fields: str = DEFAULT_PAPER_FIELDS,
    ) -> S2CitationsResult:
        """Get papers that cite this paper.

        Args:
            paper_id: S2 paper ID or identifier (DOI:..., arXiv:...)
            limit: Max citations to return per request (max 1000)
            offset: Pagination offset
            fields: Fields to include for each citing paper

        Returns:
            S2CitationsResult with citing papers accessible via .papers property

        Example:
            >>> result = await client.get_paper_citations(
            ...     "649def34f8be52c8b66281af98ae884c09aef38b",
            ...     limit=50
            ... )
            >>> for paper in result.papers:
            ...     print(f"Cited by: {paper.title}")
        """
        endpoint = f"/graph/v1/paper/{paper_id}/citations"
        params = {
            "limit": min(limit, 1000),
            "offset": offset,
            "fields": fields,
        }

        response = await self._request("GET", endpoint, params=params)
        return S2CitationsResult(**response)

    async def get_paper_references(
        self,
        paper_id: str,
        limit: int = 100,
        offset: int = 0,
        fields: str = DEFAULT_PAPER_FIELDS,
    ) -> S2ReferencesResult:
        """Get papers that this paper cites (references).

        Args:
            paper_id: S2 paper ID or identifier
            limit: Max references to return per request (max 1000)
            offset: Pagination offset
            fields: Fields to include for each referenced paper

        Returns:
            S2ReferencesResult with referenced papers accessible via .papers property

        Example:
            >>> result = await client.get_paper_references(
            ...     "649def34f8be52c8b66281af98ae884c09aef38b",
            ...     limit=50
            ... )
            >>> for paper in result.papers:
            ...     print(f"Cites: {paper.title}")
        """
        endpoint = f"/graph/v1/paper/{paper_id}/references"
        params = {
            "limit": min(limit, 1000),
            "offset": offset,
            "fields": fields,
        }

        response = await self._request("GET", endpoint, params=params)
        return S2ReferencesResult(**response)

    # -------------------------------------------------------------------------
    # Author Methods
    # -------------------------------------------------------------------------

    async def get_author(
        self,
        author_id: str,
        fields: str = DEFAULT_AUTHOR_FIELDS,
    ) -> S2Author:
        """Get author by ID.

        Args:
            author_id: S2 author ID (e.g., "26331346")
            fields: Fields to return

        Returns:
            S2Author with requested fields
        """
        endpoint = f"/graph/v1/author/{author_id}"
        params = {"fields": fields}

        response = await self._request("GET", endpoint, params=params)
        return S2Author(**response)

    async def get_author_papers(
        self,
        author_id: str,
        limit: int = 100,
        offset: int = 0,
        fields: str = DEFAULT_PAPER_FIELDS,
    ) -> S2AuthorPapersResult:
        """Get papers by author.

        Args:
            author_id: S2 author ID
            limit: Maximum papers to return
            offset: Pagination offset
            fields: Paper fields to return

        Returns:
            S2AuthorPapersResult with papers and pagination
        """
        endpoint = f"/graph/v1/author/{author_id}/papers"
        params = {
            "limit": min(limit, 1000),
            "offset": offset,
            "fields": fields,
        }

        response = await self._request("GET", endpoint, params=params)
        return S2AuthorPapersResult(**response)

    # -------------------------------------------------------------------------
    # Recommendations
    # -------------------------------------------------------------------------

    async def get_recommendations(
        self,
        positive_paper_ids: list[str],
        negative_paper_ids: list[str] | None = None,
        limit: int = 10,
        fields: str = DEFAULT_PAPER_FIELDS,
    ) -> list[S2Paper]:
        """Get paper recommendations based on positive/negative examples.

        Uses the Recommendations API to find similar papers.

        Args:
            positive_paper_ids: Papers to find similar papers for
            negative_paper_ids: Papers to avoid (optional)
            limit: Maximum recommendations (max 500)
            fields: Fields to return

        Returns:
            List of recommended S2Paper objects
        """
        endpoint = "/recommendations/v1/papers"
        params = {"limit": min(limit, 500), "fields": fields}

        body: dict[str, Any] = {"positivePaperIds": positive_paper_ids}
        if negative_paper_ids:
            body["negativePaperIds"] = negative_paper_ids

        response = await self._request("POST", endpoint, params=params, json=body)
        return [S2Paper(**p) for p in response.get("recommendedPapers", [])]

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    @retry_on_exception(
        (httpx.TimeoutException, httpx.NetworkError),
        max_attempts=3,
        min_wait_seconds=1.0,
        max_wait_seconds=10.0,
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make API request with rate limiting, caching, and 429 retry.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            json: JSON body (for POST)

        Returns:
            JSON response

        Raises:
            S2APIError: On API errors
            S2RateLimitError: On rate limit (429) after max retries
            S2NotFoundError: On 404
        """
        if not self._client:
            raise S2APIError(0, "Client not initialized. Use async context manager.", endpoint)

        # Check cache for GET requests
        cache_key_params = {**(params or {}), **(json or {})} if method == "GET" else None
        if self._cache and method == "GET":
            cached = await self._cache.get(endpoint, cache_key_params)
            if cached:
                logger.debug("Cache hit", endpoint=endpoint)
                return cached

        # 429 retry configuration
        max_429_retries = 3
        base_wait = 5.0  # Base wait time for exponential backoff

        for attempt in range(max_429_retries + 1):
            # Rate limit
            await self._rate_limiter.acquire()

            # Make request
            logger.debug("API request", method=method, endpoint=endpoint)

            if method == "GET":
                response = await self._client.get(endpoint, params=params)
            elif method == "POST":
                response = await self._client.post(endpoint, params=params, json=json)
            else:
                raise ValueError(f"Unsupported method: {method}")

            # Handle 429 with retry
            if response.status_code == 429:
                if attempt < max_429_retries:
                    retry_after = response.headers.get("Retry-After")
                    wait_time = float(retry_after) if retry_after else base_wait * (2**attempt)
                    logger.warning(
                        "rate_limited_retrying",
                        endpoint=endpoint,
                        attempt=attempt + 1,
                        wait_seconds=wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Max retries exceeded
                    raise S2RateLimitError(
                        retry_after=(
                            float(response.headers.get("Retry-After"))
                            if response.headers.get("Retry-After")
                            else None
                        ),
                        endpoint=endpoint,
                    )

            # Handle other errors (no retry)
            if response.status_code == 404:
                raise S2NotFoundError(endpoint.split("/")[-1])

            if response.status_code >= 400:
                raise S2APIError(
                    response.status_code,
                    response.text[:500],
                    endpoint,
                )

            # Success - parse and cache
            data = response.json()

            if self._cache and method == "GET":
                await self._cache.set(endpoint, cache_key_params, data)

            return data

        # Should not reach here, but safety fallback
        raise S2RateLimitError(retry_after=None, endpoint=endpoint)

    async def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if self._cache:
            return await self._cache.stats()
        return {"cache_enabled": False}
