"""HTTP client for Research-KB API.

Provides async access to the research-kb API endpoints,
replacing direct asyncpg connections with centralized API calls.

This enables:
- Centralized caching, metrics, and error handling from the API layer
- Decoupled dashboard from database implementation details
- Consistent API contract across all clients
"""

import os
from typing import Any, Optional

import httpx


class ResearchKBClient:
    """Async HTTP client for Research-KB API.

    Example:
        >>> client = ResearchKBClient()
        >>> try:
        ...     results = await client.search("instrumental variables", limit=20)
        ...     for result in results.get("results", []):
        ...         print(result["source"]["title"])
        ... finally:
        ...     await client.close()
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize the client.

        Args:
            base_url: API base URL. Defaults to RESEARCH_KB_API_URL env var
                      or http://localhost:8000
            timeout: Request timeout in seconds
        """
        _default_url = os.getenv("RESEARCH_KB_API_URL") or "http://localhost:8000"
        self.base_url: str = base_url or _default_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    # -------------------------------------------------------------------------
    # Health & Stats
    # -------------------------------------------------------------------------

    async def get_stats(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dict with sources, chunks, concepts, relationships, citations counts.

        Raises:
            httpx.HTTPStatusError: On API errors
        """
        client = await self._get_client()
        response = await client.get("/stats")
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> dict[str, Any]:
        """Check API health status.

        Returns:
            Dict with status, version, database and embedding_model status.
        """
        client = await self._get_client()
        response = await client.get("/health")
        response.raise_for_status()
        return response.json()

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 20,
        context_type: str = "balanced",
        source_filter: Optional[str] = None,
        use_graph: bool = True,
        graph_weight: float = 0.2,
        use_rerank: bool = True,
        use_expand: bool = True,
    ) -> dict[str, Any]:
        """Execute hybrid search.

        Args:
            query: Search query text
            limit: Maximum results (1-100)
            context_type: "building", "auditing", or "balanced"
            source_filter: Optional filter by source type (PAPER, TEXTBOOK)
            use_graph: Enable graph-boosted search
            graph_weight: Weight for graph signal (0-1)
            use_rerank: Enable cross-encoder reranking
            use_expand: Enable query expansion

        Returns:
            Dict with query, expanded_query, results list, and metadata.

        Raises:
            httpx.HTTPStatusError: On API errors
        """
        client = await self._get_client()
        payload = {
            "query": query,
            "limit": limit,
            "context_type": context_type,
            "use_graph": use_graph,
            "graph_weight": graph_weight,
            "use_rerank": use_rerank,
            "use_expand": use_expand,
        }
        if source_filter:
            payload["source_filter"] = source_filter

        response = await client.post("/search", json=payload)
        response.raise_for_status()
        return response.json()

    # -------------------------------------------------------------------------
    # Sources
    # -------------------------------------------------------------------------

    async def list_sources(
        self,
        limit: int = 100,
        offset: int = 0,
        source_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """List sources with optional filtering.

        Args:
            limit: Maximum results (1-1000)
            offset: Pagination offset
            source_type: Optional filter (PAPER, TEXTBOOK)

        Returns:
            Dict with sources list, total count, limit, offset.

        Raises:
            httpx.HTTPStatusError: On API errors
        """
        client = await self._get_client()
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if source_type:
            params["source_type"] = source_type

        response = await client.get("/sources", params=params)
        response.raise_for_status()
        return response.json()

    async def get_source(self, source_id: str) -> dict[str, Any]:
        """Get source details with chunks.

        Args:
            source_id: UUID of the source

        Returns:
            Dict with source details and chunks list.

        Raises:
            httpx.HTTPStatusError: On API errors (including 404)
        """
        client = await self._get_client()
        response = await client.get(f"/sources/{source_id}")
        response.raise_for_status()
        return response.json()

    async def get_source_citations(self, source_id: str) -> dict[str, Any]:
        """Get citation information for a source.

        Args:
            source_id: UUID of the source

        Returns:
            Dict with citing_sources and cited_sources lists.

        Raises:
            httpx.HTTPStatusError: On API errors
        """
        client = await self._get_client()
        response = await client.get(f"/sources/{source_id}/citations")
        response.raise_for_status()
        return response.json()

    # -------------------------------------------------------------------------
    # Concepts & Graph
    # -------------------------------------------------------------------------

    async def list_concepts(
        self,
        query: Optional[str] = None,
        limit: int = 100,
        concept_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """List or search concepts.

        Args:
            query: Optional search query
            limit: Maximum results
            concept_type: Optional filter by type

        Returns:
            Dict with concepts list and total count.
        """
        client = await self._get_client()
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["query"] = query
        if concept_type:
            params["concept_type"] = concept_type

        response = await client.get("/concepts", params=params)
        response.raise_for_status()
        return response.json()

    async def get_graph_neighborhood(
        self,
        concept_name: str,
        hops: int = 2,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get graph neighborhood for a concept.

        Args:
            concept_name: Name of the center concept
            hops: Maximum distance from center
            limit: Maximum nodes to return

        Returns:
            Dict with center node, nodes list, and edges list.
        """
        client = await self._get_client()
        params = {"hops": hops, "limit": limit}
        response = await client.get(f"/graph/neighborhood/{concept_name}", params=params)
        response.raise_for_status()
        return response.json()

    async def get_graph_path(
        self,
        concept_a: str,
        concept_b: str,
    ) -> dict[str, Any]:
        """Find shortest path between two concepts.

        Args:
            concept_a: Starting concept name
            concept_b: Ending concept name

        Returns:
            Dict with from_concept, to_concept, path list, and path_length.
        """
        client = await self._get_client()
        response = await client.get(f"/graph/path/{concept_a}/{concept_b}")
        response.raise_for_status()
        return response.json()


# Convenience function for single-use operations
async def get_api_client() -> ResearchKBClient:
    """Create a new API client instance.

    Remember to call client.close() when done, or use as context:

        client = await get_api_client()
        try:
            stats = await client.get_stats()
        finally:
            await client.close()
    """
    return ResearchKBClient()
