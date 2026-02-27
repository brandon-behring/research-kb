"""Search tools for MCP server.

Exposes the hybrid search and fast vector-only search to Claude Code.
"""

from __future__ import annotations

from typing import Literal

from fastmcp import FastMCP

from research_kb_api.service import search, SearchOptions, ContextType
from research_kb_mcp.formatters import format_search_results, format_search_results_json
from research_kb_storage import HydeConfig


def register_search_tools(mcp: FastMCP) -> None:
    """Register search tools with the MCP server."""

    @mcp.tool()
    async def research_kb_search(
        query: str,
        limit: int = 10,
        domain: str | None = None,
        context_type: Literal["building", "auditing", "balanced"] = "balanced",
        use_graph: bool = True,
        use_rerank: bool = True,
        use_expand: bool = True,
        use_citations: bool = True,
        citation_weight: float = 0.15,
        use_hyde: bool = False,
        output_format: Literal["markdown", "json"] = "markdown",
    ) -> str:
        """Search the research knowledge base across multiple domains.

        Performs hybrid search combining:
        - Full-text search (BM25) for keyword matching
        - Vector similarity (BGE-large embeddings) for semantic matching
        - Knowledge graph signals for concept relationships
        - Citation authority signals (PageRank-style boosting)

        Args:
            query: Search query (natural language or keywords)
            limit: Maximum number of results (1-50, default 10)
            domain: Knowledge domain to search:
                - None: Search all domains (default)
                - "causal_inference": Econometrics, treatment effects, IV, DiD, DML
                - "time_series": Forecasting, ARIMA, VAR, GARCH, state-space
            context_type: Search weighting strategy:
                - "building": Favor semantic breadth (20% FTS, 80% vector)
                - "auditing": Favor precision (50% FTS, 50% vector)
                - "balanced": Default balance (30% FTS, 70% vector)
            use_graph: Include knowledge graph signals (default True)
            use_rerank: Apply cross-encoder reranking (default True)
            use_expand: Expand query with synonyms (default True)
            use_citations: Enable citation authority boosting (default True)
            citation_weight: Weight for citation signal (0-1, default 0.15)
            use_hyde: Enable HyDE query expansion (default False).
                Generates a hypothetical document to improve embedding quality
                for terse queries. Requires Ollama running locally.
            output_format: Response format - "markdown" (default) or "json".
                JSON returns structured data for programmatic consumers.

        Returns:
            Markdown-formatted or JSON search results with:
            - Source title, authors, year
            - Page numbers and section headers
            - Relevant text excerpt
            - Score breakdown (FTS, vector, graph, citation)
            - Source and chunk IDs for follow-up queries

        Example queries:
            - "instrumental variables assumptions" (causal_inference)
            - "double machine learning implementation" (causal_inference)
            - "ARIMA stationarity" (time_series)
            - "VAR impulse response" (time_series)
        """
        # Validate and clamp limit
        limit = max(1, min(50, limit))
        citation_weight = max(0.0, min(1.0, citation_weight))

        # Build HyDE config if requested
        hyde_config = None
        if use_hyde:
            hyde_config = HydeConfig(enabled=True, backend="ollama")

        options = SearchOptions(
            query=query,
            limit=limit,
            context_type=ContextType(context_type),
            use_graph=use_graph,
            use_rerank=use_rerank,
            use_expand=use_expand,
            use_citations=use_citations,
            citation_weight=citation_weight,
            domain_id=domain,
            hyde_config=hyde_config,
        )

        response = await search(options)
        if output_format == "json":
            return format_search_results_json(response)
        return format_search_results(response)

    @mcp.tool()
    async def research_kb_fast_search(
        query: str,
        limit: int = 5,
        domain: str | None = None,
        output_format: Literal["markdown", "json"] = "markdown",
    ) -> str:
        """Fast vector-only search (~200ms). Skips FTS, graph, citation, reranking.

        Use this for quick lookups when latency matters more than recall.
        Results are ranked by cosine similarity to BGE-large embeddings only.

        Args:
            query: Search query (natural language or keywords)
            limit: Maximum number of results (1-20, default 5)
            domain: Knowledge domain to filter by (optional)
            output_format: Response format - "markdown" (default) or "json"

        Returns:
            Markdown-formatted or JSON search results with vector similarity scores.
        """
        limit = max(1, min(20, limit))

        options = SearchOptions(
            query=query,
            limit=limit,
            fast_mode=True,
            domain_id=domain,
            use_graph=False,
            use_rerank=False,
            use_expand=False,
            use_citations=False,
        )

        response = await search(options)
        if output_format == "json":
            return format_search_results_json(response)
        return format_search_results(response)
