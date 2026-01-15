"""Search tool for MCP server.

Exposes the hybrid search functionality (FTS + vector + graph) to Claude Code.
"""

from __future__ import annotations

from typing import Literal

from fastmcp import FastMCP

from research_kb_api.service import search, SearchOptions, ContextType
from research_kb_mcp.formatters import format_search_results


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

        Returns:
            Markdown-formatted search results with:
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
        )

        response = await search(options)
        return format_search_results(response)
