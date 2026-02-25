"""Shared enums, utilities, and imports for CLI commands.

Centralises common types and the ``run_query`` helper so that every
sub-app can import them without circular dependencies.
"""

import sys
from enum import Enum
from typing import Optional

from research_kb_pdf import EmbeddingClient
from research_kb_storage import (
    ConceptStore,
    DatabaseConfig,
    SearchQuery,
    get_connection_pool,
    search_hybrid,
    search_hybrid_v2,
    search_with_expansion,
    search_with_rerank,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OutputFormat(str, Enum):
    """Output format options."""

    markdown = "markdown"
    json = "json"
    agent = "agent"


class ContextType(str, Enum):
    """Context type for search weighting.

    - building: Favor breadth, good for initial research
    - auditing: Favor precision, good for verification
    - balanced: Default balanced approach
    """

    building = "building"
    auditing = "auditing"
    balanced = "balanced"


class ScoringMethod(str, Enum):
    """Scoring method for combining search signals.

    - weighted: Weighted sum of normalized scores (default)
    - rrf: Reciprocal Rank Fusion (parameter-free, rank-based)
    """

    weighted = "weighted"
    rrf = "rrf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_context_weights(context_type: ContextType) -> tuple[float, float]:
    """Get FTS/vector weights based on context type.

    Args:
        context_type: The context mode

    Returns:
        Tuple of (fts_weight, vector_weight)
    """
    if context_type == ContextType.building:
        return 0.2, 0.8
    elif context_type == ContextType.auditing:
        return 0.5, 0.5
    else:  # balanced
        return 0.3, 0.7


async def run_query(
    query_text: str,
    limit: int,
    context_type: ContextType,
    source_filter: Optional[str],
    use_graph: bool = True,
    graph_weight: float = 0.2,
    use_citations: bool = True,
    citation_weight: float = 0.15,
    use_rerank: bool = True,
    use_expand: bool = True,
    use_llm_expand: bool = False,
    verbose: bool = False,
    domain_id: Optional[str] = None,
    scoring_method: str = "weighted",
) -> tuple:
    """Execute the search query with graph-boosted search, expansion, and reranking.

    Args:
        query_text: The query string
        limit: Maximum results
        context_type: Context mode for weight tuning
        source_filter: Optional source type filter
        use_graph: Enable graph-boosted search (default: True)
        graph_weight: Graph score weight (default: 0.2)
        use_citations: Enable citation authority boosting (default: True)
        citation_weight: Citation score weight (default: 0.15)
        use_rerank: Enable cross-encoder reranking (default: True)
        use_expand: Enable query expansion (default: True)
        use_llm_expand: Enable LLM-based expansion (default: False)
        verbose: Show expansion details (default: False)
        domain_id: Knowledge domain filter (None = all domains)
        scoring_method: Score combination method - "weighted" or "rrf" (default: weighted)

    Returns:
        Tuple of (SearchResult list, ExpandedQuery or None)
    """
    config = DatabaseConfig()
    await get_connection_pool(config)

    # Check if concepts exist when graph search requested
    if use_graph:
        concept_count = await ConceptStore.count()
        if concept_count == 0:
            print(
                "Warning: Graph search requested but no concepts extracted.",
                file=sys.stderr,
            )
            print("Falling back to standard search (FTS + vector only).", file=sys.stderr)
            print(
                "To extract concepts: python scripts/extract_concepts.py",
                file=sys.stderr,
            )
            print("", file=sys.stderr)
            use_graph = False

    # Generate embedding for query (uses BGE query instruction prefix for better recall)
    embed_client = EmbeddingClient()
    query_embedding = embed_client.embed_query(query_text)

    # Get weights based on context type
    fts_weight, vector_weight = get_context_weights(context_type)

    # Collect active signal weights
    active_weights = [("fts", fts_weight), ("vector", vector_weight)]
    if use_graph:
        active_weights.append(("graph", graph_weight))
    if use_citations:
        active_weights.append(("citation", citation_weight))

    # Normalize weights to sum to 1.0
    total = sum(w for _, w in active_weights)
    fts_weight = fts_weight / total
    vector_weight = vector_weight / total
    if use_graph:
        graph_weight = graph_weight / total
    if use_citations:
        citation_weight = citation_weight / total

    # Build search query with all enabled signals
    search_query = SearchQuery(
        text=query_text,
        embedding=query_embedding,
        fts_weight=fts_weight,
        vector_weight=vector_weight,
        graph_weight=graph_weight if use_graph else 0.0,
        use_graph=use_graph,
        citation_weight=citation_weight if use_citations else 0.0,
        use_citations=use_citations,
        max_hops=2,
        limit=limit,
        source_filter=source_filter,
        domain_id=domain_id,
        scoring_method=scoring_method,
    )

    # Execute search with expansion if enabled
    expanded_query = None
    if use_expand or use_llm_expand:
        results, expanded_query = await search_with_expansion(
            search_query,
            use_synonyms=use_expand,
            use_graph_expansion=use_expand and use_graph,
            use_llm_expansion=use_llm_expand,
            use_rerank=use_rerank,
            rerank_top_k=limit,
        )
    elif use_rerank:
        results = await search_with_rerank(search_query, rerank_top_k=limit)
    elif use_graph:
        results = await search_hybrid_v2(search_query)
    else:
        results = await search_hybrid(search_query)

    return results, expanded_query
