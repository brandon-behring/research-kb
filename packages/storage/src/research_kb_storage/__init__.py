"""Research KB Storage - PostgreSQL storage layer.

Version: 1.0.0

This package provides:
- Database connection management (asyncpg pooling)
- SourceStore (CRUD operations for sources table)
- ChunkStore (CRUD operations for chunks table)
- CitationStore (CRUD operations for citations table)
- ConceptStore (CRUD operations for concepts table - Phase 2)
- RelationshipStore (CRUD operations for concept_relationships - Phase 2)
- ChunkConceptStore (CRUD operations for chunk_concepts - Phase 2)
- MethodStore (CRUD operations for methods table - Phase 2)
- AssumptionStore (CRUD operations for assumptions table - Phase 2)
- Hybrid search (FTS + vector similarity)
- Graph-boosted search v2 (FTS + vector + graph signals)
- Query concept extraction
- Graph queries (shortest path, neighborhood, scoring)

Exclusive DB ownership - no shared database access from other packages.
"""

from research_kb_storage.chunk_store import ChunkStore
from research_kb_storage.citation_store import CitationStore
from research_kb_storage.concept_store import ConceptStore
from research_kb_storage.chunk_concept_store import ChunkConceptStore
from research_kb_storage.connection import (
    DatabaseConfig,
    close_connection_pool,
    get_connection_pool,
)
from research_kb_storage.relationship_store import RelationshipStore
from research_kb_storage.search import SearchQuery, search_hybrid, search_hybrid_v2, search_with_rerank, search_with_expansion
from research_kb_storage.query_extractor import (
    extract_query_concepts,
    extract_query_concepts_by_similarity,
)
from research_kb_storage.source_store import SourceStore
from research_kb_storage.method_store import MethodStore
from research_kb_storage.assumption_store import AssumptionStore
from research_kb_storage.graph_queries import (
    compute_graph_score,
    compute_weighted_graph_score,
    explain_path,
    find_shortest_path,
    find_shortest_path_length,
    get_neighborhood,
    get_path_with_explanation,
    get_relationship_weight,
    RELATIONSHIP_WEIGHTS,
)
from research_kb_storage.query_expander import (
    ExpandedQuery,
    QueryExpander,
    expand_query,
)
from research_kb_storage.citation_graph import (
    build_citation_graph,
    compute_pagerank_authority,
    get_citing_sources,
    get_cited_sources,
    get_citation_stats,
    get_corpus_citation_summary,
    get_most_cited_sources,
    match_citation_to_source,
)

__version__ = "1.0.0"

__all__ = [
    # Connection
    "DatabaseConfig",
    "get_connection_pool",
    "close_connection_pool",
    # Core Stores
    "SourceStore",
    "ChunkStore",
    "CitationStore",
    # Knowledge Graph Stores (Phase 2)
    "ConceptStore",
    "RelationshipStore",
    "ChunkConceptStore",
    "MethodStore",
    "AssumptionStore",
    # Search
    "SearchQuery",
    "search_hybrid",
    "search_hybrid_v2",
    "search_with_rerank",
    "search_with_expansion",
    "extract_query_concepts",
    "extract_query_concepts_by_similarity",
    # Graph Queries (Phase 2 Step 7 + Phase 3 enhancements)
    "find_shortest_path",
    "find_shortest_path_length",
    "get_neighborhood",
    "compute_graph_score",
    "compute_weighted_graph_score",
    "explain_path",
    "get_path_with_explanation",
    "get_relationship_weight",
    "RELATIONSHIP_WEIGHTS",
    # Query Expansion (Phase 3)
    "ExpandedQuery",
    "QueryExpander",
    "expand_query",
    # Citation Graph (Phase 3)
    "build_citation_graph",
    "compute_pagerank_authority",
    "get_citing_sources",
    "get_cited_sources",
    "get_citation_stats",
    "get_corpus_citation_summary",
    "get_most_cited_sources",
    "match_citation_to_source",
]
