"""Research-KB Python Client SDK.

Provides typed Python interface for research-kb daemon and CLI.

Quick Start
-----------
>>> from research_kb_client import DaemonClient
>>> client = DaemonClient()
>>> if client.is_available():
...     results = client.search("instrumental variables")
...     for r in results.results:
...         print(f"{r.source.title}: {r.score:.2f}")

Convenience Functions
---------------------
>>> from research_kb_client import search_or_none, get_methodology_context
>>> results = search_or_none("IV assumptions")  # Returns None if unavailable
>>> context = get_methodology_context("double machine learning")  # Markdown string
"""

from .models import (
    ConceptInfo,
    ConceptNeighborhood,
    ConceptRelationship,
    HealthStatus,
    SearchResponse,
    SearchResult,
    SearchResultChunk,
    SearchResultSource,
    StatsResponse,
)
from .socket_client import (
    ConnectionError,
    DaemonClient,
    ResearchKBError,
    TimeoutError,
    get_methodology_context,
    search_or_none,
)

__all__ = [
    # Client
    "DaemonClient",
    # Models
    "SearchResponse",
    "SearchResult",
    "SearchResultChunk",
    "SearchResultSource",
    "HealthStatus",
    "StatsResponse",
    "ConceptInfo",
    "ConceptNeighborhood",
    "ConceptRelationship",
    # Errors
    "ResearchKBError",
    "ConnectionError",
    "TimeoutError",
    # Convenience
    "search_or_none",
    "get_methodology_context",
]

__version__ = "0.1.0"
