"""Prometheus metrics for Research-KB API.

Provides comprehensive observability through three metric categories:

1. RED Metrics (Rate, Errors, Duration)
   - Request counts by endpoint, method, status
   - Request duration histograms
   - In-flight request gauges

2. Resource Metrics
   - Database pool utilization
   - Embedding cache statistics

3. Business Metrics
   - Source/chunk/concept counts
   - Search result quality indicators

Usage:
    from research_kb_api.metrics import (
        instrument_request,
        track_search_results,
        update_business_metrics,
    )
"""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Generator

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.requests import Request
from starlette.responses import Response

# ==============================================================================
# RED Metrics (Rate, Errors, Duration)
# ==============================================================================

# Request counter by endpoint, method, and status
REQUEST_COUNT = Counter(
    "research_kb_requests_total",
    "Total HTTP requests",
    ["endpoint", "method", "status"],
)

# Request duration histogram by endpoint
REQUEST_DURATION = Histogram(
    "research_kb_request_duration_seconds",
    "HTTP request duration in seconds",
    ["endpoint", "method"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# In-flight requests gauge
REQUESTS_IN_PROGRESS = Gauge(
    "research_kb_requests_in_progress",
    "Number of HTTP requests currently being processed",
    ["endpoint"],
)

# ==============================================================================
# Resource Metrics
# ==============================================================================

# Database pool metrics
DB_POOL_SIZE = Gauge(
    "research_kb_db_pool_size",
    "Total database connection pool size",
)

DB_POOL_AVAILABLE = Gauge(
    "research_kb_db_pool_available",
    "Available connections in database pool",
)

# Embedding cache metrics
EMBEDDING_CACHE_SIZE = Gauge(
    "research_kb_embedding_cache_size",
    "Number of embeddings in cache",
)

EMBEDDING_CACHE_HITS = Counter(
    "research_kb_embedding_cache_hits_total",
    "Total embedding cache hits",
)

EMBEDDING_CACHE_MISSES = Counter(
    "research_kb_embedding_cache_misses_total",
    "Total embedding cache misses",
)

EMBEDDING_DURATION = Histogram(
    "research_kb_embedding_duration_seconds",
    "Time to generate embeddings",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# ==============================================================================
# Business Metrics
# ==============================================================================

# Corpus size gauges
SOURCES_TOTAL = Gauge(
    "research_kb_sources_total",
    "Total number of sources in database",
)

CHUNKS_TOTAL = Gauge(
    "research_kb_chunks_total",
    "Total number of chunks in database",
)

CONCEPTS_TOTAL = Gauge(
    "research_kb_concepts_total",
    "Total number of concepts in database",
)

RELATIONSHIPS_TOTAL = Gauge(
    "research_kb_relationships_total",
    "Total number of concept relationships in database",
)

CITATIONS_TOTAL = Gauge(
    "research_kb_citations_total",
    "Total number of citations in database",
)

# Search quality metrics
SEARCH_RESULTS_RETURNED = Histogram(
    "research_kb_search_results_returned",
    "Number of results returned per search",
    buckets=[0, 1, 2, 3, 5, 10, 20, 50, 100],
)

SEARCH_EMPTY_TOTAL = Counter(
    "research_kb_search_empty_total",
    "Total searches returning zero results",
)

SEARCH_DURATION = Histogram(
    "research_kb_search_duration_seconds",
    "Search execution time in seconds",
    ["search_type"],  # fts, vector, hybrid
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)


# ==============================================================================
# Helper Functions
# ==============================================================================


@contextmanager
def instrument_request(endpoint: str, method: str) -> Generator[None, None, None]:
    """Context manager to instrument an HTTP request.

    Tracks:
    - Request count (incremented on completion)
    - Request duration (histogram)
    - In-flight requests (gauge)

    Usage:
        with instrument_request("/search", "POST"):
            # Handle request
            pass
    """
    REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).inc()
    start_time = time.time()

    try:
        yield
    finally:
        duration = time.time() - start_time
        REQUEST_DURATION.labels(endpoint=endpoint, method=method).observe(duration)
        REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).dec()


def track_request_status(endpoint: str, method: str, status: int) -> None:
    """Record request completion with status code."""
    REQUEST_COUNT.labels(
        endpoint=endpoint,
        method=method,
        status=str(status),
    ).inc()


def track_search_results(count: int, search_type: str = "hybrid", duration: float | None = None) -> None:
    """Track search result metrics.

    Args:
        count: Number of results returned
        search_type: Type of search (fts, vector, hybrid)
        duration: Optional search duration in seconds
    """
    SEARCH_RESULTS_RETURNED.observe(count)
    if count == 0:
        SEARCH_EMPTY_TOTAL.inc()
    if duration is not None:
        SEARCH_DURATION.labels(search_type=search_type).observe(duration)


def update_business_metrics(stats: dict) -> None:
    """Update business metrics from database stats.

    Args:
        stats: Dictionary with keys: sources, chunks, concepts, relationships, citations
    """
    if "sources" in stats:
        SOURCES_TOTAL.set(stats["sources"])
    if "chunks" in stats:
        CHUNKS_TOTAL.set(stats["chunks"])
    if "concepts" in stats:
        CONCEPTS_TOTAL.set(stats["concepts"])
    if "relationships" in stats:
        RELATIONSHIPS_TOTAL.set(stats["relationships"])
    if "citations" in stats:
        CITATIONS_TOTAL.set(stats["citations"])


def track_embedding(duration: float, cache_hit: bool = False) -> None:
    """Track embedding generation metrics.

    Args:
        duration: Time to generate embedding in seconds
        cache_hit: Whether the embedding was served from cache
    """
    EMBEDDING_DURATION.observe(duration)
    if cache_hit:
        EMBEDDING_CACHE_HITS.inc()
    else:
        EMBEDDING_CACHE_MISSES.inc()


def update_pool_metrics(size: int, available: int) -> None:
    """Update database pool metrics.

    Args:
        size: Total pool size
        available: Available connections
    """
    DB_POOL_SIZE.set(size)
    DB_POOL_AVAILABLE.set(available)


# ==============================================================================
# Metrics Endpoint
# ==============================================================================


async def metrics_endpoint(request: Request) -> Response:
    """Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    Mount this at /metrics in your FastAPI app.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
