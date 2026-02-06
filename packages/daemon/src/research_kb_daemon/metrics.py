"""Prometheus metrics for research-kb daemon.

Provides daemon-specific observability:

1. RED Metrics (Rate, Errors, Duration)
   - Request counts by method and status
   - Request duration histograms
   - Active connection gauge

2. Graph Engine Metrics
   - Query duration by engine (kuzu vs postgres_fallback)
   - Engine selection counts

3. System Metrics
   - KuzuDB data freshness
   - Daemon uptime

Scraped by Prometheus on port 9001 (configured in monitoring/prometheus/prometheus.yml).
"""

from prometheus_client import Counter, Gauge, Histogram

# ==============================================================================
# RED Metrics (Rate, Errors, Duration)
# ==============================================================================

REQUEST_DURATION = Histogram(
    "daemon_request_duration_seconds",
    "Daemon JSON-RPC request duration in seconds",
    ["method", "status"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

REQUEST_COUNT = Counter(
    "daemon_requests_total",
    "Total daemon JSON-RPC requests",
    ["method", "status"],
)

ACTIVE_CONNECTIONS = Gauge(
    "daemon_active_connections",
    "Number of active client connections",
)

# ==============================================================================
# Graph Engine Metrics
# ==============================================================================

GRAPH_QUERY_DURATION = Histogram(
    "graph_query_duration_seconds",
    "Graph query duration by engine",
    ["engine"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

GRAPH_ENGINE_SELECTION = Counter(
    "graph_engine_queries_total",
    "Total graph queries by engine selection",
    ["engine"],
)

# ==============================================================================
# System Metrics
# ==============================================================================

KUZU_DATA_FRESHNESS = Gauge(
    "kuzu_data_freshness_seconds",
    "Seconds since last KuzuDB sync (0 = unknown)",
)

DAEMON_UPTIME = Gauge(
    "daemon_uptime_seconds",
    "Daemon uptime in seconds",
)

# ==============================================================================
# KuzuDB Warm-up Metrics
# ==============================================================================

KUZU_WARMUP_DURATION = Histogram(
    "kuzu_warmup_duration_seconds",
    "Duration of KuzuDB pre-warming on startup",
    buckets=[1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0],
)

KUZU_WARMUP_STATUS = Gauge(
    "kuzu_warmup_status",
    "KuzuDB warmup status: 0=pending, 1=in_progress, 2=completed, -1=failed",
)
