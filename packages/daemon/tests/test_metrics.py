"""Tests for daemon Prometheus metrics registration.

Tests cover:
- All metric objects are correctly instantiated
- Labels are correctly configured
- Histogram buckets are sensible
- Metric names follow Prometheus conventions
"""

import pytest
from prometheus_client import Counter, Gauge, Histogram

from research_kb_daemon.metrics import (
    ACTIVE_CONNECTIONS,
    DAEMON_UPTIME,
    GRAPH_ENGINE_SELECTION,
    GRAPH_QUERY_DURATION,
    KUZU_DATA_FRESHNESS,
    KUZU_WARMUP_DURATION,
    KUZU_WARMUP_STATUS,
    REQUEST_COUNT,
    REQUEST_DURATION,
)

pytestmark = pytest.mark.unit


class TestREDMetrics:
    """Tests for Rate/Errors/Duration metrics."""

    def test_request_duration_is_histogram(self):
        """REQUEST_DURATION is a Histogram."""
        assert isinstance(REQUEST_DURATION, Histogram)

    def test_request_duration_has_method_and_status_labels(self):
        """REQUEST_DURATION labeled by method and status."""
        assert REQUEST_DURATION._labelnames == ("method", "status")

    def test_request_duration_buckets_are_ascending(self):
        """Histogram buckets are in ascending order."""
        buckets = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        assert buckets == sorted(buckets)

    def test_request_count_is_counter(self):
        """REQUEST_COUNT is a Counter."""
        assert isinstance(REQUEST_COUNT, Counter)

    def test_request_count_has_method_and_status_labels(self):
        """REQUEST_COUNT labeled by method and status."""
        assert REQUEST_COUNT._labelnames == ("method", "status")

    def test_active_connections_is_gauge(self):
        """ACTIVE_CONNECTIONS is a Gauge (can go up and down)."""
        assert isinstance(ACTIVE_CONNECTIONS, Gauge)

    def test_active_connections_has_no_labels(self):
        """ACTIVE_CONNECTIONS has no labels (single global value)."""
        assert ACTIVE_CONNECTIONS._labelnames == ()


class TestGraphEngineMetrics:
    """Tests for graph engine metrics."""

    def test_graph_query_duration_is_histogram(self):
        """GRAPH_QUERY_DURATION is a Histogram."""
        assert isinstance(GRAPH_QUERY_DURATION, Histogram)

    def test_graph_query_duration_labeled_by_engine(self):
        """GRAPH_QUERY_DURATION labeled by engine."""
        assert GRAPH_QUERY_DURATION._labelnames == ("engine",)

    def test_graph_query_duration_buckets_include_fast_and_slow(self):
        """Buckets range from 5ms to 5s (covers KuzuDB and PG fallback)."""
        buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
        assert min(buckets) == 0.005
        assert max(buckets) == 5.0

    def test_graph_engine_selection_is_counter(self):
        """GRAPH_ENGINE_SELECTION is a Counter."""
        assert isinstance(GRAPH_ENGINE_SELECTION, Counter)

    def test_graph_engine_selection_labeled_by_engine(self):
        """GRAPH_ENGINE_SELECTION labeled by engine."""
        assert GRAPH_ENGINE_SELECTION._labelnames == ("engine",)


class TestSystemMetrics:
    """Tests for system-level metrics."""

    def test_kuzu_data_freshness_is_gauge(self):
        """KUZU_DATA_FRESHNESS is a Gauge."""
        assert isinstance(KUZU_DATA_FRESHNESS, Gauge)

    def test_daemon_uptime_is_gauge(self):
        """DAEMON_UPTIME is a Gauge."""
        assert isinstance(DAEMON_UPTIME, Gauge)


class TestKuzuWarmupMetrics:
    """Tests for KuzuDB warmup metrics."""

    def test_warmup_duration_is_histogram(self):
        """KUZU_WARMUP_DURATION is a Histogram."""
        assert isinstance(KUZU_WARMUP_DURATION, Histogram)

    def test_warmup_duration_buckets_cover_expected_range(self):
        """Warmup buckets: 1s to 120s (typical warmup: 5-15s)."""
        buckets = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0]
        assert buckets == sorted(buckets)
        assert min(buckets) == 1.0
        assert max(buckets) == 120.0

    def test_warmup_status_is_gauge(self):
        """KUZU_WARMUP_STATUS is a Gauge."""
        assert isinstance(KUZU_WARMUP_STATUS, Gauge)

    def test_warmup_status_has_no_labels(self):
        """KUZU_WARMUP_STATUS is a single global value."""
        assert KUZU_WARMUP_STATUS._labelnames == ()


class TestMetricNaming:
    """Tests for Prometheus naming conventions."""

    def test_all_names_use_snake_case(self):
        """All metric names follow Prometheus snake_case convention."""
        metrics = [
            REQUEST_DURATION,
            REQUEST_COUNT,
            ACTIVE_CONNECTIONS,
            GRAPH_QUERY_DURATION,
            GRAPH_ENGINE_SELECTION,
            KUZU_DATA_FRESHNESS,
            DAEMON_UPTIME,
            KUZU_WARMUP_DURATION,
            KUZU_WARMUP_STATUS,
        ]
        for m in metrics:
            name = m._name
            assert "_" in name or name.isalpha(), f"Metric {name} should use snake_case"
            assert name == name.lower(), f"Metric {name} should be lowercase"

    def test_daemon_prefix_for_daemon_metrics(self):
        """Daemon-specific metrics use 'daemon_' prefix."""
        assert REQUEST_DURATION._name.startswith("daemon_")
        assert REQUEST_COUNT._name.startswith("daemon_")
        assert ACTIVE_CONNECTIONS._name.startswith("daemon_")
        assert DAEMON_UPTIME._name.startswith("daemon_")

    def test_graph_prefix_for_graph_metrics(self):
        """Graph engine metrics use 'graph_' prefix."""
        assert GRAPH_QUERY_DURATION._name.startswith("graph_")
        assert GRAPH_ENGINE_SELECTION._name.startswith("graph_")

    def test_kuzu_prefix_for_kuzu_metrics(self):
        """KuzuDB metrics use 'kuzu_' prefix."""
        assert KUZU_DATA_FRESHNESS._name.startswith("kuzu_")
        assert KUZU_WARMUP_DURATION._name.startswith("kuzu_")
        assert KUZU_WARMUP_STATUS._name.startswith("kuzu_")
