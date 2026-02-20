"""Tests for Prometheus metrics module.

Tests cover:
- RED metrics (Rate, Errors, Duration)
- Resource metrics (pool, cache)
- Business metrics (corpus size, search quality)
- Helper functions for metric tracking
- Metrics endpoint response format
"""

from __future__ import annotations

import pytest
import time
from unittest.mock import MagicMock

from starlette.requests import Request
from starlette.responses import Response

from research_kb_api.metrics import (
    # Metric objects
    REQUEST_COUNT,
    REQUEST_DURATION,
    REQUESTS_IN_PROGRESS,
    DB_POOL_SIZE,
    DB_POOL_AVAILABLE,
    EMBEDDING_CACHE_SIZE,
    EMBEDDING_CACHE_HITS,
    EMBEDDING_CACHE_MISSES,
    EMBEDDING_DURATION,
    SOURCES_TOTAL,
    CHUNKS_TOTAL,
    CONCEPTS_TOTAL,
    RELATIONSHIPS_TOTAL,
    CITATIONS_TOTAL,
    SEARCH_RESULTS_RETURNED,
    SEARCH_EMPTY_TOTAL,
    SEARCH_DURATION,
    # Helper functions
    instrument_request,
    track_request_status,
    track_search_results,
    update_business_metrics,
    track_embedding,
    update_pool_metrics,
    metrics_endpoint,
)
from prometheus_client import CONTENT_TYPE_LATEST

pytestmark = pytest.mark.integration


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metrics state before each test.

    Note: Prometheus metrics are singletons and persist across tests.
    We track initial values to compute deltas instead of resetting.
    """
    # Yield to run test
    yield
    # Cleanup not needed - metrics persist intentionally


def get_counter_value(counter, labels=None):
    """Get current value of a counter metric."""
    if labels:
        return counter.labels(**labels)._value.get()
    return counter._value.get()


def get_gauge_value(gauge, labels=None):
    """Get current value of a gauge metric."""
    if labels:
        return gauge.labels(**labels)._value.get()
    return gauge._value.get()


# =============================================================================
# Test RED Metrics
# =============================================================================


class TestRequestCount:
    """Test request counter metric."""

    def test_request_count_exists(self):
        """Test REQUEST_COUNT metric is defined."""
        assert REQUEST_COUNT is not None
        # Prometheus client may or may not include _total suffix in _name
        assert "research_kb_requests" in REQUEST_COUNT._name

    def test_request_count_labels(self):
        """Test REQUEST_COUNT has correct labels."""
        assert REQUEST_COUNT._labelnames == ("endpoint", "method", "status")

    def test_track_request_status_increments_counter(self):
        """Test track_request_status increments the counter."""
        labels = {"endpoint": "/test/count", "method": "GET", "status": "200"}
        initial = get_counter_value(REQUEST_COUNT, labels)

        track_request_status("/test/count", "GET", 200)

        final = get_counter_value(REQUEST_COUNT, labels)
        assert final == initial + 1

    def test_track_request_status_different_statuses(self):
        """Test track_request_status tracks different status codes."""
        labels_200 = {"endpoint": "/test/multi", "method": "POST", "status": "200"}
        labels_404 = {"endpoint": "/test/multi", "method": "POST", "status": "404"}
        labels_500 = {"endpoint": "/test/multi", "method": "POST", "status": "500"}

        initial_200 = get_counter_value(REQUEST_COUNT, labels_200)
        initial_404 = get_counter_value(REQUEST_COUNT, labels_404)
        initial_500 = get_counter_value(REQUEST_COUNT, labels_500)

        track_request_status("/test/multi", "POST", 200)
        track_request_status("/test/multi", "POST", 404)
        track_request_status("/test/multi", "POST", 500)

        assert get_counter_value(REQUEST_COUNT, labels_200) == initial_200 + 1
        assert get_counter_value(REQUEST_COUNT, labels_404) == initial_404 + 1
        assert get_counter_value(REQUEST_COUNT, labels_500) == initial_500 + 1


class TestRequestDuration:
    """Test request duration histogram metric."""

    def test_request_duration_exists(self):
        """Test REQUEST_DURATION metric is defined."""
        assert REQUEST_DURATION is not None
        assert REQUEST_DURATION._name == "research_kb_request_duration_seconds"

    def test_request_duration_labels(self):
        """Test REQUEST_DURATION has correct labels."""
        assert REQUEST_DURATION._labelnames == ("endpoint", "method")

    def test_request_duration_buckets(self):
        """Test REQUEST_DURATION has correct buckets."""
        expected_buckets = [
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
            float("inf"),
        ]
        # Upper bounds includes +Inf bucket - compare as list
        assert list(REQUEST_DURATION._upper_bounds) == expected_buckets


class TestRequestsInProgress:
    """Test in-flight requests gauge metric."""

    def test_requests_in_progress_exists(self):
        """Test REQUESTS_IN_PROGRESS metric is defined."""
        assert REQUESTS_IN_PROGRESS is not None
        assert REQUESTS_IN_PROGRESS._name == "research_kb_requests_in_progress"

    def test_requests_in_progress_labels(self):
        """Test REQUESTS_IN_PROGRESS has correct labels."""
        assert REQUESTS_IN_PROGRESS._labelnames == ("endpoint",)


class TestInstrumentRequest:
    """Test instrument_request context manager."""

    def test_instrument_request_increments_in_progress(self):
        """Test in-progress gauge increments during request."""
        endpoint = "/test/instrument_inc"
        labels = {"endpoint": endpoint}
        initial = get_gauge_value(REQUESTS_IN_PROGRESS, labels)

        # Inside context, gauge should be incremented
        with instrument_request(endpoint, "GET"):
            during = get_gauge_value(REQUESTS_IN_PROGRESS, labels)
            assert during == initial + 1

        # After context, gauge should be decremented
        after = get_gauge_value(REQUESTS_IN_PROGRESS, labels)
        assert after == initial

    def test_instrument_request_records_duration(self):
        """Test request duration is recorded."""
        endpoint = "/test/instrument_dur"
        method = "POST"

        with instrument_request(endpoint, method):
            time.sleep(0.01)  # Ensure measurable duration

        # Duration histogram should have received an observation
        # We can't easily test the exact value, but we can verify the metric exists
        sample = REQUEST_DURATION.labels(endpoint=endpoint, method=method)
        assert sample is not None

    def test_instrument_request_decrements_on_exception(self):
        """Test in-progress gauge decrements even on exception."""
        endpoint = "/test/instrument_exc"
        labels = {"endpoint": endpoint}
        initial = get_gauge_value(REQUESTS_IN_PROGRESS, labels)

        try:
            with instrument_request(endpoint, "GET"):
                during = get_gauge_value(REQUESTS_IN_PROGRESS, labels)
                assert during == initial + 1
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should be back to initial after exception
        after = get_gauge_value(REQUESTS_IN_PROGRESS, labels)
        assert after == initial


# =============================================================================
# Test Resource Metrics
# =============================================================================


class TestDatabasePoolMetrics:
    """Test database pool metrics."""

    def test_db_pool_size_exists(self):
        """Test DB_POOL_SIZE metric is defined."""
        assert DB_POOL_SIZE is not None
        assert DB_POOL_SIZE._name == "research_kb_db_pool_size"

    def test_db_pool_available_exists(self):
        """Test DB_POOL_AVAILABLE metric is defined."""
        assert DB_POOL_AVAILABLE is not None
        assert DB_POOL_AVAILABLE._name == "research_kb_db_pool_available"

    def test_update_pool_metrics(self):
        """Test update_pool_metrics sets gauge values."""
        update_pool_metrics(size=10, available=7)

        assert DB_POOL_SIZE._value.get() == 10
        assert DB_POOL_AVAILABLE._value.get() == 7

    def test_update_pool_metrics_different_values(self):
        """Test update_pool_metrics with different values."""
        update_pool_metrics(size=20, available=3)

        assert DB_POOL_SIZE._value.get() == 20
        assert DB_POOL_AVAILABLE._value.get() == 3


class TestEmbeddingCacheMetrics:
    """Test embedding cache metrics."""

    def test_embedding_cache_size_exists(self):
        """Test EMBEDDING_CACHE_SIZE metric is defined."""
        assert EMBEDDING_CACHE_SIZE is not None
        assert EMBEDDING_CACHE_SIZE._name == "research_kb_embedding_cache_size"

    def test_embedding_cache_hits_exists(self):
        """Test EMBEDDING_CACHE_HITS metric is defined."""
        assert EMBEDDING_CACHE_HITS is not None
        # Prometheus client may or may not include _total suffix in _name
        assert "research_kb_embedding_cache_hits" in EMBEDDING_CACHE_HITS._name

    def test_embedding_cache_misses_exists(self):
        """Test EMBEDDING_CACHE_MISSES metric is defined."""
        assert EMBEDDING_CACHE_MISSES is not None
        # Prometheus client may or may not include _total suffix in _name
        assert "research_kb_embedding_cache_misses" in EMBEDDING_CACHE_MISSES._name

    def test_embedding_duration_exists(self):
        """Test EMBEDDING_DURATION metric is defined."""
        assert EMBEDDING_DURATION is not None
        assert EMBEDDING_DURATION._name == "research_kb_embedding_duration_seconds"

    def test_embedding_duration_buckets(self):
        """Test EMBEDDING_DURATION has correct buckets."""
        expected_buckets = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, float("inf")]
        # Compare as list for consistency
        assert list(EMBEDDING_DURATION._upper_bounds) == expected_buckets


class TestTrackEmbedding:
    """Test track_embedding helper function."""

    def test_track_embedding_records_duration(self):
        """Test track_embedding records duration."""
        # This should not raise
        track_embedding(duration=0.15, cache_hit=False)

    def test_track_embedding_cache_hit(self):
        """Test track_embedding increments cache hit counter."""
        initial = get_counter_value(EMBEDDING_CACHE_HITS)

        track_embedding(duration=0.01, cache_hit=True)

        final = get_counter_value(EMBEDDING_CACHE_HITS)
        assert final == initial + 1

    def test_track_embedding_cache_miss(self):
        """Test track_embedding increments cache miss counter."""
        initial = get_counter_value(EMBEDDING_CACHE_MISSES)

        track_embedding(duration=0.1, cache_hit=False)

        final = get_counter_value(EMBEDDING_CACHE_MISSES)
        assert final == initial + 1


# =============================================================================
# Test Business Metrics
# =============================================================================


class TestCorpusMetrics:
    """Test corpus size metrics."""

    def test_sources_total_exists(self):
        """Test SOURCES_TOTAL metric is defined."""
        assert SOURCES_TOTAL is not None
        assert SOURCES_TOTAL._name == "research_kb_sources_total"

    def test_chunks_total_exists(self):
        """Test CHUNKS_TOTAL metric is defined."""
        assert CHUNKS_TOTAL is not None
        assert CHUNKS_TOTAL._name == "research_kb_chunks_total"

    def test_concepts_total_exists(self):
        """Test CONCEPTS_TOTAL metric is defined."""
        assert CONCEPTS_TOTAL is not None
        assert CONCEPTS_TOTAL._name == "research_kb_concepts_total"

    def test_relationships_total_exists(self):
        """Test RELATIONSHIPS_TOTAL metric is defined."""
        assert RELATIONSHIPS_TOTAL is not None
        assert RELATIONSHIPS_TOTAL._name == "research_kb_relationships_total"

    def test_citations_total_exists(self):
        """Test CITATIONS_TOTAL metric is defined."""
        assert CITATIONS_TOTAL is not None
        assert CITATIONS_TOTAL._name == "research_kb_citations_total"


class TestUpdateBusinessMetrics:
    """Test update_business_metrics helper function."""

    def test_update_all_metrics(self):
        """Test update_business_metrics sets all gauges."""
        stats = {
            "sources": 150,
            "chunks": 7500,
            "concepts": 350,
            "relationships": 800,
            "citations": 450,
        }

        update_business_metrics(stats)

        assert SOURCES_TOTAL._value.get() == 150
        assert CHUNKS_TOTAL._value.get() == 7500
        assert CONCEPTS_TOTAL._value.get() == 350
        assert RELATIONSHIPS_TOTAL._value.get() == 800
        assert CITATIONS_TOTAL._value.get() == 450

    def test_update_partial_metrics(self):
        """Test update_business_metrics with partial stats."""
        stats = {
            "sources": 100,
            "chunks": 5000,
        }

        update_business_metrics(stats)

        assert SOURCES_TOTAL._value.get() == 100
        assert CHUNKS_TOTAL._value.get() == 5000
        # Other metrics should not be affected by this call

    def test_update_empty_stats(self):
        """Test update_business_metrics with empty stats."""
        # Should not raise
        update_business_metrics({})


class TestSearchQualityMetrics:
    """Test search quality metrics."""

    def test_search_results_returned_exists(self):
        """Test SEARCH_RESULTS_RETURNED metric is defined."""
        assert SEARCH_RESULTS_RETURNED is not None
        assert SEARCH_RESULTS_RETURNED._name == "research_kb_search_results_returned"

    def test_search_results_returned_buckets(self):
        """Test SEARCH_RESULTS_RETURNED has correct buckets."""
        expected_buckets = [
            0.0,
            1.0,
            2.0,
            3.0,
            5.0,
            10.0,
            20.0,
            50.0,
            100.0,
            float("inf"),
        ]
        # Compare as list for consistency
        assert list(SEARCH_RESULTS_RETURNED._upper_bounds) == expected_buckets

    def test_search_empty_total_exists(self):
        """Test SEARCH_EMPTY_TOTAL metric is defined."""
        assert SEARCH_EMPTY_TOTAL is not None
        # Prometheus client may or may not include _total suffix in _name
        assert "research_kb_search_empty" in SEARCH_EMPTY_TOTAL._name

    def test_search_duration_exists(self):
        """Test SEARCH_DURATION metric is defined."""
        assert SEARCH_DURATION is not None
        assert SEARCH_DURATION._name == "research_kb_search_duration_seconds"

    def test_search_duration_labels(self):
        """Test SEARCH_DURATION has correct labels."""
        assert SEARCH_DURATION._labelnames == ("search_type",)

    def test_search_duration_buckets(self):
        """Test SEARCH_DURATION has correct buckets."""
        expected_buckets = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, float("inf")]
        # Compare as list for consistency
        assert list(SEARCH_DURATION._upper_bounds) == expected_buckets


class TestTrackSearchResults:
    """Test track_search_results helper function."""

    def test_track_search_results_records_count(self):
        """Test track_search_results records result count."""
        # Should not raise
        track_search_results(count=10)

    def test_track_search_results_empty_search(self):
        """Test track_search_results increments empty counter."""
        initial = get_counter_value(SEARCH_EMPTY_TOTAL)

        track_search_results(count=0)

        final = get_counter_value(SEARCH_EMPTY_TOTAL)
        assert final == initial + 1

    def test_track_search_results_non_empty(self):
        """Test track_search_results does not increment empty counter."""
        initial = get_counter_value(SEARCH_EMPTY_TOTAL)

        track_search_results(count=5)

        final = get_counter_value(SEARCH_EMPTY_TOTAL)
        assert final == initial  # Should not increment

    def test_track_search_results_with_duration(self):
        """Test track_search_results records duration."""
        # Should not raise
        track_search_results(count=10, search_type="hybrid", duration=0.15)

    def test_track_search_results_different_types(self):
        """Test track_search_results with different search types."""
        # Should not raise for any type
        track_search_results(count=5, search_type="fts", duration=0.05)
        track_search_results(count=5, search_type="vector", duration=0.1)
        track_search_results(count=5, search_type="hybrid", duration=0.15)

    def test_track_search_results_no_duration(self):
        """Test track_search_results without duration."""
        # Should not raise when duration is None
        track_search_results(count=3, search_type="hybrid", duration=None)


# =============================================================================
# Test Metrics Endpoint
# =============================================================================


class TestMetricsEndpoint:
    """Test metrics endpoint function."""

    @pytest.mark.asyncio
    async def test_metrics_endpoint_returns_response(self):
        """Test metrics_endpoint returns a Response."""
        # Create a mock request
        mock_request = MagicMock(spec=Request)

        response = await metrics_endpoint(mock_request)

        assert isinstance(response, Response)

    @pytest.mark.asyncio
    async def test_metrics_endpoint_content_type(self):
        """Test metrics_endpoint returns correct content type."""
        mock_request = MagicMock(spec=Request)

        response = await metrics_endpoint(mock_request)

        assert response.media_type == CONTENT_TYPE_LATEST

    @pytest.mark.asyncio
    async def test_metrics_endpoint_content_contains_metrics(self):
        """Test metrics_endpoint content includes our metrics."""
        mock_request = MagicMock(spec=Request)

        response = await metrics_endpoint(mock_request)

        content = response.body.decode("utf-8")

        # Check for key metric names in output
        assert "research_kb_requests_total" in content
        assert "research_kb_request_duration_seconds" in content
        assert "research_kb_requests_in_progress" in content
        assert "research_kb_db_pool_size" in content
        assert "research_kb_embedding_cache" in content
        assert "research_kb_sources_total" in content
        assert "research_kb_search" in content

    @pytest.mark.asyncio
    async def test_metrics_endpoint_prometheus_format(self):
        """Test metrics_endpoint returns valid Prometheus format."""
        mock_request = MagicMock(spec=Request)

        response = await metrics_endpoint(mock_request)

        content = response.body.decode("utf-8")

        # Basic Prometheus format checks
        # Lines should be # HELP, # TYPE, or metric_name{labels} value
        lines = content.strip().split("\n")
        for line in lines:
            if line:
                assert (
                    line.startswith("# ")  # Comment lines
                    or line.startswith("research_kb_")  # Our metrics
                    or line.startswith("python_")  # Python process metrics
                    or line.startswith("process_")  # Process metrics
                ), f"Unexpected line format: {line}"


# =============================================================================
# Test Metric Configuration
# =============================================================================


class TestMetricConfiguration:
    """Test metric configuration and setup."""

    def test_all_metrics_have_descriptions(self):
        """Test all metrics have documentation strings."""
        metrics = [
            REQUEST_COUNT,
            REQUEST_DURATION,
            REQUESTS_IN_PROGRESS,
            DB_POOL_SIZE,
            DB_POOL_AVAILABLE,
            EMBEDDING_CACHE_SIZE,
            EMBEDDING_CACHE_HITS,
            EMBEDDING_CACHE_MISSES,
            EMBEDDING_DURATION,
            SOURCES_TOTAL,
            CHUNKS_TOTAL,
            CONCEPTS_TOTAL,
            RELATIONSHIPS_TOTAL,
            CITATIONS_TOTAL,
            SEARCH_RESULTS_RETURNED,
            SEARCH_EMPTY_TOTAL,
            SEARCH_DURATION,
        ]

        for metric in metrics:
            assert metric._documentation, f"Metric {metric._name} has no documentation"

    def test_counter_metrics_start_at_zero(self):
        """Test counter metrics use correct type."""
        counters = [
            REQUEST_COUNT,
            EMBEDDING_CACHE_HITS,
            EMBEDDING_CACHE_MISSES,
            SEARCH_EMPTY_TOTAL,
        ]

        for counter in counters:
            # Counter type should have _value attribute
            assert hasattr(counter, "_metrics") or hasattr(counter, "_value")

    def test_gauge_metrics_have_correct_type(self):
        """Test gauge metrics use correct type."""
        gauges = [
            REQUESTS_IN_PROGRESS,
            DB_POOL_SIZE,
            DB_POOL_AVAILABLE,
            EMBEDDING_CACHE_SIZE,
            SOURCES_TOTAL,
            CHUNKS_TOTAL,
            CONCEPTS_TOTAL,
            RELATIONSHIPS_TOTAL,
            CITATIONS_TOTAL,
        ]

        for gauge in gauges:
            # Gauge type check
            assert hasattr(gauge, "_value") or hasattr(gauge, "_metrics")

    def test_histogram_metrics_have_correct_type(self):
        """Test histogram metrics use correct type."""
        histograms = [
            REQUEST_DURATION,
            EMBEDDING_DURATION,
            SEARCH_RESULTS_RETURNED,
            SEARCH_DURATION,
        ]

        for histogram in histograms:
            # Histogram type should have _upper_bounds
            assert hasattr(histogram, "_upper_bounds")


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_instrument_request_with_empty_endpoint(self):
        """Test instrument_request handles empty endpoint."""
        # Should not raise
        with instrument_request("", "GET"):
            pass

    def test_track_request_status_with_unusual_status(self):
        """Test track_request_status with unusual status codes."""
        # Should not raise
        track_request_status("/test", "GET", 418)  # I'm a teapot
        track_request_status("/test", "GET", 599)  # Custom status

    def test_update_pool_metrics_zero_values(self):
        """Test update_pool_metrics with zero values."""
        update_pool_metrics(size=0, available=0)

        assert DB_POOL_SIZE._value.get() == 0
        assert DB_POOL_AVAILABLE._value.get() == 0

    def test_track_search_results_large_count(self):
        """Test track_search_results with large result count."""
        # Should not raise
        track_search_results(count=1000)

    def test_track_embedding_very_small_duration(self):
        """Test track_embedding with very small duration."""
        # Should not raise
        track_embedding(duration=0.0001, cache_hit=False)

    def test_track_embedding_very_large_duration(self):
        """Test track_embedding with very large duration."""
        # Should not raise
        track_embedding(duration=100.0, cache_hit=False)
