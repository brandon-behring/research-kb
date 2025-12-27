"""Tests for ExtractionMetrics - Extraction pipeline observability.

Tests:
- Recording success/failure/empty extractions
- Computed properties (rates, averages, percentiles)
- Alerting thresholds
- Prometheus export format
- Human-readable summary
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from research_kb_extraction.metrics import ExtractionMetrics


class TestExtractionMetricsInit:
    """Tests for ExtractionMetrics initialization."""

    def test_default_values(self):
        """Test default initialization values."""
        metrics = ExtractionMetrics(backend="ollama:llama3.1:8b")

        assert metrics.backend == "ollama:llama3.1:8b"
        assert metrics.total_chunks == 0
        assert metrics.successful == 0
        assert metrics.validation_failures == 0
        assert metrics.json_parse_failures == 0
        assert metrics.empty_extractions == 0
        assert metrics.total_concepts == 0
        assert metrics.total_relationships == 0
        assert metrics.latencies_ms == []

    def test_custom_backend_name(self):
        """Test custom backend name is stored."""
        metrics = ExtractionMetrics(backend="anthropic:haiku")

        assert metrics.backend == "anthropic:haiku"


class TestRecordSuccess:
    """Tests for record_success()."""

    def test_increments_counters(self):
        """Test success increments appropriate counters."""
        metrics = ExtractionMetrics(backend="test")

        metrics.record_success(concepts=5, relationships=3, latency_ms=1500.0)

        assert metrics.total_chunks == 1
        assert metrics.successful == 1
        assert metrics.total_concepts == 5
        assert metrics.total_relationships == 3
        assert 1500.0 in metrics.latencies_ms

    def test_tracks_empty_extractions(self):
        """Test empty extraction (0 concepts) is tracked."""
        metrics = ExtractionMetrics(backend="test")

        metrics.record_success(concepts=0, relationships=0, latency_ms=500.0)

        assert metrics.successful == 1
        assert metrics.empty_extractions == 1
        assert metrics.total_concepts == 0

    def test_non_empty_extraction_not_counted_as_empty(self):
        """Test extraction with concepts is not counted as empty."""
        metrics = ExtractionMetrics(backend="test")

        metrics.record_success(concepts=3, relationships=0, latency_ms=500.0)

        assert metrics.successful == 1
        assert metrics.empty_extractions == 0

    def test_multiple_successes_accumulate(self):
        """Test multiple successes accumulate correctly."""
        metrics = ExtractionMetrics(backend="test")

        metrics.record_success(concepts=3, relationships=2, latency_ms=1000.0)
        metrics.record_success(concepts=5, relationships=4, latency_ms=1500.0)
        metrics.record_success(concepts=2, relationships=1, latency_ms=800.0)

        assert metrics.total_chunks == 3
        assert metrics.successful == 3
        assert metrics.total_concepts == 10
        assert metrics.total_relationships == 7
        assert len(metrics.latencies_ms) == 3


class TestRecordFailures:
    """Tests for failure recording methods."""

    def test_record_validation_failure(self):
        """Test validation failure is recorded."""
        metrics = ExtractionMetrics(backend="test")

        metrics.record_validation_failure(latency_ms=200.0)

        assert metrics.total_chunks == 1
        assert metrics.validation_failures == 1
        assert metrics.successful == 0
        assert 200.0 in metrics.latencies_ms

    def test_record_json_failure(self):
        """Test JSON parse failure is recorded."""
        metrics = ExtractionMetrics(backend="test")

        metrics.record_json_failure(latency_ms=150.0)

        assert metrics.total_chunks == 1
        assert metrics.json_parse_failures == 1
        assert metrics.successful == 0
        assert 150.0 in metrics.latencies_ms

    def test_mixed_results(self):
        """Test mixed success and failure recording."""
        metrics = ExtractionMetrics(backend="test")

        metrics.record_success(concepts=3, relationships=1, latency_ms=1000.0)
        metrics.record_validation_failure(latency_ms=200.0)
        metrics.record_success(concepts=5, relationships=2, latency_ms=1200.0)
        metrics.record_json_failure(latency_ms=100.0)

        assert metrics.total_chunks == 4
        assert metrics.successful == 2
        assert metrics.validation_failures == 1
        assert metrics.json_parse_failures == 1


class TestComputedProperties:
    """Tests for computed property values."""

    def test_failure_rate_zero_chunks(self):
        """Test failure rate is 0 when no chunks processed."""
        metrics = ExtractionMetrics(backend="test")

        assert metrics.failure_rate == 0.0

    def test_failure_rate_calculation(self):
        """Test failure rate calculation."""
        metrics = ExtractionMetrics(backend="test")

        # 2 failures out of 10 chunks = 20%
        for _ in range(8):
            metrics.record_success(concepts=1, relationships=0, latency_ms=100.0)
        metrics.record_validation_failure(latency_ms=50.0)
        metrics.record_json_failure(latency_ms=50.0)

        assert metrics.failure_rate == pytest.approx(0.2, rel=1e-5)

    def test_empty_rate_zero_successful(self):
        """Test empty rate is 0 when no successful extractions."""
        metrics = ExtractionMetrics(backend="test")
        metrics.record_json_failure(latency_ms=100.0)

        assert metrics.empty_rate == 0.0

    def test_empty_rate_calculation(self):
        """Test empty rate calculation."""
        metrics = ExtractionMetrics(backend="test")

        # 3 empty out of 10 successful = 30%
        for _ in range(7):
            metrics.record_success(concepts=1, relationships=0, latency_ms=100.0)
        for _ in range(3):
            metrics.record_success(concepts=0, relationships=0, latency_ms=100.0)

        assert metrics.empty_rate == pytest.approx(0.3, rel=1e-5)

    def test_avg_concepts_per_chunk(self):
        """Test average concepts per chunk calculation."""
        metrics = ExtractionMetrics(backend="test")

        metrics.record_success(concepts=4, relationships=0, latency_ms=100.0)
        metrics.record_success(concepts=6, relationships=0, latency_ms=100.0)

        assert metrics.avg_concepts_per_chunk == pytest.approx(5.0, rel=1e-5)

    def test_avg_concepts_zero_successful(self):
        """Test average concepts is 0 when no successes."""
        metrics = ExtractionMetrics(backend="test")

        assert metrics.avg_concepts_per_chunk == 0.0

    def test_avg_relationships_per_chunk(self):
        """Test average relationships per chunk calculation."""
        metrics = ExtractionMetrics(backend="test")

        metrics.record_success(concepts=2, relationships=3, latency_ms=100.0)
        metrics.record_success(concepts=2, relationships=5, latency_ms=100.0)

        assert metrics.avg_relationships_per_chunk == pytest.approx(4.0, rel=1e-5)


class TestLatencyMetrics:
    """Tests for latency calculations."""

    def test_latency_p50_empty(self):
        """Test p50 is 0 when no latencies recorded."""
        metrics = ExtractionMetrics(backend="test")

        assert metrics.latency_p50 == 0.0

    def test_latency_p50_single(self):
        """Test p50 with single latency."""
        metrics = ExtractionMetrics(backend="test")
        metrics.record_success(concepts=1, relationships=0, latency_ms=500.0)

        assert metrics.latency_p50 == 500.0

    def test_latency_p50_multiple(self):
        """Test p50 calculation with multiple latencies."""
        metrics = ExtractionMetrics(backend="test")

        # Latencies: 100, 200, 300, 400, 500 -> median = 300
        for lat in [100.0, 200.0, 300.0, 400.0, 500.0]:
            metrics.record_success(concepts=1, relationships=0, latency_ms=lat)

        assert metrics.latency_p50 == 300.0

    def test_latency_p95_empty(self):
        """Test p95 is 0 when no latencies recorded."""
        metrics = ExtractionMetrics(backend="test")

        assert metrics.latency_p95 == 0.0

    def test_latency_p95_calculation(self):
        """Test p95 calculation."""
        metrics = ExtractionMetrics(backend="test")

        # 100 latencies from 1 to 100
        for i in range(1, 101):
            metrics.record_success(concepts=1, relationships=0, latency_ms=float(i))

        # p95 should be around 95
        assert 94.0 <= metrics.latency_p95 <= 96.0

    def test_total_time_seconds(self):
        """Test total time calculation in seconds."""
        metrics = ExtractionMetrics(backend="test")

        metrics.record_success(concepts=1, relationships=0, latency_ms=1000.0)
        metrics.record_success(concepts=1, relationships=0, latency_ms=2000.0)

        assert metrics.total_time_seconds == pytest.approx(3.0, rel=1e-5)

    def test_throughput_chunks_per_min(self):
        """Test throughput calculation."""
        metrics = ExtractionMetrics(backend="test")

        # 2 chunks in 1000ms = 1 second = 120 chunks/min
        metrics.record_success(concepts=1, relationships=0, latency_ms=500.0)
        metrics.record_success(concepts=1, relationships=0, latency_ms=500.0)

        assert metrics.throughput_chunks_per_min == pytest.approx(120.0, rel=1e-5)

    def test_throughput_zero_time(self):
        """Test throughput is 0 when no time elapsed."""
        metrics = ExtractionMetrics(backend="test")

        assert metrics.throughput_chunks_per_min == 0.0


class TestAlerts:
    """Tests for alerting threshold checks."""

    def test_no_alerts_when_healthy(self):
        """Test no alerts when metrics are healthy."""
        metrics = ExtractionMetrics(backend="test")

        # 95% success, 10% empty = healthy
        for _ in range(95):
            metrics.record_success(concepts=1, relationships=0, latency_ms=100.0)
        for _ in range(5):
            metrics.record_json_failure(latency_ms=50.0)

        alerts = metrics.check_alerts()
        assert len(alerts) == 0

    def test_failure_rate_alert(self):
        """Test alert when failure rate exceeds threshold."""
        metrics = ExtractionMetrics(backend="test")

        # 10% failure > 5% threshold
        for _ in range(9):
            metrics.record_success(concepts=1, relationships=0, latency_ms=100.0)
        metrics.record_validation_failure(latency_ms=50.0)

        alerts = metrics.check_alerts()

        assert len(alerts) == 1
        assert "HIGH FAILURE RATE" in alerts[0]
        assert "10.0%" in alerts[0]

    def test_empty_rate_alert(self):
        """Test alert when empty rate exceeds threshold."""
        metrics = ExtractionMetrics(backend="test")

        # 20% empty > 15% threshold
        for _ in range(8):
            metrics.record_success(concepts=1, relationships=0, latency_ms=100.0)
        for _ in range(2):
            metrics.record_success(concepts=0, relationships=0, latency_ms=100.0)

        alerts = metrics.check_alerts()

        assert len(alerts) == 1
        assert "HIGH EMPTY RATE" in alerts[0]
        assert "20.0%" in alerts[0]

    def test_multiple_alerts(self):
        """Test multiple alerts can be triggered simultaneously."""
        metrics = ExtractionMetrics(backend="test")

        # Both failure and empty rate too high
        for _ in range(6):
            metrics.record_success(concepts=1, relationships=0, latency_ms=100.0)
        for _ in range(2):
            metrics.record_success(concepts=0, relationships=0, latency_ms=100.0)
        for _ in range(2):
            metrics.record_json_failure(latency_ms=50.0)

        alerts = metrics.check_alerts()

        assert len(alerts) == 2
        alert_text = " ".join(alerts)
        assert "FAILURE RATE" in alert_text
        assert "EMPTY RATE" in alert_text


class TestPrometheusExport:
    """Tests for Prometheus format export."""

    def test_prometheus_format_structure(self):
        """Test Prometheus output has correct structure."""
        metrics = ExtractionMetrics(backend="ollama:llama3.1:8b")
        metrics.record_success(concepts=5, relationships=3, latency_ms=1500.0)

        output = metrics.to_prometheus()

        # Check HELP and TYPE comments
        assert "# HELP extraction_total" in output
        assert "# TYPE extraction_total counter" in output
        assert "# HELP extraction_successful" in output
        assert "# TYPE extraction_successful counter" in output

    def test_prometheus_backend_label(self):
        """Test backend label is included in metrics."""
        metrics = ExtractionMetrics(backend="anthropic:haiku")
        metrics.record_success(concepts=1, relationships=0, latency_ms=100.0)

        output = metrics.to_prometheus()

        assert 'backend="anthropic:haiku"' in output

    def test_prometheus_counter_values(self):
        """Test counter values are correct."""
        metrics = ExtractionMetrics(backend="test")
        metrics.record_success(concepts=5, relationships=3, latency_ms=100.0)
        metrics.record_json_failure(latency_ms=50.0)

        output = metrics.to_prometheus()

        assert 'extraction_total{backend="test"} 2' in output
        assert 'extraction_successful{backend="test"} 1' in output
        assert 'extraction_json_failures{backend="test"} 1' in output
        assert 'extraction_concepts_total{backend="test"} 5' in output

    def test_prometheus_gauge_values(self):
        """Test gauge values are correct."""
        metrics = ExtractionMetrics(backend="test")
        metrics.record_success(concepts=1, relationships=0, latency_ms=100.0)
        metrics.record_success(concepts=1, relationships=0, latency_ms=200.0)

        output = metrics.to_prometheus()

        # Latency gauges should be present
        assert "extraction_latency_p50_ms" in output
        assert "extraction_latency_p95_ms" in output
        assert "extraction_throughput_per_min" in output

    def test_save_prometheus(self):
        """Test saving Prometheus metrics to file."""
        metrics = ExtractionMetrics(backend="test")
        metrics.record_success(concepts=3, relationships=1, latency_ms=500.0)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "metrics.txt"
            metrics.save_prometheus(path)

            assert path.exists()
            content = path.read_text()
            assert "extraction_total" in content

    def test_save_creates_parent_dirs(self):
        """Test save_prometheus creates parent directories."""
        metrics = ExtractionMetrics(backend="test")

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "deep" / "nested" / "path" / "metrics.txt"
            metrics.save_prometheus(path)

            assert path.exists()


class TestSummary:
    """Tests for human-readable summary output."""

    def test_summary_with_no_data(self):
        """Test summary when no data collected."""
        metrics = ExtractionMetrics(backend="test")

        summary = metrics.summary()

        assert "test" in summary
        assert "No data collected" in summary

    def test_summary_includes_all_metrics(self):
        """Test summary includes all key metrics."""
        metrics = ExtractionMetrics(backend="ollama:llama3.1:8b")
        metrics.record_success(concepts=5, relationships=3, latency_ms=1500.0)
        metrics.record_success(concepts=3, relationships=2, latency_ms=1200.0)
        metrics.record_json_failure(latency_ms=100.0)

        summary = metrics.summary()

        assert "ollama:llama3.1:8b" in summary
        assert "Total chunks:" in summary
        assert "Successful:" in summary
        assert "Failures:" in summary
        assert "Empty extractions:" in summary
        assert "Concepts/chunk:" in summary
        assert "Latency p50:" in summary
        assert "Throughput:" in summary

    def test_summary_failure_percentage(self):
        """Test summary shows failure percentage."""
        metrics = ExtractionMetrics(backend="test")
        for _ in range(9):
            metrics.record_success(concepts=1, relationships=0, latency_ms=100.0)
        metrics.record_validation_failure(latency_ms=50.0)

        summary = metrics.summary()

        # Should show 10% failure rate
        assert "10.0%" in summary


class TestThresholdConstants:
    """Tests for alerting threshold constants."""

    def test_empty_rate_threshold_is_reasonable(self):
        """Test empty rate threshold is set reasonably."""
        assert ExtractionMetrics.EMPTY_RATE_WARN == 0.15

    def test_failure_rate_threshold_is_reasonable(self):
        """Test failure rate threshold is set reasonably."""
        assert ExtractionMetrics.FAILURE_RATE_WARN == 0.05
