"""Extraction metrics for observability.

Tracks validation failures, empty extractions, latency, and per-backend stats.
Supports Prometheus text file format for scraping.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from research_kb_common import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractionMetrics:
    """Per-run extraction metrics with alerting thresholds.

    Tracks:
    - Success/failure counts with breakdown by failure type
    - Empty extractions (valid JSON but no concepts found)
    - Latency distribution (p50, p95)
    - Concepts and relationships per chunk

    Example:
        >>> metrics = ExtractionMetrics(backend="ollama:llama3.1:8b")
        >>> metrics.record_success(concepts=5, relationships=3, latency_ms=1500.0)
        >>> metrics.record_json_failure(latency_ms=200.0)
        >>> print(metrics.summary())
    """

    backend: str
    total_chunks: int = 0
    successful: int = 0
    validation_failures: int = 0
    json_parse_failures: int = 0
    empty_extractions: int = 0
    total_concepts: int = 0
    total_relationships: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    # Alerting thresholds (can be tuned with data)
    EMPTY_RATE_WARN = 0.15  # Warn if >15% empty (lenient start)
    FAILURE_RATE_WARN = 0.05  # Warn if >5% failures

    def record_success(
        self, concepts: int, relationships: int, latency_ms: float
    ) -> None:
        """Record a successful extraction.

        Args:
            concepts: Number of concepts extracted
            relationships: Number of relationships extracted
            latency_ms: Time taken in milliseconds
        """
        self.total_chunks += 1
        self.successful += 1
        self.total_concepts += concepts
        self.total_relationships += relationships
        self.latencies_ms.append(latency_ms)
        if concepts == 0 and relationships == 0:
            self.empty_extractions += 1

    def record_validation_failure(self, latency_ms: float) -> None:
        """Record a Pydantic validation failure (valid JSON, invalid schema)."""
        self.total_chunks += 1
        self.validation_failures += 1
        self.latencies_ms.append(latency_ms)

    def record_json_failure(self, latency_ms: float) -> None:
        """Record a JSON parse failure (malformed output)."""
        self.total_chunks += 1
        self.json_parse_failures += 1
        self.latencies_ms.append(latency_ms)

    @property
    def failure_rate(self) -> float:
        """Fraction of chunks that failed (JSON or validation error)."""
        if self.total_chunks == 0:
            return 0.0
        return (self.validation_failures + self.json_parse_failures) / self.total_chunks

    @property
    def empty_rate(self) -> float:
        """Fraction of successful extractions that produced no concepts."""
        if self.successful == 0:
            return 0.0
        return self.empty_extractions / self.successful

    @property
    def avg_concepts_per_chunk(self) -> float:
        """Average concepts per successful extraction."""
        if self.successful == 0:
            return 0.0
        return self.total_concepts / self.successful

    @property
    def avg_relationships_per_chunk(self) -> float:
        """Average relationships per successful extraction."""
        if self.successful == 0:
            return 0.0
        return self.total_relationships / self.successful

    @property
    def latency_p50(self) -> float:
        """Median latency in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        return sorted_lat[len(sorted_lat) // 2]

    @property
    def latency_p95(self) -> float:
        """95th percentile latency in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def total_time_seconds(self) -> float:
        """Total time spent on extractions in seconds."""
        return sum(self.latencies_ms) / 1000.0

    @property
    def throughput_chunks_per_min(self) -> float:
        """Throughput in chunks per minute."""
        if self.total_time_seconds == 0:
            return 0.0
        return self.total_chunks / (self.total_time_seconds / 60.0)

    def check_alerts(self) -> list[str]:
        """Return list of alert messages if thresholds exceeded."""
        alerts = []
        if self.failure_rate > self.FAILURE_RATE_WARN:
            alerts.append(
                f"HIGH FAILURE RATE: {self.failure_rate:.1%} "
                f"({self.validation_failures + self.json_parse_failures}/{self.total_chunks})"
            )
        if self.empty_rate > self.EMPTY_RATE_WARN:
            alerts.append(
                f"HIGH EMPTY RATE: {self.empty_rate:.1%} "
                f"({self.empty_extractions}/{self.successful})"
            )
        return alerts

    def to_prometheus(self) -> str:
        """Export as Prometheus text format for scraping/graphing."""
        lines = [
            "# HELP extraction_total Total chunks processed",
            "# TYPE extraction_total counter",
            f'extraction_total{{backend="{self.backend}"}} {self.total_chunks}',
            "",
            "# HELP extraction_successful Successfully extracted chunks",
            "# TYPE extraction_successful counter",
            f'extraction_successful{{backend="{self.backend}"}} {self.successful}',
            "",
            "# HELP extraction_validation_failures Pydantic validation failures",
            "# TYPE extraction_validation_failures counter",
            f'extraction_validation_failures{{backend="{self.backend}"}} {self.validation_failures}',
            "",
            "# HELP extraction_json_failures JSON parse failures",
            "# TYPE extraction_json_failures counter",
            f'extraction_json_failures{{backend="{self.backend}"}} {self.json_parse_failures}',
            "",
            "# HELP extraction_empty Empty extractions (0 concepts)",
            "# TYPE extraction_empty counter",
            f'extraction_empty{{backend="{self.backend}"}} {self.empty_extractions}',
            "",
            "# HELP extraction_concepts_total Total concepts extracted",
            "# TYPE extraction_concepts_total counter",
            f'extraction_concepts_total{{backend="{self.backend}"}} {self.total_concepts}',
            "",
            "# HELP extraction_relationships_total Total relationships extracted",
            "# TYPE extraction_relationships_total counter",
            f'extraction_relationships_total{{backend="{self.backend}"}} {self.total_relationships}',
            "",
            "# HELP extraction_latency_p50_ms Median latency in milliseconds",
            "# TYPE extraction_latency_p50_ms gauge",
            f'extraction_latency_p50_ms{{backend="{self.backend}"}} {self.latency_p50:.1f}',
            "",
            "# HELP extraction_latency_p95_ms 95th percentile latency in milliseconds",
            "# TYPE extraction_latency_p95_ms gauge",
            f'extraction_latency_p95_ms{{backend="{self.backend}"}} {self.latency_p95:.1f}',
            "",
            "# HELP extraction_throughput_per_min Chunks processed per minute",
            "# TYPE extraction_throughput_per_min gauge",
            f'extraction_throughput_per_min{{backend="{self.backend}"}} {self.throughput_chunks_per_min:.2f}',
        ]
        return "\n".join(lines)

    def summary(self) -> str:
        """Human-readable summary for CLI output."""
        # Handle edge case of no data
        if self.total_chunks == 0:
            return f"\nExtraction Metrics ({self.backend})\n{'='*50}\nNo data collected.\n"

        return f"""
Extraction Metrics ({self.backend})
{'='*50}
Total chunks:      {self.total_chunks}
Successful:        {self.successful}
Failures:          {self.validation_failures + self.json_parse_failures} ({self.failure_rate:.1%})
  - Validation:    {self.validation_failures}
  - JSON parse:    {self.json_parse_failures}
Empty extractions: {self.empty_extractions} ({self.empty_rate:.1%})
Concepts/chunk:    {self.avg_concepts_per_chunk:.1f}
Relations/chunk:   {self.avg_relationships_per_chunk:.1f}
Latency p50:       {self.latency_p50:.0f}ms
Latency p95:       {self.latency_p95:.0f}ms
Throughput:        {self.throughput_chunks_per_min:.2f} chunks/min
Total time:        {self.total_time_seconds:.1f}s
"""

    def save_prometheus(self, path: Path) -> None:
        """Save Prometheus metrics to file.

        Args:
            path: File path to write metrics
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_prometheus())
        logger.info("metrics_saved", path=str(path))
