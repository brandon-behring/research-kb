"""Daemon metrics for observability.

Provides Prometheus-compatible metrics for the research-kb daemon:
- Request counts by action
- Latency histograms with quantiles
- Error counts by type
- Cache hit/miss tracking
- Uptime
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DaemonMetrics:
    """Metrics collector for daemon observability."""

    # Counters
    requests_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cache_hits: int = 0
    cache_misses: int = 0

    # Latency tracking (per action)
    latencies: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    # Timestamps
    start_time: float = field(default_factory=time.time)

    def record_request(self, action: str, duration_ms: float) -> None:
        """Record a successful request."""
        self.requests_total[action] += 1
        self.latencies[action].append(duration_ms)

        # Keep only last 1000 samples per action for memory efficiency
        if len(self.latencies[action]) > 1000:
            self.latencies[action] = self.latencies[action][-1000:]

    def record_error(self, error_type: str) -> None:
        """Record an error."""
        self.errors_total[error_type] += 1

    def record_cache_hit(self) -> None:
        """Record embedding cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record embedding cache miss."""
        self.cache_misses += 1

    def get_quantile(self, action: str, quantile: float) -> float | None:
        """Get latency quantile for an action."""
        samples = self.latencies.get(action, [])
        if not samples:
            return None
        sorted_samples = sorted(samples)
        idx = int(len(sorted_samples) * quantile)
        idx = min(idx, len(sorted_samples) - 1)
        return sorted_samples[idx]

    def uptime_seconds(self) -> float:
        """Get daemon uptime in seconds."""
        return time.time() - self.start_time

    def total_requests(self) -> int:
        """Get total request count across all actions."""
        return sum(self.requests_total.values())

    def total_errors(self) -> int:
        """Get total error count across all types."""
        return sum(self.errors_total.values())

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        # Uptime
        lines.append(f"# HELP daemon_uptime_seconds Daemon uptime in seconds")
        lines.append(f"# TYPE daemon_uptime_seconds gauge")
        lines.append(f"daemon_uptime_seconds {self.uptime_seconds():.2f}")
        lines.append("")

        # Request counts
        lines.append(f"# HELP daemon_requests_total Total requests by action")
        lines.append(f"# TYPE daemon_requests_total counter")
        for action, count in self.requests_total.items():
            lines.append(f'daemon_requests_total{{action="{action}"}} {count}')
        lines.append("")

        # Error counts
        lines.append(f"# HELP daemon_errors_total Total errors by type")
        lines.append(f"# TYPE daemon_errors_total counter")
        for error_type, count in self.errors_total.items():
            lines.append(f'daemon_errors_total{{type="{error_type}"}} {count}')
        lines.append("")

        # Latency quantiles
        lines.append(f"# HELP daemon_request_duration_ms Request duration in milliseconds")
        lines.append(f"# TYPE daemon_request_duration_ms summary")
        for action in self.latencies:
            for q, label in [(0.5, "0.5"), (0.95, "0.95"), (0.99, "0.99")]:
                value = self.get_quantile(action, q)
                if value is not None:
                    lines.append(f'daemon_request_duration_ms{{action="{action}",quantile="{label}"}} {value:.2f}')
        lines.append("")

        # Cache metrics
        lines.append(f"# HELP daemon_cache_hits_total Embedding cache hits")
        lines.append(f"# TYPE daemon_cache_hits_total counter")
        lines.append(f"daemon_cache_hits_total {self.cache_hits}")
        lines.append("")

        lines.append(f"# HELP daemon_cache_misses_total Embedding cache misses")
        lines.append(f"# TYPE daemon_cache_misses_total counter")
        lines.append(f"daemon_cache_misses_total {self.cache_misses}")
        lines.append("")

        # Cache hit rate (gauge)
        total_cache = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_cache if total_cache > 0 else 0
        lines.append(f"# HELP daemon_cache_hit_rate Embedding cache hit rate")
        lines.append(f"# TYPE daemon_cache_hit_rate gauge")
        lines.append(f"daemon_cache_hit_rate {hit_rate:.4f}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export metrics as dictionary for JSON endpoints."""
        return {
            "uptime_seconds": self.uptime_seconds(),
            "requests": dict(self.requests_total),
            "errors": dict(self.errors_total),
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0,
            },
            "latency": {
                action: {
                    "p50": self.get_quantile(action, 0.5),
                    "p95": self.get_quantile(action, 0.95),
                    "p99": self.get_quantile(action, 0.99),
                    "count": len(samples),
                }
                for action, samples in self.latencies.items()
            },
        }


# Global metrics instance
metrics = DaemonMetrics()
