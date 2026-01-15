"""Locust load testing for research-kb daemon.

Tests the daemon's Unix socket JSON-RPC 2.0 interface under load.

Usage:
    # Interactive mode (opens web UI at http://localhost:8089)
    locust -f tests/load/locustfile.py

    # Headless mode for CI (50 users, ramp 10/sec, run 2 minutes)
    locust -f tests/load/locustfile.py --headless -u 50 -r 10 --run-time 2m

    # Custom socket path
    SOCKET_PATH=/tmp/research_kb_daemon_custom.sock locust -f tests/load/locustfile.py

Targets (Phase 2):
    - p50 < 50ms
    - p99 < 200ms
    - Throughput > 100 req/s
"""

import json
import os
import random
import socket
import time
from typing import Any, Optional

from locust import User, task, between, events

# Sample queries for load testing (representative causal inference topics)
SAMPLE_QUERIES = [
    "instrumental variables",
    "double machine learning",
    "propensity score matching",
    "difference in differences",
    "regression discontinuity",
    "synthetic control",
    "causal forest",
    "LATE local average treatment effect",
    "unconfoundedness assumption",
    "selection bias",
    "backdoor criterion",
    "DAG directed acyclic graph",
    "potential outcomes",
    "treatment effect",
    "heterogeneous effects",
    "cross-fitting",
    "Neyman orthogonality",
    "overlap assumption",
    "parallel trends",
    "IV exclusion restriction",
]


class UnixSocketClient:
    """JSON-RPC 2.0 client for Unix sockets."""

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self._request_id = 0

    def _send_request(
        self, method: str, params: Optional[dict[str, Any]] = None
    ) -> tuple[dict, float]:
        """Send JSON-RPC 2.0 request and return (response, latency_ms)."""
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self._request_id,
        }

        start_time = time.time()

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(5.0)  # 5 second timeout
            sock.connect(self.socket_path)

            # Send request
            request_bytes = json.dumps(request).encode("utf-8")
            sock.sendall(request_bytes)

            # Receive response
            response_data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk
                # Check for complete JSON
                try:
                    json.loads(response_data.decode("utf-8"))
                    break
                except json.JSONDecodeError:
                    continue

            latency_ms = (time.time() - start_time) * 1000
            response = json.loads(response_data.decode("utf-8"))

            return response, latency_ms

        finally:
            sock.close()


class DaemonUser(User):
    """Locust user that makes requests to the research-kb daemon."""

    # Wait 100-500ms between requests
    wait_time = between(0.1, 0.5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        socket_path = os.environ.get(
            "SOCKET_PATH",
            f"/tmp/research_kb_daemon_{os.environ.get('USER', 'unknown')}.sock"
        )
        self.client = UnixSocketClient(socket_path)

    def _report_result(
        self,
        request_type: str,
        name: str,
        response_time: float,
        response_length: int,
        exception: Optional[Exception] = None,
    ):
        """Report request result to Locust."""
        if exception:
            events.request.fire(
                request_type=request_type,
                name=name,
                response_time=response_time,
                response_length=response_length,
                exception=exception,
            )
        else:
            events.request.fire(
                request_type=request_type,
                name=name,
                response_time=response_time,
                response_length=response_length,
                exception=None,
            )

    @task(10)  # Weight: 10x more common than other tasks
    def search_query(self):
        """Perform a search query."""
        query = random.choice(SAMPLE_QUERIES)
        try:
            response, latency_ms = self.client._send_request(
                "search",
                {"query": query, "limit": 5}
            )

            if "error" in response:
                self._report_result(
                    "search", "search",
                    latency_ms, 0,
                    Exception(response["error"].get("message", "Unknown error"))
                )
            else:
                result_count = len(response.get("result", {}).get("results", []))
                self._report_result(
                    "search", "search",
                    latency_ms, result_count,
                )

        except Exception as e:
            self._report_result("search", "search", 0, 0, e)

    @task(3)  # Weight: 3x
    def health_check(self):
        """Check daemon health."""
        try:
            response, latency_ms = self.client._send_request("health")

            if "error" in response:
                self._report_result(
                    "health", "health",
                    latency_ms, 0,
                    Exception(response["error"].get("message", "Unknown error"))
                )
            else:
                self._report_result("health", "health", latency_ms, 1)

        except Exception as e:
            self._report_result("health", "health", 0, 0, e)

    @task(2)  # Weight: 2x
    def stats_check(self):
        """Get database stats."""
        try:
            response, latency_ms = self.client._send_request("stats")

            if "error" in response:
                self._report_result(
                    "stats", "stats",
                    latency_ms, 0,
                    Exception(response["error"].get("message", "Unknown error"))
                )
            else:
                self._report_result("stats", "stats", latency_ms, 1)

        except Exception as e:
            self._report_result("stats", "stats", 0, 0, e)


# Event hooks for custom reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Print test configuration at start."""
    print("\n" + "=" * 60)
    print("Research-KB Daemon Load Test")
    print("=" * 60)
    socket_path = os.environ.get(
        "SOCKET_PATH",
        f"/tmp/research_kb_daemon_{os.environ.get('USER', 'unknown')}.sock"
    )
    print(f"Socket: {socket_path}")
    print(f"Sample queries: {len(SAMPLE_QUERIES)}")
    print("=" * 60 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print summary at end."""
    stats = environment.stats
    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)

    # Check targets
    total_requests = stats.total.num_requests
    total_failures = stats.total.num_failures

    if total_requests > 0:
        failure_rate = total_failures / total_requests * 100
        p50 = stats.total.get_response_time_percentile(0.50)
        p99 = stats.total.get_response_time_percentile(0.99)

        print(f"Total requests: {total_requests}")
        print(f"Failures: {total_failures} ({failure_rate:.1f}%)")
        print(f"p50 latency: {p50:.0f}ms (target: <50ms)")
        print(f"p99 latency: {p99:.0f}ms (target: <200ms)")

        # Check targets
        targets_met = True
        if p50 > 50:
            print("  ✗ p50 exceeds target")
            targets_met = False
        else:
            print("  ✓ p50 meets target")

        if p99 > 200:
            print("  ✗ p99 exceeds target")
            targets_met = False
        else:
            print("  ✓ p99 meets target")

        if failure_rate > 1:
            print("  ✗ Failure rate > 1%")
            targets_met = False
        else:
            print("  ✓ Failure rate acceptable")

        print("=" * 60)
        if targets_met:
            print("✓ ALL TARGETS MET")
        else:
            print("✗ SOME TARGETS MISSED")
        print("=" * 60 + "\n")
