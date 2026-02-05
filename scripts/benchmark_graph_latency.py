#!/usr/bin/env python3
"""Benchmark graph query latency via daemon socket (JSON-RPC) and direct library calls.

Measures p50/p95/p99 for representative query types:
1. Common method query ("instrumental variables")
2. North Star assumption audit ("double machine learning assumptions")
3. Terse query ("IV")
4. Multi-concept query ("difference-in-differences parallel trends")
5. Method + technique ("propensity score matching")

Outputs:
- Human-readable table to stdout
- JSON to fixtures/benchmarks/graph_latency_YYYY-MM-DD.json

Usage:
    python scripts/benchmark_graph_latency.py
    python scripts/benchmark_graph_latency.py --runs 5  # More runs for stability
"""

import argparse
import asyncio
import json
import os
import socket
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

SOCKET_PATH = f"/tmp/research_kb_daemon_{os.getenv('USER', 'unknown')}.sock"

BENCHMARK_QUERIES = [
    {
        "name": "common_method",
        "description": "Common method query",
        "method": "search",
        "params": {"query": "instrumental variables", "limit": 5},
    },
    {
        "name": "north_star",
        "description": "North Star assumption audit",
        "method": "search",
        "params": {"query": "double machine learning assumptions", "limit": 5, "use_graph": True},
    },
    {
        "name": "terse_query",
        "description": "Terse query",
        "method": "fast_search",
        "params": {"query": "IV", "limit": 5},
    },
    {
        "name": "multi_concept",
        "description": "Multi-concept query",
        "method": "search",
        "params": {
            "query": "difference-in-differences parallel trends",
            "limit": 5,
            "use_graph": True,
        },
    },
    {
        "name": "method_technique",
        "description": "Method + technique",
        "method": "search",
        "params": {"query": "propensity score matching", "limit": 5, "use_graph": True},
    },
    {
        "name": "graph_path",
        "description": "Graph path finding",
        "method": "graph_path",
        "params": {"start": "instrumental variables", "end": "endogeneity", "max_hops": 3},
    },
    {
        "name": "health_check",
        "description": "Health check (baseline)",
        "method": "health",
        "params": {},
    },
]


def send_jsonrpc(method: str, params: dict, request_id: int = 1) -> tuple[dict, float]:
    """Send a JSON-RPC request via Unix socket and return (response, duration_ms).

    Args:
        method: JSON-RPC method name
        params: Method parameters
        request_id: Request ID

    Returns:
        Tuple of (parsed response dict, duration in milliseconds)
    """
    request = json.dumps(
        {"jsonrpc": "2.0", "method": method, "params": params, "id": request_id}
    ).encode("utf-8")

    start = time.monotonic()

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(120.0)  # Graph-boosted queries can take 3-10s
        sock.connect(SOCKET_PATH)
        sock.sendall(request)
        sock.shutdown(socket.SHUT_WR)  # Signal end of request

        # Read response
        chunks = []
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
    finally:
        sock.close()

    duration_ms = (time.monotonic() - start) * 1000
    response = json.loads(b"".join(chunks).decode("utf-8"))
    return response, duration_ms


def percentile(data: list[float], p: float) -> float:
    """Compute percentile using interpolation.

    Args:
        data: Sorted list of values
        p: Percentile (0-100)

    Returns:
        Interpolated percentile value
    """
    if not data:
        return 0.0
    k = (len(data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(data):
        return data[-1]
    return data[f] + (k - f) * (data[c] - data[f])


def run_benchmarks(runs: int = 3) -> dict:
    """Execute all benchmark queries.

    Args:
        runs: Number of runs per query

    Returns:
        Benchmark results dict with per-query statistics
    """
    # Warmup: single health check to establish connection patterns
    try:
        send_jsonrpc("health", {})
    except Exception as e:
        print(f"Error: Cannot connect to daemon at {SOCKET_PATH}: {e}")
        print("Is the daemon running? Start with: research-kb-daemon")
        sys.exit(1)

    results = {}
    print(f"\nRunning {runs} iterations per query...\n")

    for query in BENCHMARK_QUERIES:
        name = query["name"]
        method = query["method"]
        params = query["params"]
        timings: list[float] = []
        errors = 0

        for i in range(runs):
            try:
                response, duration_ms = send_jsonrpc(method, params, request_id=i + 1)
                if "error" in response:
                    errors += 1
                    print(f"  {name} run {i+1}: error - {response['error']['message']}")
                else:
                    timings.append(duration_ms)
            except Exception as e:
                errors += 1
                print(f"  {name} run {i+1}: exception - {e}")

        if timings:
            timings.sort()
            result = {
                "description": query["description"],
                "method": method,
                "runs": runs,
                "errors": errors,
                "p50_ms": round(percentile(timings, 50), 1),
                "p95_ms": round(percentile(timings, 95), 1),
                "p99_ms": round(percentile(timings, 99), 1),
                "min_ms": round(min(timings), 1),
                "max_ms": round(max(timings), 1),
                "mean_ms": round(statistics.mean(timings), 1),
                "stddev_ms": round(statistics.stdev(timings), 1) if len(timings) > 1 else 0.0,
                "raw_ms": [round(t, 1) for t in timings],
            }
        else:
            result = {
                "description": query["description"],
                "method": method,
                "runs": runs,
                "errors": errors,
                "p50_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "min_ms": None,
                "max_ms": None,
                "mean_ms": None,
                "stddev_ms": None,
                "raw_ms": [],
            }

        results[name] = result

    return results


def print_table(results: dict) -> None:
    """Print human-readable benchmark table.

    Args:
        results: Benchmark results dict
    """
    print("=" * 85)
    print(f"{'Query':<35} {'p50':>8} {'p95':>8} {'p99':>8} {'mean':>8} {'errors':>7}")
    print("-" * 85)

    for name, data in results.items():
        p50 = f"{data['p50_ms']:.0f}ms" if data["p50_ms"] is not None else "N/A"
        p95 = f"{data['p95_ms']:.0f}ms" if data["p95_ms"] is not None else "N/A"
        p99 = f"{data['p99_ms']:.0f}ms" if data["p99_ms"] is not None else "N/A"
        mean = f"{data['mean_ms']:.0f}ms" if data["mean_ms"] is not None else "N/A"
        errors = str(data["errors"])

        print(f"{data['description']:<35} {p50:>8} {p95:>8} {p99:>8} {mean:>8} {errors:>7}")

    print("=" * 85)


def save_results(results: dict, output_dir: Path) -> Path:
    """Save benchmark results to JSON.

    Args:
        results: Benchmark results dict
        output_dir: Directory to save JSON file

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_path = output_dir / f"graph_latency_{date_str}.json"

    output = {
        "timestamp": datetime.now().isoformat(),
        "socket_path": SOCKET_PATH,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output_path


def main() -> None:
    """Run benchmarks and output results."""
    parser = argparse.ArgumentParser(description="Benchmark graph query latency via daemon")
    parser.add_argument("--runs", type=int, default=3, help="Runs per query (default: 3)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "fixtures" / "benchmarks",
        help="Output directory for JSON results",
    )
    args = parser.parse_args()

    print(f"Daemon socket: {SOCKET_PATH}")
    print(f"Benchmark queries: {len(BENCHMARK_QUERIES)}")
    print(f"Runs per query: {args.runs}")

    results = run_benchmarks(runs=args.runs)
    print_table(results)

    output_path = save_results(results, args.output_dir)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
