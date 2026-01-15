# Load Testing for Research-KB Daemon

This directory contains Locust-based load tests for the research-kb daemon Unix socket interface.

## Requirements

```bash
pip install locust
```

## Usage

### Interactive Mode (Web UI)

```bash
locust -f tests/load/locustfile.py
```

Then open http://localhost:8089 in your browser.

### Headless Mode (CI)

```bash
# 50 concurrent users, ramp 10/sec, run for 2 minutes
locust -f tests/load/locustfile.py --headless -u 50 -r 10 --run-time 2m
```

### Custom Socket Path

```bash
SOCKET_PATH=/tmp/research_kb_daemon_custom.sock locust -f tests/load/locustfile.py
```

## Targets (Phase 2)

| Metric | Target |
|--------|--------|
| p50 latency | < 50ms |
| p99 latency | < 200ms |
| Failure rate | < 1% |
| Throughput | > 100 req/s |

## Test Scenarios

The load test includes weighted tasks:

| Task | Weight | Description |
|------|--------|-------------|
| search | 10 | Hybrid search queries (most common) |
| health | 3 | Health check endpoint |
| stats | 2 | Database statistics |

## Sample Queries

20 representative causal inference queries are used:
- Instrumental variables
- Double machine learning
- Propensity score matching
- Difference-in-differences
- Regression discontinuity
- And more...

## Interpreting Results

The test summary shows:
- Total requests and failure count
- p50 and p99 latency vs targets
- Pass/fail status for each target

Example output:
```
============================================================
LOAD TEST SUMMARY
============================================================
Total requests: 5000
Failures: 12 (0.2%)
p50 latency: 25ms (target: <50ms)
p99 latency: 145ms (target: <200ms)
  ✓ p50 meets target
  ✓ p99 meets target
  ✓ Failure rate acceptable
============================================================
✓ ALL TARGETS MET
============================================================
```
