# Service Level Objectives (SLOs)

This document defines the Service Level Objectives for research-kb, providing clear targets for system reliability and performance.

## Overview

SLOs are internal targets that help maintain service quality. They should be more strict than any external SLAs to provide buffer for improvements and incident response.

## Daemon SLOs

### Availability

| SLO | Target | Measurement | Alert Threshold |
|-----|--------|-------------|-----------------|
| Uptime | 99.9% | `up{job="research_kb_daemon"}` | < 99.5% over 24h |

**Rationale**: The daemon is the primary search interface for Claude Code integration. 99.9% allows ~8.7 hours of downtime per year.

### Latency

| SLO | Target | Measurement | Alert Threshold |
|-----|--------|-------------|-----------------|
| p50 latency | < 500ms | `daemon_request_duration_ms{quantile="0.5"}` | > 1000ms for 5m |
| p95 latency | < 2000ms | `daemon_request_duration_ms{quantile="0.95"}` | > 3000ms for 5m |
| p99 latency | < 5000ms | `daemon_request_duration_ms{quantile="0.99"}` | > 5000ms for 5m |

**Rationale**: Graph-boosted search can take 1-2s for complex queries. p95 target of 2s ensures good user experience for most queries.

### Error Rate

| SLO | Target | Measurement | Alert Threshold |
|-----|--------|-------------|-----------------|
| Error rate | < 0.1% | `rate(daemon_errors_total) / rate(daemon_requests_total)` | > 5% for 5m |

**Error types tracked**:
- `json_parse`: Invalid JSON in request
- `missing_action`: Request missing action field
- `unknown_action`: Unrecognized action
- `internal`: Unhandled exceptions

### Cache Performance

| SLO | Target | Measurement | Alert Threshold |
|-----|--------|-------------|-----------------|
| Cache hit rate | > 80% | `daemon_cache_hit_rate` | < 50% for 15m |

**Rationale**: High cache hit rate reduces embedding computation latency. Low hit rate may indicate memory pressure or workload changes.

## API SLOs (When Running)

### Availability

| SLO | Target | Measurement |
|-----|--------|-------------|
| Uptime | 99.5% | `up{job="research_kb_api"}` |

### Latency

| SLO | Target | Measurement |
|-----|--------|-------------|
| p95 latency | < 1000ms | `histogram_quantile(0.95, http_request_duration_seconds_bucket)` |

### Error Rate

| SLO | Target | Measurement |
|-----|--------|-------------|
| 5xx rate | < 1% | `rate(http_requests_total{status=~"5.."}) / rate(http_requests_total)` |

## Extraction SLOs

### Quality

| SLO | Target | Measurement | Alert Threshold |
|-----|--------|-------------|-----------------|
| Failure rate | < 5% | `extraction_failure_rate` | > 5% for 10m |
| Empty rate | < 15% | `extraction_empty_rate` | > 15% for 10m |

**Rationale**: Some chunks (tables, figures) naturally produce empty extractions. 15% empty rate is acceptable for mixed-content PDFs.

### Throughput

| SLO | Target | Measurement |
|-----|--------|-------------|
| Throughput | > 30 chunks/min | `extraction_throughput_per_min` |

**Note**: Throughput depends on hardware. Target assumes NVIDIA GPU with 8GB+ VRAM and concurrency=2.

## Error Budgets

Error budgets represent the amount of unreliability allowed before SLO violation:

| Service | SLO | Monthly Budget |
|---------|-----|----------------|
| Daemon availability | 99.9% | 43.8 minutes |
| API availability | 99.5% | 3.6 hours |
| Error rate | 0.1% | 1 in 1000 requests |

## Measurement Windows

- **Real-time alerts**: 1-5 minute windows for immediate issues
- **SLO tracking**: 30-day rolling window for trend analysis
- **Error budget**: Calendar month reset

## Monitoring Stack

```
┌─────────────────┐     ┌─────────────────┐
│  Daemon :9001   │────▶│   Prometheus    │
│  /metrics       │     │    :9090        │
└─────────────────┘     └────────┬────────┘
                                 │
┌─────────────────┐              ▼
│   API :8000     │────▶┌─────────────────┐
│  /metrics       │     │    Grafana      │
└─────────────────┘     │    :3000        │
                        └─────────────────┘
```

## Usage

Start monitoring stack:
```bash
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

Access:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## Related Documentation

- [CLAUDE.md](../CLAUDE.md) - CLI usage and operational commands
- [monitoring/prometheus/alerts.yml](../monitoring/prometheus/alerts.yml) - Alert rule definitions
- [monitoring/grafana/dashboards/research_kb.json](../monitoring/grafana/dashboards/research_kb.json) - Grafana dashboard
