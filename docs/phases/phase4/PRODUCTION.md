# Phase 4: Production

**Status**: 📋 **PLANNED**
**Target**: Weeks 7-8

---

## Overview

Phase 4 delivers production infrastructure including FastAPI REST API, authentication, observability, and deployment automation.

---

## Planned Deliverables

### 1. FastAPI REST API

**Endpoints**:

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/search` | Hybrid search with all options |
| GET | `/sources` | List all sources |
| GET | `/sources/{id}` | Source details |
| GET | `/stats` | Database statistics |
| GET | `/concepts/{query}` | Concept search |
| GET | `/graph/{concept}` | Graph neighborhood |
| GET | `/health` | Health check |

**Example Request**:
```bash
curl -X POST http://localhost:8000/search \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "instrumental variables",
    "limit": 10,
    "context_type": "building",
    "use_graph": true,
    "rerank": true
  }'
```

**Files to Create**:
- `packages/api/src/research_kb_api/main.py`
- `packages/api/src/research_kb_api/routes/`
- `packages/api/src/research_kb_api/schemas.py`

---

### 2. Authentication & Rate Limiting

**Options**:
- **Simple**: API key authentication
- **Advanced**: OAuth2/JWT for multi-user

**Rate Limiting**:
| Tier | Requests/hour | Concurrent |
|------|--------------|------------|
| Free | 100 | 5 |
| Standard | 1000 | 20 |
| Premium | 10000 | 100 |

**Implementation**:
- `slowapi` for rate limiting
- Redis for session/token storage (optional)

---

### 3. Observability

**Already Implemented**:
- Structured logging via `structlog`
- OpenTelemetry tracing hooks in `research_kb_common`

**To Add**:
- Prometheus metrics endpoint (`/metrics`)
- Grafana dashboard templates
- Error tracking (Sentry integration)

**Key Metrics**:
- Query latency (p50, p95, p99)
- Search result quality scores
- Embedding server health
- Database connection pool usage

---

### 4. Deployment Automation

**Local (Docker Compose)**:
```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    depends_on: [postgres, embedding-server]

  postgres:
    image: ankane/pgvector:latest
    volumes: [pgdata:/var/lib/postgresql/data]

  embedding-server:
    build: ./packages/pdf-tools
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

**Production (Kubernetes)**:
- Deployment manifests
- Service definitions
- Ingress configuration
- ConfigMaps and Secrets
- HPA for autoscaling

**Files to Create**:
- `docker-compose.prod.yml`
- `kubernetes/` directory with manifests

---

## Success Criteria

| Metric | Target |
|--------|--------|
| API response time | <200ms (p95) |
| Uptime | 99.5% |
| Concurrent users | 100+ |
| Documentation | OpenAPI spec complete |

---

## Previous Phase

← [Phase 3: Enhanced Retrieval](../phase3/ENHANCED_RETRIEVAL.md)
