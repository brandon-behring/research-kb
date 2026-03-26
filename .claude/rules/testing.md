---
paths:
  - "tests/**"
  - "**/tests/**"
  - "conftest.py"
---

# Testing & CI/CD

## Test Commands by Marker

```bash
pytest -m "unit"                    # ~1,570 tests (fast, mocked)
pytest -m "integration"             # ~343 tests (needs PostgreSQL)
pytest -m "requires_embedding"      # ~19 tests (needs embed_server)
pytest -m "requires_reranker"       # ~20 tests (needs reranker model)
pytest -m "requires_ollama"         # ~32 tests (needs Ollama)

# CI-safe (excludes all service-dependent tests)
pytest -m "unit and not requires_embedding and not requires_ollama and not requires_reranker and not requires_grobid"
pytest -m "integration and not requires_embedding and not requires_ollama and not requires_reranker"

# By package
pytest packages/cli/tests/ -v
pytest packages/storage/tests/ -v
pytest packages/pdf-tools/tests/ -v
pytest packages/extraction/tests/ -v
```

## Test Patterns

- All tests use `pytest-asyncio` with `asyncio_mode = auto`
- Function-scoped event loops
- Mock fixtures: `mock_ollama`, `mock_embedding_client`
- Float comparisons: use `pytest.approx(value, rel=1e-5)`

## CI/CD Tiers

1. **PR Checks** (<10 min): Unit + integration with mocked services, pytest-cov coverage (XML), doc freshness gate
2. **Manual Integration** (15 min, `workflow_dispatch`): Search pipeline + quality + script tests + doc freshness (`audit_docs.py`, `generate_status.py --check`)
3. **Full Rebuild** (45 min, `workflow_dispatch`): Demo data load, embedding generation, retrieval eval against YAML test cases with per-domain metrics
