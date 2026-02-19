"""Manning Library ingestion system.

Catalog-centric architecture: manning_catalog.yaml is the single source of truth.
Files stay where they are (Documents/ for Manning, fixtures/ for legacy).

Usage:
    python -m scripts.manning catalog --generate
    python -m scripts.manning audit
    python -m scripts.manning cleanup --fix-domains
    python -m scripts.manning ingest --tier 1 --quiet
"""
