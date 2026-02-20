# Scripts Reference

Active scripts for the research-kb system. For archived (obsolete) scripts, see `archive/`.

## Ingestion

| Script | Purpose | Dependencies |
|--------|---------|--------------|
| `ingest_corpus.py` | Main corpus ingestion pipeline | DB, embed_server |
| `ingest_missing_textbooks.py` | Ingest textbooks not yet in DB | DB, embed_server |
| `ingest_missing_papers.py` | Ingest papers not yet in DB | DB, embed_server |
| `ingest_arxiv_papers.py` | Ingest from arXiv by ID | DB, embed_server |
| `ingest_mmm_papers.py` | Ingest MMM conference papers | DB, embed_server |
| `ingest_interview_prep.py` | Ingest interview prep materials | DB, embed_server |
| `ingest_rag_llm_textbooks.py` | Ingest RAG/LLM textbooks | DB, embed_server |
| `process_ingestion_queue.py` | Process queued ingestion jobs | DB, embed_server |
| `retry_dlq.py` | Retry failed ingestion from DLQ | DB, embed_server |

## Extraction

| Script | Purpose | Dependencies |
|--------|---------|--------------|
| `extract_concepts.py` | Extract concepts from chunks via LLM | DB, Ollama or Anthropic |
| `extract_citations.py` | Extract citation links between sources | DB |
| `deduplicate_concepts.py` | Merge duplicate concepts | DB |

## Knowledge Graph

| Script | Purpose | Dependencies |
|--------|---------|--------------|
| `sync_kuzu.py` | Sync PostgreSQL concepts/rels to KuzuDB | DB |
| `build_citation_graph.py` | Build citation authority graph | DB |
| `compute_bibliographic_coupling.py` | Compute biblio-coupling scores | DB |
| `discover_cross_domain_links.py` | Find cross-domain concept links | DB |
| `link_interview_to_concepts.py` | Link interview prep to KB concepts | DB |

## Evaluation & Validation

| Script | Purpose | Dependencies |
|--------|---------|--------------|
| `eval_retrieval.py` | Evaluate search retrieval quality | DB, embed_server |
| `eval_assumptions.py` | Evaluate assumption audit quality | DB |
| `eval_interview_readiness.py` | Score KB readiness for interview prep | DB |
| `eval_scoring_methods.py` | Compare search scoring strategies | DB |
| `validate_known_answers.py` | Validate against known-good queries | DB, embed_server |
| `validate_seed_concepts.py` | Validate seed concept extraction | DB |
| `validate_glossary.py` | Cross-validate glossary terms with KB | DB |
| `validate_proactive_context.py` | Validate proactive context retrieval | DB |
| `run_quality_checks.py` | Run data quality metrics | DB |
| `benchmark_backends.py` | Benchmark extraction backends | Ollama |
| `benchmark_graph_latency.py` | Benchmark KuzuDB graph queries | DB |

## Maintenance & Repair

| Script | Purpose | Dependencies |
|--------|---------|--------------|
| `diagnose_data_quality.py` | Diagnose data quality issues | DB |
| `diagnose_orphans.py` | Find orphaned chunks/concepts | DB |
| `repair_orphan_sources.py` | Re-link orphaned sources | DB |
| `relink_orphans.py` | Re-link orphaned chunks | DB |
| `dedup_chunks.py` | Deduplicate chunks | DB |
| `identify_ocr_needed.py` | Find PDFs needing OCR | DB |

## Domain Management

| Script | Purpose | Dependencies |
|--------|---------|--------------|
| `add_missing_domains.py` | Add domain tags to untagged sources | DB |
| `check_domains.py` | Report domain coverage | DB |
| `tag_untagged_sources.py` | Tag sources without domain metadata | DB |
| `sync_chunk_domains.py` | Sync domain tags from sources to chunks | DB |

## Discovery

| Script | Purpose | Dependencies |
|--------|---------|--------------|
| `s2_auto_discover.py` | Auto-discover papers via Semantic Scholar | S2 API |
| `discover_golden_candidates.py` | Find candidate golden dataset papers | DB |
| `build_golden_dataset.py` | Build golden evaluation dataset | DB |
| `generate_golden_dataset.py` | Generate golden query/answer pairs | DB |

## Method Enrichment

| Script | Purpose | Dependencies |
|--------|---------|--------------|
| `populate_assumption_cache.py` | Cache assumptions for top methods | DB, Ollama or Anthropic |
| `enrich_method_attributes.py` | Enrich method metadata | DB |
| `process_method_enrichment.py` | Process method enrichment queue | DB |

## Documentation & Dev Tools

| Script | Purpose | Dependencies |
|--------|---------|--------------|
| `generate_status.py` | Generate current status report | DB |
| `generate_package_docs.py` | Auto-generate package documentation | Filesystem |
| `audit_docs.py` | Detect documentation drift | Filesystem |
| `tag_tests.py` | Auto-tag test files with pytest markers | Filesystem |
| `test_mcp_queries.py` | Test MCP server queries | DB |

## Shell Scripts

| Script | Purpose | Dependencies |
|--------|---------|--------------|
| `backup_db.sh` | Backup PostgreSQL database | DB |
| `test_backup_restore.sh` | Test backup/restore cycle | DB |
| `docker-safe.sh` | Safe wrapper for docker compose | Docker |
| `install_daemon.sh` | Install daemon systemd service | Systemd |
| `install_reranker.sh` | Install reranker service | pip |
| `install_s2_discovery.sh` | Install S2 discovery tools | pip |
| `configure_ollama_perf.sh` | Configure Ollama performance | Ollama |
| `run_local_validation.sh` | Run full local validation suite | DB, embed_server |

## Subpackages

### `archive/`
Archived scripts from earlier development phases.

### `systemd/`
Systemd service/timer definitions.
- `research-kb-sync.service` — KuzuDB sync service
- `research-kb-sync.timer` — Daily sync timer (3am)
- `research-kb-reranker.service` — Reranker model service

## Archived

15 scripts moved to `archive/` (Phase G). These were one-time utilities, superseded pipelines, or deprecated backends. See git history for original context.
