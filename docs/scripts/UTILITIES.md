# Script Utilities Reference

> Categorized index of the ~60 scripts in `scripts/`. Each key script includes purpose, usage, and important flags. One-off scripts live in `scripts/archive/`.

---

## Ingestion

### `ingest_corpus.py`

Ingests Phase 1 corpus: 2 textbooks (Pearl, Angrist/Pischke) and 12 arXiv papers. Reports total chunk count and validates ~500 target.

```bash
python scripts/ingest_corpus.py
python scripts/ingest_corpus.py --quiet    # Errors + summary only
```

### `ingest_missing_textbooks.py`

Scans `fixtures/textbooks/` for PDFs, checks which are already ingested (by file hash), and ingests missing ones with auto-extracted metadata.

```bash
python scripts/ingest_missing_textbooks.py
python scripts/ingest_missing_textbooks.py --quiet          # Minimal output
python scripts/ingest_missing_textbooks.py --quiet --json   # JSON report
```

### `ingest_missing_papers.py`

Scans `fixtures/papers/` for PDFs, checks which are already ingested (by file hash), and ingests missing ones with auto-extracted metadata.

```bash
python scripts/ingest_missing_papers.py
```

### `ingest_arxiv_papers.py`

Ingests papers from `fixtures/papers/` subdirectories (arxiv, acquired) not covered by `ingest_missing_papers.py`.

```bash
python scripts/ingest_arxiv_papers.py
```

### `embed_missing.py`

Generates embeddings for chunks that lack them. Used after loading demo data from fixtures. Requires the embedding server to be running.

```bash
python -m research_kb_pdf.embed_server &   # Start embed server first
python scripts/embed_missing.py
python scripts/embed_missing.py --batch 64 --limit 1000
```

### `load_demo_data.py`

Fast-path demo setup: loads pre-extracted data from JSON fixtures directly into PostgreSQL.

```bash
python scripts/load_demo_data.py
python scripts/load_demo_data.py --data-dir /path/to/fixtures --domain your_domain
```

### `setup_demo.py`

Full demo pipeline orchestrator: download, ingest, and optional concept extraction.

```bash
python scripts/setup_demo.py
python scripts/setup_demo.py --skip-download --extract
```

---

## Extraction

### `extract_concepts.py`

Extracts concepts from text chunks using LLM, deduplicates by canonical name, stores relationships and chunk-concept links, and optionally syncs to KuzuDB.

```bash
# Ollama (GPU, fast)
python scripts/extract_concepts.py --backend ollama --model llama3.1:8b --limit 1000

# Anthropic (higher quality)
python scripts/extract_concepts.py --backend anthropic --model haiku --concurrency 4

# Other flags
python scripts/extract_concepts.py --source-id <uuid> --skip-sync --dry-run
```

### `sync_kuzu.py`

Exports concepts and relationships from PostgreSQL and loads them into KuzuDB for fast graph traversal.

```bash
python scripts/sync_kuzu.py                          # Full sync
python scripts/sync_kuzu.py --incremental             # Delta only
python scripts/sync_kuzu.py --verify-only             # Check consistency
python scripts/sync_kuzu.py --kuzu-path /custom/path  # Custom KuzuDB location
```

### `extract_citations.py`

Processes PDFs with GROBID for each source without citations, stores extracted citation records.

```bash
python scripts/extract_citations.py
python scripts/extract_citations.py --dry-run --limit 10 --type paper
```

### `build_citation_graph.py`

Matches extracted citations to corpus sources, creates `source_citations` edges, and computes PageRank-style authority scores.

```bash
python scripts/build_citation_graph.py
python scripts/build_citation_graph.py --skip-pagerank
```

### `populate_assumption_cache.py`

Pre-populates `method_assumption_cache` for top interview methods using Anthropic Haiku (or Ollama).

```bash
python scripts/populate_assumption_cache.py
python scripts/populate_assumption_cache.py --backend ollama --method "IV" --dry-run
```

---

## Evaluation

### `eval_retrieval.py`

Evaluates retrieval quality using Hit Rate@K, MRR, NDCG@K, and Concept Recall against the golden dataset (55 YAML test cases).

```bash
python scripts/eval_retrieval.py
python scripts/eval_retrieval.py --per-domain --verbose
python scripts/eval_retrieval.py --tag causal_inference --output results.json
```

### `run_quality_checks.py`

Runs all quality validation tests and generates comprehensive reports.

```bash
python scripts/run_quality_checks.py
python scripts/run_quality_checks.py --strict --json --output report.json
```

### `validate_known_answers.py`

Tests search quality by running curated queries with known expected results from the golden dataset.

```bash
python scripts/validate_known_answers.py
```

### `eval_assumptions.py`

Evaluates assumption auditing (North Star feature) quality against golden dataset using recall, precision, F1, and importance-weighted metrics.

```bash
python scripts/eval_assumptions.py
python scripts/eval_assumptions.py --method "IV" --verbose
```

### `validate_seed_concepts.py`

Validates concept extraction quality against seed concepts using exact, fuzzy, and semantic matching strategies.

```bash
python scripts/validate_seed_concepts.py
python scripts/validate_seed_concepts.py --type method --confidence 0.8 --no-semantic
```

### `build_golden_dataset.py`

Uses SQL + FTS to find target chunks matching specific queries within target domains. Prefer `discover_golden_candidates.py` for new work.

```bash
python scripts/build_golden_dataset.py
```

### `discover_golden_candidates.py`

Discovers candidate chunks for expanding the golden eval dataset via FTS or hybrid search.

```bash
python scripts/discover_golden_candidates.py --domain causal_inference
python scripts/discover_golden_candidates.py --hybrid --dry-run
```

---

## Maintenance

### `audit_docs.py`

Documentation drift detector: runs 12 prioritized checks (CLI commands, packages, paths, backends, API routes, freshness, auto-generated sections).

```bash
python scripts/audit_docs.py                # Full audit
python scripts/audit_docs.py --pre-commit   # Fast checks only (used by pre-commit hook)
python scripts/audit_docs.py --ci           # Exit 2 on failure (used in CI)
```

### `generate_status.py`

Generates `docs/status/CURRENT_STATUS.md` from actual database state.

```bash
python scripts/generate_status.py           # Update CURRENT_STATUS.md
python scripts/generate_status.py --check   # Check freshness (CI gate)
python scripts/generate_status.py --stdout  # Print to stdout
```

### `generate_readme_sections.py`

Auto-generates README sections marked with `AUTO-GEN` markers from source code introspection (CLI commands, MCP tools, packages).

```bash
python scripts/generate_readme_sections.py
python scripts/generate_readme_sections.py --check     # Verify freshness
python scripts/generate_readme_sections.py --dry-run   # Preview changes
```

### `generate_package_docs.py`

Auto-generates package READMEs from code (FastAPI routes, Streamlit pages).

```bash
python scripts/generate_package_docs.py
```

### `backup_db.sh`

Backs up PostgreSQL database to timestamped file in `backups/` (last 5 kept).

```bash
./scripts/backup_db.sh                     # Standard backup
./scripts/backup_db.sh --pre-extraction    # Before extraction runs
./scripts/backup_db.sh latest              # Show latest backup path
./scripts/backup_db.sh --path-only         # Print path without backing up
```

### `docker-safe.sh`

Safe docker compose wrapper that intercepts destructive operations (`down -v`, `volume rm`, `system prune`) with data count warnings and confirmation.

```bash
alias dc='./scripts/docker-safe.sh'
dc up -d       # Works normally
dc down -v     # Warns, shows data counts, requires 'DELETE' confirmation
```

### `install_daemon.sh`

Installs `research-kb-daemon` as a user systemd service.

```bash
./scripts/install_daemon.sh install     # Install + enable
./scripts/install_daemon.sh uninstall   # Remove service
./scripts/install_daemon.sh status      # Check status
```

### `tag_tests.py`

Analyzes test files and applies pytest markers (`unit`, `integration`, `requires_embedding`, etc.) based on directory structure and content patterns.

```bash
python scripts/tag_tests.py --verbose   # Show proposed changes
python scripts/tag_tests.py --apply     # Apply markers
```

### `mypy_baseline_check.py`

Enforces mypy baseline by blocking new type errors while celebrating fixes. Compares current mypy output against `.mypy_baseline.txt`.

```bash
python scripts/mypy_baseline_check.py            # Check against baseline
python scripts/mypy_baseline_check.py --update   # Update baseline file
```

---

## Data Quality

### `dedup_chunks.py`

Removes same-source, same-`content_hash` duplicate chunks, keeping earliest version and reassigning `chunk_concepts` to keeper.

```bash
python scripts/dedup_chunks.py --verbose   # Preview
python scripts/dedup_chunks.py --apply     # Execute deduplication
```

### `deduplicate_concepts.py`

Finds concepts where one is plural of another (e.g., "eigenvalue" vs "eigenvalues") and merges, keeping singular as canonical.

```bash
python scripts/deduplicate_concepts.py --limit 100   # Preview
python scripts/deduplicate_concepts.py --execute      # Apply merges
```

### `diagnose_data_quality.py`

Read-only diagnostic: checks chunk duplicates, source near-duplicates (pg_trgm similarity), and golden chunk domain verification.

```bash
python scripts/diagnose_data_quality.py
```

### `diagnose_orphans.py`

Diagnoses orphan concepts (0 relationships) and categorizes as salvageable (has `chunk_concepts`) or dead (no chunk links, low confidence).

```bash
python scripts/diagnose_orphans.py
python scripts/diagnose_orphans.py --output orphans.json
```

### `relink_orphans.py`

Re-links orphan concepts using semantic similarity to connect isolated concepts back into the knowledge graph with `RELATED_TO` edges.

```bash
python scripts/relink_orphans.py --threshold 0.85 --limit 500   # Preview
python scripts/relink_orphans.py --threshold 0.85 --execute      # Apply
```

### `repair_orphan_sources.py`

Fixes sources with 0 chunks by re-running embedding on sources that didn't complete chunking/embedding due to server errors.

```bash
python scripts/repair_orphan_sources.py
```

---

## Discovery

### `s2_auto_discover.py`

Main orchestrator for S2 Auto-Discovery Pipeline: discovers papers via Semantic Scholar API and queues for ingestion.

```bash
python scripts/s2_auto_discover.py search "double machine learning"
python scripts/s2_auto_discover.py topics
python scripts/s2_auto_discover.py author "Chernozhukov"
python scripts/s2_auto_discover.py traverse --domain causal_inference
```

### `discover_cross_domain_links.py`

Discovers semantically similar concepts across knowledge domains using embedding similarity and stores high-confidence matches.

```bash
python scripts/discover_cross_domain_links.py --dry-run
python scripts/discover_cross_domain_links.py --threshold 0.9 --full
```

---

## Benchmarking

### `benchmark_backends.py`

Benchmarks extraction backend performance (Ollama vs Anthropic) for concept extraction throughput.

```bash
python scripts/benchmark_backends.py
```

### `benchmark_graph_latency.py`

Benchmarks KuzuDB graph query latency (cold/warm) for graph_path, neighborhood, and scoring operations.

```bash
python scripts/benchmark_graph_latency.py
```

### `eval_scoring_methods.py`

Evaluates different scoring methods (weighted sum vs RRF) on the golden dataset.

```bash
python scripts/eval_scoring_methods.py
```

---

## Other Utilities

| Script | Purpose |
|--------|---------|
| `add_missing_domains.py` | Register missing domains in PostgreSQL |
| `check_domains.py` | Verify domain registration and source counts |
| `compute_bibliographic_coupling.py` | Compute bibliographic coupling scores between sources |
| `configure_ollama_perf.sh` | Apply Ollama performance settings (flash attention, parallel streams) |
| `download_demo_corpus.sh` | Download arXiv papers for demo corpus |
| `enrich_method_attributes.py` | Enrich method concepts with structured attributes |
| `export_demo_data.py` | Export demo data to JSON fixtures |
| `generate_golden_dataset.py` | Generate golden evaluation dataset |
| `identify_ocr_needed.py` | Identify PDFs that need OCR preprocessing |
| `ingest_mmm_papers.py` | Ingest marketing mix modeling papers |
| `ingest_rag_llm_textbooks.py` | Ingest RAG/LLM textbooks batch |
| `install_reranker.sh` | Install BGE reranker model |
| `install_s2_discovery.sh` | Install S2 discovery dependencies |
| `install_services.sh` | Install all systemd services |
| `process_ingestion_queue.py` | Process queued ingestion jobs |
| `process_method_enrichment.py` | Process method enrichment pipeline |
| `retry_dlq.py` | Retry items from dead-letter queue |
| `run_local_validation.sh` | Run full local validation suite |
| `sync_chunk_domains.py` | Sync domain tags from sources to chunks |
| `tag_untagged_sources.py` | Auto-tag sources missing domain metadata |
| `test_backup_restore.sh` | Test backup and restore procedure |
| `test_mcp_queries.py` | Test MCP server query responses |
| `validate_glossary.py` | Validate glossary terms against KB concepts |

---

## Archive

One-off scripts in `scripts/archive/`. These were used during specific phases and are preserved for reference. See individual file headers for context.

| Script | Original Purpose |
|--------|-----------------|
| `ab_test_embeddings.py` | A/B test different embedding models |
| `backfill_concept_embeddings.py` | Backfill embeddings for existing concepts |
| `build_llama_cpp_cuda.sh` | Build llama.cpp with CUDA support |
| `compare_haiku_models.py` | Compare Haiku model versions for extraction |
| `compare_search_configs.py` | Compare search weight configurations |
| `cross_reference_available.py` | Cross-reference available papers |
| `download_gguf_model.sh` | Download GGUF model files |
| `eval_phase3.py` | Phase 3 retrieval evaluation |
| `extract_top10.sh` | Extract top 10 results for analysis |
| `ingest_golden_pdfs.py` | Ingest golden evaluation PDFs |
| `install_llama_cpp_cuda.sh` | Install llama.cpp CUDA build |
| `master_plan_validation.py` | Phase 2 master plan validation |
| `validate_gold_eval.py` | Validate golden evaluation dataset |
| `validate_proactive_context.py` | Validate ProactiveContext integration |
