---
paths:
  - "scripts/**"
  - "packages/cli/**"
---

# CLI Usage & Script Operations

## Search and Retrieval

```bash
# 3-way default: FTS + vector + citation; graph disabled pending re-extraction
research-kb search query "instrumental variables"            # Default (FTS+vector+citation)
research-kb search query "test" --no-citations               # Without citation authority
research-kb search query "IV" --citation-weight 0.25         # Boost citation influence
research-kb search query "IV" --context building             # Context-tuned weights
research-kb search query "IV" --graph                        # Enable graph signal (if KG current)
```

## Source Management

```bash
research-kb sources list                                     # List sources
research-kb sources stats                                    # Database statistics
research-kb sources extraction-status                        # Extraction pipeline stats
```

## Knowledge Graph

```bash
research-kb graph concepts "IV"                              # Concept search
research-kb graph neighborhood "DML" --hops 2                # Graph exploration
research-kb graph path "IV" "unconfoundedness"               # Shortest path
research-kb graph explain "DML" "cross-fitting"              # Explain with evidence + synthesis
research-kb graph explain "IV" "endogeneity" --style research
research-kb graph explain "DML" "overlap" --no-llm           # Graph + evidence only
```

## Citation Network

```bash
research-kb citations list <source>                          # List citations
research-kb citations cited-by <source>                      # Sources citing this one
research-kb citations cites <source>                         # Sources this one cites
research-kb citations stats                                  # Corpus citation statistics
research-kb citations similar <source>                       # Similar by shared refs
```

## Assumption Auditing (North Star Feature)

```bash
research-kb search audit-assumptions "double machine learning"
research-kb search audit-assumptions "IV" --no-ollama           # Graph only
research-kb search audit-assumptions "DML" --format json
research-kb search audit-assumptions "RDD" --domain time_series --scope applied
```

## Literature Review

```bash
research-kb review generate "double machine learning"                     # Educational
research-kb review generate "instrumental variables" --style research     # Research-style
research-kb review generate "causal forests" --no-llm --format json       # Graph+search only
```

## Semantic Scholar Discovery

```bash
research-kb discover search "double machine learning"
research-kb discover topics
research-kb discover author "Chernozhukov"
research-kb enrich citations                           # Enrich with S2 metadata
research-kb enrich status
research-kb enrich job-status
```

## Data Operations

```bash
python scripts/ingest_corpus.py                  # Ingest corpus
python scripts/extract_concepts.py --limit 1000  # Extract concepts (requires Ollama)
python scripts/eval_retrieval.py                 # Validate retrieval quality
python scripts/eval_retrieval.py --use-citations # Validate with citation signal
python scripts/run_quality_checks.py             # Quality metrics
```

## Ingestion: Three-Phase GPU Workflow

**CRITICAL: Docling and embed_server CANNOT share the RTX 2070 GPU simultaneously.**
Combined VRAM (~6GB) exceeds available memory and causes OOM/system freeze.

All ingestion scripts default to `--no-embed` (extraction only). Use the three-phase workflow:

```bash
# Phase 1: EXTRACT (Docling on GPU, embed_server OFF)
kill $(pgrep -f embed_server)              # Free GPU for Docling
python scripts/ingest_missing_textbooks.py  # --no-embed is default

# Phase 2: EMBED (embed_server on GPU, Docling done)
python -m research_kb_pdf.embed_server &    # Start embed_server
python scripts/backfill_embeddings.py --batch-size 8

# Phase 3: CITATIONS (CPU only)
python scripts/build_citation_graph.py      # Match citations + PageRank
```

**Scripts with `--no-embed` default (safe):**
- `ingest_missing_textbooks.py` — textbooks from fixtures/textbooks/
- `ingest_missing_papers.py` — papers from fixtures/papers/
- `ingest_healthcare_reference_cache.py` — healthcare markdown/PDFs
- `rechunk_corpus.py` — re-extract existing sources
- `mass_ingest_catalog.py` — bulk catalog ingestion

**To override (dangerous):** Pass `--with-embed` — script will abort if embed_server is detected.

**GPU guard:** `gpu_guard.py` provides VRAM ceiling (0.35), adaptive batch sizing, and VRAMMonitor.

**Error Recovery:**
- Failed files logged to `data/dlq/failed_pdfs.jsonl`
- Database errors: check PostgreSQL connection and disk space
- OOM during extraction: embed_server was likely running — kill it and retry

## Extraction Profiles

**Fast Profile** (Ollama, GPU): ~50 chunks/min on NVIDIA GPU (8GB+ VRAM)
```bash
python scripts/extract_concepts.py \
  --backend ollama --model llama3.1:8b --concurrency 2 \
  --metrics-file /tmp/extraction_metrics.txt
```

**Quality Profile** (Anthropic): Higher accuracy, ~20 chunks/min
```bash
python scripts/extract_concepts.py \
  --backend anthropic --model haiku --concurrency 4
```

**Ollama Optimization** (already applied via systemd override):
```bash
# /etc/systemd/system/ollama.service.d/override.conf
OLLAMA_FLASH_ATTENTION=1   # Enable flash attention
OLLAMA_NUM_PARALLEL=2      # Allow 2 parallel streams
OLLAMA_KV_CACHE_TYPE=q8_0  # Quantized KV cache
```
