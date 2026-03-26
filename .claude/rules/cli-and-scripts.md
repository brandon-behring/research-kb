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

## Ingestion Best Practices

```bash
# Recommended: quiet mode for Claude Code monitoring (minimal output)
python scripts/ingest_missing_textbooks.py --quiet

# JSON output for programmatic parsing
python scripts/ingest_missing_textbooks.py --quiet --json > ingestion_report.json
```

**Error Recovery:**
- Failed files with `recoverable: true` can be re-ingested later
- Memory errors indicate PDF too large (contact maintainer)
- Embedding service errors: ensure embed_server is running
- Database errors: check PostgreSQL connection and disk space

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
