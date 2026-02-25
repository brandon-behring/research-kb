# Adding Your Own Domain

This tutorial walks through adding a new knowledge domain to research-kb. By default, research-kb ships with 19 domains including `causal_inference`, `time_series`, `rag_llm`, `software_engineering`, `econometrics`, `deep_learning`, and more (see full list with `research-kb sources stats`). You can add any research domain following this guide.

## Prerequisites

- research-kb installed and running (PostgreSQL + pgvector)
- Python 3.11+
- PDFs or papers to ingest

## Overview

Adding a domain involves 5 steps:

1. Register the domain in PostgreSQL
2. Configure domain-specific prompts (optional)
3. Ingest source documents
4. Extract concepts
5. Verify with search

## Step 1: Register the Domain

Create a SQL migration to register your domain. Copy the pattern from an existing migration:

```bash
# Use the next available migration number
cp packages/storage/migrations/015_software_engineering_domain.sql \
   packages/storage/migrations/016_your_domain.sql
```

Edit the migration to define your domain:

```sql
INSERT INTO domains (id, name, description, concept_types, relationship_types) VALUES
(
    'your_domain_id',
    'Your Domain Name',
    'Description of what this domain covers',
    ARRAY['method', 'assumption', 'problem', 'definition', 'theorem',
          'concept', 'principle', 'technique', 'model'],
    ARRAY['REQUIRES', 'USES', 'ADDRESSES', 'GENERALIZES', 'SPECIALIZES',
          'ALTERNATIVE_TO', 'EXTENDS', 'RELATED_TO']
)
ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    description = EXCLUDED.description;
```

Apply the migration:

```bash
psql -h localhost -U postgres -d research_kb -f packages/storage/migrations/016_your_domain.sql
```

## Step 2: Configure Domain Prompts (Optional)

Domain-specific prompts improve concept extraction quality. Edit:

```
packages/extraction/src/research_kb_extraction/domain_prompts.py
```

Add your domain's abbreviation list and concept guidance. See existing domains for the pattern:

```python
DOMAIN_ABBREVIATIONS["your_domain_id"] = {
    "ML": "Machine Learning",
    "DL": "Deep Learning",
    # ... domain-specific abbreviations
}

DOMAIN_CONCEPT_GUIDANCE["your_domain_id"] = """
Focus on extracting:
- Core methods and techniques
- Key assumptions and their implications
- Problem formulations
- Relationships between methods
"""
```

If you skip this step, the default prompts will be used. They work well for most technical domains.

## Step 3: Ingest Source Documents

### Option A: Ingest PDFs via CLI

```bash
# Ingest all PDFs in a directory
research-kb ingest /path/to/your/pdfs --domain your_domain_id

# Ingest a single PDF
research-kb ingest /path/to/paper.pdf --domain your_domain_id
```

### Option B: Ingest via script (for batch processing)

```bash
python scripts/ingest_corpus.py --domain your_domain_id --source-dir /path/to/pdfs
```

### Option C: Load from pre-exported fixtures

If you've already exported fixtures (sources.json, chunks.json, etc.):

```bash
python scripts/load_demo_data.py --data-dir /path/to/fixtures --domain your_domain_id
```

## Step 4: Extract Concepts and Build Graph

After ingestion, extract concepts from the text chunks:

```bash
# Extract concepts (requires Ollama or configured LLM)
python scripts/extract_concepts.py --domain your_domain_id

# Sync concepts to KuzuDB (enables graph queries)
python scripts/sync_kuzu.py

# Generate embeddings for new chunks
python -m research_kb_pdf.embed_server &
python scripts/embed_missing.py
```

## Step 5: Verify

### Search your domain

```bash
# Text search
research-kb search query "your search term" --domain your_domain_id

# Check stats
research-kb sources stats
```

### Check domain registration

```bash
psql -h localhost -U postgres -d research_kb -c "SELECT id, name FROM domains;"
```

Expected output includes your new domain.

### Build evaluation queries (recommended)

Create golden evaluation queries for your domain to measure retrieval quality:

```bash
# Discover candidate queries from ingested content
python scripts/discover_golden_candidates.py --domain your_domain_id

# Build golden dataset
python scripts/build_golden_dataset.py
```

Then evaluate:

```bash
python scripts/eval_retrieval.py --per-domain
```

## Domain Checklist

```
[ ] Migration SQL created and applied
[ ] Domain prompts configured (optional but recommended)
[ ] Sources ingested (PDFs, papers, or fixtures)
[ ] Concepts extracted
[ ] KuzuDB synced
[ ] Embeddings generated
[ ] Search verified with sample queries
[ ] Golden evaluation queries created (recommended)
```

## Troubleshooting

### "Domain not found" errors

Verify the migration was applied:

```sql
SELECT * FROM domains WHERE id = 'your_domain_id';
```

### Poor search quality

1. Check that embeddings were generated: `python scripts/embed_missing.py`
2. Verify chunks exist: `SELECT COUNT(*) FROM chunks WHERE domain_id = 'your_domain_id';`
3. Add domain-specific abbreviations to `domain_prompts.py`
4. Create more specific evaluation queries

### Cross-domain concepts

After adding a new domain, run cross-domain concept linking to discover connections:

```bash
python scripts/sync_kuzu.py  # Re-syncs all domains
```

This enables graph queries that traverse domain boundaries (e.g., finding "CI/CD" concepts that relate to "evaluation pipeline" across software_engineering and rag_llm domains).
