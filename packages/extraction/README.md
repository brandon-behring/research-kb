# Research KB Extraction

Concept extraction package for the research knowledge base with multiple LLM backend options.

## Features

- **Multiple Backends**: OllamaClient, InstructorOllamaClient, LlamaCppClient, AnthropicClient
- **ConceptExtractor**: Extract concepts and relationships from text chunks
- **Deduplicator**: Canonical name normalization and embedding-based deduplication
- **Metrics**: Extraction quality and performance tracking

## Backend Comparison

| Backend | Speed | Cost | GPU | JSON Method | Best For |
|---------|-------|------|-----|-------------|----------|
| **OllamaClient** | Fast | Free | Recommended | Native JSON mode | Default local use |
| **InstructorOllamaClient** | Fast | Free | Recommended | Pydantic + retry | Validation-critical |
| **LlamaCppClient** | Fastest | Free | Required | Grammar-based | Maximum throughput |
| **AnthropicClient** | Medium | $$$ | No | tool_use schema | Highest quality |

### Configuration Reference

| Backend | Default Model | Key Config |
|---------|---------------|------------|
| ollama | llama3.1:8b | `num_ctx=4096`, `temperature=0.1` |
| instructor | llama3.1:8b | `max_retries=3` (auto-retry on schema fail) |
| llamacpp | Meta-Llama-3.1-8B-Q4_K_M.gguf | `n_gpu_layers=20` (~8GB VRAM) |
| anthropic | claude-3-5-haiku-latest | Models: haiku, haiku-3.5, sonnet, opus |

### Backend Selection

```bash
# Default Ollama
python scripts/extract_concepts.py --backend ollama

# With Pydantic validation
python scripts/extract_concepts.py --backend instructor

# Direct GPU (fastest)
python scripts/extract_concepts.py --backend llamacpp

# API-based (highest quality)
python scripts/extract_concepts.py --backend anthropic --model haiku
python scripts/extract_concepts.py --backend anthropic --model opus
```

## Installation

```bash
pip install -e packages/extraction
```

## Usage

### Basic Extraction

```python
from research_kb_extraction import OllamaClient, ConceptExtractor

async def extract():
    async with ConceptExtractor() as extractor:
        result = await extractor.extract_from_text("""
            Instrumental variables (IV) estimation is used to address
            endogeneity problems in econometric analysis.
        """)

        for concept in result.concepts:
            print(f"{concept.name}: {concept.concept_type}")
```

### Deduplication

```python
from research_kb_extraction import Deduplicator

dedup = Deduplicator()

# Expand abbreviations to canonical form
canonical = dedup.to_canonical_name("IV")  # "instrumental variables"
canonical = dedup.to_canonical_name("DiD")  # "difference-in-differences"

# Deduplicate batch of concepts
matches = await dedup.deduplicate_batch(extracted_concepts)
```

### Using Different Backends

```python
from research_kb_extraction import InstructorOllamaClient, LlamaCppClient, AnthropicClient

# Instructor with auto-retry on validation errors
async with InstructorOllamaClient() as client:
    result = await client.extract_concepts(text, max_retries=3)

# LlamaCpp for maximum throughput
async with LlamaCppClient(model_path="path/to/model.gguf") as client:
    result = await client.extract_concepts(text)

# Anthropic for highest quality
async with AnthropicClient(model="opus") as client:
    result = await client.extract_concepts(text)
```

## Configuration

### Ollama

- Default model: `llama3.1:8b`
- Server: `http://localhost:11434`
- GPU acceleration recommended (NVIDIA GPU with 8GB+ VRAM)

### LlamaCpp

- Model path: `models/*.gguf` (download separately)
- GPU layers: Set `n_gpu_layers` based on VRAM
- See `scripts/download_gguf_model.sh` for model setup

### Anthropic

- API key: Set `ANTHROPIC_API_KEY` environment variable
- Models: haiku (fast/cheap), sonnet (balanced), opus (best quality)

## Concept Types

| Type | Description | Examples |
|------|-------------|----------|
| `method` | Statistical methods | IV, DiD, matching |
| `assumption` | Required conditions | parallel trends, unconfoundedness |
| `problem` | Issues methods address | endogeneity, selection bias |
| `definition` | Formal definitions | ATE, LATE |
| `theorem` | Mathematical results | backdoor criterion |

## Relationship Types

| Type | Description | Example |
|------|-------------|---------|
| `REQUIRES` | Method requires assumption | IV -> relevance |
| `USES` | Method uses technique | Matching -> propensity scores |
| `ADDRESSES` | Method solves problem | IV -> endogeneity |
| `GENERALIZES` | Broader concept | Panel -> DiD |
| `SPECIALIZES` | Narrower concept | LATE -> ATE |
| `ALTERNATIVE_TO` | Competing approaches | Matching vs Regression |
| `EXTENDS` | Builds upon | DML -> ML + CI |

## Graph Storage

All graph queries are handled by PostgreSQL using recursive CTEs. This provides:
- ACID transactions with chunk/concept data
- No additional infrastructure needed
- Efficient 2-3 hop queries for typical use cases

See `docs/ROADMAP.md` Phase 5 for future Neo4j considerations when scale requires advanced graph analytics.

## Testing

```bash
cd packages/extraction
pytest
```
