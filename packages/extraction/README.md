# Research KB Extraction

Concept extraction package for the research knowledge base using Ollama LLM.

## Features

- **OllamaClient**: GPU-accelerated LLM wrapper for structured JSON output
- **ConceptExtractor**: Extract concepts and relationships from text chunks
- **Deduplicator**: Canonical name normalization and embedding-based deduplication
- **GraphSyncService**: Sync concepts to Neo4j graph database

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

### Neo4j Sync

```python
from research_kb_extraction import GraphSyncService

async with GraphSyncService() as sync:
    # Sync concept to Neo4j
    await sync.sync_concept(
        concept_id=uuid,
        name="instrumental variables",
        canonical_name="instrumental variables",
        concept_type="method",
    )

    # Find related concepts
    related = await sync.find_related_concepts(concept_id, max_hops=2)
```

## Configuration

### Ollama

- Default model: `llama3.1:8b`
- Server: `http://localhost:11434`
- GPU acceleration recommended (RTX 2070 SUPER or better)

### Neo4j

- URI: `bolt://localhost:7687`
- Auth: `neo4j/research_kb_dev`

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
| `REQUIRES` | Method requires assumption | IV → relevance |
| `USES` | Method uses technique | Matching → propensity scores |
| `ADDRESSES` | Method solves problem | IV → endogeneity |
| `GENERALIZES` | Broader concept | Panel → DiD |
| `SPECIALIZES` | Narrower concept | LATE → ATE |
| `ALTERNATIVE_TO` | Competing approaches | Matching vs Regression |
| `EXTENDS` | Builds upon | DML → ML + CI |

## Testing

```bash
cd packages/extraction
pytest
```
