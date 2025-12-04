# Concept Extraction Skill

This skill teaches agents how to extract and build knowledge graphs from research literature.

## When to Use

### During Ingestion
- Extracting concepts from newly ingested papers
- Building concept graph for a research domain
- Enriching chunks with concept annotations

### During Analysis
- Identifying key concepts in search results
- Finding conceptual connections across papers
- Mapping research landscape

### For Quality Assurance
- Validating extraction quality
- Deduplicating concepts
- Verifying relationships

## What Qualifies as a Concept

### Concept Types

**Methods** (`ConceptType.METHOD`)
- Identification or estimation techniques
- Examples: instrumental variables, difference-in-differences, matching
- Characteristics: Usually actionable, have requirements/assumptions

**Assumptions** (`ConceptType.ASSUMPTION`)
- Identification conditions required by methods
- Examples: unconfoundedness, parallel trends, exogeneity
- Characteristics: Often untestable, mathematically formalized

**Problems** (`ConceptType.PROBLEM`)
- Identification challenges that methods address
- Examples: endogeneity, confounding, selection bias
- Characteristics: Obstacles to causal inference

**Definitions** (`ConceptType.DEFINITION`)
- Estimands and target quantities
- Examples: ATE, ATT, LATE, CATE
- Characteristics: Well-defined mathematical objects

**Theorems** (`ConceptType.THEOREM`)
- Formal results with proofs
- Examples: Frisch-Waugh-Lovell, Gauss-Markov
- Characteristics: Mathematically proven statements

### Recognition Patterns

**Methods** appear with phrases like:
- "We use [method] to..."
- "The [method] approach..."
- "[Method] estimates the..."
- "Identification via [method]..."

**Assumptions** appear with phrases like:
- "We assume that..."
- "Under the assumption of..."
- "Requires [assumption] to hold..."
- "If [assumption] is satisfied..."

**Problems** appear with phrases like:
- "The problem of [problem]..."
- "[Problem] arises when..."
- "To address [problem], we..."
- "[Problem] can bias estimates..."

## Extraction Pipeline

### 1. Chunk Processing

```python
from research_kb_extraction import ConceptExtractor, OllamaClient

# Initialize extractor
ollama = OllamaClient(model="llama3.1:8b", gpu=True)
extractor = ConceptExtractor(ollama_client=ollama, confidence_threshold=0.7)

# Extract from chunk
chunk_text = """
Instrumental variables (IV) is an econometric technique used to identify
causal effects in the presence of endogeneity. The method requires three
key assumptions: relevance, exclusion restriction, and exogeneity.
"""

extraction = await extractor.extract_from_chunk(chunk_text)

# Result:
# extraction.concepts = [
#     ExtractedConcept(
#         name="instrumental variables",
#         concept_type="method",
#         definition="Econometric technique for causal identification...",
#         aliases=["IV", "IVs"],
#         confidence=0.92
#     ),
#     ExtractedConcept(
#         name="endogeneity",
#         concept_type="problem",
#         confidence=0.88
#     ),
#     # ... more concepts
# ]
# extraction.relationships = [
#     ExtractedRelationship(
#         source="instrumental variables",
#         target="relevance",
#         type="REQUIRES"
#     ),
#     # ... more relationships
# ]
```

### 2. Deduplication

```python
from research_kb_extraction import Deduplicator

dedup = Deduplicator()

# Canonicalize names
canonical = dedup.to_canonical_name("DiD")  # → "difference-in-differences"
canonical = dedup.to_canonical_name("2SLS")  # → "two-stage least squares"

# Check if concepts are duplicates
is_dup = dedup.are_duplicates(
    "Difference in Differences",
    "difference-in-differences"
)  # → True

# Merge concept variants
concepts = [
    Concept(name="IV", canonical_name="instrumental variables", ...),
    Concept(name="Instrumental Variables", canonical_name="instrumental variables", ...),
]
merged = dedup.merge_concepts(concepts)  # → Single concept with combined aliases
```

**Built-in Abbreviations:**
The deduplicator includes 36+ causal inference abbreviations:
- IV → instrumental variables
- DiD/DD → difference-in-differences
- 2SLS/TSLS → two-stage least squares
- PSM → propensity score matching
- RDD/RD → regression discontinuity design
- ATE/ATT/LATE/CATE → treatment effect variants
- ...and more

### 3. Embedding Generation

```python
from research_kb_pdf import EmbeddingClient

embed_client = EmbeddingClient()

# Generate concept embedding from name + definition
concept_text = f"{concept.canonical_name}. {concept.definition}"
embedding = embed_client.embed(concept_text)

# Store with embedding for semantic deduplication
await ConceptStore.create(
    name=concept.name,
    canonical_name=concept.canonical_name,
    concept_type=concept.concept_type,
    definition=concept.definition,
    embedding=embedding,
    confidence_score=concept.confidence,
)
```

### 4. Storage and Linking

```python
from research_kb_storage import ConceptStore, RelationshipStore, ChunkConceptStore

# Store concept
concept_id = await ConceptStore.create(
    name="instrumental variables",
    canonical_name="instrumental variables",
    concept_type=ConceptType.METHOD,
    aliases=["IV", "IVs", "instrument"],
    definition="...",
    embedding=embedding,
    extraction_method="ollama:llama3.1:8b",
    confidence_score=0.92,
)

# Store relationship
await RelationshipStore.create(
    source_concept_id=iv_concept.id,
    target_concept_id=endogeneity_concept.id,
    relationship_type=RelationshipType.ADDRESSES,
    confidence_score=0.85,
    evidence_chunk_ids=[chunk.id],
)

# Link chunk to concepts
await ChunkConceptStore.create(
    chunk_id=chunk.id,
    concept_id=concept_id,
    mention_type="defines",  # or "reference", "example"
    relevance_score=0.95,
)
```

## Building Concept Hierarchies

### Relationship Types

**REQUIRES** - Method needs assumption
```python
# IV requires relevance, exclusion restriction, exogeneity
RelationshipType.REQUIRES
```

**USES** - Method employs technique
```python
# IV uses 2SLS for estimation
RelationshipType.USES
```

**ADDRESSES** - Method solves problem
```python
# IV addresses endogeneity
RelationshipType.ADDRESSES
```

**GENERALIZES** / **SPECIALIZES** - Hierarchical relationships
```python
# Panel methods generalize DiD
RelationshipType.GENERALIZES
RelationshipType.SPECIALIZES
```

**ALTERNATIVE_TO** - Competing approaches
```python
# Matching vs regression adjustment
RelationshipType.ALTERNATIVE_TO
```

**EXTENDS** - Builds upon
```python
# Double ML extends matching with ML
RelationshipType.EXTENDS
```

## Linking Concepts Across Sources

### Concept Unification

When extracting from multiple papers:

1. **Exact match**: Same canonical_name
   ```python
   existing = await ConceptStore.get_by_canonical_name("instrumental variables")
   ```

2. **Fuzzy match**: Check aliases
   ```python
   # Search for "IV" finds "instrumental variables" via aliases
   ```

3. **Semantic match**: Embedding similarity > 0.95
   ```python
   similar = await ConceptStore.find_similar(
       embedding=new_concept_embedding,
       threshold=0.95,
   )
   ```

### Enrichment Strategy

When finding existing concept:
1. Merge aliases: `concept.aliases = list(set(existing.aliases + new.aliases))`
2. Enhance definition: Combine if both non-empty
3. Increase confidence: Repeated extraction signals validity
4. Add evidence: Track all chunks mentioning concept

## Handling Synonyms and Aliases

### Common Patterns

| Canonical Name | Aliases |
|----------------|---------|
| instrumental variables | IV, IVs, instrument, instruments |
| difference-in-differences | DiD, DD, diff-in-diff |
| unconfoundedness | CIA, conditional independence, ignorability, selection on observables |
| average treatment effect | ATE |
| propensity score matching | PSM, propensity matching |

### Alias Management

```python
# Add new alias
concept.aliases.append("new_alias")
await ConceptStore.update(concept.id, aliases=concept.aliases)

# Query by alias
all_concepts = await ConceptStore.list_all()
matches = [c for c in all_concepts if "IV" in c.aliases]
```

## Quality Validation

### Confidence Thresholds

| Confidence | Status | Action |
|------------|--------|--------|
| >= 0.9 | High | Auto-accept |
| 0.7 - 0.9 | Medium | Auto-accept, flag for review |
| 0.5 - 0.7 | Low | Manual review required |
| < 0.5 | Very low | Reject or manual review |

### Validation Script

```bash
# Run validation against seed concepts
python scripts/validate_seed_concepts.py

# Output formats
python scripts/validate_seed_concepts.py --output text      # Terminal
python scripts/validate_seed_concepts.py --output json      # CI/CD
python scripts/validate_seed_concepts.py --output markdown  # Docs

# Filter by type
python scripts/validate_seed_concepts.py --type method --confidence 0.8
```

### CLI Monitoring

```bash
# Check extraction status
research-kb extraction-status

# Output:
# Total concepts extracted: 73
# Validated concepts: 4 (5.5%)
# Concepts by type:
#   method             53
#   assumption         10
#   problem             6
# Average confidence: 0.86
# Chunk coverage: 4/3631 (0.1%)
```

## Batch Extraction Pipeline

```bash
# Extract concepts from all chunks
python scripts/extract_concepts.py --batch-size 50 --gpu

# With dry-run
python scripts/extract_concepts.py --dry-run

# Resume from checkpoint
python scripts/extract_concepts.py --resume --checkpoint checkpoints/batch_50.json
```

## Best Practices

### 1. Pre-extraction
- Ensure chunks are well-formed (clean section boundaries)
- Check embedding server is running
- Verify Ollama model downloaded: `ollama list`

### 2. During Extraction
- Monitor confidence scores
- Check deduplication rate (should merge ~30-40% variants)
- Watch for model hallucinations (concepts not in text)

### 3. Post-extraction
- Run seed concept validation
- Manually review low-confidence extractions
- Verify relationship edges make sense
- Check for orphan concepts (no relationships)

### 4. Quality Signals

**Good extraction:**
- Confidence > 0.75
- Definition present and accurate
- Aliases captured
- Relationships to other concepts
- Clear concept_type classification

**Poor extraction:**
- Generic names ("Method", "Approach")
- Vague definitions
- No aliases when abbreviations exist
- Isolated (no relationships)
- Wrong concept_type

## Python API Reference

### ConceptExtractor

```python
from research_kb_extraction import ConceptExtractor, OllamaClient

ollama = OllamaClient(model="llama3.1:8b")
extractor = ConceptExtractor(
    ollama_client=ollama,
    confidence_threshold=0.7,
    enable_relationships=True,
)

# Single chunk
result = await extractor.extract_from_chunk(chunk_text)

# Batch processing
results = await extractor.extract_batch(chunks, batch_size=10)
```

### Deduplicator

```python
from research_kb_extraction import Deduplicator

dedup = Deduplicator()

# Canonical names
canonical = dedup.to_canonical_name("text")  # lowercase, hyphens, expand abbrevs
clean = dedup.normalize_name("text")         # just lowercase + strip

# Duplicate detection
is_dup = dedup.are_duplicates(name1, name2)  # fuzzy matching
similarity = dedup.similarity(name1, name2)  # 0.0-1.0

# Merge concepts
merged = dedup.merge_concepts(concept_list)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Ollama not running | `ollama serve` or check `systemctl --user status ollama` |
| Model not found | `ollama pull llama3.1:8b` |
| Low extraction rate | Lower confidence threshold, check prompt |
| Too many duplicates | Review ABBREVIATION_MAP, add missing aliases |
| Hallucinated concepts | Increase confidence threshold, add negative examples |
| Slow extraction | Enable GPU: `ollama run llama3.1:8b --verbose` should show CUDA |
| Empty relationships | Check relationship extraction enabled, review prompt |

## See Also

- **Assumption Tracking Skill**: Specialized guidance for assumptions
- **Research Context Retrieval Skill**: Using extracted concepts for search
- **PDF Ingestion Skill**: Preparing chunks for extraction
