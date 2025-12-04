# Phase 3 Research: Enhanced Retrieval

**Date**: 2025-12-02
**Status**: Research Complete

---

## 1. Cross-Encoder Re-ranking

### Overview

Two-stage retrieval is the standard industry approach:
1. **Stage 1**: Bi-encoder (fast) retrieves top-k candidates
2. **Stage 2**: Cross-encoder (accurate) re-ranks candidates

### Why Cross-Encoders?

- **Bi-encoders** encode query and documents separately → fast but less accurate
- **Cross-encoders** encode query+document together → slower but higher accuracy
- Cross-encoders perform attention across query and document simultaneously

### Recommended Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | Fast | Good |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 33M | Medium | Better |
| `cross-encoder/ms-marco-electra-base` | 110M | Slow | Best |

**Recommendation**: Start with `ms-marco-MiniLM-L-6-v2` for balance of speed/accuracy.

### Implementation Pattern

```python
from sentence_transformers import CrossEncoder

# Initialize once
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_results(query: str, chunks: list[Chunk], top_k: int = 10) -> list[Chunk]:
    """Re-rank search results using cross-encoder."""
    # Create query-document pairs
    pairs = [[query, chunk.content] for chunk in chunks]

    # Score all pairs
    scores = cross_encoder.predict(pairs)

    # Sort by score
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    return [chunk for chunk, score in ranked[:top_k]]
```

### Integration with Current Architecture

```
Current: Query → FTS + Vector → Combine → Return top-k
Proposed: Query → FTS + Vector → Combine → Cross-Encoder Re-rank → Return top-k
```

**Key Design Decisions**:
- Re-rank top-50 candidates down to top-10
- Cache cross-encoder model in memory
- Add `--no-rerank` CLI flag for comparison

### Sources

- [Sentence Transformers: Retrieve & Re-Rank](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
- [Cross-Encoders — Sentence Transformers](https://sbert.net/examples/cross_encoder/applications/README.html)
- [OpenAI Cookbook: Search Reranking](https://cookbook.openai.com/examples/search_reranking_with_cross-encoders)
- [The aRt of RAG Part 3: Reranking with Cross Encoders](https://medium.com/@rossashman/the-art-of-rag-part-3-reranking-with-cross-encoders-688a16b64669)

---

## 2. Query Expansion

### Overview

Query expansion bridges the semantic gap between user queries and document content by adding synonyms, related terms, or reformulated queries.

### Approaches

#### A. Synonym-Based (Controlled)

- Use predefined synonym mappings
- High precision, low recall
- Best for domain-specific vocabulary

```python
SYNONYMS = {
    "iv": ["instrumental variables", "instrumental variable", "2sls"],
    "did": ["difference-in-differences", "diff-in-diff"],
    "ate": ["average treatment effect"],
    # ... more mappings
}

def expand_query(query: str) -> str:
    """Expand query with known synonyms."""
    expanded = query
    for abbrev, synonyms in SYNONYMS.items():
        if abbrev in query.lower():
            expanded += " OR " + " OR ".join(synonyms)
    return expanded
```

**Advantage**: Deterministic, auditable, no false positives

#### B. Embedding-Based (Automatic)

- Find nearest neighbors in embedding space
- Higher recall, may introduce noise
- Useful for discovering related concepts

```python
def expand_with_embeddings(query: str, k: int = 5) -> list[str]:
    """Find semantically similar terms to expand query."""
    query_embedding = embed(query)
    # Find nearest concept embeddings
    similar_concepts = find_nearest_concepts(query_embedding, k=k)
    return [c.name for c in similar_concepts]
```

#### C. Concept-Based (Graph-Aware)

- Leverage our knowledge graph
- Find related concepts via graph traversal
- Most domain-appropriate for this project

```python
def expand_with_graph(query: str, hops: int = 1) -> list[str]:
    """Expand query using concept graph relationships."""
    # Extract concepts from query
    query_concepts = extract_concepts(query)

    # Find related concepts via graph
    related = []
    for concept in query_concepts:
        neighbors = get_neighborhood(concept.id, hops=hops)
        related.extend([c.name for c in neighbors])

    return related
```

### Recommended Approach for Phase 3

**Hybrid Strategy**:
1. **Explicit synonyms** for common abbreviations (IV, DiD, ATE)
2. **Graph expansion** for related concepts
3. **NO automatic embedding expansion** (risk of noise)

### Integration

```python
class QueryExpander:
    def __init__(self, synonym_map: dict, graph_service: GraphService):
        self.synonyms = synonym_map
        self.graph = graph_service

    def expand(self, query: str, use_graph: bool = True) -> ExpandedQuery:
        # Step 1: Synonym expansion
        expanded_terms = self._expand_synonyms(query)

        # Step 2: Graph expansion (if enabled)
        if use_graph:
            concepts = extract_concepts(query)
            for concept in concepts:
                related = self.graph.get_related(concept.id, hops=1)
                expanded_terms.extend(related)

        return ExpandedQuery(
            original=query,
            expanded_terms=expanded_terms,
            fts_query=self._build_fts_query(query, expanded_terms)
        )
```

### Sources

- [Semantic approaches for query expansion: taxonomy, challenges](https://pmc.ncbi.nlm.nih.gov/articles/PMC11935759/)
- [Synonymic Query Expansion for Smarter Search](https://sathishsaravanan.com/blog/synonymic-query-expansion/)
- [Hybrid query expansion using lexical resources and word embeddings](https://dl.acm.org/doi/10.1016/j.ins.2019.12.002)
- [Query Expansion Using Word Embeddings](https://www.researchgate.net/publication/310823543_Query_Expansion_Using_Word_Embeddings)

---

## 3. Multi-hop Reasoning Chains

### Overview

Multi-hop reasoning connects disparate concepts through intermediate relationships, enabling answers to questions like:

> "What assumptions does double machine learning require?"
>
> Path: DML → uses → cross-fitting → requires → sample splitting

### Current Capability

Already implemented in `packages/storage/src/research_kb_storage/graph_queries.py`:
- `find_shortest_path()` - paths between concepts
- `get_neighborhood()` - N-hop expansion
- `compute_graph_score()` - graph-based relevance

### Enhancement Opportunities

1. **Chain Explanation**: Return the reasoning path, not just the score
2. **Path Weighting**: Weight paths by relationship types (REQUIRES > USES)
3. **Confidence Propagation**: Decay confidence along path

```python
def explain_connection(concept_a: str, concept_b: str) -> str:
    """Generate human-readable explanation of concept connection."""
    path = find_shortest_path(concept_a, concept_b)

    if not path:
        return f"No direct connection found between {concept_a} and {concept_b}."

    explanation = f"{concept_a}"
    for rel_type in path["relationship_types"]:
        explanation += f" → ({rel_type.lower()}) → "
    explanation += concept_b

    return explanation
```

---

## 4. Implementation Priority

### Phase 3A: Cross-Encoder Re-ranking (2-3 hours)

1. Add `sentence-transformers` to pdf-tools dependencies
2. Create `packages/pdf-tools/src/research_kb_pdf/reranker.py`
3. Integrate into search pipeline
4. Add `--rerank/--no-rerank` CLI flag
5. Benchmark: Compare P@5 with/without re-ranking

### Phase 3B: Query Expansion (2-3 hours)

1. Create synonym mapping for causal inference terms
2. Add `QueryExpander` class to storage package
3. Integrate with FTS query building
4. Add `--expand-query/--no-expand` CLI flag
5. Test with abbreviation-heavy queries

### Phase 3C: Reasoning Chains (1-2 hours)

1. Add path explanation to CLI output
2. Enhance graph command with chain display
3. Optional: Add chain confidence scoring

---

## 5. Files to Create/Modify

### New Files
- `packages/pdf-tools/src/research_kb_pdf/reranker.py`
- `packages/storage/src/research_kb_storage/query_expander.py`
- `fixtures/concepts/synonym_map.yaml`

### Modified Files
- `packages/storage/src/research_kb_storage/search.py` (add re-ranking hook)
- `packages/cli/src/research_kb_cli/main.py` (add CLI flags)
- `packages/pdf-tools/pyproject.toml` (add sentence-transformers)

---

## 6. Expected Improvements

| Technique | Expected P@5 Improvement | Latency Impact |
|-----------|-------------------------|----------------|
| Cross-encoder re-ranking | +5-15% | +50-100ms |
| Synonym expansion | +3-5% | Negligible |
| Graph expansion | Already integrated | Already measured |

**Combined**: Expect +10-20% improvement in retrieval quality.
