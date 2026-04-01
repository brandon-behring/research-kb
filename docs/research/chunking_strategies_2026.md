# Chunking Strategies Research: 2024-2026

**Date**: 2026-03-30
**Context**: Evaluating chunking approaches for research-kb (1,756 technical books/papers,
36 domains, BGE-large-en-v1.5 1024-dim, 512 token context).

**Current approach**: Docling HybridChunker, max_tokens=300, flat chunks with section metadata.

---

## 1. Contextual Chunking (Anthropic, September 2024)

### How it works

Prepend a short (50-100 token) context snippet to each chunk before embedding.
The context explains the chunk's position within the larger document.

**Prompt template:**
> "Please give a short succinct context to situate this chunk within the overall
> document for the purposes of improving search retrieval of the chunk."

The full document is sent as context, and the LLM generates a prefix for each chunk.
Apply to both vector embeddings (Contextual Embeddings) and BM25 indexing (Contextual BM25).

### Published results

| Configuration | Retrieval Failure Reduction |
|---|---|
| Contextual Embeddings alone | 35% (5.7% -> 3.7%) |
| Contextual Embeddings + Contextual BM25 | 49% (5.7% -> 2.9%) |
| + Reranking | 67% (5.7% -> 1.9%) |

Measured across codebases, academic papers, and fiction. Top-20 chunk retrieval.

### Cost

$1.02 per million document tokens (with prompt caching).
research-kb estimate: ~1.4M chunks * ~200 tokens avg = ~280M tokens = ~$286 one-time cost.

### Implementation complexity for research-kb: 3/5

- Requires LLM call per chunk (1.4M chunks = significant API cost)
- Requires re-embedding entire corpus
- Storage: Same flat schema, add `context_prefix` column
- Search logic: No change (context is baked into embeddings)

### Pros

- Largest published improvement numbers for RAG retrieval
- Minimal schema change (add one column)
- Works with existing embedding model (BGE-large)
- Disambiguates domain-specific terms naturally

### Cons

- **$286 estimated cost** for full corpus (Haiku 4.5 pricing)
- Context prefix consumes embedding tokens (50-100 of 512 budget)
- Requires stable corpus (re-contextualizing on every change)
- Quality depends on LLM context generation quality

### Relevance to research-kb: HIGH

research-kb's main problem — chunks from technical books losing context — is exactly
what contextual retrieval was designed to solve. A chunk saying "the backdoor criterion
states..." is much more findable when prefixed with "[Causality by Judea Pearl, Chapter 3,
Section 3.3: Interventions]".

**Note**: A "Reconstructing Context" paper (arXiv:2504.19754, April 2025) found contextual
retrieval only marginally exceeded late chunking on nDCG@5 (0.317 vs 0.309). Voyage AI's
`voyage-context-3` (July 2025) claims to achieve similar results WITHOUT the LLM step,
outperforming Anthropic's approach by 6.76% via learned contextualization during embedding.
This could be a cheaper alternative if we decide to switch embedding models.

**Sources**:
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Reconstructing Context (arXiv)](https://arxiv.org/html/2504.19754v1)
- [Voyage-context-3](https://blog.voyageai.com/2025/07/23/voyage-context-3/)

---

## 2. Hierarchical / Parent-Child Chunking

### How it works

Create chunks at multiple levels of the document hierarchy:
- **Parent chunks**: Section-level (500-2000 tokens), provide broad context
- **Child chunks**: Paragraph-level (100-500 tokens), provide precise matching

During retrieval:
1. Search against child chunks (high precision)
2. Fetch parent context for display and reranking
3. Deduplicate siblings from the same parent

### Database schema change

```
chunk_hierarchy:
  parent_chunk_id (FK -> chunks.id)
  child_chunk_id  (FK -> chunks.id)
  depth           (int, nesting level)
```

### Implementation complexity for research-kb: 4/5

- New database table and migration
- Search pipeline changes (retrieve children, expand to parents)
- Deduplication logic for sibling chunks
- Re-chunking and re-embedding entire corpus at two granularities
- ~2x storage (parent + child embeddings)

### Pros

- Natural for technical books (chapter > section > paragraph)
- Best retrieval precision (small child chunks match precisely)
- Rich context for display (parent provides framing)
- Can navigate up/down the hierarchy

### Cons

- ~2x storage and embedding cost
- More complex search logic
- Parent-child relationships must be maintained during ingestion
- Not well-suited for flat documents (papers with no hierarchy)

### Relevance to research-kb: MEDIUM-HIGH

Good fit for textbooks (structured hierarchy). Less useful for papers
(typically flat). Worth considering but higher implementation cost
than contextual chunking for potentially similar gains.

**Source**: [Weaviate Chunking Strategies](https://weaviate.io/blog/chunking-strategies-for-rag)

---

## 3. Docling's Built-in Hierarchical Capabilities

### What's available NOW (already in research-kb's stack)

Docling provides TWO chunkers that research-kb could use:

**HierarchicalChunker**: One chunk per document element. Preserves full document tree.
Each chunk carries heading context and provenance metadata (JSON Pointer to source element).

**HybridChunker** (currently used): Starts from HierarchicalChunker output, then:
1. Splits oversized chunks (based on tokenizer limit)
2. Merges undersized successive chunks (if `merge_peers=True`)

### Key parameters currently set

- `max_tokens=300` (of 512 BGE budget)
- `merge_peers=True` (merge adjacent sibling chunks)
- Metadata: `section`, `heading_level`, `chunking_method`

### What's NOT being used

- **HierarchicalChunker** (full document tree, one chunk per element)
- `repeat_table_header` (True by default — already working)
- `omit_header_on_overflow` (False by default)
- Full provenance metadata (JSON Pointer references to source elements)
- Heading breadcrumbs (available via `chunk.meta.headings` — partially used)

### Low-hanging fruit

1. **Store full heading breadcrumbs** (not just last heading): Currently only
   `safe_headings[-1]` is stored. Storing the full path enables contextual search.
2. **Use HierarchicalChunker alongside HybridChunker**: Create parent chunks
   (section-level from HierarchicalChunker) and child chunks (from HybridChunker).
3. **Prepend heading path to chunk text** before embedding — a zero-cost
   approximation of Anthropic's contextual retrieval.

### Implementation complexity: 1-2/5

Already part of the stack. No new dependencies. Low risk.

**Source**: [Docling Chunking Docs](https://docling-project.github.io/docling/concepts/chunking/)

---

## 4. Late Chunking (Jina AI, 2024)

### How it works

Traditional: chunk text -> embed each chunk independently.
Late chunking: embed the FULL document -> use token-level embeddings to create
chunk embeddings that retain global context.

1. Pass entire document through embedding model
2. Get per-token embeddings (aware of full document context)
3. Apply chunk boundaries to segment the token embeddings
4. Pool (mean/max) within each segment to get chunk embeddings

### Key limitation

Requires a long-context embedding model that produces per-token embeddings.
BGE-large-en-v1.5 has a 512-token context limit — **too short for late chunking**.

Models that support it: jina-embeddings-v2 (8K context), nomic-embed-text (8K),
and similar long-context models.

### Implementation complexity for research-kb: 5/5

- Would require switching from BGE-large to a long-context embedding model
- Re-embedding entire corpus
- Custom embedding pipeline (not standard sentence-transformers)
- Not supported by standard embedding APIs

### Relevance to research-kb: LOW

Incompatible with current embedding model (BGE-large 512 tokens).
Would require a full embedding model migration, which is a separate
decision with broad implications.

**Source**: [Jina Late Chunking Paper](https://arxiv.org/pdf/2409.04701)

---

## 5. Multi-Resolution Indexing

### How it works

Index the same content at 3 levels simultaneously:
- **Level 1** (paragraph): ~100 tokens, high precision
- **Level 2** (section): ~500 tokens, balanced
- **Level 3** (chapter): ~2000 tokens, high recall

Query all levels, merge results with level-weighted scoring.

### Implementation complexity for research-kb: 4/5

- 3x storage and embedding cost
- Complex merge/dedup logic across levels
- Query routing to appropriate granularity
- Most invasive schema change

### Relevance to research-kb: LOW-MEDIUM

The gains over parent-child (2 levels) don't justify the 3x cost.
Parent-child covers the same use case with less complexity.

---

## 6. Proposition-Based Chunking (Dense X Retrieval)

### How it works

Convert text into atomic propositions (self-contained factual statements)
before embedding. Each proposition is independently searchable.

Example:
- Input: "Pearl introduced the backdoor criterion in 1995 to identify causal effects."
- Propositions:
  - "Pearl introduced the backdoor criterion"
  - "The backdoor criterion was introduced in 1995"
  - "The backdoor criterion is used to identify causal effects"

### Implementation complexity for research-kb: 5/5

- Requires LLM call per chunk to extract propositions
- Massively increases chunk count (3-5x)
- Lossy for mathematical content (propositions don't preserve equations)
- Not suitable for technical books with dense notation

### Relevance to research-kb: LOW

Poor fit for mathematical/technical content. Equations and formal
notation don't decompose well into propositions. Very high cost.

---

## 7. Agentic Chunking / Semantic Boundary Detection

### How it works

Use an LLM to identify natural chunk boundaries by reading the text
and deciding where semantic topic shifts occur.

### Implementation complexity for research-kb: 4/5

- LLM call per document (expensive for 1,756 sources)
- Hard to make deterministic/reproducible
- Docling's structure-aware chunking already captures most boundaries

### Relevance to research-kb: LOW

Docling's HybridChunker already uses document structure (headings,
sections) to identify boundaries. Agentic chunking would be redundant
and much more expensive.

---

## Recommendation Matrix

| Approach | Relevance | Complexity | Expected Gain | Cost | Recommend |
|---|---|---|---|---|---|
| **Contextual Chunking** | HIGH | 3/5 | ~35-67% | ~$286 | **Yes (Phase 2)** |
| **Parent-Child (Docling)** | MEDIUM-HIGH | 2/5 | ~15-25% | ~$0 | **Yes (Phase 1)** |
| **Heading Path Prepend** | HIGH | 1/5 | ~10-20% | $0 | **Yes (Immediate)** |
| Late Chunking | LOW | 5/5 | ~15% | Model switch | No |
| Multi-Resolution | LOW-MEDIUM | 4/5 | ~20% | 3x storage | No |
| Proposition-Based | LOW | 5/5 | Varies | Very high | No |
| Agentic Chunking | LOW | 4/5 | ~10% | High | No |

---

## Recommended Implementation Order

### Immediate: Heading Path Prepend (Zero Cost)

Currently, only `safe_headings[-1]` (the last heading) is stored in chunk metadata.
Docling provides the full heading breadcrumb. Change the chunker to prepend the
full heading path to chunk text before embedding:

```
Current:  "The backdoor criterion states that..."
Improved: "Causality > Chapter 3 > Section 3.3 > The backdoor criterion states that..."
```

This is a free approximation of contextual retrieval. No API cost. No schema change.
Requires re-embedding changed chunks only.

**Measure impact with eval_compare.py before proceeding.**

### Phase 1: Docling Parent-Child Chunking (Low Cost)

Use Docling's HierarchicalChunker to create section-level parent chunks
alongside the existing HybridChunker paragraph-level child chunks.

1. Add `chunk_hierarchy` table (parent_chunk_id, child_chunk_id, depth)
2. During ingestion, run both chunkers
3. Embed parent chunks alongside child chunks
4. Search against child chunks; expand to parent context for display

**Measure impact with eval_compare.py before proceeding.**

### Phase 2: Anthropic Contextual Retrieval (Medium Cost)

If Phase 1 gains are insufficient, add LLM-generated context prefixes.
This is the most proven approach but costs ~$286 for the full corpus.

1. Use Haiku 4.5 to generate context prefix for each chunk
2. Store prefix in new `context_prefix` column
3. Re-embed with prefix prepended
4. Prompt caching reduces cost for multi-chunk documents

**Only proceed after Phase 1 results are measured.**

### Critical Principle

**Never change chunking without measuring the impact.**
Every change must be:
1. Run baseline eval BEFORE
2. Apply change to a subset (2-3 domains)
3. Run eval on changed subset
4. Compare with eval_compare.py
5. Only roll out corpus-wide if metrics improve

---

## Key Academic References

- Anthropic Contextual Retrieval (Sept 2024): [anthropic.com](https://www.anthropic.com/news/contextual-retrieval)
- Late Chunking (Jina AI, Aug 2024): [arXiv:2409.04701](https://arxiv.org/html/2409.04701v2)
- Dense X Retrieval / Propositions (EMNLP 2024): [aclanthology.org](https://aclanthology.org/2024.emnlp-main.845/)
- Reconstructing Context (April 2025): [arXiv:2504.19754](https://arxiv.org/html/2504.19754v1)
- TopoChunker dual-agent (March 2026): [arXiv:2603.18409](https://arxiv.org/html/2603.18409)
- MDKeyChunker for Markdown (March 2026): [arXiv:2603.23533](https://arxiv.org/html/2603.23533)
- Clinical Chunking Evaluation (2025): [PMC12649634](https://pmc.ncbi.nlm.nih.gov/articles/PMC12649634/)
- Vectara NAACL 2025: Chunking config matters as much as embedding model choice
- AI21 Multi-Scale Study: [ai21.com](https://www.ai21.com/blog/query-dependent-chunking/)
- Voyage-context-3 (July 2025): [voyageai.com](https://blog.voyageai.com/2025/07/23/voyage-context-3/)
- Docling Chunking: [docling-project.github.io](https://docling-project.github.io/docling/concepts/chunking/)
- LlamaIndex AutoMergingRetriever: [developers.llamaindex.ai](https://developers.llamaindex.ai/python/examples/retrievers/auto_merging_retriever/)

---

## Baseline (2026-03-30)

Current system: Docling HybridChunker, max_tokens=300, 3-way search (FTS+vector+citation).

| Metric | Value |
|---|---|
| Hit@K | 92.5% (99/107) |
| MRR | 0.787 |
| NDCG@5 | 0.768 |
| NDCG@10 | 0.776 |
| Failures | 8/107 |

Weak domains: data_science (33% hit), sql (67%), algorithms (75%), software_engineering (67%).
All improvements must be measured against this baseline using `scripts/eval_compare.py`.
