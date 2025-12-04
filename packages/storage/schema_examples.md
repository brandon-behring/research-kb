# Schema Usage Examples

## Purpose
Demonstrate how the minimal schema (sources + chunks) handles different source types using JSONB metadata.

---

## Example 1: Textbook Source

```sql
-- Insert Pearl's Causality textbook
INSERT INTO sources (
    source_type,
    title,
    authors,
    year,
    file_path,
    file_hash,
    metadata
) VALUES (
    'textbook',
    'Causality: Models, Reasoning, and Inference',
    ARRAY['Judea Pearl'],
    2009,
    '/home/user/library/textbooks/pearl_causality.pdf',
    'sha256:abc123...',
    jsonb_build_object(
        'isbn', '978-0521895606',
        'publisher', 'Cambridge University Press',
        'edition', 2,
        'toc_extracted', true,
        'total_pages', 464
    )
);

-- Insert theorem chunk from textbook
INSERT INTO chunks (
    source_id,
    content,
    content_hash,
    location,
    page_start,
    page_end,
    embedding,
    metadata
) VALUES (
    (SELECT id FROM sources WHERE file_hash = 'sha256:abc123...'),
    'The backdoor criterion states that a set of variables Z satisfies the backdoor criterion relative to (X, Y) if...',
    'sha256:chunk456...',
    'Chapter 3, Section 3.3, Theorem 3.3.1, p. 73',
    73,
    74,
    '[0.123, 0.456, ...]'::vector(1024),  -- BGE embedding
    jsonb_build_object(
        'chunk_type', 'theorem',
        'chapter_num', 3,
        'section_num', '3.3',
        'theorem_name', 'Backdoor Criterion',
        'theorem_number', '3.3.1',
        'has_proof', true
    )
);
```

---

## Example 2: Research Paper Source

```sql
-- Insert Chernozhukov 2018 paper
INSERT INTO sources (
    source_type,
    title,
    authors,
    year,
    file_path,
    file_hash,
    metadata
) VALUES (
    'paper',
    'Double/debiased machine learning for treatment and structural parameters',
    ARRAY['Victor Chernozhukov', 'Denis Chetverikov', 'Mert Demirer', 'Esther Duflo', 'Christian Hansen', 'Whitney Newey', 'James Robins'],
    2018,
    '/home/user/library/papers/chernozhukov_2018_dml.pdf',
    'sha256:def789...',
    jsonb_build_object(
        'doi', '10.1111/ectj.12097',
        'journal', 'Econometrics Journal',
        'volume', 21,
        'issue', 1,
        'pages', 'C1-C68',
        'citations_count', 1200,
        'authority_tier', 'canonical',
        'grobid_processed', true
    )
);

-- Insert abstract chunk from paper
INSERT INTO chunks (
    source_id,
    content,
    content_hash,
    location,
    page_start,
    page_end,
    embedding,
    metadata
) VALUES (
    (SELECT id FROM sources WHERE file_hash = 'sha256:def789...'),
    'We revisit the classic semiparametric problem of inference on a low-dimensional parameter θ0 in the presence of high-dimensional nuisance parameters...',
    'sha256:chunk789...',
    'Abstract, p. 1',
    1,
    1,
    '[0.789, 0.012, ...]'::vector(1024),
    jsonb_build_object(
        'chunk_type', 'abstract',
        'section', 'abstract',
        'paragraph_num', 1,
        'key_concepts', ARRAY['double_machine_learning', 'cross_fitting', 'neyman_orthogonality']
    )
);
```

---

## Example 3: Code Repository Source

```sql
-- Insert scikit-learn repository
INSERT INTO sources (
    source_type,
    title,
    authors,
    year,
    file_path,
    file_hash,
    metadata
) VALUES (
    'code_repo',
    'scikit-learn/sklearn.linear_model',
    ARRAY['scikit-learn developers'],
    2023,
    '/home/user/repos/scikit-learn',
    'sha256:ghi012...',
    jsonb_build_object(
        'git_url', 'https://github.com/scikit-learn/scikit-learn',
        'commit_sha', 'abc123def456',
        'branch', 'main',
        'language', 'python',
        'license', 'BSD-3-Clause'
    )
);

-- Insert code chunk (function implementation)
INSERT INTO chunks (
    source_id,
    content,
    content_hash,
    location,
    page_start,
    page_end,
    embedding,
    metadata
) VALUES (
    (SELECT id FROM sources WHERE file_hash = 'sha256:ghi012...'),
    'class LogisticRegression(BaseEstimator, LinearClassifierMixin):\n    """Logistic Regression classifier.\n\n    Parameters\n    ----------\n    penalty : {''l1'', ''l2'', ''elasticnet'', None}, default=''l2''\n        Regularization penalty...',
    'sha256:chunk345...',
    'sklearn/linear_model/_logistic.py:LogisticRegression:lines 1200-1450',
    NULL,  -- No page numbers for code
    NULL,
    '[0.345, 0.678, ...]'::vector(1024),
    jsonb_build_object(
        'chunk_type', 'class_definition',
        'file_path', 'sklearn/linear_model/_logistic.py',
        'start_line', 1200,
        'end_line', 1450,
        'class_name', 'LogisticRegression',
        'language', 'python',
        'ast_node_type', 'ClassDef'
    )
);
```

---

## Future Extension Examples (Using JSONB)

### Flashcard Generation (Before Dedicated Table)

```sql
-- Add flashcard to existing chunk
UPDATE chunks
SET metadata = metadata || jsonb_build_object(
    'flashcard', jsonb_build_object(
        'front', 'What is the backdoor criterion?',
        'back', 'A criterion for identifying valid adjustment sets in causal inference. A set Z satisfies the backdoor criterion if it blocks all backdoor paths from X to Y.',
        'difficulty', 'medium',
        'next_review', '2025-12-01'
    )
)
WHERE id = 'chunk-uuid-here';

-- Query chunks with flashcards
SELECT
    content,
    metadata->>'flashcard'->>'front' AS question,
    metadata->>'flashcard'->>'back' AS answer
FROM chunks
WHERE metadata ? 'flashcard';
```

### Concept Linking (Before Dedicated Concepts Table)

```sql
-- Add concept tags to chunk
UPDATE chunks
SET metadata = metadata || jsonb_build_object(
    'concepts', ARRAY['instrumental_variables', 'endogeneity', 'exclusion_restriction'],
    'concept_types', jsonb_build_object(
        'instrumental_variables', 'method',
        'endogeneity', 'problem',
        'exclusion_restriction', 'assumption'
    )
)
WHERE content_hash = 'sha256:chunk456...';

-- Query chunks by concept
SELECT content, location
FROM chunks
WHERE metadata->'concepts' ? 'instrumental_variables';
```

### Parent-Child Chunk Relationships

```sql
-- Add hierarchical relationship
UPDATE chunks
SET metadata = metadata || jsonb_build_object(
    'parent_chunk_id', 'parent-uuid-here',
    'hierarchy_level', 'subsection',
    'sibling_chunks', ARRAY['sibling1-uuid', 'sibling2-uuid']
)
WHERE id = 'child-chunk-uuid';
```

---

## Validation Queries

### Check Schema Extensibility

```sql
-- Textbook chunks with theorem metadata
SELECT
    title,
    location,
    metadata->>'theorem_name' AS theorem_name,
    metadata->>'chapter_num' AS chapter
FROM chunks c
JOIN sources s ON c.source_id = s.id
WHERE s.source_type = 'textbook'
  AND c.metadata->>'chunk_type' = 'theorem';

-- Papers by authority tier
SELECT
    title,
    authors,
    metadata->>'authority_tier' AS tier,
    (metadata->>'citations_count')::int AS citations
FROM sources
WHERE source_type = 'paper'
ORDER BY (metadata->>'citations_count')::int DESC;

-- Code chunks by language
SELECT
    title,
    metadata->>'file_path' AS file,
    metadata->>'class_name' AS class_name
FROM chunks c
JOIN sources s ON c.source_id = s.id
WHERE s.source_type = 'code_repo'
  AND c.metadata->>'language' = 'python';
```

### Test FTS and Vector Search

```sql
-- Full-text search for "backdoor criterion"
SELECT
    s.title,
    c.location,
    ts_rank(c.fts_vector, plainto_tsquery('english', 'backdoor criterion')) AS rank
FROM chunks c
JOIN sources s ON c.source_id = s.id
WHERE c.fts_vector @@ plainto_tsquery('english', 'backdoor criterion')
ORDER BY rank DESC
LIMIT 10;

-- Vector similarity search (requires actual embedding)
-- SELECT
--     s.title,
--     c.location,
--     c.embedding <=> '[query_embedding]'::vector(1024) AS distance
-- FROM chunks c
-- JOIN sources s ON c.source_id = s.id
-- ORDER BY distance
-- LIMIT 10;
```

---

## Migration Path: JSONB → Dedicated Table

When a use case solidifies (e.g., flashcards become core feature):

### Step 1: Create Dedicated Table

```sql
CREATE TABLE flashcards (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    front TEXT NOT NULL,
    back TEXT NOT NULL,
    difficulty TEXT,  -- easy, medium, hard
    next_review TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_flashcards_chunk ON flashcards(chunk_id);
CREATE INDEX idx_flashcards_review ON flashcards(next_review);
```

### Step 2: Migrate Data from JSONB

```sql
INSERT INTO flashcards (chunk_id, front, back, difficulty, next_review)
SELECT
    id AS chunk_id,
    (metadata->'flashcard'->>'front')::TEXT AS front,
    (metadata->'flashcard'->>'back')::TEXT AS back,
    (metadata->'flashcard'->>'difficulty')::TEXT AS difficulty,
    (metadata->'flashcard'->>'next_review')::TIMESTAMPTZ AS next_review
FROM chunks
WHERE metadata ? 'flashcard';

-- Optional: Clean up JSONB after migration
UPDATE chunks
SET metadata = metadata - 'flashcard'
WHERE metadata ? 'flashcard';
```

### Step 3: Update Application Code

```python
# Old: Query JSONB
flashcard_data = chunk.metadata.get("flashcard")

# New: Query dedicated table
flashcard = session.query(Flashcard).filter_by(chunk_id=chunk.id).first()
```

---

## Summary

✓ **Minimal schema handles 3 source types** (textbook, paper, code)
✓ **JSONB enables experimentation** without migrations
✓ **Clear migration path** when use cases solidify
✓ **Examples demonstrate extensibility** for flashcards, concepts, hierarchies
