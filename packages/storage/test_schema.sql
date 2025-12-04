-- Test Script: Validate Schema with 3 Sample Sources
-- Purpose: Verify schema works for textbook, paper, and code repo
--
-- Run after: docker-compose up postgres
-- Command: psql -U postgres -d research_kb < test_schema.sql

-- ===========================================================================
-- Setup: Clean slate
-- ===========================================================================
TRUNCATE TABLE chunks CASCADE;
TRUNCATE TABLE sources CASCADE;

-- ===========================================================================
-- Test 1: Textbook Source (Pearl's Causality)
-- ===========================================================================
WITH textbook AS (
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
        '/test/pearl_causality.pdf',
        'test_hash_textbook_001',
        jsonb_build_object(
            'isbn', '978-0521895606',
            'publisher', 'Cambridge University Press',
            'total_pages', 464
        )
    ) RETURNING id
)
INSERT INTO chunks (source_id, content, content_hash, location, page_start, page_end, metadata)
SELECT
    id,
    'The backdoor criterion states that a set of variables Z satisfies the backdoor criterion relative to (X, Y) if...',
    'test_chunk_textbook_001',
    'Chapter 3, Section 3.3, p. 73',
    73,
    74,
    jsonb_build_object(
        'chunk_type', 'theorem',
        'chapter_num', 3,
        'section_num', '3.3',
        'theorem_name', 'Backdoor Criterion'
    )
FROM textbook;

-- ===========================================================================
-- Test 2: Research Paper Source (Chernozhukov 2018)
-- ===========================================================================
WITH paper AS (
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
        'Double/debiased machine learning',
        ARRAY['Victor Chernozhukov', 'Denis Chetverikov'],
        2018,
        '/test/chernozhukov_2018.pdf',
        'test_hash_paper_001',
        jsonb_build_object(
            'doi', '10.1111/ectj.12097',
            'journal', 'Econometrics Journal',
            'authority_tier', 'canonical'
        )
    ) RETURNING id
)
INSERT INTO chunks (source_id, content, content_hash, location, page_start, page_end, metadata)
SELECT
    id,
    'We revisit the classic semiparametric problem of inference on a low-dimensional parameter θ0...',
    'test_chunk_paper_001',
    'Abstract, p. 1',
    1,
    1,
    jsonb_build_object(
        'chunk_type', 'abstract',
        'section', 'abstract',
        'key_concepts', ARRAY['double_machine_learning', 'cross_fitting']
    )
FROM paper;

-- ===========================================================================
-- Test 3: Code Repository Source (scikit-learn)
-- ===========================================================================
WITH code_repo AS (
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
        'scikit-learn/linear_model',
        ARRAY['scikit-learn developers'],
        2023,
        '/test/scikit-learn',
        'test_hash_code_001',
        jsonb_build_object(
            'git_url', 'https://github.com/scikit-learn/scikit-learn',
            'language', 'python',
            'license', 'BSD-3-Clause'
        )
    ) RETURNING id
)
INSERT INTO chunks (source_id, content, content_hash, location, metadata)
SELECT
    id,
    'class LogisticRegression(BaseEstimator):\n    """Logistic Regression classifier."""\n    pass',
    'test_chunk_code_001',
    'sklearn/linear_model/_logistic.py:lines 1200-1250',
    jsonb_build_object(
        'chunk_type', 'class_definition',
        'file_path', 'sklearn/linear_model/_logistic.py',
        'start_line', 1200,
        'end_line', 1250,
        'class_name', 'LogisticRegression'
    )
FROM code_repo;

-- ===========================================================================
-- Validation Queries
-- ===========================================================================

\echo ''
\echo '========================================='
\echo 'Validation Test 1: Sources by Type'
\echo '========================================='
SELECT
    source_type,
    title,
    array_length(authors, 1) AS author_count,
    year,
    jsonb_object_keys(metadata) AS metadata_keys
FROM sources
ORDER BY source_type;

\echo ''
\echo '========================================='
\echo 'Validation Test 2: Chunks by Source Type'
\echo '========================================='
SELECT
    s.source_type,
    s.title,
    c.location,
    c.metadata->>'chunk_type' AS chunk_type,
    length(c.content) AS content_length
FROM chunks c
JOIN sources s ON c.source_id = s.id
ORDER BY s.source_type;

\echo ''
\echo '========================================='
\echo 'Validation Test 3: JSONB Queries'
\echo '========================================='

\echo 'Textbook theorems:'
SELECT
    s.title,
    c.metadata->>'theorem_name' AS theorem_name,
    c.metadata->>'chapter_num' AS chapter
FROM chunks c
JOIN sources s ON c.source_id = s.id
WHERE c.metadata->>'chunk_type' = 'theorem';

\echo ''
\echo 'Papers by authority tier:'
SELECT
    title,
    metadata->>'authority_tier' AS tier,
    metadata->>'journal' AS journal
FROM sources
WHERE source_type = 'paper'
  AND metadata ? 'authority_tier';

\echo ''
\echo 'Code chunks by language:'
SELECT
    s.title,
    c.metadata->>'file_path' AS file,
    c.metadata->>'class_name' AS class_name
FROM chunks c
JOIN sources s ON c.source_id = s.id
WHERE s.metadata->>'language' = 'python';

\echo ''
\echo '========================================='
\echo 'Validation Test 4: FTS Index'
\echo '========================================='
SELECT
    s.title,
    c.location,
    ts_rank(c.fts_vector, plainto_tsquery('english', 'backdoor criterion')) AS rank
FROM chunks c
JOIN sources s ON c.source_id = s.id
WHERE c.fts_vector @@ plainto_tsquery('english', 'backdoor criterion')
ORDER BY rank DESC;

\echo ''
\echo '========================================='
\echo 'Validation Test 5: Idempotency (file_hash UNIQUE)'
\echo '========================================='
\echo 'Attempting to insert duplicate source (should fail):'
DO $$
BEGIN
    INSERT INTO sources (source_type, title, authors, file_hash)
    VALUES ('textbook', 'Duplicate Test', ARRAY['Test'], 'test_hash_textbook_001');
    RAISE NOTICE 'ERROR: Duplicate insert succeeded (should have failed!)';
EXCEPTION WHEN unique_violation THEN
    RAISE NOTICE 'SUCCESS: Duplicate prevented by UNIQUE constraint on file_hash';
END $$;

\echo ''
\echo '========================================='
\echo 'Validation Test 6: Cascade Delete'
\echo '========================================='
SELECT COUNT(*) AS total_chunks_before_delete FROM chunks;

DELETE FROM sources WHERE file_hash = 'test_hash_code_001';

SELECT COUNT(*) AS total_chunks_after_delete FROM chunks;
SELECT COUNT(*) AS deleted_source_chunks
FROM chunks c
LEFT JOIN sources s ON c.source_id = s.id
WHERE s.id IS NULL;

\echo ''
\echo '========================================='
\echo 'Summary: All Tests'
\echo '========================================='
SELECT
    (SELECT COUNT(*) FROM sources) AS total_sources,
    (SELECT COUNT(*) FROM chunks) AS total_chunks,
    (SELECT COUNT(DISTINCT source_type) FROM sources) AS distinct_source_types;

\echo ''
\echo '✓ Schema validation complete!'
\echo 'Next: Set up Docker Compose and run this script'
