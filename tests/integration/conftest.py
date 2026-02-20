"""Shared fixtures for integration tests.

Provides database connection pool and synthetic test data for validating
the full search pipeline without real PDFs, embedding servers, or LLMs.

Uses the same db_pool pattern as packages/storage/conftest.py.
"""

import math
import os
import random
from uuid import uuid4

import pytest_asyncio
from research_kb_contracts import ConceptType, RelationshipType, SourceType
from research_kb_storage import (
    ChunkConceptStore,
    ChunkStore,
    CitationStore,
    ConceptStore,
    DatabaseConfig,
    RelationshipStore,
    SourceStore,
    close_connection_pool,
    get_connection_pool,
)


# Test database name - NEVER use production database for tests
TEST_DATABASE_NAME = os.environ.get("TEST_DATABASE_NAME", "research_kb_test")
PRODUCTION_DATABASE_NAME = "research_kb"


class ProductionDatabaseError(Exception):
    """Raised when test attempts to modify production database."""

    pass


def _verify_not_production(database_name: str) -> None:
    """Safety check: refuse to run destructive operations on production DB."""
    if database_name == PRODUCTION_DATABASE_NAME:
        raise ProductionDatabaseError(
            f"REFUSING to run test fixture against production database "
            f"'{PRODUCTION_DATABASE_NAME}'!\n"
            f"Tests must use '{TEST_DATABASE_NAME}' or another test database.\n"
            f"Set TEST_DATABASE_NAME environment variable to override."
        )


@pytest_asyncio.fixture(scope="function")
async def db_pool():
    """Create database connection pool for each test function.

    This fixture:
    - Creates a fresh connection pool for each test
    - Cleans the database before the test runs
    - Closes the pool after the test completes
    - REFUSES to connect to production database
    """
    _verify_not_production(TEST_DATABASE_NAME)

    await close_connection_pool()

    config = DatabaseConfig(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=TEST_DATABASE_NAME,
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
    )
    pool = await get_connection_pool(config)

    async with pool.acquire() as conn:
        current_db = await conn.fetchval("SELECT current_database()")
        _verify_not_production(current_db)

        await conn.execute(
            "TRUNCATE TABLE chunk_concepts, concept_relationships, "
            "chunks, concepts, sources, citations, methods, assumptions CASCADE"
        )

    yield pool

    await close_connection_pool()


# ---------------------------------------------------------------------------
# Synthetic embedding helpers
# ---------------------------------------------------------------------------


def _make_embedding(seed: int, dim: int = 1024) -> list[float]:
    """Generate a deterministic pseudo-random unit vector.

    Different seeds produce roughly orthogonal vectors in high dimensions.
    Same seed always returns the same vector (deterministic tests).
    """
    rng = random.Random(seed)
    raw = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _blend_embeddings(base: list[float], target: list[float], alpha: float = 0.7) -> list[float]:
    """Blend two embeddings: result = alpha * base + (1-alpha) * target.

    Higher alpha → closer to base. Returns normalized vector.
    Used to create embeddings with controlled cosine similarity.
    """
    blended = [alpha * b + (1 - alpha) * t for b, t in zip(base, target)]
    norm = math.sqrt(sum(x * x for x in blended))
    return [x / norm for x in blended]


# ---------------------------------------------------------------------------
# Seed vectors (deterministic, roughly orthogonal in 1024-dim)
# ---------------------------------------------------------------------------

# Topic clusters — each seed produces a distinct direction
IV_VECTOR = _make_embedding(seed=42)  # instrumental variables cluster
DML_VECTOR = _make_embedding(seed=99)  # double ML cluster
RAG_VECTOR = _make_embedding(seed=200)  # RAG/LLM cluster
UNRELATED_VECTOR = _make_embedding(seed=777)  # noise / unrelated


@pytest_asyncio.fixture
async def search_corpus(db_pool):
    """Seed a minimal but realistic corpus for search pipeline tests.

    Creates:
    - 5 sources (3 textbooks + 2 papers, across causal_inference + rag_llm)
    - ~25 chunks with controlled embeddings
    - 15 concepts (methods, assumptions, definitions) with relationships
    - 3 citation links
    - chunk-concept links

    Returns a dict with all created entities for assertion in tests.
    """
    # ------------------------------------------------------------------
    # 0. Seed domains (FK target for concepts and chunks)
    # ------------------------------------------------------------------
    # Use ON CONFLICT so this is idempotent (domains may already exist
    # from migrations in the local test DB).
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO domains (id, name, description, concept_types, relationship_types)
            VALUES
                ('causal_inference', 'Causal Inference',
                 'Econometrics and causal ML',
                 ARRAY['method','assumption','problem','definition','theorem','concept','principle','technique','model'],
                 ARRAY['REQUIRES','USES','ADDRESSES','GENERALIZES','SPECIALIZES','ALTERNATIVE_TO','EXTENDS','RELATED_TO']),
                ('rag_llm', 'RAG & LLM',
                 'Retrieval-augmented generation and language models',
                 ARRAY['method','technique','model','concept','problem','definition','assumption','theorem','principle'],
                 ARRAY['REQUIRES','USES','ADDRESSES','GENERALIZES','SPECIALIZES','ALTERNATIVE_TO','EXTENDS','RELATED_TO'])
            ON CONFLICT (id) DO NOTHING
        """
        )

    # ------------------------------------------------------------------
    # 1. Sources
    # ------------------------------------------------------------------
    src_pearl = await SourceStore.create(
        source_type=SourceType.TEXTBOOK,
        title="Causality: Models, Reasoning, and Inference",
        authors=["Pearl, Judea"],
        year=2009,
        file_hash=f"sha256:pearl_{uuid4().hex[:8]}",
        metadata={"domain": "causal_inference"},
    )

    src_angrist = await SourceStore.create(
        source_type=SourceType.TEXTBOOK,
        title="Mostly Harmless Econometrics",
        authors=["Angrist, Joshua", "Pischke, Jorn-Steffen"],
        year=2008,
        file_hash=f"sha256:angrist_{uuid4().hex[:8]}",
        metadata={"domain": "causal_inference"},
    )

    src_chernozhukov = await SourceStore.create(
        source_type=SourceType.PAPER,
        title="Double/Debiased Machine Learning for Treatment Effects",
        authors=["Chernozhukov, Victor", "Chetverikov, Denis"],
        year=2018,
        file_hash=f"sha256:chernoz_{uuid4().hex[:8]}",
        metadata={"domain": "causal_inference"},
    )

    src_rag_textbook = await SourceStore.create(
        source_type=SourceType.TEXTBOOK,
        title="Retrieval-Augmented Generation: Foundations and Applications",
        authors=["Smith, Alice"],
        year=2024,
        file_hash=f"sha256:rag_tb_{uuid4().hex[:8]}",
        metadata={"domain": "rag_llm"},
        domain_id="rag_llm",
    )

    src_rag_paper = await SourceStore.create(
        source_type=SourceType.PAPER,
        title="Improving RAG with Knowledge Graphs",
        authors=["Jones, Bob"],
        year=2024,
        file_hash=f"sha256:rag_paper_{uuid4().hex[:8]}",
        metadata={"domain": "rag_llm"},
        domain_id="rag_llm",
    )

    sources = {
        "pearl": src_pearl,
        "angrist": src_angrist,
        "chernozhukov": src_chernozhukov,
        "rag_textbook": src_rag_textbook,
        "rag_paper": src_rag_paper,
    }

    # ------------------------------------------------------------------
    # 2. Citations (chernozhukov cites pearl + angrist; rag_paper cites rag_textbook)
    # ------------------------------------------------------------------
    cit1 = await CitationStore.create(
        source_id=src_chernozhukov.id,
        raw_string="Pearl, J. (2009). Causality. Cambridge University Press.",
        title="Causality",
        authors=["Pearl, Judea"],
        year=2009,
        extraction_method="synthetic",
        metadata={"cited_source_id": str(src_pearl.id)},
    )
    cit2 = await CitationStore.create(
        source_id=src_chernozhukov.id,
        raw_string="Angrist and Pischke (2008). Mostly Harmless Econometrics.",
        title="Mostly Harmless Econometrics",
        authors=["Angrist, Joshua", "Pischke, Jorn-Steffen"],
        year=2008,
        extraction_method="synthetic",
        metadata={"cited_source_id": str(src_angrist.id)},
    )
    cit3 = await CitationStore.create(
        source_id=src_rag_paper.id,
        raw_string="Smith (2024). RAG Foundations.",
        title="Retrieval-Augmented Generation: Foundations and Applications",
        authors=["Smith, Alice"],
        year=2024,
        extraction_method="synthetic",
        metadata={"cited_source_id": str(src_rag_textbook.id)},
    )

    # ------------------------------------------------------------------
    # 3. Chunks with controlled embeddings
    # ------------------------------------------------------------------
    chunks = {}

    # --- Causal inference chunks (IV cluster) ---
    chunks["iv_definition"] = await ChunkStore.create(
        source_id=src_pearl.id,
        content=(
            "An instrumental variable is a variable that affects the treatment "
            "but has no direct effect on the outcome except through the treatment. "
            "Instrumental variables provide identification when unobserved "
            "confounding is present."
        ),
        content_hash=f"sha256:iv_def_{uuid4().hex[:8]}",
        location="Chapter 7, p. 247",
        embedding=IV_VECTOR,
        metadata={"chunk_type": "definition"},
    )

    chunks["iv_assumptions"] = await ChunkStore.create(
        source_id=src_angrist.id,
        content=(
            "The instrumental variables estimator requires three key assumptions: "
            "relevance (the instrument must affect treatment), exclusion restriction "
            "(the instrument affects outcome only through treatment), and independence "
            "(the instrument is as-if randomly assigned)."
        ),
        content_hash=f"sha256:iv_assum_{uuid4().hex[:8]}",
        location="Chapter 4, p. 113",
        embedding=_blend_embeddings(IV_VECTOR, DML_VECTOR, alpha=0.85),
        metadata={"chunk_type": "theorem"},
    )

    chunks["iv_example"] = await ChunkStore.create(
        source_id=src_angrist.id,
        content=(
            "A classic example of instrumental variables is the use of quarter "
            "of birth as an instrument for years of schooling in estimating "
            "the returns to education."
        ),
        content_hash=f"sha256:iv_ex_{uuid4().hex[:8]}",
        location="Chapter 4, p. 130",
        embedding=_blend_embeddings(IV_VECTOR, UNRELATED_VECTOR, alpha=0.6),
        metadata={"chunk_type": "example"},
    )

    # --- DML cluster ---
    chunks["dml_overview"] = await ChunkStore.create(
        source_id=src_chernozhukov.id,
        content=(
            "Double machine learning uses cross-fitting and Neyman-orthogonal "
            "scores to provide root-n consistent and approximately unbiased "
            "estimates of treatment effects using machine learning nuisance "
            "parameter estimates."
        ),
        content_hash=f"sha256:dml_over_{uuid4().hex[:8]}",
        location="Section 1, p. 1",
        embedding=DML_VECTOR,
        metadata={"chunk_type": "overview"},
    )

    chunks["dml_assumptions"] = await ChunkStore.create(
        source_id=src_chernozhukov.id,
        content=(
            "Double machine learning requires unconfoundedness (conditional "
            "independence of treatment and potential outcomes given covariates), "
            "overlap (bounded propensity scores), and sufficient convergence "
            "rates for nuisance estimators."
        ),
        content_hash=f"sha256:dml_assum_{uuid4().hex[:8]}",
        location="Section 2, p. 5",
        embedding=_blend_embeddings(DML_VECTOR, IV_VECTOR, alpha=0.8),
        metadata={"chunk_type": "theorem"},
    )

    # --- RAG cluster ---
    chunks["rag_overview"] = await ChunkStore.create(
        source_id=src_rag_textbook.id,
        content=(
            "Retrieval-augmented generation combines a retrieval system with "
            "a language model. The retriever finds relevant documents from a "
            "knowledge base, and the generator conditions its output on both "
            "the query and retrieved context."
        ),
        content_hash=f"sha256:rag_over_{uuid4().hex[:8]}",
        location="Chapter 1, p. 1",
        embedding=RAG_VECTOR,
        metadata={"chunk_type": "overview"},
        domain_id="rag_llm",
    )

    chunks["rag_chunking"] = await ChunkStore.create(
        source_id=src_rag_textbook.id,
        content=(
            "Effective chunking strategies are critical for RAG quality. "
            "Chunks must be semantically coherent and sized to fit the context "
            "window while preserving enough context for meaningful retrieval."
        ),
        content_hash=f"sha256:rag_chunk_{uuid4().hex[:8]}",
        location="Chapter 3, p. 45",
        embedding=_blend_embeddings(RAG_VECTOR, UNRELATED_VECTOR, alpha=0.7),
        metadata={"chunk_type": "technique"},
        domain_id="rag_llm",
    )

    chunks["rag_kg_integration"] = await ChunkStore.create(
        source_id=src_rag_paper.id,
        content=(
            "Knowledge graph integration with RAG systems improves factual "
            "accuracy by providing structured relationships between entities. "
            "Graph-enhanced retrieval can surface relevant context that pure "
            "vector similarity misses."
        ),
        content_hash=f"sha256:rag_kg_{uuid4().hex[:8]}",
        location="Section 3, p. 12",
        embedding=_blend_embeddings(RAG_VECTOR, IV_VECTOR, alpha=0.75),
        metadata={"chunk_type": "technique"},
        domain_id="rag_llm",
    )

    # --- Unrelated noise chunk ---
    chunks["unrelated"] = await ChunkStore.create(
        source_id=src_pearl.id,
        content=(
            "The history of probability theory stretches back to the work "
            "of Pascal and Fermat in the seventeenth century, well before "
            "the modern formalization by Kolmogorov."
        ),
        content_hash=f"sha256:unrelated_{uuid4().hex[:8]}",
        location="Appendix A, p. 380",
        embedding=UNRELATED_VECTOR,
        metadata={"chunk_type": "historical"},
    )

    # ------------------------------------------------------------------
    # 4. Concepts + relationships
    # ------------------------------------------------------------------
    concepts = {}

    # Methods
    concepts["iv_method"] = await ConceptStore.create(
        name="Instrumental Variables",
        canonical_name=f"instrumental_variables_{uuid4().hex[:8]}",
        concept_type=ConceptType.METHOD,
        aliases=["IV", "2SLS", "TSLS"],
        definition="Method for estimating causal effects using exogenous variation",
        domain_id="causal_inference",
    )

    concepts["dml_method"] = await ConceptStore.create(
        name="Double Machine Learning",
        canonical_name=f"double_machine_learning_{uuid4().hex[:8]}",
        concept_type=ConceptType.METHOD,
        aliases=["DML", "debiased ML"],
        definition="Cross-fitting approach for ML-based causal inference",
        domain_id="causal_inference",
    )

    concepts["rag_method"] = await ConceptStore.create(
        name="Retrieval-Augmented Generation",
        canonical_name=f"rag_{uuid4().hex[:8]}",
        concept_type=ConceptType.METHOD,
        aliases=["RAG"],
        definition="Combining retrieval with generation for grounded LLM output",
        domain_id="rag_llm",
    )

    # Assumptions
    concepts["exclusion"] = await ConceptStore.create(
        name="Exclusion Restriction",
        canonical_name=f"exclusion_restriction_{uuid4().hex[:8]}",
        concept_type=ConceptType.ASSUMPTION,
        definition="Instrument affects outcome only through treatment",
        domain_id="causal_inference",
    )

    concepts["relevance"] = await ConceptStore.create(
        name="Relevance",
        canonical_name=f"relevance_{uuid4().hex[:8]}",
        concept_type=ConceptType.ASSUMPTION,
        definition="Instrument must be correlated with treatment",
        domain_id="causal_inference",
    )

    concepts["unconfoundedness"] = await ConceptStore.create(
        name="Unconfoundedness",
        canonical_name=f"unconfoundedness_{uuid4().hex[:8]}",
        concept_type=ConceptType.ASSUMPTION,
        aliases=["CIA", "conditional independence"],
        definition="No unmeasured confounders given covariates",
        domain_id="causal_inference",
    )

    concepts["overlap"] = await ConceptStore.create(
        name="Overlap",
        canonical_name=f"overlap_{uuid4().hex[:8]}",
        concept_type=ConceptType.ASSUMPTION,
        definition="Positive probability of treatment for all covariate values",
        domain_id="causal_inference",
    )

    # Definitions
    concepts["confounding"] = await ConceptStore.create(
        name="Confounding",
        canonical_name=f"confounding_{uuid4().hex[:8]}",
        concept_type=ConceptType.DEFINITION,
        definition="When a common cause affects both treatment and outcome",
        domain_id="causal_inference",
    )

    concepts["causal_effect"] = await ConceptStore.create(
        name="Causal Effect",
        canonical_name=f"causal_effect_{uuid4().hex[:8]}",
        concept_type=ConceptType.DEFINITION,
        definition="The effect of a treatment on an outcome in a causal model",
        domain_id="causal_inference",
    )

    # RAG domain concepts (use DEFINITION type for compatibility with
    # test databases that may not have migration 004 applied)
    concepts["vector_search"] = await ConceptStore.create(
        name="Vector Search",
        canonical_name=f"vector_search_{uuid4().hex[:8]}",
        concept_type=ConceptType.DEFINITION,
        definition="Finding similar items by embedding distance",
        domain_id="rag_llm",
    )

    concepts["chunking"] = await ConceptStore.create(
        name="Chunking",
        canonical_name=f"chunking_{uuid4().hex[:8]}",
        concept_type=ConceptType.DEFINITION,
        definition="Splitting documents into retrieval units",
        domain_id="rag_llm",
    )

    # ------------------------------------------------------------------
    # 5. Relationships
    # ------------------------------------------------------------------
    # IV --REQUIRES--> exclusion restriction
    await RelationshipStore.create(
        source_concept_id=concepts["iv_method"].id,
        target_concept_id=concepts["exclusion"].id,
        relationship_type=RelationshipType.REQUIRES,
        strength=0.95,
        confidence_score=0.9,
    )

    # IV --REQUIRES--> relevance
    await RelationshipStore.create(
        source_concept_id=concepts["iv_method"].id,
        target_concept_id=concepts["relevance"].id,
        relationship_type=RelationshipType.REQUIRES,
        strength=0.95,
        confidence_score=0.9,
    )

    # IV --ADDRESSES--> confounding
    await RelationshipStore.create(
        source_concept_id=concepts["iv_method"].id,
        target_concept_id=concepts["confounding"].id,
        relationship_type=RelationshipType.ADDRESSES,
        strength=0.85,
        confidence_score=0.85,
    )

    # DML --REQUIRES--> unconfoundedness
    await RelationshipStore.create(
        source_concept_id=concepts["dml_method"].id,
        target_concept_id=concepts["unconfoundedness"].id,
        relationship_type=RelationshipType.REQUIRES,
        strength=0.95,
        confidence_score=0.9,
    )

    # DML --REQUIRES--> overlap
    await RelationshipStore.create(
        source_concept_id=concepts["dml_method"].id,
        target_concept_id=concepts["overlap"].id,
        relationship_type=RelationshipType.REQUIRES,
        strength=0.90,
        confidence_score=0.85,
    )

    # RAG --USES--> vector_search
    await RelationshipStore.create(
        source_concept_id=concepts["rag_method"].id,
        target_concept_id=concepts["vector_search"].id,
        relationship_type=RelationshipType.USES,
        strength=0.9,
        confidence_score=0.9,
    )

    # RAG --USES--> chunking
    await RelationshipStore.create(
        source_concept_id=concepts["rag_method"].id,
        target_concept_id=concepts["chunking"].id,
        relationship_type=RelationshipType.USES,
        strength=0.85,
        confidence_score=0.85,
    )

    # ------------------------------------------------------------------
    # 6. Chunk-concept links
    # ------------------------------------------------------------------
    # IV chunks linked to IV concepts
    await ChunkConceptStore.create(
        chunk_id=chunks["iv_definition"].id,
        concept_id=concepts["iv_method"].id,
        mention_type="defines",
        relevance_score=0.95,
    )
    await ChunkConceptStore.create(
        chunk_id=chunks["iv_assumptions"].id,
        concept_id=concepts["iv_method"].id,
        mention_type="reference",
        relevance_score=0.9,
    )
    await ChunkConceptStore.create(
        chunk_id=chunks["iv_assumptions"].id,
        concept_id=concepts["exclusion"].id,
        mention_type="defines",
        relevance_score=0.95,
    )
    await ChunkConceptStore.create(
        chunk_id=chunks["iv_assumptions"].id,
        concept_id=concepts["relevance"].id,
        mention_type="defines",
        relevance_score=0.9,
    )

    # DML chunks linked to DML concepts
    await ChunkConceptStore.create(
        chunk_id=chunks["dml_overview"].id,
        concept_id=concepts["dml_method"].id,
        mention_type="defines",
        relevance_score=0.95,
    )
    await ChunkConceptStore.create(
        chunk_id=chunks["dml_assumptions"].id,
        concept_id=concepts["dml_method"].id,
        mention_type="reference",
        relevance_score=0.85,
    )
    await ChunkConceptStore.create(
        chunk_id=chunks["dml_assumptions"].id,
        concept_id=concepts["unconfoundedness"].id,
        mention_type="defines",
        relevance_score=0.9,
    )

    # RAG chunks linked to RAG concepts
    await ChunkConceptStore.create(
        chunk_id=chunks["rag_overview"].id,
        concept_id=concepts["rag_method"].id,
        mention_type="defines",
        relevance_score=0.95,
    )
    await ChunkConceptStore.create(
        chunk_id=chunks["rag_chunking"].id,
        concept_id=concepts["chunking"].id,
        mention_type="reference",
        relevance_score=0.9,
    )
    await ChunkConceptStore.create(
        chunk_id=chunks["rag_kg_integration"].id,
        concept_id=concepts["vector_search"].id,
        mention_type="reference",
        relevance_score=0.8,
    )

    return {
        "sources": sources,
        "chunks": chunks,
        "concepts": concepts,
        "citations": [cit1, cit2, cit3],
    }
