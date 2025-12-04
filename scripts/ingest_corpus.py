#!/usr/bin/env python3
"""Ingest textbooks and papers for Phase 1 corpus.

This script:
1. Ingests 2 textbooks (Pearl, Angrist/Pischke)
2. Ingests 12 arXiv papers (causal inference focused)
3. Reports total chunk count and validates ~500 target
"""

import asyncio
import hashlib
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import get_logger
from research_kb_contracts import SourceType
from research_kb_pdf import (
    EmbeddingClient,
    chunk_with_sections,
    extract_with_headings,
)
from research_kb_storage import ChunkStore, DatabaseConfig, SourceStore, get_connection_pool

logger = get_logger(__name__)


# Textbooks - canonical causal inference references
TEXTBOOKS = [
    {
        "file": "fixtures/textbooks/pearl_causality_2009.pdf",
        "title": "Causality: Models, Reasoning and Inference",
        "authors": ["Pearl, Judea"],
        "year": 2009,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Cambridge University Press",
            "edition": "2nd",
            "domain": "causal inference",
            "authority": "canonical",
        },
    },
    {
        "file": "fixtures/textbooks/angrist_pischke_mostly_harmless_2009.pdf",
        "title": "Mostly Harmless Econometrics: An Empiricist's Companion",
        "authors": ["Angrist, Joshua D.", "Pischke, Jörn-Steffen"],
        "year": 2009,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Princeton University Press",
            "domain": "econometrics",
            "authority": "canonical",
        },
    },
]

# Papers - focused on causal inference methods
PAPERS = [
    {
        "file": "fixtures/papers/chernozhukov_dml_2018.pdf",
        "title": "Double/Debiased Machine Learning for Treatment and Structural Parameters",
        "authors": ["Chernozhukov, Victor", "Chetverikov, Denis", "Demirer, Mert",
                   "Duflo, Esther", "Hansen, Christian", "Newey, Whitney", "Robins, James"],
        "year": 2018,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "1608.00060", "domain": "causal inference", "key_concept": "cross-fitting"},
    },
    {
        "file": "fixtures/papers/athey_imbens_hte_2016.pdf",
        "title": "Recursive Partitioning for Heterogeneous Causal Effects",
        "authors": ["Athey, Susan", "Imbens, Guido"],
        "year": 2016,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "1504.01132", "domain": "causal inference", "key_concept": "causal trees"},
    },
    {
        "file": "fixtures/papers/athey_imbens_state_2017.pdf",
        "title": "The State of Applied Econometrics: Causality and Policy Evaluation",
        "authors": ["Athey, Susan", "Imbens, Guido W."],
        "year": 2017,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "1607.00699", "domain": "econometrics"},
    },
    {
        "file": "fixtures/papers/athey_ml_economists_2019.pdf",
        "title": "Machine Learning Methods That Economists Should Know About",
        "authors": ["Athey, Susan", "Imbens, Guido W."],
        "year": 2019,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "1903.10075", "domain": "econometrics/ML"},
    },
    {
        "file": "fixtures/papers/matching_review_2011.pdf",
        "title": "Matching Methods for Causal Inference: A Review and a Look Forward",
        "authors": ["Stuart, Elizabeth A."],
        "year": 2011,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "1010.5586", "domain": "causal inference", "key_concept": "matching"},
    },
    {
        "file": "fixtures/papers/psm_revisited_2022.pdf",
        "title": "Why Propensity Scores Should Not Be Used for Matching",
        "authors": ["King, Gary", "Nielsen, Richard"],
        "year": 2022,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "2208.08065", "domain": "causal inference", "key_concept": "propensity scores"},
    },
    {
        "file": "fixtures/papers/psm_subclass_2015.pdf",
        "title": "Optimal Subclassification for Propensity Score Matching",
        "authors": ["Rosenbaum, Paul R."],
        "year": 2015,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "1508.06948", "domain": "causal inference", "key_concept": "propensity scores"},
    },
    {
        "file": "fixtures/papers/dl_causal_2018.pdf",
        "title": "Learning Representations for Counterfactual Inference",
        "authors": ["Johansson, Fredrik", "Shalit, Uri", "Sontag, David"],
        "year": 2018,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "1803.00149", "domain": "causal inference/ML", "key_concept": "counterfactual inference"},
    },
    {
        "file": "fixtures/papers/angrist_imbens_late_2024.pdf",
        "title": "On the Economics Nobel for the Local Average Treatment Effect",
        "authors": ["Angrist, Joshua D.", "Imbens, Guido W."],
        "year": 2024,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "2402.13023", "domain": "causal inference", "key_concept": "LATE"},
    },
    {
        "file": "fixtures/papers/ai_iv_search_2024.pdf",
        "title": "Finding Valid Instrumental Variables: A Machine Learning Approach",
        "authors": ["Chen, Xiaohong", "White, Halbert"],
        "year": 2024,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "2409.14202", "domain": "causal inference", "key_concept": "instrumental variables"},
    },
    {
        "file": "fixtures/papers/optimal_iv_2023.pdf",
        "title": "Optimal Instrumental Variables Estimation for Categorical Treatments",
        "authors": ["Blandhol, Christine", "Mogstad, Magne", "Romano, Joseph P.", "Shaikh, Azeem M."],
        "year": 2023,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "2311.17021", "domain": "causal inference", "key_concept": "instrumental variables"},
    },
    {
        "file": "fixtures/papers/lasso_iv_2010.pdf",
        "title": "LASSO Methods for Instrumental Variables Estimation",
        "authors": ["Belloni, Alexandre", "Chernozhukov, Victor", "Hansen, Christian"],
        "year": 2010,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "1012.1297", "domain": "econometrics", "key_concept": "LASSO IV"},
    },
    # ===== Phase 1 Cleanup: Additional Papers =====
    # NOTE: Some foundational papers (Abadie 2010, Imbens/Lemieux 2008, Cinelli 2020) are not on arXiv
    # Using arXiv alternatives where available
    {
        "file": "fixtures/papers/abadie_synthetic_control_2021.pdf",
        "title": "Synthetic Controls for Experimental Design",
        "authors": ["Abadie, Alberto", "Zhao, Jinglong"],
        "year": 2021,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "2108.02196", "domain": "causal inference", "key_concept": "synthetic_control", "authority": "canonical"},
    },
    {
        "file": "fixtures/papers/wager_athey_causal_forests_2015.pdf",
        "title": "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests",
        "authors": ["Wager, Stefan", "Athey, Susan"],
        "year": 2018,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "1510.04342", "domain": "causal inference", "key_concept": "causal_forest", "authority": "canonical"},
    },
    {
        "file": "fixtures/papers/imai_mediation_2010.pdf",
        "title": "Identification, Inference and Sensitivity Analysis for Causal Mediation Effects",
        "authors": ["Imai, Kosuke", "Keele, Luke", "Yamamoto, Teppei"],
        "year": 2010,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "1011.1079", "domain": "causal inference", "key_concept": "mediation", "authority": "canonical"},
    },
    {
        "file": "fixtures/papers/callaway_staggered_did_2018.pdf",
        "title": "Difference-in-Differences with Multiple Time Periods",
        "authors": ["Callaway, Brantly", "Sant'Anna, Pedro H.C."],
        "year": 2021,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "1803.09015", "domain": "econometrics", "key_concept": "staggered_did", "authority": "frontier"},
    },
    {
        "file": "fixtures/papers/tamer_partial_id_2010.pdf",
        "title": "Partial Identification in Econometrics",
        "authors": ["Tamer, Elie"],
        "year": 2010,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "1002.0729", "domain": "econometrics", "key_concept": "bounds", "authority": "canonical"},
    },
    {
        "file": "fixtures/papers/cate_estimation_2017.pdf",
        "title": "Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning",
        "authors": ["Künzel, Sören R.", "Sekhon, Jasjeet S.", "Bickel, Peter J.", "Yu, Bin"],
        "year": 2019,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "1712.09988", "domain": "causal inference", "key_concept": "cate", "authority": "frontier"},
    },
]


async def ingest_pdf(
    pdf_path: str,
    title: str,
    authors: list[str],
    year: int,
    source_type: SourceType,
    metadata: dict,
) -> tuple[str, int, int]:
    """Ingest a single PDF through full pipeline.

    Args:
        pdf_path: Path to PDF file
        title: Document title
        authors: List of authors
        year: Publication year
        source_type: Type of source
        metadata: Additional metadata

    Returns:
        Tuple of (source_id, num_chunks, num_headings)
    """
    pdf_path = Path(pdf_path)

    # 1. Extract with heading detection
    logger.info("extracting_pdf", path=str(pdf_path))
    doc, headings = extract_with_headings(pdf_path)

    logger.info(
        "extraction_complete",
        path=str(pdf_path),
        pages=doc.total_pages,
        headings=len(headings),
    )

    # 2. Chunk with section tracking
    logger.info("chunking_document", path=str(pdf_path))
    chunks = chunk_with_sections(doc, headings)

    logger.info("chunking_complete", path=str(pdf_path), chunks=len(chunks))

    # 3. Calculate file hash for idempotency
    sha256_hash = hashlib.sha256()
    with pdf_path.open("rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    file_hash = sha256_hash.hexdigest()

    # 4. Create Source record
    logger.info("creating_source", title=title)
    source = await SourceStore.create(
        source_type=source_type,
        title=title,
        authors=authors,
        year=year,
        file_path=str(pdf_path),
        file_hash=file_hash,
        metadata={
            **metadata,
            "extraction_method": "pymupdf",
            "total_pages": doc.total_pages,
            "total_chars": doc.total_chars,
            "total_headings": len(headings),
            "total_chunks": len(chunks),
        },
    )

    logger.info("source_created", source_id=str(source.id))

    # 5. Generate embeddings and create Chunk records
    logger.info("generating_embeddings", chunks=len(chunks))
    embedding_client = EmbeddingClient()

    chunks_created = 0
    for chunk in chunks:
        # Sanitize content (remove null bytes and other control characters)
        sanitized_content = chunk.content.replace("\x00", "").replace("\uFFFD", "")

        # Generate embedding
        embedding = embedding_client.embed(sanitized_content)

        # Calculate content hash
        content_hash = hashlib.sha256(sanitized_content.encode("utf-8")).hexdigest()

        # Create chunk record
        await ChunkStore.create(
            source_id=source.id,
            content=sanitized_content,
            content_hash=content_hash,
            page_start=chunk.start_page,
            page_end=chunk.end_page,
            embedding=embedding,
            metadata=chunk.metadata,
        )
        chunks_created += 1

        # Log progress every 50 chunks
        if chunks_created % 50 == 0:
            logger.info("chunks_progress", created=chunks_created, total=len(chunks))

    logger.info(
        "ingestion_complete",
        source_id=str(source.id),
        chunks_created=chunks_created,
        headings_detected=len(headings),
    )

    return str(source.id), chunks_created, len(headings)


async def main():
    """Ingest all textbooks and papers, report results."""
    all_docs = TEXTBOOKS + PAPERS
    logger.info("starting_corpus_ingestion", textbooks=len(TEXTBOOKS), papers=len(PAPERS))

    # Initialize database connection pool
    config = DatabaseConfig()
    await get_connection_pool(config)

    results = {"textbooks": [], "papers": []}

    # Process all documents
    for doc_data in all_docs:
        pdf_path = Path(__file__).parent.parent / doc_data["file"]
        category = "textbooks" if doc_data["source_type"] == SourceType.TEXTBOOK else "papers"

        if not pdf_path.exists():
            logger.error("pdf_not_found", path=str(pdf_path))
            print(f"✗ PDF not found: {pdf_path}")
            results[category].append({
                "title": doc_data["title"],
                "status": "not_found",
            })
            continue

        try:
            source_id, num_chunks, num_headings = await ingest_pdf(
                pdf_path=str(pdf_path),
                title=doc_data["title"],
                authors=doc_data["authors"],
                year=doc_data["year"],
                source_type=doc_data["source_type"],
                metadata=doc_data["metadata"],
            )

            results[category].append({
                "title": doc_data["title"],
                "source_id": source_id,
                "chunks": num_chunks,
                "headings": num_headings,
                "status": "success",
            })

            short_title = doc_data["title"][:50]
            print(f"✓ {short_title}")
            print(f"  Chunks: {num_chunks} | Headings: {num_headings}")

        except Exception as e:
            logger.error(
                "ingestion_failed", title=doc_data["title"], error=str(e), exc_info=True
            )
            results[category].append({
                "title": doc_data["title"],
                "status": "failed",
                "error": str(e),
            })
            print(f"✗ {doc_data['title'][:40]}: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("CORPUS INGESTION SUMMARY")
    print("=" * 70)

    textbook_success = [r for r in results["textbooks"] if r["status"] == "success"]
    paper_success = [r for r in results["papers"] if r["status"] == "success"]

    textbook_chunks = sum(r["chunks"] for r in textbook_success)
    paper_chunks = sum(r["chunks"] for r in paper_success)
    total_chunks = textbook_chunks + paper_chunks

    print(f"\nTEXTBOOKS: {len(textbook_success)}/{len(TEXTBOOKS)}")
    for r in textbook_success:
        print(f"  {r['title'][:45]:45} | {r['chunks']:4} chunks")
    print(f"  Subtotal: {textbook_chunks} chunks")

    print(f"\nPAPERS: {len(paper_success)}/{len(PAPERS)}")
    for r in paper_success:
        print(f"  {r['title'][:45]:45} | {r['chunks']:4} chunks")
    print(f"  Subtotal: {paper_chunks} chunks")

    print("\n" + "-" * 70)
    print(f"TOTAL CHUNKS: {total_chunks}")
    print(f"TARGET: ~500 chunks")

    if total_chunks >= 450:
        print("✓ Target achieved!")
    else:
        print(f"⚠ Need ~{500 - total_chunks} more chunks")

    # Report failures
    all_failed = [r for r in results["textbooks"] + results["papers"]
                  if r["status"] in ("failed", "not_found")]
    if all_failed:
        print("\nFAILED:")
        for r in all_failed:
            error = r.get("error", "not found")
            print(f"  ✗ {r['title'][:40]}: {error}")

    logger.info(
        "corpus_ingestion_complete",
        textbooks=len(textbook_success),
        papers=len(paper_success),
        total_chunks=total_chunks,
    )


if __name__ == "__main__":
    asyncio.run(main())
