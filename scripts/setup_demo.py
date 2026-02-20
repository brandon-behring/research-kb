#!/usr/bin/env python3
"""Set up the demo corpus: download, ingest, and optionally extract concepts.

Usage:
    python scripts/setup_demo.py                    # Full setup (download + ingest)
    python scripts/setup_demo.py --skip-download     # Ingest only (papers already downloaded)
    python scripts/setup_demo.py --extract           # Also run concept extraction (needs Ollama)
    python scripts/setup_demo.py --seed-only         # Load pre-extracted concepts from fixtures
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

# Add packages to path for development mode
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "storage" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "common" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "contracts" / "src"))

DEMO_DIR = PROJECT_ROOT / "fixtures" / "demo"
PAPERS_DIR = DEMO_DIR / "papers"

# Demo corpus metadata: arXiv papers with domain tags
DEMO_PAPERS = [
    # Causal Inference
    {
        "file": "chernozhukov_dml_2018.pdf",
        "title": "Double/Debiased Machine Learning for Treatment and Structural Parameters",
        "authors": ["Chernozhukov", "Chetverikov", "Demirer", "Duflo", "Hansen", "Newey", "Robins"],
        "year": 2018,
        "domain": "causal_inference",
        "arxiv_id": "1608.00060",
    },
    {
        "file": "wager_athey_causal_forests_2018.pdf",
        "title": "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests",
        "authors": ["Wager", "Athey"],
        "year": 2018,
        "domain": "causal_inference",
        "arxiv_id": "1510.04342",
    },
    {
        "file": "kunzel_metalearners_2019.pdf",
        "title": "Metalearners for Estimating Heterogeneous Treatment Effects",
        "authors": ["Kunzel", "Sekhon", "Bickel", "Yu"],
        "year": 2019,
        "domain": "causal_inference",
        "arxiv_id": "1712.09988",
    },
    {
        "file": "athey_imbens_recursive_partitioning_2016.pdf",
        "title": "Recursive Partitioning for Heterogeneous Causal Effects",
        "authors": ["Athey", "Imbens"],
        "year": 2016,
        "domain": "causal_inference",
        "arxiv_id": "1504.01132",
    },
    {
        "file": "callaway_santanna_staggered_did_2021.pdf",
        "title": "Difference-in-Differences with Multiple Time Periods",
        "authors": ["Callaway", "Sant'Anna"],
        "year": 2021,
        "domain": "causal_inference",
        "arxiv_id": "1803.09015",
    },
    {
        "file": "abadie_zhao_synthetic_controls_2021.pdf",
        "title": "Synthetic Controls: Methods for Comparative Case Studies",
        "authors": ["Abadie", "Zhao"],
        "year": 2021,
        "domain": "causal_inference",
        "arxiv_id": "2108.02196",
    },
    {
        "file": "imai_keele_tingley_mediation_2010.pdf",
        "title": "A General Approach to Causal Mediation Analysis",
        "authors": ["Imai", "Keele", "Tingley"],
        "year": 2010,
        "domain": "causal_inference",
        "arxiv_id": "1011.1079",
    },
    {
        "file": "athey_imbens_state_of_applied_econometrics_2017.pdf",
        "title": "The State of Applied Econometrics: Causality and Policy Evaluation",
        "authors": ["Athey", "Imbens"],
        "year": 2017,
        "domain": "causal_inference",
        "arxiv_id": "1607.00699",
    },
    {
        "file": "athey_imbens_ml_methods_economists_2019.pdf",
        "title": "Machine Learning Methods That Economists Should Know About",
        "authors": ["Athey", "Imbens"],
        "year": 2019,
        "domain": "causal_inference",
        "arxiv_id": "1903.10075",
    },
    # RAG / LLM
    {
        "file": "lewis_rag_2020.pdf",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": [
            "Lewis",
            "Perez",
            "Piktus",
            "Petroni",
            "Karpukhin",
            "Goyal",
            "Kuttler",
            "Lewis",
            "Yih",
            "Rocktaschel",
            "Riedel",
            "Kiela",
        ],
        "year": 2020,
        "domain": "rag_llm",
        "arxiv_id": "2005.11401",
    },
    {
        "file": "gao_rag_survey_2024.pdf",
        "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
        "authors": ["Gao", "Xiong", "Gao", "Jia", "Pan", "Bi", "Dai", "Sun", "Wang", "Wang"],
        "year": 2024,
        "domain": "rag_llm",
        "arxiv_id": "2312.10997",
    },
    {
        "file": "vaswani_attention_2017.pdf",
        "title": "Attention Is All You Need",
        "authors": [
            "Vaswani",
            "Shazeer",
            "Parmar",
            "Uszkoreit",
            "Jones",
            "Gomez",
            "Kaiser",
            "Polosukhin",
        ],
        "year": 2017,
        "domain": "deep_learning",
        "arxiv_id": "1706.03762",
    },
    {
        "file": "wei_chain_of_thought_2022.pdf",
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "authors": ["Wei", "Wang", "Schuurmans", "Bosma", "Xia", "Chi", "Le", "Zhou"],
        "year": 2022,
        "domain": "rag_llm",
        "arxiv_id": "2203.02155",
    },
    {
        "file": "izacard_grave_passage_retrieval_2021.pdf",
        "title": "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering",
        "authors": ["Izacard", "Grave"],
        "year": 2021,
        "domain": "rag_llm",
        "arxiv_id": "2104.08691",
    },
    # Knowledge Graphs + LLMs
    {
        "file": "pan_unifying_llms_kgs_2024.pdf",
        "title": "Unifying Large Language Models and Knowledge Graphs: A Roadmap",
        "authors": ["Pan", "Luo", "Wang", "Chen", "Wang", "Wu"],
        "year": 2024,
        "domain": "rag_llm",
        "arxiv_id": "2306.08302",
    },
    {
        "file": "edge_graphrag_2024.pdf",
        "title": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
        "authors": ["Edge", "Trinh", "Cheng", "Bradley", "Chao", "Mody", "Truitt", "Larson"],
        "year": 2024,
        "domain": "rag_llm",
        "arxiv_id": "2404.16130",
    },
    {
        "file": "hu_empowering_llms_kgs_2023.pdf",
        "title": "Empowering Language Models with Knowledge Graph Reasoning",
        "authors": ["Hu", "Bi", "Wu", "Yao", "Chen", "Frank"],
        "year": 2023,
        "domain": "rag_llm",
        "arxiv_id": "2308.14522",
    },
    # Additional foundational papers
    {
        "file": "brown_gpt3_2020.pdf",
        "title": "Language Models are Few-Shot Learners",
        "authors": ["Brown", "Mann", "Ryder", "Subbiah"],
        "year": 2020,
        "domain": "deep_learning",
        "arxiv_id": "2005.14165",
    },
    {
        "file": "devlin_bert_2019.pdf",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": ["Devlin", "Chang", "Lee", "Toutanova"],
        "year": 2019,
        "domain": "deep_learning",
        "arxiv_id": "1810.04805",
    },
    {
        "file": "touvron_llama2_2023.pdf",
        "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        "authors": ["Touvron", "Martin", "Stone"],
        "year": 2023,
        "domain": "deep_learning",
        "arxiv_id": "2307.09288",
    },
]


def download_papers() -> int:
    """Run the download script. Returns number of papers downloaded."""
    script = PROJECT_ROOT / "scripts" / "download_demo_corpus.sh"
    result = subprocess.run(
        ["bash", str(script), str(PAPERS_DIR)],
        check=True,
    )
    return len(list(PAPERS_DIR.glob("*.pdf")))


async def ingest_papers() -> dict:
    """Ingest demo papers into the database."""
    from research_kb_storage import DatabaseConfig, get_connection_pool

    config = DatabaseConfig()
    pool = await get_connection_pool(config)

    ingested = 0
    skipped = 0
    failed = 0

    try:
        for paper in DEMO_PAPERS:
            pdf_path = PAPERS_DIR / paper["file"]
            if not pdf_path.exists():
                print(f"  [skip] {paper['file']} (not downloaded)")
                skipped += 1
                continue

            # Check if already ingested
            async with pool.acquire() as conn:
                existing = await conn.fetchval(
                    "SELECT id FROM sources WHERE title = $1",
                    paper["title"],
                )
                if existing:
                    print(f"  [exists] {paper['title'][:60]}...")
                    skipped += 1
                    continue

            try:
                # Use the ingestion pipeline
                from research_kb_pdf import extract_pdf, chunk_document
                from research_kb_pdf.embed_client import EmbeddingClient

                print(f"  [ingest] {paper['title'][:60]}...")

                # Extract text
                doc = extract_pdf(str(pdf_path))
                chunks = chunk_document(doc, target_tokens=300, overlap_tokens=50)

                # Store source
                async with pool.acquire() as conn:
                    source_id = await conn.fetchval(
                        """INSERT INTO sources (title, authors, year, source_type, metadata)
                        VALUES ($1, $2, $3, 'paper', $4::jsonb)
                        RETURNING id""",
                        paper["title"],
                        paper["authors"],
                        paper["year"],
                        json.dumps(
                            {
                                "domain": paper["domain"],
                                "arxiv_id": paper.get("arxiv_id"),
                            }
                        ),
                    )

                    # Store chunks
                    for i, chunk in enumerate(chunks):
                        await conn.execute(
                            """INSERT INTO chunks
                            (source_id, chunk_index, content, page_start, page_end)
                            VALUES ($1, $2, $3, $4, $5)""",
                            source_id,
                            i,
                            chunk.text,
                            chunk.page_start if hasattr(chunk, "page_start") else None,
                            chunk.page_end if hasattr(chunk, "page_end") else None,
                        )

                print(f"           -> {len(chunks)} chunks stored")
                ingested += 1

            except Exception as e:
                print(f"  [FAIL] {paper['file']}: {e}")
                failed += 1

    finally:
        await pool.close()

    return {"ingested": ingested, "skipped": skipped, "failed": failed}


async def embed_chunks() -> int:
    """Generate embeddings for all chunks without them."""
    from research_kb_storage import DatabaseConfig, get_connection_pool

    config = DatabaseConfig()
    pool = await get_connection_pool(config)

    try:
        async with pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")
            if count == 0:
                print("  All chunks already have embeddings.")
                return 0

            print(f"  {count} chunks need embeddings...")
            print("  (Embedding requires the embed server running)")
            print("  Start with: python -m research_kb_pdf.embed_server &")
            return count
    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser(description="Set up the research-kb demo corpus")
    parser.add_argument("--skip-download", action="store_true", help="Skip paper download")
    parser.add_argument(
        "--extract", action="store_true", help="Run concept extraction (needs Ollama)"
    )
    parser.add_argument(
        "--seed-only", action="store_true", help="Load pre-extracted concepts from fixtures"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("research-kb Demo Corpus Setup")
    print("=" * 60)

    # Step 1: Download
    if not args.skip_download and not args.seed_only:
        print("\n--- Step 1: Downloading papers from arXiv ---")
        count = download_papers()
        print(f"  {count} papers available in {PAPERS_DIR}")
    else:
        print("\n--- Step 1: Download skipped ---")

    # Step 2: Ingest
    if not args.seed_only:
        print("\n--- Step 2: Ingesting papers into PostgreSQL ---")
        result = asyncio.run(ingest_papers())
        print(
            f"  Result: {result['ingested']} ingested, {result['skipped']} skipped, {result['failed']} failed"
        )

        # Step 3: Embed
        print("\n--- Step 3: Checking embeddings ---")
        need_embed = asyncio.run(embed_chunks())
        if need_embed > 0:
            print(f"  Run: python -m research_kb_pdf.embed_server &")
            print(f"  Then: python scripts/embed_missing.py")

    # Step 4: Optional concept extraction
    if args.extract:
        print("\n--- Step 4: Extracting concepts (Ollama) ---")
        subprocess.run(
            [
                sys.executable,
                "scripts/extract_concepts.py",
                "--backend",
                "ollama",
                "--limit",
                "500",
            ],
            check=True,
        )
        print("\n--- Step 5: Syncing KuzuDB ---")
        subprocess.run(
            [sys.executable, "scripts/sync_kuzu.py"],
            check=True,
        )

    # Step 5: Pre-extracted seed data
    if args.seed_only:
        seed_file = DEMO_DIR / "seed_data.sql"
        if seed_file.exists():
            print("\n--- Loading pre-extracted seed data ---")
            subprocess.run(
                ["psql", "-f", str(seed_file)],
                env={"PGDATABASE": "research_kb"},
                check=True,
            )
        else:
            print(f"\n  Seed file not found: {seed_file}")
            print("  Run full setup instead: python scripts/setup_demo.py")

    print("\n" + "=" * 60)
    print("Setup complete! Try:")
    print('  research-kb query "instrumental variables"')
    print("  research-kb stats")
    print("=" * 60)


if __name__ == "__main__":
    main()
