#!/usr/bin/env python3
"""Ingest RAG/LLM textbooks from fixtures/textbooks/rag_llm/ into research-kb.

Follows the same pattern as ingest_missing_textbooks.py:
1. Scans fixtures/textbooks/rag_llm/ for all PDFs
2. Checks which ones are already ingested (by file hash)
3. Ingests missing ones with auto-extracted metadata

These textbooks expand the corpus from causal inference into RAG/LLM/AI domains,
supporting cross-domain knowledge graph queries and interview preparation.

Usage:
    python scripts/ingest_rag_llm_textbooks.py          # Normal output
    python scripts/ingest_rag_llm_textbooks.py --quiet   # Errors + summary only
    python scripts/ingest_rag_llm_textbooks.py --quiet --json  # JSON output
"""

import argparse
import asyncio
import hashlib
import json as json_module
import re
import sys
import time
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import EmbeddingError, StorageError, configure_logging, get_logger
from research_kb_contracts import SourceType
from research_kb_pdf import EmbeddingClient, chunk_with_sections, extract_with_headings
from research_kb_storage import ChunkStore, DatabaseConfig, SourceStore, get_connection_pool

logger = get_logger(__name__)

# Domain tag for all ingested textbooks
DOMAIN = "rag_llm"

# Metadata overrides for known textbooks (filename prefix → metadata)
KNOWN_TEXTBOOKS: dict[str, dict] = {
    "Enterprise_RAG": {
        "title": "Enterprise RAG",
        "authors": [],
        "year": 2025,
    },
    "A_Simple_Guide_to_Retrieval_Augmented": {
        "title": "A Simple Guide to Retrieval Augmented Generation",
        "authors": [],
        "year": 2024,
    },
    "LLMs_in_Production": {
        "title": "LLMs in Production",
        "authors": [],
        "year": 2024,
    },
    "Knowledge_Graphs_and_LLMs": {
        "title": "Knowledge Graphs and LLMs in Action",
        "authors": [],
        "year": 2024,
    },
    "Transformers_in_Action": {
        "title": "Transformers in Action",
        "authors": [],
        "year": 2024,
    },
    "AI_Agents_in_Action": {
        "title": "AI Agents in Action",
        "authors": [],
        "year": 2025,
    },
    "AI_Agents_and_Applications": {
        "title": "AI Agents and Applications",
        "authors": [],
        "year": 2025,
    },
    "Introduction_to_Generative_AI": {
        "title": "Introduction to Generative AI (2nd Edition)",
        "authors": [],
        "year": 2025,
    },
    "Build_an_LLM_Application": {
        "title": "Build an LLM Application from Scratch",
        "authors": [],
        "year": 2025,
    },
    "Building_Reliable_AI_Systems": {
        "title": "Building Reliable AI Systems",
        "authors": [],
        "year": 2025,
    },
    "Graph_Neural_Networks_in_Action": {
        "title": "Graph Neural Networks in Action",
        "authors": [],
        "year": 2024,
    },
    "Financial_AI_in_Practice": {
        "title": "Financial AI in Practice",
        "authors": [],
        "year": 2025,
    },
    "AI_Applications_Made_Easy": {
        "title": "AI Applications Made Easy",
        "authors": [],
        "year": 2025,
    },
}


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def parse_filename_for_metadata(filename: str) -> dict:
    """Extract metadata from filename, using KNOWN_TEXTBOOKS first."""
    for prefix, meta in KNOWN_TEXTBOOKS.items():
        if filename.startswith(prefix):
            return meta.copy()

    # Generic fallback
    name = filename.replace(".pdf", "")
    name = re.sub(r'_v\d+_MEAP.*', '', name)
    name = re.sub(r'\s*\(\d+\)\s*', '', name)
    title = name.replace("_", " ").strip()
    return {"title": title, "authors": [], "year": None}


async def ingest_textbook(
    pdf_path: str,
    title: str,
    authors: list[str],
    year: int | None,
    quiet: bool = False,
) -> tuple[str, int, int]:
    """Ingest a single textbook PDF.

    Returns: (source_id, chunks_created, headings_found)
    """
    if not quiet:
        logger.info("extracting_pdf", path=pdf_path)

    doc, headings = extract_with_headings(pdf_path)

    metadata = {
        "domain": DOMAIN,
        "extraction_method": "pymupdf",
        "total_pages": doc.total_pages,
        "total_chars": doc.total_chars,
        "total_headings": len(headings),
        "auto_ingested": True,
        "source": "ingest_rag_llm_textbooks",
    }

    if not quiet:
        logger.info("chunking_document", path=pdf_path)
    chunks = chunk_with_sections(doc, headings, target_tokens=300)
    metadata["total_chunks"] = len(chunks)

    if not quiet:
        logger.info("chunking_complete", path=pdf_path, chunks=len(chunks))

    file_hash = compute_file_hash(pdf_path)

    if not quiet:
        logger.info("creating_source", title=title)
    source = await SourceStore.create(
        source_type=SourceType.TEXTBOOK,
        title=title,
        authors=authors,
        year=year,
        file_path=pdf_path,
        file_hash=file_hash,
        metadata=metadata,
    )

    embedding_client = EmbeddingClient()
    texts = [chunk.content for chunk in chunks]

    if not quiet:
        logger.info("generating_embeddings", chunks=len(chunks))

    embeddings = embedding_client.embed_batch(texts, batch_size=32)

    chunks_data = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
        chunks_data.append({
            "source_id": source.id,
            "content": chunk.content,
            "content_hash": content_hash,
            "page_start": chunk.start_page,
            "page_end": chunk.end_page,
            "embedding": embedding,
            "metadata": {
                "section_header": chunk.metadata.get("section", ""),
                "chunk_index": i,
                "domain": DOMAIN,
            },
        })

    BATCH_SIZE = 100
    chunks_created = 0
    for i in range(0, len(chunks_data), BATCH_SIZE):
        batch = chunks_data[i:i + BATCH_SIZE]
        await ChunkStore.batch_create(batch)
        chunks_created += len(batch)

    if not quiet:
        logger.info("ingestion_complete",
                    source_id=str(source.id),
                    chunks=chunks_created,
                    headings=len(headings))

    return str(source.id), chunks_created, len(headings)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest RAG/LLM textbooks from fixtures/textbooks/rag_llm/."
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Minimal output (one line per PDF + final summary)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output summary as JSON (for programmatic parsing)",
    )
    return parser.parse_args()


async def main():
    """Ingest all RAG/LLM textbooks."""
    args = parse_args()
    quiet = args.quiet
    json_output = args.json

    if quiet:
        configure_logging(level="ERROR")

    textbooks_dir = Path(__file__).parent.parent / "fixtures" / "textbooks" / "rag_llm"

    if not textbooks_dir.exists():
        if json_output:
            print(json_module.dumps({"error": f"{textbooks_dir} does not exist"}))
        else:
            print(f"Error: {textbooks_dir} does not exist")
        return

    all_pdfs = sorted(textbooks_dir.glob("*.pdf"))
    if not quiet:
        print(f"Found {len(all_pdfs)} PDFs in {textbooks_dir}")

    config = DatabaseConfig()
    await get_connection_pool(config)

    to_ingest = []
    already_ingested = 0

    for pdf_path in all_pdfs:
        file_hash = compute_file_hash(str(pdf_path))
        existing = await SourceStore.get_by_file_hash(file_hash)

        if existing:
            already_ingested += 1
            if not quiet:
                logger.info("already_ingested", path=pdf_path.name, title=existing.title)
        else:
            to_ingest.append(pdf_path)

    if not quiet:
        print(f"Already ingested: {already_ingested}")
        print(f"To ingest: {len(to_ingest)}")

    if not to_ingest:
        if json_output:
            print(json_module.dumps({
                "domain": DOMAIN,
                "success_count": 0,
                "failed_count": 0,
                "total_chunks": 0,
                "already_ingested": already_ingested,
                "failed_files": [],
            }))
        elif not quiet:
            print("Nothing to ingest!")
        return

    results: dict[str, list] = {"success": [], "failed": []}

    for i, pdf_path in enumerate(to_ingest):
        if not quiet and not json_output:
            print(f"\n[{i+1}/{len(to_ingest)}] Processing: {pdf_path.name}")

        meta = parse_filename_for_metadata(pdf_path.name)
        start_time = time.time()

        try:
            source_id, num_chunks, num_headings = await ingest_textbook(
                pdf_path=str(pdf_path),
                title=meta["title"],
                authors=meta["authors"],
                year=meta.get("year"),
                quiet=quiet,
            )

            elapsed = time.time() - start_time
            results["success"].append({
                "file": pdf_path.name,
                "title": meta["title"],
                "chunks": num_chunks,
                "elapsed_seconds": round(elapsed, 1),
            })

            if quiet and not json_output:
                print(f"  {pdf_path.name}: {num_chunks} chunks ({elapsed:.0f}s)")
            elif not quiet and not json_output:
                print(f"  {num_chunks} chunks created")

        except (MemoryError, OSError) as e:
            error_type = "memory_exhausted" if isinstance(e, MemoryError) else "file_io_error"
            results["failed"].append({
                "file": pdf_path.name,
                "error": f"{error_type}: {str(e)[:100]}",
                "recoverable": False,
            })
            if not json_output:
                print(f"  {pdf_path.name}: {error_type}")

        except (EmbeddingError, ConnectionError) as e:
            results["failed"].append({
                "file": pdf_path.name,
                "error": "Embedding service failure (retries exhausted)",
                "recoverable": True,
            })
            if not json_output:
                print(f"  {pdf_path.name}: embedding service failure (recoverable)")

        except StorageError as e:
            results["failed"].append({
                "file": pdf_path.name,
                "error": f"Database error: {str(e)[:100]}",
                "recoverable": True,
            })
            if not json_output:
                print(f"  {pdf_path.name}: database error (recoverable)")

        except Exception as e:
            results["failed"].append({
                "file": pdf_path.name,
                "error": str(e)[:100],
                "recoverable": False,
            })
            if not json_output:
                print(f"  {pdf_path.name}: {str(e)[:60]}")

    total_chunks = sum(r["chunks"] for r in results["success"])

    if json_output:
        summary = {
            "domain": DOMAIN,
            "success_count": len(results["success"]),
            "failed_count": len(results["failed"]),
            "total_chunks": total_chunks,
            "already_ingested": already_ingested,
            "failed_files": [
                {"file": r["file"], "error": r["error"], "recoverable": r.get("recoverable", False)}
                for r in results["failed"]
            ],
        }
        print(json_module.dumps(summary, indent=2))
    elif quiet:
        print(f"\nIngested: {len(results['success'])} textbooks | {total_chunks} chunks")
        if results["failed"]:
            recoverable = sum(1 for r in results["failed"] if r.get("recoverable", False))
            print(f"Failed: {len(results['failed'])} ({recoverable} recoverable)")
    else:
        print("\n" + "=" * 70)
        print(f"RAG/LLM INGESTION SUMMARY (domain: {DOMAIN})")
        print("=" * 70)
        print(f"Success: {len(results['success'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"Total new chunks: {total_chunks}")

        if results["failed"]:
            print("\nFailed files:")
            for r in results["failed"]:
                status = "(recoverable)" if r.get("recoverable", False) else "(not recoverable)"
                print(f"  - {r['file']}: {r['error'][:60]} {status}")

        print(f"\nNext: Run concept extraction overnight:")
        print(f"  nohup python scripts/extract_concepts.py \\")
        print(f"    --backend ollama --model llama3.1:8b \\")
        print(f"    --concurrency 2 --metrics-file /tmp/{DOMAIN}_extraction_metrics.txt \\")
        print(f"    > /tmp/{DOMAIN}_extraction.log 2>&1 &")
        print(f"\n  Then: python scripts/sync_kuzu.py")


if __name__ == "__main__":
    asyncio.run(main())
