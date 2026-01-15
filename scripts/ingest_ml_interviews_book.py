#!/usr/bin/env python3
"""Ingest ML Interviews book by Susan Shu Chang.

This is a one-off script to add the O'Reilly ML Interviews book
to research-kb for the job applications project.

Usage:
    cd ~/Claude/research-kb
    source venv/bin/activate
    python scripts/ingest_ml_interviews_book.py
"""

import asyncio
import glob
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


# Find the book file (uses glob to handle special characters in filename)
_pdf_pattern = "$HOME/Claude/job_applications/Machine Learning Interviews*.pdf"
_pdf_files = glob.glob(_pdf_pattern)
_pdf_file = _pdf_files[0] if _pdf_files else None

# The book to ingest
ML_INTERVIEWS_BOOK = {
    "file": _pdf_file,
    "title": "Machine Learning Interviews: Kickstart Your Machine Learning and Data Career",
    "authors": ["Chang, Susan Shu"],
    "year": 2024,
    "source_type": SourceType.TEXTBOOK,
    "metadata": {
        "publisher": "O'Reilly Media",
        "isbn": "978-1-098-14654-2",
        "domain": "interview_prep",
        "authority": "canonical",
        "companion_url": "https://susanshu.substack.com",
    },
}


async def ingest_book():
    """Ingest the ML Interviews book."""
    book = ML_INTERVIEWS_BOOK
    pdf_path = Path(book["file"])

    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return False

    # Calculate file hash
    sha256_hash = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    file_hash = sha256_hash.hexdigest()

    logger.info(f"Ingesting: {book['title']}")
    logger.info(f"File hash: {file_hash}")

    # Initialize connection pool (global)
    config = DatabaseConfig()
    await get_connection_pool(config)

    try:
        # Check if already ingested
        existing = await SourceStore.get_by_file_hash(file_hash)
        if existing:
            logger.info(f"Already ingested: {existing.title} (ID: {existing.id})")
            return True

        # Extract text with headings
        logger.info("Extracting text...")
        doc, headings = extract_with_headings(str(pdf_path))
        logger.info(f"Extracted {len(doc.pages)} pages, {len(headings)} headings")

        # Chunk with section tracking
        logger.info("Chunking...")
        chunks = chunk_with_sections(doc, headings)
        logger.info(f"Created {len(chunks)} chunks")

        # Create source record
        source = await SourceStore.create(
            source_type=book["source_type"],
            title=book["title"],
            authors=book["authors"],
            year=book["year"],
            file_path=str(pdf_path),
            file_hash=file_hash,
            metadata={
                **book["metadata"],
                "extraction_method": "pymupdf",
                "total_pages": doc.total_pages,
                "total_chars": doc.total_chars,
                "total_headings": len(headings),
                "total_chunks": len(chunks),
            },
        )
        logger.info(f"Created source: {source.id}")

        # Generate embeddings and store chunks
        logger.info("Generating embeddings (this may take a few minutes)...")
        embedding_client = EmbeddingClient()

        chunks_created = 0
        for i, chunk in enumerate(chunks):
            # Sanitize content
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

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks")

        logger.info(f"✅ Successfully ingested {book['title']}")
        logger.info(f"   Source ID: {source.id}")
        logger.info(f"   Chunks: {chunks_created}")
        return True

    except Exception as e:
        logger.error(f"Failed to ingest: {e}")
        raise


if __name__ == "__main__":
    success = asyncio.run(ingest_book())
    sys.exit(0 if success else 1)
