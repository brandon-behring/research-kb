"""PDF Dispatcher - Orchestrates ingestion with GROBID→PyMuPDF fallback.

Provides:
- PDFDispatcher class for orchestrating PDF ingestion
- GROBID→PyMuPDF fallback logic for metadata extraction
- Full chunking and embedding pipeline
- Integration with Dead Letter Queue for error tracking
- Idempotency via file hash checking

Phase 1.5.1: Extended to include chunking + embedding storage.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from research_kb_common import get_logger
from research_kb_contracts import Source, SourceType

from research_kb_pdf.bibtex_generator import citation_to_bibtex
from research_kb_pdf.chunker import chunk_with_sections, TextChunk
from research_kb_pdf.dlq import DeadLetterQueue
from research_kb_pdf.embedding_client import EmbeddingClient
from research_kb_pdf.grobid_client import GrobidClient, ExtractedPaper
from research_kb_pdf.pymupdf_extractor import extract_with_headings

# Will be imported from storage package
try:
    from research_kb_storage import ChunkStore, CitationStore, SourceStore
except ImportError:
    # For testing without storage package installed
    SourceStore = None  # type: ignore
    ChunkStore = None  # type: ignore
    CitationStore = None  # type: ignore

logger = get_logger(__name__)


@dataclass
class IngestResult:
    """Result of PDF ingestion including source and chunk statistics.

    Attributes:
        source: Created Source record
        chunk_count: Number of chunks created
        headings_detected: Number of headings detected for section tracking
        extraction_method: Method used for text extraction ("grobid" or "pymupdf")
        grobid_metadata_extracted: Whether GROBID was used for metadata
        citations_extracted: Number of citations extracted and stored
    """

    source: Source
    chunk_count: int
    headings_detected: int
    extraction_method: str
    grobid_metadata_extracted: bool = False
    citations_extracted: int = 0


class PDFDispatcher:
    """Orchestrates PDF ingestion with GROBID→PyMuPDF fallback.

    Pipeline (Phase 1.5.1 extended):
    1. Calculate file hash for idempotency check
    2. Check if already ingested via SourceStore.get_by_file_hash()
    3. Try GROBID for metadata extraction (if service available)
    4. Extract text with PyMuPDF (always, for chunking)
    5. Detect headings and chunk with section tracking
    6. Generate embeddings via EmbeddingClient
    7. Store chunks via ChunkStore.batch_create
    8. Log failures to DLQ for manual review
    9. Return IngestResult with source and chunk statistics

    Example:
        >>> dispatcher = PDFDispatcher(
        ...     grobid_url="http://localhost:8070",
        ...     dlq_path="data/dlq/failed_pdfs.jsonl"
        ... )
        >>> result = await dispatcher.ingest_pdf(
        ...     pdf_path="/data/paper.pdf",
        ...     source_type=SourceType.PAPER,
        ...     title="Deep Learning",
        ...     authors=["LeCun", "Bengio"],
        ... )
        >>> print(f"Created {result.chunk_count} chunks")
    """

    def __init__(
        self,
        grobid_url: str = "http://localhost:8070",
        dlq_path: str | Path = "data/dlq/failed_pdfs.jsonl",
        embedding_socket_path: str = "/tmp/research_kb_embed.sock",
    ):
        """Initialize PDF dispatcher.

        Args:
            grobid_url: GROBID service URL
            dlq_path: Path to dead letter queue JSONL file
            embedding_socket_path: Path to embedding server Unix socket
        """
        self.grobid_client = GrobidClient(grobid_url=grobid_url)
        self.dlq = DeadLetterQueue(dlq_path)
        self.embed_client = EmbeddingClient(socket_path=embedding_socket_path)
        logger.info(
            "dispatcher_initialized",
            grobid_url=grobid_url,
            dlq_path=str(dlq_path),
            embedding_socket=embedding_socket_path,
        )

    def _calculate_file_hash(self, pdf_path: str | Path) -> str:
        """Calculate SHA256 hash of PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            SHA256 hex digest

        Raises:
            FileNotFoundError: If PDF doesn't exist
        """
        pdf_path = Path(pdf_path)  # Convert to Path if string
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        sha256_hash = hashlib.sha256()
        with pdf_path.open("rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    async def ingest_pdf(
        self,
        pdf_path: str | Path,
        source_type: SourceType,
        title: str,
        domain_id: str,
        authors: Optional[list[str]] = None,
        year: Optional[int] = None,
        metadata: Optional[dict] = None,
        force_pymupdf: bool = False,
        skip_embedding: bool = False,
    ) -> IngestResult:
        """Ingest a PDF with full chunking and embedding pipeline.

        Phase 1.5.1: Extended to include chunking + embedding + storage.

        Args:
            pdf_path: Path to PDF file
            source_type: Type of source (textbook, paper, code_repo)
            title: Source title
            authors: List of author names
            year: Publication year
            metadata: Optional metadata (doi, arxiv_id, etc.)
            force_pymupdf: Skip GROBID metadata extraction
            skip_embedding: Skip embedding generation (for testing)

        Returns:
            IngestResult with source and chunk statistics

        Raises:
            FileNotFoundError: If PDF doesn't exist
            ValueError: If ingestion fails completely (logged to DLQ)
            ConnectionError: If embedding server not available

        Example:
            >>> result = await dispatcher.ingest_pdf(
            ...     pdf_path="papers/attention.pdf",
            ...     source_type=SourceType.PAPER,
            ...     title="Attention Is All You Need",
            ...     authors=["Vaswani", "Shazeer"],
            ...     year=2017,
            ...     metadata={"arxiv_id": "1706.03762"}
            ... )
            >>> print(f"Created {result.chunk_count} chunks")
        """
        pdf_path = Path(pdf_path)

        # Calculate file hash for idempotency
        try:
            file_hash = self._calculate_file_hash(pdf_path)
            logger.info("file_hash_calculated", path=str(pdf_path), hash=file_hash[:16])
        except FileNotFoundError:
            logger.error("pdf_not_found", path=str(pdf_path))
            raise

        # Check for existing source (idempotency)
        if SourceStore is not None:
            existing = await SourceStore.get_by_file_hash(file_hash)
            if existing:
                # Return existing with chunk count
                chunk_count = (
                    await ChunkStore.count_by_source(existing.id) if ChunkStore is not None else 0
                )
                logger.info(
                    "source_already_exists",
                    source_id=str(existing.id),
                    file_path=str(pdf_path),
                    chunk_count=chunk_count,
                )
                return IngestResult(
                    source=existing,
                    chunk_count=chunk_count,
                    headings_detected=existing.metadata.get("total_headings", 0),
                    extraction_method=existing.metadata.get("extraction_method", "unknown"),
                    grobid_metadata_extracted="grobid"
                    in existing.metadata.get("extraction_method", ""),
                )

        logger.info(
            "ingestion_started",
            path=str(pdf_path),
            source_type=source_type.value,
        )

        # Try GROBID for metadata extraction (optional, per /iterate Q1)
        grobid_metadata = {}
        grobid_doc: Optional[ExtractedPaper] = None
        grobid_used = False
        if not force_pymupdf:
            try:
                if self.grobid_client.is_alive():
                    grobid_doc = self.grobid_client.process_pdf(pdf_path)
                    grobid_metadata = {
                        "grobid_sections": len(grobid_doc.sections),
                        "grobid_abstract": (
                            grobid_doc.metadata.abstract[:200]
                            if grobid_doc.metadata.abstract
                            else None
                        ),
                        "grobid_citations_count": len(grobid_doc.citations),
                    }
                    grobid_used = True
                    logger.info(
                        "grobid_metadata_extracted",
                        path=str(pdf_path),
                        sections=len(grobid_doc.sections),
                        citations=len(grobid_doc.citations),
                    )
                else:
                    logger.warning(
                        "grobid_not_available",
                        path=str(pdf_path),
                        fallback="pymupdf_only",
                    )
            except Exception as e:
                logger.warning(
                    "grobid_extraction_failed",
                    path=str(pdf_path),
                    error=str(e),
                    fallback="pymupdf_only",
                )

        # Always use PyMuPDF for text extraction and chunking
        try:
            # Extract with heading detection
            doc, headings = extract_with_headings(pdf_path)

            logger.info(
                "pymupdf_extraction_success",
                path=str(pdf_path),
                pages=doc.total_pages,
                chars=doc.total_chars,
                headings=len(headings),
            )

            # Chunk with section tracking
            chunks = chunk_with_sections(doc, headings)
            logger.info("chunking_complete", path=str(pdf_path), chunks=len(chunks))

            # Create Source record
            if SourceStore is None:
                raise ValueError("SourceStore not available (testing mode)")

            source = await SourceStore.create(
                source_type=source_type,
                title=title,
                file_hash=file_hash,
                domain_id=domain_id,
                authors=authors or [],
                year=year,
                file_path=str(pdf_path),
                metadata={
                    **(metadata or {}),
                    **grobid_metadata,
                    "extraction_method": "grobid+pymupdf" if grobid_used else "pymupdf",
                    "total_pages": doc.total_pages,
                    "total_chars": doc.total_chars,
                    "total_headings": len(headings),
                    "total_chunks": len(chunks),
                },
            )
            logger.info("source_created", source_id=str(source.id))

            # Generate embeddings and store chunks
            chunks_created = await self._store_chunks_with_embeddings(
                source=source,
                chunks=chunks,
                skip_embedding=skip_embedding,
            )

            # Store citations if GROBID extracted them (Phase 1.5.2)
            citations_stored = 0
            if grobid_doc and grobid_doc.citations and CitationStore is not None:
                citations_stored = await self._store_citations(
                    source=source,
                    grobid_doc=grobid_doc,
                )

            logger.info(
                "ingestion_complete",
                source_id=str(source.id),
                chunks_created=chunks_created,
                citations_stored=citations_stored,
                headings_detected=len(headings),
            )

            return IngestResult(
                source=source,
                chunk_count=chunks_created,
                headings_detected=len(headings),
                extraction_method="grobid+pymupdf" if grobid_used else "pymupdf",
                grobid_metadata_extracted=grobid_used,
                citations_extracted=citations_stored,
            )

        except Exception as extraction_error:
            # Extraction failed - log to DLQ
            logger.error(
                "extraction_failed",
                path=str(pdf_path),
                error=str(extraction_error),
            )

            dlq_entry = self.dlq.add(
                file_path=pdf_path,
                error=extraction_error,
                metadata={
                    "source_type": source_type.value,
                    "title": title,
                    "domain_id": domain_id,
                    "error": str(extraction_error),
                    "file_size": pdf_path.stat().st_size,
                },
            )

            raise ValueError(
                f"PDF ingestion failed (DLQ entry: {dlq_entry.id}). Error: {extraction_error}"
            ) from extraction_error

    async def _store_chunks_with_embeddings(
        self,
        source: Source,
        chunks: list[TextChunk],
        skip_embedding: bool = False,
    ) -> int:
        """Generate embeddings and store chunks to database.

        Args:
            source: Parent Source record
            chunks: List of TextChunk objects from chunker
            skip_embedding: Skip embedding generation (for testing)

        Returns:
            Number of chunks created

        Raises:
            ConnectionError: If embedding server not available
            StorageError: If chunk storage fails
        """
        if ChunkStore is None:
            raise ValueError("ChunkStore not available (testing mode)")

        if not chunks:
            return 0

        logger.info("generating_embeddings", source_id=str(source.id), chunks=len(chunks))

        # Prepare chunk data for batch creation
        chunks_data = []

        for chunk in chunks:
            # Sanitize content (remove null bytes and control characters)
            sanitized_content = chunk.content.replace("\x00", "").replace("\ufffd", "")

            # Calculate content hash for idempotency
            content_hash = hashlib.sha256(sanitized_content.encode("utf-8")).hexdigest()

            # Generate embedding (unless skipped)
            embedding = None
            if not skip_embedding:
                try:
                    embedding = self.embed_client.embed(sanitized_content)
                except ConnectionError as e:
                    logger.error(
                        "embedding_server_unavailable",
                        source_id=str(source.id),
                        error=str(e),
                    )
                    raise

            chunks_data.append(
                {
                    "source_id": source.id,
                    "content": sanitized_content,
                    "content_hash": content_hash,
                    "page_start": chunk.start_page,
                    "page_end": chunk.end_page,
                    "embedding": embedding,
                    "metadata": chunk.metadata,
                }
            )

            # Log progress every 50 chunks
            if len(chunks_data) % 50 == 0:
                logger.info(
                    "embedding_progress",
                    source_id=str(source.id),
                    processed=len(chunks_data),
                    total=len(chunks),
                )

        # Batch create all chunks
        created_chunks = await ChunkStore.batch_create(chunks_data)

        logger.info(
            "chunks_stored",
            source_id=str(source.id),
            count=len(created_chunks),
        )

        return len(created_chunks)

    async def _store_citations(
        self,
        source: Source,
        grobid_doc: ExtractedPaper,
    ) -> int:
        """Store citations extracted by GROBID with generated BibTeX.

        Phase 1.5.2: Integration of GROBID citations with CitationStore.

        Args:
            source: Parent Source record
            grobid_doc: GROBID extraction result containing citations

        Returns:
            Number of citations stored

        Raises:
            StorageError: If citation storage fails
        """
        if CitationStore is None:
            raise ValueError("CitationStore not available (testing mode)")

        if not grobid_doc.citations:
            return 0

        logger.info(
            "storing_citations",
            source_id=str(source.id),
            count=len(grobid_doc.citations),
        )

        # Prepare citation data for batch creation
        citations_data = []
        for citation in grobid_doc.citations:
            # Generate BibTeX entry
            try:
                bibtex = citation_to_bibtex(citation)
            except Exception as e:
                logger.warning(
                    "bibtex_generation_failed",
                    source_id=str(source.id),
                    title=citation.title,
                    error=str(e),
                )
                bibtex = None

            citations_data.append(
                {
                    "source_id": source.id,
                    "authors": citation.authors,
                    "title": citation.title,
                    "year": citation.year,
                    "venue": citation.venue,
                    "doi": citation.doi,
                    "arxiv_id": citation.arxiv_id,
                    "raw_string": citation.raw_string,
                    "bibtex": bibtex,
                    "extraction_method": "grobid",
                    "confidence_score": None,  # GROBID doesn't provide confidence
                    "metadata": {},
                }
            )

        # Batch create all citations
        created_citations = await CitationStore.batch_create(citations_data)

        logger.info(
            "citations_stored",
            source_id=str(source.id),
            count=len(created_citations),
        )

        return len(created_citations)

    async def retry_from_dlq(self, entry_id: str) -> Optional[IngestResult]:
        """Retry ingestion for a DLQ entry.

        Args:
            entry_id: UUID of the DLQ entry

        Returns:
            IngestResult if successful, None if entry not found

        Raises:
            ValueError: If retry fails again

        Example:
            >>> result = await dispatcher.retry_from_dlq(entry.id)
            >>> if result:
            ...     print(f"Created {result.chunk_count} chunks")
        """
        entry = self.dlq.get(entry_id)
        if not entry:
            logger.warning("dlq_entry_not_found", entry_id=entry_id)
            return None

        logger.info(
            "dlq_retry_started",
            entry_id=entry_id,
            file_path=entry.file_path,
            retry_count=entry.retry_count,
        )

        # Extract metadata from DLQ entry
        source_type_str = entry.metadata.get("source_type", "paper")
        source_type = SourceType(source_type_str)
        title = entry.metadata.get("title", "Unknown")
        domain_id = entry.metadata.get("domain_id", "causal_inference")

        try:
            # Retry ingestion
            result = await self.ingest_pdf(
                pdf_path=entry.file_path,
                source_type=source_type,
                title=title,
                domain_id=domain_id,
            )

            # Success - remove from DLQ
            self.dlq.remove(entry_id)
            logger.info(
                "dlq_retry_success",
                entry_id=entry_id,
                source_id=str(result.source.id),
                chunks=result.chunk_count,
            )
            return result

        except Exception as e:
            # Retry failed - add back to DLQ with incremented retry count
            self.dlq.add(
                file_path=entry.file_path,
                error=e,
                retry_count=entry.retry_count + 1,
                metadata=entry.metadata,
            )

            # Remove old entry
            self.dlq.remove(entry_id)

            logger.error(
                "dlq_retry_failed",
                entry_id=entry_id,
                retry_count=entry.retry_count + 1,
                error=str(e),
            )
            raise
