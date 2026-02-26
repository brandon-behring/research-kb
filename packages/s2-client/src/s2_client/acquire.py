"""Paper acquisition module with deduplication.

Downloads open-access papers and saves them to the fixtures directory,
with deduplication checks against existing sources.
"""

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from research_kb_common import get_logger

from s2_client.errors import S2Error
from s2_client.models import S2Paper

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Default fixtures directory (relative to research-kb)
DEFAULT_FIXTURES_DIR = Path.home() / "Claude" / "research-kb" / "fixtures" / "papers"


class AcquisitionError(S2Error):
    """Error during paper acquisition."""

    pass


@dataclass
class AcquisitionResult:
    """Result of an acquisition run."""

    acquired: list[tuple[S2Paper, Path]] = field(default_factory=list)
    skipped_existing: list[S2Paper] = field(default_factory=list)
    skipped_paywall: list[S2Paper] = field(default_factory=list)
    skipped_no_url: list[S2Paper] = field(default_factory=list)
    failed: list[tuple[S2Paper, str]] = field(default_factory=list)
    acquisition_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_summary_dict(self) -> dict[str, Any]:
        """Convert to summary dict."""
        return {
            "acquired": len(self.acquired),
            "skipped_existing": len(self.skipped_existing),
            "skipped_paywall": len(self.skipped_paywall),
            "skipped_no_url": len(self.skipped_no_url),
            "failed": len(self.failed),
            "total_processed": (
                len(self.acquired)
                + len(self.skipped_existing)
                + len(self.skipped_paywall)
                + len(self.skipped_no_url)
                + len(self.failed)
            ),
            "acquisition_time": self.acquisition_time.isoformat(),
        }


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Sanitize text for use in filename.

    Args:
        text: Text to sanitize
        max_length: Maximum length of result

    Returns:
        Sanitized string safe for filenames
    """
    # Remove/replace problematic characters
    text = re.sub(r'[<>:"/\\|?*]', "", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\-.]", "", text)
    text = text.lower()

    return text[:max_length]


def generate_filename(paper: S2Paper) -> str:
    """Generate standardized filename for a paper.

    Format: {first_author_lastname}_{title_snippet}_{year}.pdf

    Args:
        paper: S2Paper to generate filename for

    Returns:
        Generated filename
    """
    # Get first author's last name
    author = "unknown"
    if paper.first_author_name:
        parts = paper.first_author_name.split()
        if parts:
            author = sanitize_filename(parts[-1], max_length=20)

    # Get title snippet
    title = sanitize_filename(paper.title or "untitled", max_length=30)

    # Get year
    year = paper.year or "nd"

    return f"{author}_{title}_{year}.pdf"


def compute_file_hash(content: bytes) -> str:
    """Compute SHA256 hash of file content.

    Args:
        content: File bytes

    Returns:
        Hex-encoded SHA256 hash
    """
    return hashlib.sha256(content).hexdigest()


class PaperAcquisition:
    """Handles downloading and deduplicating papers.

    Example:
        >>> async with PaperAcquisition() as acq:
        ...     result = await acq.acquire_papers(papers)
        ...     print(f"Acquired {len(result.acquired)} papers")

    Attributes:
        fixtures_dir: Directory to save downloaded papers
        existing_hashes: Set of file hashes for existing papers
        existing_s2_ids: Set of S2 paper IDs already in DB
        existing_dois: Set of DOIs already in DB
        existing_arxiv_ids: Set of arXiv IDs already in DB
    """

    def __init__(
        self,
        fixtures_dir: Path | None = None,
        existing_hashes: set[str] | None = None,
        existing_s2_ids: set[str] | None = None,
        existing_dois: set[str] | None = None,
        existing_arxiv_ids: set[str] | None = None,
    ) -> None:
        """Initialize acquisition handler.

        Args:
            fixtures_dir: Directory to save papers
            existing_hashes: File hashes of existing papers
            existing_s2_ids: S2 paper IDs already in database
            existing_dois: DOIs already in database
            existing_arxiv_ids: arXiv IDs already in database
        """
        self.fixtures_dir = fixtures_dir or DEFAULT_FIXTURES_DIR
        self.existing_hashes = existing_hashes or set()
        self.existing_s2_ids = existing_s2_ids or set()
        self.existing_dois = existing_dois or set()
        self.existing_arxiv_ids = existing_arxiv_ids or set()
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "PaperAcquisition":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            follow_redirects=True,
            headers={"User-Agent": "s2-client/1.0.0 (research-kb acquisition)"},
        )
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()

    def is_duplicate(self, paper: S2Paper) -> bool:
        """Check if paper already exists in database.

        Uses multiple identifier checks:
        1. S2 paper ID
        2. DOI
        3. arXiv ID

        File hash is checked after download.

        Args:
            paper: Paper to check

        Returns:
            True if paper exists (duplicate)
        """
        if paper.paper_id and paper.paper_id in self.existing_s2_ids:
            return True
        if paper.doi and paper.doi in self.existing_dois:
            return True
        if paper.arxiv_id and paper.arxiv_id in self.existing_arxiv_ids:
            return True
        return False

    def get_pdf_url(self, paper: S2Paper) -> str | None:
        """Get best available PDF URL for paper.

        Priority:
        1. S2 open access PDF URL
        2. arXiv PDF URL

        Args:
            paper: Paper to get URL for

        Returns:
            PDF URL or None if not available
        """
        # Check S2 open access URL
        if paper.open_access_pdf and paper.open_access_pdf.url:
            return paper.open_access_pdf.url

        # Fall back to arXiv
        if paper.arxiv_id:
            return f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"

        return None

    async def download_pdf(self, url: str) -> bytes | None:
        """Download PDF from URL.

        Args:
            url: URL to download from

        Returns:
            PDF bytes or None on failure
        """
        if not self._client:
            raise AcquisitionError("Client not initialized. Use async context manager.")

        try:
            logger.info("Downloading PDF", url=url)
            response = await self._client.get(url)
            response.raise_for_status()

            # Verify it's actually a PDF
            content = response.content
            if not content.startswith(b"%PDF"):
                logger.warning("Downloaded content is not a PDF", url=url)
                return None

            return content

        except httpx.HTTPStatusError as e:
            logger.error("HTTP error downloading PDF", url=url, status=e.response.status_code)
            return None
        except httpx.RequestError as e:
            logger.error("Request error downloading PDF", url=url, error=str(e))
            return None

    async def acquire_paper(self, paper: S2Paper) -> tuple[Path | None, str | None]:
        """Acquire a single paper.

        Args:
            paper: Paper to acquire

        Returns:
            Tuple of (save_path, error_message). save_path is None on failure.
        """
        # Get PDF URL
        pdf_url = self.get_pdf_url(paper)
        if not pdf_url:
            return None, "no_url"

        # Download
        content = await self.download_pdf(pdf_url)
        if not content:
            return None, "download_failed"

        # Check for duplicate by hash
        file_hash = compute_file_hash(content)
        if file_hash in self.existing_hashes:
            return None, "duplicate_hash"

        # Generate filename and save
        filename = generate_filename(paper)
        save_path = self.fixtures_dir / filename

        # Handle collision
        if save_path.exists():
            # Add hash prefix to make unique
            base = save_path.stem
            save_path = self.fixtures_dir / f"{base}_{file_hash[:8]}.pdf"

        save_path.write_bytes(content)

        # Save S2 metadata sidecar for ingestion pipeline
        sidecar_path = save_path.with_suffix(".s2.json")
        self._save_metadata_sidecar(paper, sidecar_path, file_hash)

        # Track this hash to prevent duplicates within same run
        self.existing_hashes.add(file_hash)

        logger.info(
            "Paper acquired",
            title=paper.title[:50] if paper.title else "Unknown",
            path=str(save_path),
            sidecar=str(sidecar_path),
            size_kb=len(content) // 1024,
        )

        return save_path, None

    def _save_metadata_sidecar(self, paper: S2Paper, sidecar_path: Path, file_hash: str) -> None:
        """Save S2 metadata as JSON sidecar for ingestion pipeline.

        The sidecar contains rich metadata that would otherwise be lost
        when using filename-based extraction.

        Args:
            paper: S2Paper with metadata
            sidecar_path: Path to write JSON sidecar
            file_hash: SHA256 hash of the PDF content
        """
        import json
        from datetime import datetime, timezone

        sidecar_data = {
            # Core metadata for ingest_pdf()
            "title": paper.title,
            "authors": [a.name for a in (paper.authors or []) if a.name],
            "year": paper.year,
            # S2-specific metadata (stored in sources.metadata JSONB)
            "s2_paper_id": paper.paper_id,
            "s2_corpus_id": paper.corpus_id,
            "doi": paper.doi,
            "arxiv_id": paper.arxiv_id,
            "citation_count": paper.citation_count,
            "influential_citation_count": paper.influential_citation_count,
            "is_open_access": paper.is_open_access,
            "fields_of_study": [f.get("category") for f in (paper.s2_fields_of_study or [])],
            "venue": paper.venue,
            "abstract": paper.abstract,
            # Provenance
            "file_hash": file_hash,
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "sidecar_version": "1.0",
        }

        sidecar_path.write_text(json.dumps(sidecar_data, indent=2, ensure_ascii=False))

    async def acquire_papers(
        self,
        papers: list[S2Paper],
        max_concurrent: int = 3,
    ) -> AcquisitionResult:
        """Acquire multiple papers with deduplication.

        Args:
            papers: Papers to acquire
            max_concurrent: Maximum concurrent downloads

        Returns:
            AcquisitionResult with acquired/skipped/failed papers
        """
        result = AcquisitionResult()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def acquire_one(paper: S2Paper) -> None:
            async with semaphore:
                # Check for pre-existing duplicate
                if self.is_duplicate(paper):
                    result.skipped_existing.append(paper)
                    return

                # Check open access
                if not paper.is_open_access:
                    result.skipped_paywall.append(paper)
                    return

                # Try to acquire
                path, error = await self.acquire_paper(paper)

                if path:
                    result.acquired.append((paper, path))
                elif error == "no_url":
                    result.skipped_no_url.append(paper)
                elif error == "duplicate_hash":
                    result.skipped_existing.append(paper)
                else:
                    result.failed.append((paper, error or "unknown"))

        # Process all papers
        await asyncio.gather(*[acquire_one(p) for p in papers])

        return result


async def load_existing_identifiers() -> tuple[set[str], set[str], set[str], set[str]]:
    """Load existing paper identifiers from PostgreSQL.

    Queries the sources table for existing S2 IDs, DOIs, arXiv IDs, and file hashes.

    Returns:
        Tuple of (s2_ids, dois, arxiv_ids, file_hashes)
    """
    # Import here to avoid circular dependency
    try:
        from research_kb_storage import DatabaseConfig, get_connection_pool
    except ImportError:
        logger.warning("research_kb_storage not available, dedup will be limited")
        return set(), set(), set(), set()

    try:
        config = DatabaseConfig()
        pool = await get_connection_pool(config)

        async with pool.acquire() as conn:
            # Query for identifiers
            rows = await conn.fetch("""
                SELECT
                    file_hash,
                    metadata->>'s2_paper_id' as s2_id,
                    metadata->>'doi' as doi,
                    metadata->>'arxiv_id' as arxiv_id
                FROM sources
            """)

        s2_ids = {r["s2_id"] for r in rows if r["s2_id"]}
        dois = {r["doi"] for r in rows if r["doi"]}
        arxiv_ids = {r["arxiv_id"] for r in rows if r["arxiv_id"]}
        file_hashes = {r["file_hash"] for r in rows if r["file_hash"]}

        logger.info(
            "Loaded existing identifiers",
            s2_ids=len(s2_ids),
            dois=len(dois),
            arxiv_ids=len(arxiv_ids),
            file_hashes=len(file_hashes),
        )

        return s2_ids, dois, arxiv_ids, file_hashes

    except Exception as e:
        logger.error("Failed to load existing identifiers", error=str(e))
        return set(), set(), set(), set()
