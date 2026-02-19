"""MEAP version selection and file deduplication.

For each book directory in Documents/:
- Parse all PDFs, extract MEAP version from filename: r'_v(\\d+)_MEAP'
- If MEAP: pick highest version number
- If published (no version): pick file WITHOUT (N) suffix, or most recent by mtime
- Skip EPUBs (PDF pipeline only)
- Return one canonical Path per book
"""

import hashlib
import re
from pathlib import Path


# MEAP version pattern: _v7_MEAP, _v18_MEAP, etc.
MEAP_VERSION_RE = re.compile(r"_v(\d+)_MEAP")

# Download duplicate pattern: (1), (2), etc.
DOWNLOAD_DUP_RE = re.compile(r"\s*\(\d+\)")


def compute_file_hash(file_path: str | Path) -> str:
    """Compute SHA256 hash of a file.

    Replicates scripts/ingest_missing_textbooks.py:39 logic.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def parse_meap_version(filename: str) -> int | None:
    """Extract MEAP version number from filename.

    Args:
        filename: PDF filename like 'Enterprise_RAG_v3_MEAP.pdf'

    Returns:
        Version number (e.g. 3) or None if not a MEAP file.
    """
    match = MEAP_VERSION_RE.search(filename)
    if match:
        return int(match.group(1))
    return None


def is_download_duplicate(filename: str) -> bool:
    """Check if filename has a download duplicate suffix like (1), (2).

    These are browser re-download artifacts, not real content variants.
    """
    stem = Path(filename).stem
    return bool(DOWNLOAD_DUP_RE.search(stem))


def pick_best_pdf(book_dir: Path) -> Path | None:
    """Pick the best (canonical) PDF from a book directory.

    Strategy:
    1. Collect all .pdf files (skip .epub)
    2. Separate into MEAP and published files
    3. For MEAP: pick highest version, prefer non-duplicate copy
    4. For published: pick file without (N) suffix, or most recent by mtime

    Args:
        book_dir: Directory containing one book's files.

    Returns:
        Path to the best PDF, or None if no PDFs found.
    """
    pdfs = sorted(book_dir.glob("*.pdf"))
    if not pdfs:
        return None

    # Separate MEAP vs published PDFs
    meap_pdfs: list[tuple[int, Path]] = []
    published_pdfs: list[Path] = []

    for pdf in pdfs:
        version = parse_meap_version(pdf.name)
        if version is not None:
            meap_pdfs.append((version, pdf))
        else:
            published_pdfs.append(pdf)

    # Prefer published over MEAP (published = final version)
    if published_pdfs:
        # Among published PDFs, prefer non-duplicate
        non_dupes = [p for p in published_pdfs if not is_download_duplicate(p.name)]
        if non_dupes:
            return non_dupes[0]
        # All are duplicates — pick most recent by mtime
        return max(published_pdfs, key=lambda p: p.stat().st_mtime)

    # Only MEAP files — pick highest version, prefer non-duplicate
    if meap_pdfs:
        max_version = max(v for v, _ in meap_pdfs)
        latest_meaps = [(v, p) for v, p in meap_pdfs if v == max_version]
        # Prefer non-duplicate copy of latest version
        non_dupes = [(v, p) for v, p in latest_meaps if not is_download_duplicate(p.name)]
        if non_dupes:
            return non_dupes[0][1]
        return latest_meaps[0][1]

    return None


def get_all_pdfs(book_dir: Path) -> list[Path]:
    """Get all PDF files in a book directory (for catalog metadata)."""
    return sorted(book_dir.glob("*.pdf"))


def get_meap_version_for_book(book_dir: Path) -> int | None:
    """Get the highest MEAP version found in a book directory.

    Returns None if the book has no MEAP PDFs (i.e., it's published).
    """
    max_version = None
    for pdf in book_dir.glob("*.pdf"):
        version = parse_meap_version(pdf.name)
        if version is not None:
            if max_version is None or version > max_version:
                max_version = version
    return max_version
