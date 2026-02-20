"""Generate and load manning_catalog.yaml from the Documents/ directory.

The catalog is the single source of truth for Manning book metadata.
First run generates a draft with domain_id='unclassified', tier=0.
User then edits to assign domains and tiers.

Subsequent runs can merge new books into existing catalog without
overwriting manual classifications.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .dedup import get_all_pdfs, get_meap_version_for_book, pick_best_pdf

# Default Documents/ location
DEFAULT_BOOKS_DIR = Path.home() / "Documents" / "Manning_Books_research_kb"

# Catalog lives alongside the scripts
CATALOG_PATH = Path(__file__).parent / "manning_catalog.yaml"

# Parse author + version from manning_owned.md
VERSION_RE = re.compile(r"version:\s*(\d+),\s*last updated:\s*([\d-]+)")


@dataclass
class ManningBook:
    """A single Manning book with metadata from catalog + filesystem."""

    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    meap_version: int | None = None
    domain_id: str = "unclassified"
    tier: int = 0
    best_pdf: Path | None = None
    all_pdfs: list[Path] = field(default_factory=list)
    github_repo: str | None = None


def parse_manning_owned(owned_path: Path) -> dict[str, dict[str, Any]]:
    """Parse manning_owned.md to extract authors, versions, and dates.

    Returns:
        Dict mapping title → {authors, meap_version, last_updated}
    """
    if not owned_path.exists():
        return {}

    text = owned_path.read_text()
    entries: dict[str, dict[str, Any]] = {}
    lines = text.strip().split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Skip empty lines and non-title lines
        if not line or line.startswith("version:") or line.startswith("Foreword"):
            i += 1
            continue

        # Skip known non-book lines
        if any(
            line.startswith(prefix)
            for prefix in [
                "pdf",
                "audio",
                "resume",
                "open in",
                "Download",
                "With chapters",
            ]
        ):
            i += 1
            continue

        # Potential title line — look ahead for author line
        potential_title = line.strip()

        # Clean up title — remove leading/trailing whitespace, quotes
        if potential_title.startswith('"') and potential_title.endswith('"'):
            potential_title = potential_title.strip('"')

        # Check next lines for author and version info
        author_line = ""
        version_info: dict[str, Any] = {}

        j = i + 1
        while j < len(lines) and j <= i + 4:
            next_line = lines[j].strip()
            if not next_line:
                j += 1
                continue
            if next_line.startswith("Foreword"):
                j += 1
                continue
            if next_line.startswith("version:"):
                vm = VERSION_RE.search(next_line)
                if vm:
                    version_info["meap_version"] = int(vm.group(1))
                    version_info["last_updated"] = vm.group(2)
                j += 1
                continue
            # Skip non-content lines
            if any(
                next_line.startswith(prefix)
                for prefix in [
                    "pdf",
                    "audio",
                    "resume",
                    "open in",
                    "Download",
                    "With chapters",
                ]
            ):
                j += 1
                continue
            # First real non-skip line after title = author
            if not author_line:
                author_line = next_line
            j += 1
            break

        # Parse authors from author line
        authors: list[str] = []
        if author_line:
            # Remove "with ..." secondary authors prefix
            author_str = re.sub(r"\s+with\s+.*", "", author_line)
            # Split on " and " or ", "
            parts = re.split(r",\s*|\s+and\s+", author_str)
            authors = [a.strip().removeprefix("and ").strip() for a in parts if a.strip()]

        # Only record if it looks like a book title (has at least 2 words or is known)
        if len(potential_title.split()) >= 2 or potential_title in (
            "Modern C",
            "Fast Python",
        ):
            entries[potential_title] = {
                "authors": authors,
                **version_info,
            }

        i = j if j > i + 1 else i + 1

    return entries


def scan_books_directory(books_dir: Path) -> list[ManningBook]:
    """Scan the Manning books directory to discover all books.

    Each subdirectory is treated as one book. Directory name = book title.

    Returns:
        List of ManningBook with filesystem info populated.
    """
    if not books_dir.exists():
        raise FileNotFoundError(f"Books directory not found: {books_dir}")

    books: list[ManningBook] = []
    for entry in sorted(books_dir.iterdir()):
        if not entry.is_dir():
            continue

        title = entry.name
        best_pdf = pick_best_pdf(entry)
        all_pdfs = get_all_pdfs(entry)
        meap_version = get_meap_version_for_book(entry)

        book = ManningBook(
            title=title,
            meap_version=meap_version,
            best_pdf=best_pdf,
            all_pdfs=all_pdfs,
        )
        books.append(book)

    return books


def generate_catalog(
    books_dir: Path = DEFAULT_BOOKS_DIR,
    owned_path: Path | None = None,
    output_path: Path = CATALOG_PATH,
    merge_existing: bool = True,
) -> list[ManningBook]:
    """Generate manning_catalog.yaml from filesystem scan.

    If merge_existing=True and catalog already exists, preserves
    manual classifications (domain_id, tier, authors) for known books
    and only adds new discoveries.

    Args:
        books_dir: Path to Manning books directory.
        owned_path: Path to manning_owned.md for author/version metadata.
        output_path: Where to write the catalog YAML.
        merge_existing: Whether to preserve existing manual edits.

    Returns:
        List of ManningBook entries written to catalog.
    """
    # Scan filesystem
    books = scan_books_directory(books_dir)

    # Parse manning_owned.md for extra metadata
    if owned_path is None:
        owned_path = Path(__file__).parent.parent.parent / "manning_owned.md"
    owned_data = parse_manning_owned(owned_path)

    # Load existing catalog if merging
    existing: dict[str, dict] = {}
    if merge_existing and output_path.exists():
        existing_catalog = load_catalog_raw(output_path)
        for entry in existing_catalog.get("books", []):
            existing[entry["title"]] = entry

    # Enrich books with owned metadata and existing catalog data
    for book in books:
        # Try exact match in owned data, then fuzzy
        owned_entry = owned_data.get(book.title)
        if owned_entry is None:
            # Try case-insensitive match
            for owned_title, data in owned_data.items():
                if owned_title.lower() == book.title.lower():
                    owned_entry = data
                    break

        if owned_entry:
            if not book.authors and owned_entry.get("authors"):
                book.authors = owned_entry["authors"]
            if owned_entry.get("meap_version") and book.meap_version is None:
                book.meap_version = owned_entry["meap_version"]

        # Merge with existing catalog entries (preserve manual edits)
        if book.title in existing:
            ex = existing[book.title]
            if ex.get("domain_id", "unclassified") != "unclassified":
                book.domain_id = ex["domain_id"]
            if ex.get("tier", 0) != 0:
                book.tier = ex["tier"]
            if ex.get("authors") and not book.authors:
                book.authors = ex["authors"]
            if ex.get("year"):
                book.year = ex["year"]
            if ex.get("github_repo"):
                book.github_repo = ex["github_repo"]

    # Write catalog YAML
    catalog_data = {
        "version": 1,
        "books_dir": str(books_dir),
        "total_books": len(books),
        "books": [],
    }

    for book in books:
        entry: dict[str, Any] = {
            "title": book.title,
            "authors": book.authors,
            "year": book.year,
            "domain_id": book.domain_id,
            "tier": book.tier,
            "meap_version": book.meap_version,
            "best_pdf": str(book.best_pdf) if book.best_pdf else None,
            "pdf_count": len(book.all_pdfs),
        }
        if book.github_repo:
            entry["github_repo"] = book.github_repo
        catalog_data["books"].append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(
            catalog_data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    return books


def load_catalog_raw(path: Path = CATALOG_PATH) -> dict:
    """Load raw catalog YAML as dict."""
    if not path.exists():
        raise FileNotFoundError(
            f"Catalog not found: {path}\nRun: python -m scripts.manning catalog --generate"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def load_catalog(path: Path = CATALOG_PATH) -> list[ManningBook]:
    """Load catalog and return structured ManningBook list.

    Resolves best_pdf paths back to Path objects.
    """
    data = load_catalog_raw(path)
    books: list[ManningBook] = []

    for entry in data.get("books", []):
        best_pdf = Path(entry["best_pdf"]) if entry.get("best_pdf") else None
        book = ManningBook(
            title=entry["title"],
            authors=entry.get("authors", []),
            year=entry.get("year"),
            meap_version=entry.get("meap_version"),
            domain_id=entry.get("domain_id", "unclassified"),
            tier=entry.get("tier", 0),
            best_pdf=best_pdf,
            all_pdfs=[],  # Not stored in catalog — re-scan if needed
            github_repo=entry.get("github_repo"),
        )
        books.append(book)

    return books
