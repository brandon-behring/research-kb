"""Cross-reference catalog, disk, and database to produce a gap report.

Matches three data sources:
1. Catalog (manning_catalog.yaml) — what you own
2. Disk (Documents/ + fixtures/textbooks/) — what's on disk
3. DB (sources table) — what's ingested

Uses fuzzy title matching because DB titles are often messy
(e.g., "Aditya Bhargava Grokking Algorithms An illustrated guide for...").
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import DEFAULT_BOOKS_DIR, load_catalog

# Add packages to path for imports
_project_root = Path(__file__).parent.parent.parent
for pkg in ("storage", "contracts", "common"):
    sys.path.insert(0, str(_project_root / "packages" / pkg / "src"))

from research_kb_storage import DatabaseConfig, get_connection_pool  # noqa: E402


@dataclass
class AuditRow:
    """One row in the audit report."""

    title: str
    domain_id: str
    tier: int
    on_disk: bool
    best_pdf: Path | None
    ingested: bool = False
    ingested_count: int = 0  # how many times (for duplicates)
    db_domains: list[str] = field(default_factory=list)
    db_source_ids: list[str] = field(default_factory=list)
    db_chunk_counts: list[int] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


@dataclass
class AuditSummary:
    """Summary statistics from the audit."""

    catalog_count: int = 0
    on_disk_count: int = 0
    ingested_count: int = 0
    missing_count: int = 0
    duplicate_count: int = 0
    domain_missing_count: int = 0
    domain_mismatch_count: int = 0


def normalize_for_match(title: str) -> str:
    """Normalize a title for fuzzy matching.

    Strips punctuation, lowercases, removes common suffixes.
    """
    t = title.lower().strip()
    # Remove edition suffixes for matching
    t = t.replace(", second edition", "").replace(", third edition", "")
    t = t.replace("second edition", "").replace("third edition", "")
    # Remove parentheticals
    import re

    t = re.sub(r"[()]", "", t)
    # Remove punctuation except spaces
    t = re.sub(r"[^\w\s]", "", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def fuzzy_title_match(catalog_title: str, db_title: str, threshold: float = 0.6) -> bool:
    """Check if a DB title matches a catalog title.

    Uses normalized substring matching — if the normalized catalog title
    appears as a substring of the normalized DB title, it's a match.
    Also checks Jaccard similarity of word sets as fallback.

    Args:
        catalog_title: Clean title from catalog.
        db_title: Potentially messy title from DB.
        threshold: Minimum Jaccard similarity for word-set matching.
    """
    norm_catalog = normalize_for_match(catalog_title)
    norm_db = normalize_for_match(db_title)

    # Exact normalized match
    if norm_catalog == norm_db:
        return True

    # Substring match (DB title often has author prefix + subtitle)
    if norm_catalog in norm_db or norm_db in norm_catalog:
        return True

    # Jaccard similarity on word sets
    words_catalog = set(norm_catalog.split())
    words_db = set(norm_db.split())
    if not words_catalog or not words_db:
        return False

    intersection = words_catalog & words_db
    union = words_catalog | words_db
    jaccard = len(intersection) / len(union)
    return jaccard >= threshold


async def fetch_db_sources(pool) -> list[dict]:
    """Fetch all sources from the database with chunk counts."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                s.id,
                s.title,
                s.file_hash,
                s.file_path,
                COALESCE(s.metadata->>'domain', 'none') AS domain,
                COUNT(c.id) AS chunk_count
            FROM sources s
            LEFT JOIN chunks c ON c.source_id = s.id
            GROUP BY s.id, s.title, s.file_hash, s.file_path, domain
            ORDER BY s.title
            """
        )
    return [dict(r) for r in rows]


async def run_audit(
    catalog_path: Path | None = None,
    books_dir: Path = DEFAULT_BOOKS_DIR,
) -> tuple[list[AuditRow], AuditSummary]:
    """Run the full audit, returning rows and summary.

    Cross-references:
    1. Catalog books (from YAML)
    2. Disk presence (checks best_pdf exists)
    3. DB sources (fuzzy title match)
    """
    books = load_catalog(catalog_path) if catalog_path else load_catalog()

    config = DatabaseConfig()
    pool = await get_connection_pool(config)
    db_sources = await fetch_db_sources(pool)

    rows: list[AuditRow] = []
    summary = AuditSummary(catalog_count=len(books))

    # Track which DB sources got matched (to find unmatched ones)
    matched_db_ids: set[str] = set()

    for book in books:
        on_disk = book.best_pdf is not None and book.best_pdf.exists()
        if on_disk:
            summary.on_disk_count += 1

        row = AuditRow(
            title=book.title,
            domain_id=book.domain_id,
            tier=book.tier,
            on_disk=on_disk,
            best_pdf=book.best_pdf,
        )

        # Find matching DB sources
        for src in db_sources:
            if fuzzy_title_match(book.title, src["title"]):
                row.ingested = True
                row.ingested_count += 1
                row.db_domains.append(src["domain"])
                row.db_source_ids.append(str(src["id"]))
                row.db_chunk_counts.append(src["chunk_count"])
                matched_db_ids.add(str(src["id"]))

        # Detect issues
        if row.ingested_count > 1:
            row.issues.append("DUPLICATE")
            summary.duplicate_count += 1

        if row.ingested and any(d == "none" for d in row.db_domains):
            row.issues.append("DOMAIN_MISSING")
            summary.domain_missing_count += 1

        if (
            row.ingested
            and book.domain_id != "unclassified"
            and row.db_domains
            and all(d != book.domain_id and d != "none" for d in row.db_domains)
        ):
            row.issues.append("DOMAIN_MISMATCH")
            summary.domain_mismatch_count += 1

        if row.ingested:
            summary.ingested_count += 1
        else:
            summary.missing_count += 1

        rows.append(row)

    return rows, summary


def format_audit_report(rows: list[AuditRow], summary: AuditSummary) -> str:
    """Format audit results as a readable table."""
    lines: list[str] = []

    # Header
    lines.append(
        f"{'Title':<45} | {'Domain':<16} | {'Tier':>4} | {'Disk':>4} | "
        f"{'Ingested':>8} | {'DB Domain':<16} | Issues"
    )
    lines.append("-" * 130)

    for row in rows:
        disk_str = "YES" if row.on_disk else "NO"
        if row.ingested_count > 1:
            ingest_str = f"YES(x{row.ingested_count})"
        elif row.ingested:
            ingest_str = "YES"
        else:
            ingest_str = "NO"

        db_domain_str = ", ".join(sorted(set(row.db_domains))) if row.db_domains else "-"
        issues_str = ", ".join(row.issues) if row.issues else "-"

        title_display = row.title[:44] if len(row.title) > 44 else row.title

        lines.append(
            f"{title_display:<45} | {row.domain_id:<16} | {row.tier:>4} | "
            f"{disk_str:>4} | {ingest_str:>8} | {db_domain_str:<16} | {issues_str}"
        )

    # Summary
    lines.append("")
    lines.append("=" * 80)
    lines.append(
        f"Catalog: {summary.catalog_count} books | "
        f"On disk: {summary.on_disk_count} | "
        f"Ingested: {summary.ingested_count} | "
        f"Missing: {summary.missing_count}"
    )
    lines.append(
        f"Duplicates: {summary.duplicate_count} | "
        f"Missing domain: {summary.domain_missing_count} | "
        f"Domain mismatch: {summary.domain_mismatch_count}"
    )
    lines.append("=" * 80)

    return "\n".join(lines)
