#!/usr/bin/env python3
"""Build owned inventory: cross-reference DB sources with on-disk PDFs.

Outputs docs/owned_inventory.json with ground truth of what we own.

Usage:
    python scripts/build_owned_inventory.py
    python scripts/build_owned_inventory.py --quiet   # JSON only to stdout
"""

import asyncio
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

# Ensure packages are importable
ROOT = Path(__file__).resolve().parent.parent
for pkg in ["storage", "contracts", "common"]:
    sys.path.insert(0, str(ROOT / "packages" / pkg / "src"))

from research_kb_storage import SourceStore, get_connection_pool


# Directories to scan for PDFs
PDF_SCAN_DIRS = [
    ROOT / "fixtures" / "papers",
    ROOT / "fixtures" / "textbooks",
    Path.home() / "Claude" / "annuity-pricing" / "references",
    Path.home() / "Claude" / "job_applications" / "resources",
]


def scan_pdfs(dirs: list[Path]) -> dict[str, Path]:
    """Scan directories recursively for PDFs. Returns {stem_lower: path}."""
    pdfs: dict[str, Path] = {}
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.pdf"):
            key = p.stem.lower().replace(" ", "_").replace("-", "_")
            pdfs[key] = p
    return pdfs


def normalize_title(title: str) -> str:
    """Normalize title for fuzzy matching against filenames."""
    return (
        title.lower()
        .replace(":", "")
        .replace(",", "")
        .replace("'", "")
        .replace('"', "")
        .replace("(", "")
        .replace(")", "")
        .replace("  ", " ")
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
    )


def title_matches_filename(title: str, filename_stem: str) -> bool:
    """Check if a source title plausibly matches a filename stem."""
    norm_title = normalize_title(title)
    norm_file = filename_stem.lower().replace("-", "_").replace(" ", "_")

    # Direct substring match (either direction)
    if norm_title in norm_file or norm_file in norm_title:
        return True

    # Check first 3 significant words match
    title_words = [w for w in norm_title.split("_") if len(w) > 2][:3]
    if title_words and all(w in norm_file for w in title_words):
        return True

    return False


async def main() -> None:
    quiet = "--quiet" in sys.argv

    # Connect to DB
    pool = await get_connection_pool()

    # Fetch all sources from DB
    all_sources = []
    offset = 0
    batch_size = 500
    while True:
        batch = await SourceStore.list_all(limit=batch_size, offset=offset)
        all_sources.extend(batch)
        if len(batch) < batch_size:
            break
        offset += batch_size

    if not quiet:
        print(f"DB sources: {len(all_sources)}")

    # Scan filesystem
    on_disk = scan_pdfs(PDF_SCAN_DIRS)
    if not quiet:
        print(f"PDFs on disk: {len(on_disk)}")

    # Cross-reference: for each DB source, check if a matching PDF exists
    type_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    sources_data = []

    matched_files: set[str] = set()

    for src in all_sources:
        type_counts[src.source_type.value] += 1
        domain_counts[src.domain_id] += 1

        # Try to match against on-disk PDFs
        found_on_disk = False
        matched_path = None

        # Check file_path field first
        if src.file_path:
            fp = Path(src.file_path)
            if fp.exists():
                found_on_disk = True
                matched_path = str(fp)
                key = fp.stem.lower().replace(" ", "_").replace("-", "_")
                matched_files.add(key)

        # Fuzzy match against scanned PDFs
        if not found_on_disk:
            for key, path in on_disk.items():
                if title_matches_filename(src.title, key):
                    found_on_disk = True
                    matched_path = str(path)
                    matched_files.add(key)
                    break

        sources_data.append(
            {
                "title": src.title,
                "authors": src.authors,
                "type": src.source_type.value,
                "domain": src.domain_id,
                "year": src.year,
                "in_db": True,
                "on_disk": found_on_disk,
                "file_path": matched_path,
            }
        )

    # Find orphan PDFs (on disk but NOT in DB)
    orphans = []
    for key, path in sorted(on_disk.items()):
        if key not in matched_files:
            orphans.append({"filename": path.name, "path": str(path)})

    # Sort sources by domain then title
    sources_data.sort(key=lambda s: (s["domain"], s["title"]))

    # Build output
    inventory = {
        "generated": str(date.today()),
        "summary": {
            "total_in_db": len(all_sources),
            "pdfs_on_disk": len(on_disk),
            "by_type": dict(type_counts.most_common()),
            "orphan_pdfs": len(orphans),
        },
        "by_domain": dict(domain_counts.most_common()),
        "thin_domains": {
            k: v for k, v in domain_counts.most_common() if v < 5
        },
        "sources": sources_data,
        "orphan_pdfs": orphans,
    }

    # Write output
    out_path = ROOT / "docs" / "owned_inventory.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(inventory, indent=2, default=str))

    if quiet:
        print(json.dumps(inventory["summary"], indent=2))
    else:
        print(f"\nWritten to {out_path}")
        print(f"  Total DB sources: {inventory['summary']['total_in_db']}")
        print(f"  PDFs on disk: {inventory['summary']['pdfs_on_disk']}")
        print(f"  Orphan PDFs (on disk, not in DB): {inventory['summary']['orphan_pdfs']}")
        print(f"\n  By type: {dict(type_counts.most_common())}")
        print(f"\n  Thin domains (<5 sources):")
        for domain, count in sorted(inventory["thin_domains"].items()):
            print(f"    {domain}: {count}")

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
