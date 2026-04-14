#!/usr/bin/env python3
"""Validate classifier accuracy against database ground truth.

Compares domain assignments from three sources:
1. Catalog (catalog_books_r2.json) — what the classification pipeline assigned
2. Database (sources.domain_id) — what was stored at ingestion time
3. Classifier (pdftotext keywords) — what classify_library_books.py would produce

Usage:
    python scripts/validate_classifier.py              # Baseline validation (100 books)
    python scripts/validate_classifier.py --limit 200  # More samples
    python scripts/validate_classifier.py --check-skips # Check 102 suspect skips against DB
"""

import argparse
import asyncio
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

import asyncpg

REPO_ROOT = Path(__file__).parent.parent
CATALOG_PATH = REPO_ROOT / "fixtures" / "library_catalog" / "catalog_books_r2.json"
CHECKPOINT_PATH = REPO_ROOT / "fixtures" / "library_catalog" / ".mass_ingest_checkpoint.json"
LIBRARY_DIR = REPO_ROOT / "fixtures" / "library_books"


def extract_text_pdftotext(path: Path, pages: int = 3) -> str:
    """Extract text from first N pages via pdftotext."""
    try:
        result = subprocess.run(
            ["pdftotext", "-f", "1", "-l", str(pages), str(path), "-"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout[:3000]
    except (subprocess.TimeoutExpired, Exception):
        return ""


def extract_page_count(path: Path) -> int:
    """Get page count via pdfinfo."""
    try:
        result = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, timeout=10)
        for line in result.stdout.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return 0


# Import domain rules from classify_library_books.py
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from classify_library_books import DOMAIN_RULES


def classify_domain_current(title: str, text: str = "") -> str:
    """Replicate the CURRENT classifier logic (title-only, Pass 1 only)."""
    title_lower = title.lower()
    for domain_id, keywords in DOMAIN_RULES:
        for kw in keywords:
            clean_kw = kw.lstrip("^")
            if re.search(clean_kw, title_lower):
                return domain_id
    return "uncategorized"


async def validate_baseline(limit: int = 100) -> dict:
    """Compare catalog domain vs DB domain vs classifier domain."""
    catalog = json.loads(CATALOG_PATH.read_text())
    checkpoint = json.loads(CHECKPOINT_PATH.read_text())
    completed = list(checkpoint["completed"])[:limit]

    catalog_by_hash = {b["sha256"]: b for b in catalog}

    pool = await asyncpg.create_pool(
        host="localhost",
        port=5432,
        database="research_kb",
        user="postgres",
        password="postgres",
        min_size=1,
        max_size=3,
    )

    results = []
    try:
        async with pool.acquire() as conn:
            for i, file_hash in enumerate(completed):
                cat_entry = catalog_by_hash.get(file_hash)
                if not cat_entry:
                    continue

                # Get DB domain
                db_domain = await conn.fetchval(
                    "SELECT domain_id FROM sources WHERE file_hash = $1", file_hash
                )

                # Get catalog domain
                cat_domain = cat_entry.get("domain", "unknown")

                # Find file on disk
                filename = Path(cat_entry.get("full_path", "")).name
                local_path = LIBRARY_DIR / filename
                if not local_path.exists():
                    # Try case-insensitive
                    matches = list(LIBRARY_DIR.glob(f"*{filename[-20:]}*"))
                    local_path = matches[0] if matches else None

                # Run classifier
                title = cat_entry.get("title", filename)
                if local_path and local_path.exists():
                    text = extract_text_pdftotext(local_path, pages=3)
                    pages = extract_page_count(local_path)
                else:
                    text = ""
                    pages = 0

                classifier_domain = classify_domain_current(title, text)

                # Classification method used in catalog
                r2 = cat_entry.get("r2_classification", {})
                method = r2.get("method", "unknown") if isinstance(r2, dict) else "unknown"

                results.append(
                    {
                        "title": title[:60],
                        "catalog": cat_domain,
                        "db": db_domain or "NOT_IN_DB",
                        "classifier": classifier_domain,
                        "pages": pages,
                        "method": method,
                        "cat_eq_db": cat_domain == db_domain if db_domain else None,
                        "cls_eq_db": classifier_domain == db_domain if db_domain else None,
                        "cls_eq_cat": classifier_domain == cat_domain,
                    }
                )

                if (i + 1) % 25 == 0:
                    print(f"  Processed {i+1}/{len(completed)}...")

    finally:
        await pool.close()

    return {"results": results, "count": len(results)}


async def check_suspect_skips() -> None:
    """Check 102 suspect skips against DB for duplicates."""
    catalog = json.loads(CATALOG_PATH.read_text())

    # Find suspect skips (domain-relevant books marked as skip)
    keywords = [
        "machine learning",
        "deep learning",
        "neural",
        "statistics",
        "probability",
        "algorithm",
        "optimization",
        "reinforcement",
        "causal",
        "econometric",
        "finance",
        "portfolio",
        "time series",
        "bayesian",
        "regression",
        "python",
        "programming",
        "software",
        "database",
        "sql",
        "linear algebra",
        "calculus",
        "differential",
        "topology",
        "data science",
        "artificial intelligence",
        "nlp",
        "transformer",
        "recommender",
        "signal processing",
        "control theory",
    ]

    suspect_skips = []
    for b in catalog:
        if b.get("domain") != "skip":
            continue
        title = b.get("title", b["filename"]).lower()
        if any(kw in title for kw in keywords):
            suspect_skips.append(b)

    pool = await asyncpg.create_pool(
        host="localhost",
        port=5432,
        database="research_kb",
        user="postgres",
        password="postgres",
        min_size=1,
        max_size=2,
    )

    duplicates = []
    missing = []

    try:
        async with pool.acquire() as conn:
            for b in suspect_skips:
                existing = await conn.fetchval(
                    "SELECT title FROM sources WHERE file_hash = $1", b["sha256"]
                )
                title = b.get("title", b["filename"])[:70]
                if existing:
                    duplicates.append({"skip_title": title, "db_title": existing[:70]})
                else:
                    # What domain would the classifier assign?
                    domain = classify_domain_current(title)
                    missing.append({"title": title, "suggested_domain": domain})
    finally:
        await pool.close()

    print(f"\n=== SUSPECT SKIPS: {len(suspect_skips)} books ===\n")
    print(f"Already in DB (duplicates): {len(duplicates)}")
    print(f"NOT in DB (missing — should reclassify): {len(missing)}")

    if duplicates:
        print(f"\n--- Duplicates (correctly skipped) ---")
        for d in duplicates[:15]:
            print(f"  SKIP: {d['skip_title']}")
            print(f"    DB: {d['db_title']}")
        if len(duplicates) > 15:
            print(f"  ... and {len(duplicates)-15} more")

    if missing:
        print(f"\n--- Missing (should reclassify) ---")
        for m in missing[:20]:
            print(f"  [{m['suggested_domain']}] {m['title']}")
        if len(missing) > 20:
            print(f"  ... and {len(missing)-20} more")


def print_results(data: dict) -> None:
    """Print validation results."""
    results = data["results"]

    # Header
    print(f"\n{'='*90}")
    print(f"CLASSIFIER VALIDATION — {data['count']} books")
    print(f"{'='*90}\n")

    # Agreement rates
    cat_eq_db = [r for r in results if r["cat_eq_db"] is True]
    cls_eq_db = [r for r in results if r["cls_eq_db"] is True]
    cls_eq_cat = [r for r in results if r["cls_eq_cat"]]
    in_db = [r for r in results if r["db"] != "NOT_IN_DB"]

    print(
        f"Catalog == DB:     {len(cat_eq_db)}/{len(in_db)} ({100*len(cat_eq_db)/max(len(in_db),1):.1f}%)"
    )
    print(
        f"Classifier == DB:  {len(cls_eq_db)}/{len(in_db)} ({100*len(cls_eq_db)/max(len(in_db),1):.1f}%)"
    )
    print(
        f"Classifier == Cat: {len(cls_eq_cat)}/{len(results)} ({100*len(cls_eq_cat)/max(len(results),1):.1f}%)"
    )

    # Disagreements
    disagreements = [r for r in results if r["db"] != "NOT_IN_DB" and not r["cat_eq_db"]]
    if disagreements:
        print(f"\n--- Catalog != DB ({len(disagreements)} cases) ---")
        for r in disagreements[:20]:
            print(f"  {r['title']}")
            print(f"    catalog={r['catalog']}  db={r['db']}  classifier={r['classifier']}")

    # Classifier failures
    cls_failures = [r for r in results if r["db"] != "NOT_IN_DB" and not r["cls_eq_db"]]
    if cls_failures:
        print(f"\n--- Classifier != DB ({len(cls_failures)} cases) ---")
        for r in cls_failures[:20]:
            print(f"  {r['title']}")
            print(f"    classifier={r['classifier']}  db={r['db']}  catalog={r['catalog']}")

    # Uncategorized by classifier
    uncategorized = [r for r in results if r["classifier"] == "uncategorized"]
    if uncategorized:
        print(f"\n--- Classifier returned 'uncategorized' ({len(uncategorized)}) ---")
        for r in uncategorized[:10]:
            print(f"  [{r['db']}] {r['title']}")

    # Domain confusion matrix
    if cls_failures:
        print(f"\n--- Domain confusion (classifier → DB) ---")
        confusion = Counter((r["classifier"], r["db"]) for r in cls_failures)
        for (cls, db), count in confusion.most_common(15):
            print(f"  {cls:25s} → {db:25s} ({count})")

    # Classification method breakdown
    methods = Counter(r["method"] for r in results)
    print(f"\n--- Catalog classification methods ---")
    for m, c in methods.most_common():
        print(f"  {m}: {c}")


def main():
    parser = argparse.ArgumentParser(description="Validate classifier accuracy")
    parser.add_argument("--limit", type=int, default=100, help="Number of books to validate")
    parser.add_argument("--check-skips", action="store_true", help="Check suspect skips against DB")
    args = parser.parse_args()

    if args.check_skips:
        asyncio.run(check_suspect_skips())
    else:
        print(f"Validating classifier on {args.limit} ingested books...")
        data = asyncio.run(validate_baseline(args.limit))
        print_results(data)


if __name__ == "__main__":
    main()
