#!/usr/bin/env python3
"""Build a curated programming-books ingest manifest.

This script scans the existing catalog plus a downloads directory, then writes:

- programming_ingest_manifest.json: curated PDF shortlist compatible with
  scripts/mass_ingest_catalog.py --catalog
- programming_ingest_review.csv: review sheet covering selected, deferred,
  duplicate, and filtered candidates considered for the shortlist
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = REPO_ROOT / "fixtures" / "library_catalog" / "catalog_books_r2.json"
DEFAULT_DOWNLOADS_DIR = Path("/home/brandon_behring/Downloads/acqusition_books/acqusition_books")
DEFAULT_LIBRARY_ROOT = REPO_ROOT / "fixtures" / "library_books"
DEFAULT_OUTPUT_JSON = (
    REPO_ROOT / "fixtures" / "library_catalog" / "programming_ingest_manifest.json"
)
DEFAULT_OUTPUT_CSV = REPO_ROOT / "fixtures" / "library_catalog" / "programming_ingest_review.csv"

ALLOWED_DOMAINS = {
    "software_engineering",
    "algorithms",
    "sql",
    "functional_programming",
}
INGESTABLE_SUFFIXES = {".pdf"}
DEFERRED_SUFFIXES = {".epub", ".mobi", ".azw", ".djvu"}
MIN_CATALOG_PRIORITY = 35


@dataclass(frozen=True)
class ManualRule:
    key: str
    patterns: tuple[str, ...]
    domain: str
    priority_score: int
    tier: str
    reason: str


@dataclass
class Candidate:
    title: str
    authors: list[str]
    year: int | None
    publisher: str | None
    filename: str
    full_path: str
    file_size_mb: float
    is_book: bool
    source_kind: str
    planned_fixtures_path: str
    domain: str | None = None
    catalog_domain: str | None = None
    catalog_priority: int | None = None
    r2_isbns: list[str] = field(default_factory=list)
    matched_rule: ManualRule | None = None

    @property
    def suffix(self) -> str:
        return Path(self.filename).suffix.lower()

    @property
    def is_early_access(self) -> bool:
        text = combined_text(self.title, self.filename)
        return any(pattern in text for pattern in EARLY_ACCESS_PATTERNS)

    @property
    def has_copy_suffix(self) -> bool:
        return bool(re.search(r"\(\d+\)", Path(self.filename).stem))


MANUAL_RULES = (
    ManualRule(
        key="pragmatic_programmer",
        patterns=("pragmatic programmer",),
        domain="software_engineering",
        priority_score=98,
        tier="core",
        reason="Core software-craft classic with strong general programming value.",
    ),
    ManualRule(
        key="unit_testing_principles_practices_patterns",
        patterns=("unit testing principles practices and patterns",),
        domain="software_engineering",
        priority_score=97,
        tier="core",
        reason="High-signal testing practice book for production code quality.",
    ),
    ManualRule(
        key="fluent_python",
        patterns=("fluent python",),
        domain="software_engineering",
        priority_score=96,
        tier="core",
        reason="Deep Python fluency and language-idiom coverage.",
    ),
    ManualRule(
        key="effective_python",
        patterns=("effective python",),
        domain="software_engineering",
        priority_score=95,
        tier="core",
        reason="Modern Python best practices for maintainable production code.",
    ),
    ManualRule(
        key="refactoring",
        patterns=("refactoring",),
        domain="software_engineering",
        priority_score=94,
        tier="core",
        reason="Foundational design-improvement and code transformation reference.",
    ),
    ManualRule(
        key="head_first_python",
        patterns=("head first python",),
        domain="software_engineering",
        priority_score=91,
        tier="core",
        reason="Accessible Python programming foundation within the shortlist.",
    ),
    ManualRule(
        key="api_design_patterns",
        patterns=("api design patterns",),
        domain="software_engineering",
        priority_score=90,
        tier="core",
        reason="Useful production API design guidance with reusable patterns.",
    ),
    ManualRule(
        key="database_internals",
        patterns=("database internals",),
        domain="software_engineering",
        priority_score=89,
        tier="core",
        reason="Strong systems book for storage, indexing, and distributed data fundamentals.",
    ),
    ManualRule(
        key="software_engineering_for_data_scientists",
        patterns=("software engineering for data scientists",),
        domain="software_engineering",
        priority_score=88,
        tier="core",
        reason="Bridges notebook-heavy work with software engineering discipline.",
    ),
    ManualRule(
        key="fundamentals_of_data_engineering",
        patterns=("fundamentals of data engineering",),
        domain="software_engineering",
        priority_score=87,
        tier="core",
        reason="Production-oriented data engineering fundamentals for systems-minded programming.",
    ),
    ManualRule(
        key="data_structures_and_algorithms_in_python",
        patterns=("data structures and algorithms in python",),
        domain="algorithms",
        priority_score=86,
        tier="core",
        reason="Algorithms and data structures grounded in Python implementations.",
    ),
    ManualRule(
        key="introduction_to_algorithms",
        patterns=("introduction to algorithms",),
        domain="algorithms",
        priority_score=84,
        tier="core",
        reason="Canonical algorithms reference for fundamentals and interview-adjacent breadth.",
    ),
    ManualRule(
        key="learning_sql",
        patterns=("learning sql",),
        domain="sql",
        priority_score=78,
        tier="supporting",
        reason="Supporting SQL fundamentals for day-to-day data work.",
    ),
    ManualRule(
        key="sql_cookbook",
        patterns=("sql cookbook",),
        domain="sql",
        priority_score=76,
        tier="supporting",
        reason="Supporting query reference with practical SQL patterns.",
    ),
    ManualRule(
        key="haskell_in_depth",
        patterns=("haskell in depth",),
        domain="functional_programming",
        priority_score=74,
        tier="supporting",
        reason="Functional programming depth and type-driven design perspective.",
    ),
    ManualRule(
        key="designing_data_intensive_applications",
        patterns=("designing data intensive applications",),
        domain="software_engineering",
        priority_score=92,
        tier="deferred",
        reason="High-value systems book discovered as EPUB and deferred until a PDF copy exists.",
    ),
)

MANUAL_RULES_BY_KEY = {rule.key: rule for rule in MANUAL_RULES}
SCOPE_KEYWORDS = (
    "programming",
    "programmer",
    "algorithm",
    "data structure",
    "sql",
    "refactoring",
    "unit testing",
    "api design",
    "software design",
    "software engineering",
    "database internals",
    "data intensive",
    "data engineering",
    "concurrency",
    "functional programming",
    "haskell",
    "scala",
    "design pattern",
    "head first python",
    "fluent python",
    "effective python",
    "python concurrency",
    "well grounded python",
    "publishing python packages",
)
EARLY_ACCESS_PATTERNS = (" meap", " v1 meap", " v2 meap", " v3 meap", "early access", "preview")
EXCLUDED_PATTERNS = (
    "cheat sheet",
    "leetcode",
    "interview",
    "certified",
    "certification",
    "study guide",
    "exam",
    "quiz",
    "syllabus",
    "workbook",
    "verification letter",
    "instructor manual",
    "instructor s manual",
    "solutions manual",
    "glossary",
    "nanodegree",
)
TOO_SPECIFIC_PATTERNS = (
    "google cloud",
    "aws ",
    "kubernetes",
    "serverless",
    "akka",
    "mathematica",
    "julia",
    "swagger",
    "openapi",
    "computer networks",
    "platform engineering",
)
OUT_OF_SCOPE_PATTERNS = (
    "machine learning",
    "deep learning",
    "causal inference",
    "time series",
    "forecasting",
    "finance",
    "portfolio",
    "trading",
    "econometric",
    "statistics",
    "physics",
    "bayesian",
    "data science",
)
DOMAIN_PATTERNS: dict[str, tuple[str, ...]] = {
    "sql": (" sql", "sql ", "learning sql", "sql cookbook", "database", "transaction"),
    "algorithms": (
        "algorithm",
        "data structure",
        "computational complexity",
        "dynamic programming",
    ),
    "functional_programming": ("functional programming", "haskell", "scala", "lambda calculus"),
    "software_engineering": (
        "software engineering",
        "software design",
        "refactoring",
        "unit testing",
        "api design",
        "programmer",
        "programming",
        "concurrency",
        "database internals",
        "data engineering",
    ),
}


def normalize_text(value: str) -> str:
    normalized = re.sub(r"[^0-9a-z]+", " ", value.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def combined_text(*parts: str) -> str:
    return normalize_text(" ".join(part for part in parts if part))


def parse_year(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", value)
    return int(match.group(0)) if match else None


def normalize_authors(authors: Any) -> list[str]:
    if authors is None:
        return []
    if isinstance(authors, list):
        return [str(author).strip() for author in authors if str(author).strip()]
    if isinstance(authors, str):
        pieces = re.split(r"\s*(?:;| and | & )\s*", authors)
        return [piece.strip() for piece in pieces if piece.strip()]
    return [str(authors).strip()]


def normalize_isbns(values: Any) -> list[str]:
    if not values:
        return []
    raw_values = values if isinstance(values, list) else [values]
    normalized: list[str] = []
    for raw in raw_values:
        digits = re.sub(r"[^0-9X]", "", str(raw).upper())
        if len(digits) in {10, 13} and digits not in normalized:
            normalized.append(digits)
    return normalized


def clean_title(raw_title: str, fallback_filename: str) -> str:
    title = raw_title or Path(fallback_filename).stem
    title = title.replace("Anna’s Archive", "").replace("Annas Archive", "")
    title = title.replace("_", " ")
    title = re.sub(r"\s+", " ", title).strip(" -_,")
    return title


def match_rule(title: str, filename: str) -> ManualRule | None:
    text = combined_text(title, filename)
    for rule in MANUAL_RULES:
        if all(pattern in text for pattern in rule.patterns):
            return rule
    return None


def infer_domain(title: str, filename: str, catalog_domain: str | None) -> str | None:
    if catalog_domain in ALLOWED_DOMAINS:
        return catalog_domain

    text = combined_text(title, filename)
    for domain in ("sql", "algorithms", "functional_programming", "software_engineering"):
        if any(pattern in text for pattern in DOMAIN_PATTERNS[domain]):
            return domain
    return None


def is_scope_candidate(
    title: str, filename: str, catalog_domain: str | None, priority: int | None
) -> bool:
    if match_rule(title, filename):
        return True
    text = combined_text(title, filename)
    if any(keyword in text for keyword in SCOPE_KEYWORDS):
        return True
    return catalog_domain in ALLOWED_DOMAINS and (priority or 0) >= MIN_CATALOG_PRIORITY


def build_work_key(candidate: Candidate) -> str:
    if candidate.matched_rule:
        return f"rule:{candidate.matched_rule.key}"
    if candidate.r2_isbns:
        return f"isbn:{candidate.r2_isbns[0]}"

    title_key = normalize_text(candidate.title)
    title_key = re.sub(r"\b(second|third|edition|ed|revised|anniversary)\b", "", title_key).strip()
    author_key = normalize_text(candidate.authors[0]) if candidate.authors else ""
    return f"title:{title_key}|author:{author_key}"


def copy_preference_key(candidate: Candidate) -> tuple[int, int, int, int, int, int, int]:
    return (
        1 if not candidate.is_early_access else 0,
        1 if candidate.suffix in INGESTABLE_SUFFIXES else 0,
        candidate.year or 0,
        1 if candidate.source_kind == "repo_library" else 0,
        candidate.catalog_priority or 0,
        0 if candidate.has_copy_suffix else 1,
        int(candidate.file_size_mb * 100),
    )


def review_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    status_order = {
        "selected_core": 0,
        "selected_supporting": 1,
        "deferred_non_pdf": 2,
        "duplicate": 3,
        "filtered_early_access": 4,
        "filtered_excluded": 5,
        "filtered_too_specific": 6,
        "filtered_out_of_scope": 7,
        "filtered_curated_out": 8,
    }
    return (status_order.get(row["status"], 99), -(row["priority_score"] or 0), row["title"])


def build_library_lookup(library_root: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    if not library_root.exists():
        return lookup
    for path in library_root.iterdir():
        if path.is_file():
            lookup[path.name.lower()] = path
    return lookup


def candidate_from_catalog(
    entry: dict[str, Any], library_lookup: dict[str, Path], library_root: Path
) -> Candidate | None:
    title = clean_title(str(entry.get("title") or ""), str(entry.get("filename") or ""))
    filename = str(entry.get("filename") or "").strip()
    if not filename:
        return None

    catalog_domain = entry.get("domain")
    catalog_priority = entry.get("priority_score")
    if not is_scope_candidate(title, filename, catalog_domain, catalog_priority):
        return None

    local_path = library_lookup.get(filename.lower())
    current_path = local_path or Path(str(entry.get("full_path") or "")).expanduser()
    planned_path = library_root / filename
    candidate = Candidate(
        title=title,
        authors=normalize_authors(entry.get("authors")),
        year=entry.get("year"),
        publisher=entry.get("publisher"),
        filename=filename,
        full_path=str(current_path),
        file_size_mb=float(entry.get("file_size_mb") or 0.0),
        is_book=bool(entry.get("is_book", True)),
        source_kind="repo_library" if local_path else "repo_catalog",
        planned_fixtures_path=str(planned_path),
        domain=infer_domain(title, filename, catalog_domain),
        catalog_domain=catalog_domain,
        catalog_priority=catalog_priority if isinstance(catalog_priority, int) else None,
        r2_isbns=normalize_isbns(entry.get("r2_isbns")),
    )
    candidate.matched_rule = match_rule(candidate.title, candidate.filename)
    return candidate


def candidate_from_download(path: Path, library_root: Path) -> Candidate | None:
    filename = path.name
    stem = path.stem
    parts = [part.strip() for part in stem.split(" -- ")]
    title = clean_title(parts[0] if parts else stem, filename)
    if not is_scope_candidate(title, filename, None, None):
        return None

    authors = normalize_authors(parts[1]) if len(parts) >= 2 else []
    year = parse_year(parts[2] if len(parts) >= 3 else stem)
    publisher = parts[3].strip() if len(parts) >= 4 and parts[3].strip() else None

    candidate = Candidate(
        title=title,
        authors=authors,
        year=year,
        publisher=publisher,
        filename=filename,
        full_path=str(path),
        file_size_mb=round(path.stat().st_size / (1024 * 1024), 2),
        is_book=True,
        source_kind="downloads",
        planned_fixtures_path=str(library_root / filename),
        domain=infer_domain(title, filename, None),
        catalog_domain=None,
        catalog_priority=None,
        r2_isbns=normalize_isbns(re.findall(r"\b(?:97[89]\d{10}|\d{9}[\dX])\b", stem)),
    )
    candidate.matched_rule = match_rule(candidate.title, candidate.filename)
    return candidate


def load_candidates(catalog_path: Path, downloads_dir: Path, library_root: Path) -> list[Candidate]:
    library_lookup = build_library_lookup(library_root)
    catalog_entries = json.loads(catalog_path.read_text())
    candidates = [
        candidate
        for entry in catalog_entries
        if (candidate := candidate_from_catalog(entry, library_lookup, library_root)) is not None
    ]

    if downloads_dir.exists():
        for path in sorted(downloads_dir.iterdir()):
            if not path.is_file():
                continue
            candidate = candidate_from_download(path, library_root)
            if candidate is not None:
                candidates.append(candidate)

    return candidates


def group_candidates(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    grouped: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        grouped.setdefault(build_work_key(candidate), []).append(candidate)
    return grouped


def classify_group(best: Candidate, group: list[Candidate]) -> str:
    text = combined_text(best.title, best.filename)
    rule = best.matched_rule or next(
        (candidate.matched_rule for candidate in group if candidate.matched_rule), None
    )
    domain = resolved_domain(best, group)

    if any(pattern in text for pattern in EXCLUDED_PATTERNS):
        return "filtered_excluded"
    if any(pattern in text for pattern in TOO_SPECIFIC_PATTERNS):
        return "filtered_too_specific"
    if domain not in ALLOWED_DOMAINS and rule is None:
        return "filtered_out_of_scope"
    if best.is_early_access and best.suffix in INGESTABLE_SUFFIXES:
        return "filtered_early_access"
    if rule is None:
        return "filtered_curated_out"
    if best.suffix in DEFERRED_SUFFIXES or best.suffix not in INGESTABLE_SUFFIXES:
        return "deferred_non_pdf"
    if rule.tier == "supporting":
        return "selected_supporting"
    return "selected_core"


def selection_reason(best: Candidate, group: list[Candidate], status: str) -> str:
    rule = best.matched_rule or next(
        (candidate.matched_rule for candidate in group if candidate.matched_rule), None
    )
    if rule is None:
        return "Curated out of the v1 shortlist after broad programming review."

    reason = rule.reason
    duplicates = len(group) - 1
    if duplicates > 0:
        reason += f" Preferred over {duplicates} alternate "
        if duplicates > 1:
            reason += "copies."
        else:
            reason += "copy."
    if status == "deferred_non_pdf":
        reason += " Non-PDF format is tracked for review but excluded from the ingest queue."
    return reason


def best_catalog_hint(group: list[Candidate]) -> Candidate | None:
    catalog_candidates = [
        candidate for candidate in group if candidate.catalog_priority is not None
    ]
    if not catalog_candidates:
        return None
    return max(catalog_candidates, key=lambda candidate: candidate.catalog_priority or 0)


def resolved_domain(candidate: Candidate, group: list[Candidate]) -> str | None:
    rule = candidate.matched_rule or next(
        (item.matched_rule for item in group if item.matched_rule), None
    )
    if rule is not None:
        return rule.domain
    if candidate.domain in ALLOWED_DOMAINS:
        return candidate.domain
    if candidate.catalog_domain in ALLOWED_DOMAINS:
        return candidate.catalog_domain
    return candidate.domain


def build_manifest_entry(best: Candidate, group: list[Candidate], status: str) -> dict[str, Any]:
    hint = best_catalog_hint(group)
    return {
        "filename": best.filename,
        "full_path": best.full_path,
        "file_size_mb": round(best.file_size_mb, 2),
        "is_book": best.is_book,
        "domain": resolved_domain(best, group),
        "priority_score": (
            best.matched_rule.priority_score if best.matched_rule else best.catalog_priority or 0
        ),
        "title": best.title,
        "authors": best.authors,
        "year": best.year,
        "publisher": best.publisher,
        "r2_isbns": best.r2_isbns,
        "source_kind": best.source_kind,
        "selection_reason": selection_reason(best, group, status),
        "catalog_domain": best.catalog_domain or (hint.catalog_domain if hint else None),
        "catalog_priority": (
            best.catalog_priority
            if best.catalog_priority is not None
            else (hint.catalog_priority if hint else None)
        ),
        "planned_fixtures_path": best.planned_fixtures_path,
    }


def build_review_rows(group: list[Candidate], best: Candidate, status: str) -> list[dict[str, Any]]:
    hint = best_catalog_hint(group)
    rows: list[dict[str, Any]] = []
    selected_priority = (
        best.matched_rule.priority_score if best.matched_rule else best.catalog_priority or 0
    )
    chosen_title = best.title
    for candidate in sorted(group, key=copy_preference_key, reverse=True):
        row_status = status if candidate is best else "duplicate"
        notes: list[str] = []
        if candidate.is_early_access:
            notes.append("early_access")
        if candidate.catalog_priority is not None:
            notes.append(f"catalog_priority={candidate.catalog_priority}")
        if candidate.source_kind == "downloads":
            notes.append("download_source")
        if row_status == "duplicate":
            notes.append("deduplicated under preferred copy")
        if row_status == "deferred_non_pdf":
            notes.append("non_pdf")
        if row_status.startswith("filtered_"):
            notes.append("not_in_curated_manifest")

        rows.append(
            {
                "status": row_status,
                "title": candidate.title,
                "domain": resolved_domain(candidate, group) or candidate.catalog_domain or "",
                "priority_score": (
                    selected_priority if candidate is best else candidate.catalog_priority or 0
                ),
                "source_kind": candidate.source_kind,
                "current_path": candidate.full_path,
                "planned_fixtures_path": candidate.planned_fixtures_path,
                "duplicate_of": "" if candidate is best else chosen_title,
                "notes": "; ".join(notes),
                "catalog_domain": candidate.catalog_domain or (hint.catalog_domain if hint else ""),
                "catalog_priority": (
                    candidate.catalog_priority
                    if candidate.catalog_priority is not None
                    else (hint.catalog_priority if hint else "")
                ),
            }
        )
    return rows


def build_programming_manifest(
    catalog_path: Path, downloads_dir: Path, library_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = load_candidates(catalog_path, downloads_dir, library_root)
    grouped = group_candidates(candidates)

    manifest: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for group in grouped.values():
        best = max(group, key=copy_preference_key)
        # Group-level rule propagation lets a newer download inherit the existing catalog hint.
        if best.matched_rule is None:
            best.matched_rule = next(
                (candidate.matched_rule for candidate in group if candidate.matched_rule), None
            )
        status = classify_group(best, group)
        review_rows.extend(build_review_rows(group, best, status))
        if status.startswith("selected_"):
            manifest.append(build_manifest_entry(best, group, status))

    manifest.sort(key=lambda entry: (-entry["priority_score"], entry["title"]))
    review_rows.sort(key=review_sort_key)
    return manifest, review_rows


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "title",
        "domain",
        "priority_score",
        "source_kind",
        "current_path",
        "planned_fixtures_path",
        "duplicate_of",
        "notes",
        "catalog_domain",
        "catalog_priority",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the curated programming ingest manifest.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="Catalog JSON path")
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=DEFAULT_DOWNLOADS_DIR,
        help="Directory with recently acquired books",
    )
    parser.add_argument(
        "--library-root",
        type=Path,
        default=DEFAULT_LIBRARY_ROOT,
        help="Local fixtures/library_books directory",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output manifest JSON path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output review CSV path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest, review_rows = build_programming_manifest(
        catalog_path=args.catalog,
        downloads_dir=args.downloads_dir,
        library_root=args.library_root,
    )
    write_json(args.output_json, manifest)
    write_csv(args.output_csv, review_rows)

    print(f"Wrote {len(manifest)} manifest entries -> {args.output_json}")
    print(f"Wrote {len(review_rows)} review rows -> {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
