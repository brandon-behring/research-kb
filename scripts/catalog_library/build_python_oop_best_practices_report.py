#!/usr/bin/env python3
"""Build a curated Python OOP best-practices inventory.

This script scans a small set of high-signal roots under /home/brandon_behring
and writes:

- python_oop_best_practices.md: curated study list for Python OOP/design work
- python_oop_best_practices_sources.csv: source inventory backing the report
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIBRARY_ROOT = REPO_ROOT / "fixtures" / "library_books"
DEFAULT_DOWNLOADS_ROOT = Path("/home/brandon_behring/Downloads")
DEFAULT_MANNING_ROOT = Path("/home/brandon_behring/Documents/Manning_Books_research_kb")
DEFAULT_COURSE_LEARNING_ROOT = Path("/home/brandon_behring/Claude/course_learning")
DEFAULT_OUTPUT_MD = REPO_ROOT / "fixtures" / "library_catalog" / "python_oop_best_practices.md"
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT / "fixtures" / "library_catalog" / "python_oop_best_practices_sources.csv"
)

BOOK_SUFFIXES = {".pdf", ".epub"}
PREVIEW_PATTERNS = (" meap", " preview", " early access")
IGNORED_DIR_NAMES = {
    "__pycache__",
    "__macosx",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "site-packages",
}

SECTION_TITLES = {
    "core_python_oop": "Core Python OOP",
    "structured_notes": "Structured Notes",
    "supporting_classics": "Supporting Classics",
    "appendix": "Appendix",
}
SECTION_ORDER = {
    "core_python_oop": 0,
    "structured_notes": 1,
    "supporting_classics": 2,
    "appendix": 3,
}
SOURCE_TYPE_LABELS = {
    "book_pdf": "PDF book",
    "book_epub": "EPUB book",
    "course_notes": "Structured course notes",
}


@dataclass(frozen=True)
class ResourceSpec:
    key: str
    title: str
    tier: str
    order: int
    aliases: tuple[str, ...]
    oop_relevance: str
    appendix_only: bool = False


@dataclass
class Candidate:
    spec: ResourceSpec
    path: Path
    source_root: str

    @property
    def source_type(self) -> str:
        if self.path.is_dir():
            return "course_notes"
        return "book_pdf" if self.path.suffix.lower() == ".pdf" else "book_epub"

    @property
    def year(self) -> int | None:
        return parse_year(str(self.path))

    @property
    def edition_rank(self) -> int:
        return parse_edition_rank(str(self.path))

    @property
    def is_preview(self) -> bool:
        text = normalize_text(str(self.path))
        return any(pattern.strip() in text for pattern in PREVIEW_PATTERNS)

    @property
    def has_copy_suffix(self) -> bool:
        stem = self.path.stem
        return bool(re.search(r"\([1-9]\d?\)$", stem) or re.search(r"(?:[_ -])[1-9]\d?$", stem))

    @property
    def size_bytes(self) -> int:
        if self.path.is_dir():
            return 0
        try:
            return self.path.stat().st_size
        except OSError:
            return 0


@dataclass
class ReportEntry:
    title: str
    tier: str
    order: int
    source_type: str
    status: str
    oop_relevance: str
    current_path: str
    preferred_path: str
    duplicate_of: str
    notes: str


BOOK_SPECS = (
    ResourceSpec(
        key="software_design_python_programmers",
        title="Software Design for Python Programmers",
        tier="core_python_oop",
        order=10,
        aliases=(
            "software design for python programmers",
            "software design in python",
        ),
        oop_relevance=(
            "Direct coverage of class design, encapsulation, delegation, loose "
            "coupling, inheritance, and design patterns in Python."
        ),
    ),
    ResourceSpec(
        key="practices_python_pro",
        title="Practices of the Python Pro",
        tier="core_python_oop",
        order=20,
        aliases=("practices of the python pro",),
        oop_relevance=(
            "Strong treatment of separation of concerns, abstraction, "
            "encapsulation, inheritance tradeoffs, and loose coupling."
        ),
    ),
    ResourceSpec(
        key="well_grounded_python_developer",
        title="The Well-Grounded Python Developer",
        tier="core_python_oop",
        order=30,
        aliases=("well grounded python developer",),
        oop_relevance=(
            "Useful Python design guide for APIs, the object model, "
            "composition, inheritance, and maintainable style."
        ),
    ),
    ResourceSpec(
        key="fluent_python",
        title="Fluent Python",
        tier="core_python_oop",
        order=40,
        aliases=("fluent python",),
        oop_relevance=(
            "Best source here for Python's data model, protocols, dunder "
            "methods, composition, and idiomatic class design."
        ),
    ),
    ResourceSpec(
        key="effective_python",
        title="Effective Python",
        tier="core_python_oop",
        order=50,
        aliases=("effective python",),
        oop_relevance=(
            "Concrete Python best practices for class APIs, inheritance, mixins, "
            "composition, and maintainable production code."
        ),
    ),
    ResourceSpec(
        key="refactoring",
        title="Refactoring",
        tier="supporting_classics",
        order=60,
        aliases=("refactoring",),
        oop_relevance=(
            "Essential support text for improving class design safely and "
            "cleaning up object-oriented code over time."
        ),
    ),
    ResourceSpec(
        key="pragmatic_programmer",
        title="The Pragmatic Programmer",
        tier="supporting_classics",
        order=70,
        aliases=("pragmatic programmer",),
        oop_relevance=(
            "General software craft classic that sharpens coupling, cohesion, "
            "maintainability, and pragmatic design judgment."
        ),
    ),
    ResourceSpec(
        key="api_design_patterns",
        title="API Design Patterns",
        tier="supporting_classics",
        order=80,
        aliases=("api design patterns",),
        oop_relevance=(
            "Helpful support text for interface and object boundary design, "
            "especially when classes back public APIs."
        ),
    ),
    ResourceSpec(
        key="unit_testing_principles_practices_patterns",
        title="Unit Testing Principles, Practices, and Patterns",
        tier="supporting_classics",
        order=90,
        aliases=("unit testing principles practices and patterns",),
        oop_relevance=(
            "Testing discipline that makes refactoring class behavior and "
            "verifying object interactions much safer."
        ),
    ),
    ResourceSpec(
        key="generic_oop_pdf",
        title="OOP.pdf",
        tier="appendix",
        order=999,
        aliases=("oop pdf",),
        oop_relevance=(
            "Generic OOP reference with a weak title match; keep it as a low-"
            "confidence appendix item only."
        ),
        appendix_only=True,
    ),
)

NOTE_SPECS = (
    ResourceSpec(
        key="manning_software_design_python_notes",
        title="Software Design for Python Programmers Notes",
        tier="structured_notes",
        order=110,
        aliases=("manning_software_design_python",),
        oop_relevance=(
            "Chapter-by-chapter notes covering class design, encapsulation, "
            "delegation, inheritance, and design patterns."
        ),
    ),
    ResourceSpec(
        key="manning_practices_python_pro_notes",
        title="Practices of the Python Pro Notes",
        tier="structured_notes",
        order=120,
        aliases=("manning_practices_python_pro",),
        oop_relevance=(
            "Structured notes on abstraction, encapsulation, inheritance "
            "exceptions, loose coupling, and practical design tradeoffs."
        ),
    ),
    ResourceSpec(
        key="manning_well_grounded_python_dev_notes",
        title="The Well-Grounded Python Developer Notes",
        tier="structured_notes",
        order=130,
        aliases=("manning_well_grounded_python_dev",),
        oop_relevance=(
            "Organized notes on Python APIs, object-oriented programming, "
            "composition, inheritance, and code style."
        ),
    ),
    ResourceSpec(
        key="manning_publishing_python_packages_notes",
        title="Publishing Python Packages Notes",
        tier="structured_notes",
        order=140,
        aliases=("manning_publishing_python_packages",),
        oop_relevance=(
            "Supporting notes on tests, code quality tooling, CI, and packaging "
            "practices that reinforce maintainable Python design."
        ),
    ),
)

FILE_SPECS = BOOK_SPECS
NOTE_SPECS_BY_ALIAS = {spec.aliases[0]: spec for spec in NOTE_SPECS}


def normalize_text(value: str) -> str:
    normalized = re.sub(r"[^0-9a-z]+", " ", value.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def parse_year(value: str) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", value)
    return int(match.group(0)) if match else None


def parse_edition_rank(value: str) -> int:
    text = normalize_text(value)
    patterns = (
        (r"\b4th\b|\bfourth edition\b", 4),
        (r"\b3rd\b|\bthird edition\b", 3),
        (r"\b2nd\b|\bsecond edition\b|\b20th anniversary\b", 2),
        (r"\b1st\b|\bfirst edition\b", 1),
    )
    for pattern, rank in patterns:
        if re.search(pattern, text):
            return rank
    return 0


def action_needed(status: str, source_type: str) -> str:
    if status == "already_in_library":
        return "Already staged in fixtures/library_books."
    if status == "move_from_downloads":
        return "Move from Downloads into fixtures/library_books if you want it staged with the main library."
    if status == "external_reference":
        if source_type == "course_notes":
            return "Keep in place as an external note set."
        return "Keep in place as an external reference copy."
    if status == "meap_only":
        return "Use only as a preview copy until a stable final edition is available."
    if status == "duplicate":
        return "Optional alternate copy; preferred copy already identified."
    return "Review manually."


def classify_status(candidate: Candidate) -> str:
    if candidate.source_root in {"downloads_acquisitions", "downloads_top"}:
        return "move_from_downloads"
    if candidate.source_root == "library":
        return "already_in_library"
    return "external_reference"


def location_priority(candidate: Candidate) -> int:
    priorities = {
        "library": 4,
        "downloads_acquisitions": 3,
        "downloads_top": 2,
        "manning": 1,
        "course_learning": 1,
    }
    return priorities.get(candidate.source_root, 0)


def preference_key(candidate: Candidate) -> tuple[int, int, int, int, int, int, str]:
    return (
        1 if not candidate.is_preview else 0,
        1 if candidate.source_type != "book_epub" else 0,
        candidate.year or 0,
        candidate.edition_rank,
        0 if candidate.has_copy_suffix else 1,
        location_priority(candidate),
        candidate.path.name.lower(),
    )


def match_file_spec(path: Path) -> ResourceSpec | None:
    text = normalize_text(f"{path.parent.name} {path.name}")
    best_match: ResourceSpec | None = None
    best_length = -1
    for spec in FILE_SPECS:
        if spec.key == "generic_oop_pdf":
            if path.name.lower() == "oop.pdf":
                return spec
            continue
        for alias in spec.aliases:
            if alias in text and len(alias) > best_length:
                best_match = spec
                best_length = len(alias)
    return best_match


def should_skip_path(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    return any(part in lowered_parts for part in IGNORED_DIR_NAMES)


def iter_book_files(root: Path, *, recursive: bool) -> Iterable[Path]:
    if not root.exists():
        return []
    iterable = root.rglob("*") if recursive else root.iterdir()
    return (
        path
        for path in iterable
        if path.is_file() and path.suffix.lower() in BOOK_SUFFIXES and not should_skip_path(path)
    )


def collect_file_candidates(
    library_root: Path, downloads_root: Path, manning_root: Path
) -> list[Candidate]:
    candidates: list[Candidate] = []
    acquisitions_root = downloads_root / "acqusition_books" / "acqusition_books"
    scan_roots = (
        ("library", library_root, True),
        ("downloads_acquisitions", acquisitions_root, True),
        ("downloads_top", downloads_root, False),
        ("manning", manning_root, True),
    )
    for source_root, root, recursive in scan_roots:
        for path in iter_book_files(root, recursive=recursive):
            spec = match_file_spec(path)
            if spec is None:
                continue
            candidates.append(Candidate(spec=spec, path=path, source_root=source_root))
    return candidates


def collect_note_candidates(course_learning_root: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    if not course_learning_root.exists():
        return candidates
    for path in sorted(course_learning_root.iterdir()):
        if not path.is_dir():
            continue
        spec = NOTE_SPECS_BY_ALIAS.get(path.name)
        if spec is None:
            continue
        candidates.append(Candidate(spec=spec, path=path, source_root="course_learning"))
    return candidates


def group_candidates(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    grouped: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.spec.key, []).append(candidate)
    return grouped


def make_entry(
    candidate: Candidate,
    *,
    tier: str,
    status: str,
    preferred_candidate: Candidate,
    duplicate_of: str,
    notes: str,
) -> ReportEntry:
    return ReportEntry(
        title=candidate.spec.title,
        tier=tier,
        order=candidate.spec.order,
        source_type=candidate.source_type,
        status=status,
        oop_relevance=candidate.spec.oop_relevance,
        current_path=str(candidate.path),
        preferred_path=str(preferred_candidate.path),
        duplicate_of=duplicate_of,
        notes=notes,
    )


def summarize_duplicate_notes(candidate: Candidate, preferred_candidate: Candidate) -> str:
    reasons: list[str] = []
    if candidate.is_preview:
        reasons.append("preview_copy")
    if candidate.source_type == "book_epub":
        reasons.append("epub_alternate")
    if candidate.year and preferred_candidate.year and candidate.year < preferred_candidate.year:
        reasons.append("older_edition")
    if candidate.has_copy_suffix:
        reasons.append("filename_copy_suffix")
    if not reasons:
        reasons.append("alternate_copy")
    return "; ".join(reasons)


def build_entries(
    library_root: Path,
    downloads_root: Path,
    manning_root: Path,
    course_learning_root: Path,
) -> list[ReportEntry]:
    candidates = collect_file_candidates(library_root, downloads_root, manning_root)
    candidates.extend(collect_note_candidates(course_learning_root))
    grouped = group_candidates(candidates)

    entries: list[ReportEntry] = []
    for spec in sorted((*BOOK_SPECS, *NOTE_SPECS), key=lambda item: item.order):
        group = grouped.get(spec.key, [])
        if not group:
            continue

        ordered_group = sorted(group, key=preference_key, reverse=True)
        preferred_candidate = ordered_group[0]
        stable_candidates = [candidate for candidate in ordered_group if not candidate.is_preview]

        if spec.appendix_only:
            entries.append(
                make_entry(
                    preferred_candidate,
                    tier="appendix",
                    status=classify_status(preferred_candidate),
                    preferred_candidate=preferred_candidate,
                    duplicate_of="",
                    notes="low_confidence_generic_title",
                )
            )
            for candidate in ordered_group[1:]:
                entries.append(
                    make_entry(
                        candidate,
                        tier="appendix",
                        status="duplicate",
                        preferred_candidate=preferred_candidate,
                        duplicate_of=spec.title,
                        notes=summarize_duplicate_notes(candidate, preferred_candidate),
                    )
                )
            continue

        if not stable_candidates:
            entries.append(
                make_entry(
                    preferred_candidate,
                    tier="appendix",
                    status="meap_only",
                    preferred_candidate=preferred_candidate,
                    duplicate_of="",
                    notes="preview_only_copy",
                )
            )
            for candidate in ordered_group[1:]:
                entries.append(
                    make_entry(
                        candidate,
                        tier="appendix",
                        status="duplicate",
                        preferred_candidate=preferred_candidate,
                        duplicate_of=spec.title,
                        notes=summarize_duplicate_notes(candidate, preferred_candidate),
                    )
                )
            continue

        preferred_candidate = stable_candidates[0]
        entries.append(
            make_entry(
                preferred_candidate,
                tier=spec.tier,
                status=classify_status(preferred_candidate),
                preferred_candidate=preferred_candidate,
                duplicate_of="",
                notes=action_needed(
                    classify_status(preferred_candidate), preferred_candidate.source_type
                ),
            )
        )
        for candidate in ordered_group:
            if candidate.path == preferred_candidate.path:
                continue
            entries.append(
                make_entry(
                    candidate,
                    tier="appendix",
                    status="duplicate",
                    preferred_candidate=preferred_candidate,
                    duplicate_of=spec.title,
                    notes=summarize_duplicate_notes(candidate, preferred_candidate),
                )
            )
    return sorted(entries, key=entry_sort_key)


def entry_sort_key(entry: ReportEntry) -> tuple[int, int, int, str]:
    status_rank = {"duplicate": 1, "meap_only": 2}
    return (
        SECTION_ORDER[entry.tier],
        entry.order,
        status_rank.get(entry.status, 0),
        entry.current_path.lower(),
    )


def render_primary_entry(entry: ReportEntry) -> str:
    return "\n".join(
        [
            f"### {entry.title}",
            f"- Why it matters: {entry.oop_relevance}",
            f"- Source type: {SOURCE_TYPE_LABELS[entry.source_type]}",
            f"- Status: `{entry.status}`",
            f"- Current path: `{entry.current_path}`",
            f"- Preferred copy: `{entry.preferred_path}`",
            f"- Action needed: {action_needed(entry.status, entry.source_type)}",
            "",
        ]
    )


def render_appendix(entries: list[ReportEntry]) -> str:
    grouped: dict[str, list[ReportEntry]] = {}
    for entry in entries:
        heading = entry.duplicate_of or entry.title
        grouped.setdefault(heading, []).append(entry)

    lines = ["## Appendix", ""]
    for heading in sorted(grouped):
        lines.append(f"### {heading}")
        for entry in grouped[heading]:
            if entry.status == "duplicate":
                lines.append(
                    f"- `duplicate` {SOURCE_TYPE_LABELS[entry.source_type]}: `{entry.current_path}`"
                )
                lines.append(f"- Preferred copy: `{entry.preferred_path}`")
                lines.append(f"- Notes: {entry.notes}")
            else:
                lines.append(
                    f"- `{entry.status}` {SOURCE_TYPE_LABELS[entry.source_type]}: `{entry.current_path}`"
                )
                lines.append(f"- Preferred copy: `{entry.preferred_path}`")
                lines.append(f"- Why it matters: {entry.oop_relevance}")
                lines.append(f"- Notes: {entry.notes}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_markdown(entries: list[ReportEntry]) -> str:
    primary_entries = [entry for entry in entries if entry.tier != "appendix"]
    appendix_entries = [entry for entry in entries if entry.tier == "appendix"]

    lines = [
        "# Python OOP Best Practices Inventory",
        "",
        "Curated from the highest-signal Python-design books and structured note trees found under `/home/brandon_behring`.",
        "",
    ]
    for tier in ("core_python_oop", "structured_notes", "supporting_classics"):
        section_entries = [entry for entry in primary_entries if entry.tier == tier]
        if not section_entries:
            continue
        lines.append(f"## {SECTION_TITLES[tier]}")
        lines.append("")
        for entry in section_entries:
            lines.append(render_primary_entry(entry).rstrip())
            lines.append("")
    if appendix_entries:
        lines.append(render_appendix(appendix_entries).rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_python_oop_report(
    *,
    library_root: Path,
    downloads_root: Path,
    manning_root: Path,
    course_learning_root: Path,
) -> tuple[str, list[dict[str, str]]]:
    entries = build_entries(
        library_root=library_root,
        downloads_root=downloads_root,
        manning_root=manning_root,
        course_learning_root=course_learning_root,
    )
    markdown = render_markdown(entries)
    rows = [
        {
            "tier": entry.tier,
            "title": entry.title,
            "source_type": entry.source_type,
            "status": entry.status,
            "oop_relevance": entry.oop_relevance,
            "current_path": entry.current_path,
            "preferred_path": entry.preferred_path,
            "duplicate_of": entry.duplicate_of,
            "notes": entry.notes,
        }
        for entry in entries
    ]
    return markdown, rows


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tier",
        "title",
        "source_type",
        "status",
        "oop_relevance",
        "current_path",
        "preferred_path",
        "duplicate_of",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a curated Python OOP best-practices report."
    )
    parser.add_argument(
        "--library-root",
        type=Path,
        default=DEFAULT_LIBRARY_ROOT,
        help="fixtures/library_books root",
    )
    parser.add_argument(
        "--downloads-root",
        type=Path,
        default=DEFAULT_DOWNLOADS_ROOT,
        help="Downloads root containing standalone files and acqusition_books",
    )
    parser.add_argument(
        "--manning-root",
        type=Path,
        default=DEFAULT_MANNING_ROOT,
        help="Documents/Manning_Books_research_kb root",
    )
    parser.add_argument(
        "--course-learning-root",
        type=Path,
        default=DEFAULT_COURSE_LEARNING_ROOT,
        help="Claude/course_learning root",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Markdown report output path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="CSV source output path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    markdown, rows = build_python_oop_report(
        library_root=args.library_root,
        downloads_root=args.downloads_root,
        manning_root=args.manning_root,
        course_learning_root=args.course_learning_root,
    )
    write_markdown(args.output_md, markdown)
    write_csv(args.output_csv, rows)
    print(f"Wrote report -> {args.output_md}")
    print(f"Wrote {len(rows)} source rows -> {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
