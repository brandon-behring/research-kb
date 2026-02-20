#!/usr/bin/env python3
"""Cross-reference AVAILABLE.md against research-kb database.

Finds papers that are locally available but not yet ingested into research-kb.

Usage:
    python scripts/cross_reference_available.py
    python scripts/cross_reference_available.py --output quality-reports/gaps.json
"""

import asyncio
import json
import re
import sys
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_storage import get_connection_pool, DatabaseConfig


@dataclass
class AvailablePaper:
    """Paper from AVAILABLE.md."""

    title: str
    authors: list[str]
    year: Optional[str]
    file_path: str
    directory: str


@dataclass
class IngestedSource:
    """Source from research-kb database."""

    id: str
    title: str
    file_path: str


@dataclass
class MatchResult:
    """Result of matching an available paper."""

    available: AvailablePaper
    match_type: str  # "exact", "fuzzy", "none"
    match_score: float
    matched_source: Optional[IngestedSource] = None


def parse_available_md(filepath: str) -> list[AvailablePaper]:
    """Parse AVAILABLE.md into list of papers.

    Format:
    ## `~/path/to/directory`
    **Count**: N

    - **Title** (year)
      - File: `filename.pdf`
      - Authors: Author1 and Author2
    """
    papers = []
    content = Path(filepath).read_text()

    current_directory = None
    current_title = None
    current_year = None
    current_file = None
    current_authors = []

    lines = content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Directory header: ## `~/path/to/dir`
        if line.startswith("## `") and line.endswith("`"):
            current_directory = line[4:-1]

        # Paper entry: - **Title** (year)
        elif line.startswith("- **"):
            # Save previous paper if exists
            if current_title and current_file:
                papers.append(
                    AvailablePaper(
                        title=current_title,
                        authors=current_authors,
                        year=current_year,
                        file_path=current_file,
                        directory=current_directory or "",
                    )
                )

            # Parse new paper
            match = re.match(r"- \*\*(.+?)\*\*\s*\((\d{4}|n\.d\.|various)\)", line)
            if match:
                current_title = match.group(1)
                year_str = match.group(2)
                current_year = year_str if year_str not in ("n.d.", "various") else None
            else:
                # Title without year
                match = re.match(r"- \*\*(.+?)\*\*", line)
                if match:
                    current_title = match.group(1)
                    current_year = None

            current_file = None
            current_authors = []

        # File line: - File: `filename.pdf`
        elif line.startswith("- File:"):
            match = re.search(r"`([^`]+)`", line)
            if match:
                filename = match.group(1)
                if current_directory:
                    current_file = f"{current_directory}/{filename}"
                else:
                    current_file = filename

        # Authors line: - Authors: ...
        elif line.startswith("- Authors:"):
            authors_str = line[10:].strip()
            # Parse authors (handle "and", ",", truncation)
            authors_str = re.sub(r"\s+and\s+", ", ", authors_str)
            current_authors = [a.strip() for a in authors_str.split(",") if a.strip()]

        i += 1

    # Don't forget last paper
    if current_title and current_file:
        papers.append(
            AvailablePaper(
                title=current_title,
                authors=current_authors,
                year=current_year,
                file_path=current_file,
                directory=current_directory or "",
            )
        )

    return papers


async def get_ingested_sources() -> list[IngestedSource]:
    """Query all sources from research-kb database."""
    config = DatabaseConfig()
    pool = await get_connection_pool(config)

    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT id, title, file_path FROM sources")
        return [
            IngestedSource(id=str(row["id"]), title=row["title"], file_path=row["file_path"] or "")
            for row in rows
        ]


def normalize_title(title: str) -> str:
    """Normalize title for comparison."""
    # Lowercase, remove punctuation, collapse whitespace
    title = title.lower()
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def extract_filename_stem(filepath: str) -> str:
    """Extract filename without extension."""
    return Path(filepath).stem.lower()


def fuzzy_match_score(s1: str, s2: str) -> float:
    """Calculate fuzzy match score using SequenceMatcher."""
    return SequenceMatcher(None, s1, s2).ratio()


def match_paper(paper: AvailablePaper, sources: list[IngestedSource]) -> MatchResult:
    """Match an available paper against ingested sources."""
    paper_title_norm = normalize_title(paper.title)
    paper_filename = extract_filename_stem(paper.file_path)

    best_match = None
    best_score = 0.0
    best_type = "none"

    for source in sources:
        source_title_norm = normalize_title(source.title)
        source_filename = extract_filename_stem(source.file_path)

        # Check exact title match
        if paper_title_norm == source_title_norm:
            return MatchResult(
                available=paper,
                match_type="exact_title",
                match_score=1.0,
                matched_source=source,
            )

        # Check filename match
        if paper_filename and source_filename:
            if paper_filename == source_filename:
                return MatchResult(
                    available=paper,
                    match_type="exact_filename",
                    match_score=1.0,
                    matched_source=source,
                )

        # Fuzzy title match
        title_score = fuzzy_match_score(paper_title_norm, source_title_norm)
        if title_score > best_score:
            best_score = title_score
            best_match = source
            best_type = "fuzzy_title"

        # Fuzzy filename match (if both have meaningful filenames)
        if len(paper_filename) > 5 and len(source_filename) > 5:
            filename_score = fuzzy_match_score(paper_filename, source_filename)
            if filename_score > best_score:
                best_score = filename_score
                best_match = source
                best_type = "fuzzy_filename"

    # Threshold for considering a match
    if best_score >= 0.8:
        return MatchResult(
            available=paper,
            match_type=best_type,
            match_score=best_score,
            matched_source=best_match,
        )
    elif best_score >= 0.6:
        return MatchResult(
            available=paper,
            match_type="uncertain",
            match_score=best_score,
            matched_source=best_match,
        )
    else:
        return MatchResult(
            available=paper,
            match_type="none",
            match_score=best_score,
            matched_source=None,
        )


def categorize_paper(paper: AvailablePaper) -> str:
    """Categorize paper by topic based on title/path."""
    title_lower = paper.title.lower()
    path_lower = paper.file_path.lower()

    if any(kw in title_lower for kw in ["causal", "treatment effect", "propensity"]):
        return "causal_inference"
    elif any(kw in title_lower for kw in ["time series", "forecasting", "arima"]):
        return "time_series"
    elif any(kw in title_lower for kw in ["graph", "network", "knowledge graph"]):
        return "knowledge_graphs"
    elif any(kw in title_lower for kw in ["bayesian", "probabilistic"]):
        return "bayesian"
    elif any(kw in title_lower for kw in ["llm", "language model", "transformer", "attention"]):
        return "llm"
    elif any(kw in title_lower for kw in ["machine learning", "deep learning", "neural"]):
        return "ml"
    elif "manning" in path_lower:
        return "manning_book"
    else:
        return "other"


def prioritize_paper(paper: AvailablePaper) -> str:
    """Assign priority based on importance."""
    title_lower = paper.title.lower()

    # High priority: canonical works
    high_priority_keywords = [
        "box",
        "jenkins",
        "time series analysis",
        "pearl",
        "causality",
        "neapolitan",
        "bayesian network",
        "koller",
        "probabilistic graphical",
        "imbens",
        "rubin",
    ]
    if any(kw in title_lower for kw in high_priority_keywords):
        return "high"

    # Medium: causal inference, time series, graphs
    category = categorize_paper(paper)
    if category in ("causal_inference", "time_series", "knowledge_graphs", "bayesian"):
        return "medium"

    return "low"


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Cross-reference AVAILABLE.md vs research-kb")
    parser.add_argument(
        "--available",
        default="$HOME/Claude/lever_of_archimedes/knowledge/master_bibliography/AVAILABLE.md",
        help="Path to AVAILABLE.md",
    )
    parser.add_argument(
        "--output",
        default="quality-reports/cross_reference.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--report",
        default="quality-reports/cross_reference_report.md",
        help="Output markdown report path",
    )
    args = parser.parse_args()

    print(f"Parsing {args.available}...")
    available_papers = parse_available_md(args.available)
    print(f"  Found {len(available_papers)} available papers")

    print("Querying research-kb database...")
    ingested_sources = await get_ingested_sources()
    print(f"  Found {len(ingested_sources)} ingested sources")

    print("Matching papers...")
    results = {
        "not_ingested": [],
        "already_ingested": [],
        "uncertain_matches": [],
    }

    for paper in available_papers:
        match = match_paper(paper, ingested_sources)

        paper_dict = asdict(paper)
        paper_dict["category"] = categorize_paper(paper)
        paper_dict["priority"] = prioritize_paper(paper)

        if match.match_type in ("exact_title", "exact_filename", "fuzzy_title", "fuzzy_filename"):
            results["already_ingested"].append(
                {
                    **paper_dict,
                    "match_type": match.match_type,
                    "match_score": match.match_score,
                    "matched_title": match.matched_source.title if match.matched_source else None,
                }
            )
        elif match.match_type == "uncertain":
            results["uncertain_matches"].append(
                {
                    **paper_dict,
                    "match_score": match.match_score,
                    "possible_match": match.matched_source.title if match.matched_source else None,
                }
            )
        else:
            results["not_ingested"].append(paper_dict)

    # Sort not_ingested by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    results["not_ingested"].sort(key=lambda x: priority_order.get(x["priority"], 3))

    # Write JSON output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote JSON to {output_path}")

    # Write markdown report
    report_path = Path(args.report)
    with open(report_path, "w") as f:
        f.write("# Cross-Reference Report: Available vs Ingested\n\n")
        f.write(f"**Generated**: {__import__('datetime').datetime.now().isoformat()}\n\n")
        f.write(f"- Available papers: {len(available_papers)}\n")
        f.write(f"- Ingested sources: {len(ingested_sources)}\n")
        f.write(f"- Already ingested: {len(results['already_ingested'])}\n")
        f.write(f"- Not ingested: {len(results['not_ingested'])}\n")
        f.write(f"- Uncertain matches: {len(results['uncertain_matches'])}\n\n")

        f.write("---\n\n")
        f.write("## High Priority - Not Ingested\n\n")
        high_priority = [p for p in results["not_ingested"] if p["priority"] == "high"]
        for p in high_priority:
            f.write(f"- **{p['title']}** ({p.get('year', 'n.d.')})\n")
            f.write(f"  - Path: `{p['file_path']}`\n")
            f.write(f"  - Category: {p['category']}\n\n")

        f.write("---\n\n")
        f.write("## Medium Priority - Not Ingested\n\n")
        medium_priority = [p for p in results["not_ingested"] if p["priority"] == "medium"]
        for p in medium_priority[:20]:  # Limit to first 20
            f.write(f"- **{p['title']}** ({p.get('year', 'n.d.')})\n")
            f.write(f"  - Category: {p['category']}\n\n")
        if len(medium_priority) > 20:
            f.write(f"... and {len(medium_priority) - 20} more\n\n")

        f.write("---\n\n")
        f.write("## Uncertain Matches (Review Needed)\n\n")
        for p in results["uncertain_matches"][:10]:
            f.write(f"- **{p['title']}**\n")
            f.write(f"  - Score: {p['match_score']:.2f}\n")
            f.write(f"  - Possible match: {p['possible_match']}\n\n")

    print(f"Wrote report to {report_path}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Not ingested (total): {len(results['not_ingested'])}")
    print(f"  High priority: {len(high_priority)}")
    print(f"  Medium priority: {len(medium_priority)}")
    print(f"Uncertain matches: {len(results['uncertain_matches'])}")
    print(f"Already ingested: {len(results['already_ingested'])}")


if __name__ == "__main__":
    asyncio.run(main())
