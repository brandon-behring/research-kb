#!/usr/bin/env python3
"""Validate a markdown glossary against research-kb knowledge base.

Reads a markdown glossary file, extracts terms, and queries research-kb
for each term. Reports: matched terms, missing terms, definition drift.

Usage:
    python scripts/validate_glossary.py ~/Claude/causal_inference_mastery/docs/GLOSSARY.md
    python scripts/validate_glossary.py --json <glossary_path>
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_storage import DatabaseConfig, get_connection_pool


def extract_glossary_terms(filepath: Path) -> list[dict]:
    """Extract terms from a markdown glossary.

    Supports these patterns:
    - **Term**: Definition
    - ## Term
    - ### Term
    - - **Term** — Definition
    - - **Term**: Definition
    - Term (abbreviation): Definition

    Args:
        filepath: Path to markdown glossary

    Returns:
        List of dicts with 'term' and optional 'definition' keys
    """
    terms = []
    content = filepath.read_text()

    for line in content.split("\n"):
        line = line.strip()

        # Skip empty lines, comments, horizontal rules
        if not line or line.startswith("---") or line.startswith("```"):
            continue

        # Pattern: **Term**: Definition or **Term** — Definition
        bold_match = re.match(r"^[-*]?\s*\*\*(.+?)\*\*\s*[:—-]\s*(.*)", line)
        if bold_match:
            term = bold_match.group(1).strip()
            definition = bold_match.group(2).strip()
            if len(term) > 1 and len(term) < 100:
                terms.append({"term": term, "definition": definition})
            continue

        # Pattern: ## Term or ### Term (but not # which is title)
        heading_match = re.match(r"^#{2,4}\s+(.+)", line)
        if heading_match:
            term = heading_match.group(1).strip()
            # Skip entries that look like instructions or contain parenthetical counts
            if re.search(r"\(\d+\s+\w+\)", term) or term.lower().startswith("to "):
                continue
            # Skip headings that are section names like "Core Concepts"
            if (
                term.lower()
                not in {
                    "core concepts",
                    "methods",
                    "assumptions",
                    "glossary",
                    "references",
                    "index",
                    "table of contents",
                    "overview",
                    "variance estimators",
                    "estimands",
                    "problems",
                    "theorems",
                    "implementation-specific terms",
                    "test terminology",
                    "monte carlo metrics",
                    "dgp parameters",
                    "method-specific terms",
                    "acronym reference",
                }
                and not term.endswith("-Specific")
                and not term.endswith("Terms")
            ):
                terms.append({"term": term, "definition": ""})
            continue

    return terms


async def validate_terms(terms: list[dict], conn) -> list[dict]:
    """Validate glossary terms against research-kb.

    For each term, checks:
    1. Concept existence (canonical_name match)
    2. Chunk content presence (at least one chunk mentions it)
    3. Definition similarity (basic keyword overlap)

    Args:
        terms: List of term dicts
        conn: asyncpg connection

    Returns:
        List of validation result dicts
    """
    results = []

    for entry in terms:
        term = entry["term"]
        definition = entry.get("definition", "")

        # Check concepts table (exact or fuzzy)
        concept = await conn.fetchrow(
            """
            SELECT id, canonical_name, concept_type, name
            FROM concepts
            WHERE canonical_name ILIKE $1
               OR name ILIKE $1
            LIMIT 1
            """,
            f"%{term}%",
        )

        # Check chunk content
        chunk_count = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM chunks
            WHERE content ILIKE $1
            """,
            f"%{term}%",
        )

        # Determine status
        if concept and chunk_count > 0:
            status = "matched"
        elif concept:
            status = "concept_only"
        elif chunk_count > 0:
            status = "content_only"
        else:
            status = "missing"

        result = {
            "term": term,
            "status": status,
            "concept_found": concept is not None,
            "concept_name": concept["canonical_name"] if concept else None,
            "concept_type": concept["concept_type"] if concept else None,
            "chunk_count": chunk_count,
        }

        # Basic definition drift detection: if we have both a definition
        # and a concept, check if key words overlap
        if definition and concept:
            def_words = set(definition.lower().split())
            concept_words = set(concept["name"].lower().split())
            overlap = len(def_words & concept_words)
            result["definition_overlap"] = overlap

        results.append(result)

    return results


async def main():
    """Run glossary validation."""
    parser = argparse.ArgumentParser(description="Validate glossary against research-kb")
    parser.add_argument("glossary", type=Path, help="Path to markdown glossary file")
    parser.add_argument("--json", action="store_true", help="JSON output only")
    args = parser.parse_args()

    if not args.glossary.exists():
        print(f"Error: {args.glossary} does not exist")
        sys.exit(1)

    # Extract terms
    terms = extract_glossary_terms(args.glossary)
    if not terms:
        print("No glossary terms found in file")
        sys.exit(1)

    # Connect to database
    config = DatabaseConfig()
    pool = await get_connection_pool(config)

    async with pool.acquire() as conn:
        results = await validate_terms(terms, conn)

    # Compute summary
    matched = sum(1 for r in results if r["status"] == "matched")
    concept_only = sum(1 for r in results if r["status"] == "concept_only")
    content_only = sum(1 for r in results if r["status"] == "content_only")
    missing = sum(1 for r in results if r["status"] == "missing")

    summary = {
        "glossary": str(args.glossary),
        "total_terms": len(terms),
        "matched": matched,
        "concept_only": concept_only,
        "content_only": content_only,
        "missing": missing,
        "coverage_pct": (
            round((matched + concept_only + content_only) / len(terms) * 100, 1) if terms else 0
        ),
    }

    if args.json:
        output = {"summary": summary, "results": results}
        print(json.dumps(output, indent=2))
        return

    # Human-readable output
    print("=" * 70)
    print(f"GLOSSARY VALIDATION: {args.glossary.name}")
    print("=" * 70)
    print(f"\nTerms extracted: {len(terms)}")
    print(f"Matched (concept + content): {matched}")
    print(f"Concept only (no chunks): {concept_only}")
    print(f"Content only (no concept): {content_only}")
    print(f"Missing: {missing}")
    print(f"Coverage: {summary['coverage_pct']}%")

    if missing > 0:
        print(f"\nMissing terms ({missing}):")
        print("-" * 50)
        for r in results:
            if r["status"] == "missing":
                print(f"  - {r['term']}")

    if concept_only > 0:
        print(f"\nConcept-only terms (no chunks, {concept_only}):")
        print("-" * 50)
        for r in results:
            if r["status"] == "concept_only":
                print(f"  - {r['term']} ({r['concept_type']})")

    print()
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
