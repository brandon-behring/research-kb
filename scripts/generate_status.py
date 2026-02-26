#!/usr/bin/env python3
"""Generate CURRENT_STATUS.md from actual database state.

This script queries the database and generates accurate status documentation,
preventing drift between documentation and reality.

Usage:
    python scripts/generate_status.py           # Generate status doc
    python scripts/generate_status.py --check   # Check if docs match reality (for CI)
    python scripts/generate_status.py --stdout  # Print to stdout instead of file
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add packages to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

import asyncpg


def get_phase_status(stats: dict) -> dict:
    """Derive Phase status from actual state using hybrid approach.

    - Phase 1-2: Data-based (tables populated)
    - Phase 3: File existence (features implemented)
    - Phase 4: Package existence (api/dashboard)
    """
    repo_root = Path(__file__).parent.parent

    # Phase 1: Foundation - data exists
    phase1_complete = stats.get("sources", 0) > 0 and stats.get("chunks", 0) > 0

    # Phase 1.5: PDF Ingestion - substantial data
    phase15_complete = phase1_complete and stats.get("chunks", 0) > 1000

    # Phase 2: Knowledge Graph - concepts extracted
    phase2_complete = stats.get("concepts", 0) > 0 and stats.get("relationships", 0) > 0

    # Phase 3: Enhanced Retrieval - feature files exist
    phase3_features = [
        repo_root / "fixtures" / "concepts" / "synonym_map.yaml",
        repo_root / "packages" / "pdf-tools" / "src" / "research_kb_pdf" / "rerank_client.py",
        repo_root / "packages" / "storage" / "src" / "research_kb_storage" / "query_expander.py",
        repo_root / "packages" / "storage" / "src" / "research_kb_storage" / "citation_graph.py",
    ]
    phase3_complete = all(f.exists() for f in phase3_features)

    # Phase 4: API & Dashboard - packages exist
    phase4_packages = [
        repo_root / "packages" / "api" / "src" / "research_kb_api" / "main.py",
        repo_root / "packages" / "dashboard" / "src" / "research_kb_dashboard" / "app.py",
    ]
    phase4_complete = all(p.exists() for p in phase4_packages)

    return {
        "phase1": (
            ("✅ Complete", "PostgreSQL + pgvector operational")
            if phase1_complete
            else ("📋 Pending", "Setup required")
        ),
        "phase15": (
            (
                "✅ Complete",
                f"{stats.get('sources', 0):,} sources, {stats.get('chunks', 0):,} chunks",
            )
            if phase15_complete
            else ("📋 Pending", "Ingestion required")
        ),
        "phase2": (
            (
                "✅ Complete",
                f"{stats.get('concepts', 0):,} concepts, {stats.get('relationships', 0):,} relationships",
            )
            if phase2_complete
            else ("📋 Pending", "Concept extraction required")
        ),
        "phase3": (
            ("✅ Complete", "Query expansion, reranking, citation graph")
            if phase3_complete
            else ("📋 Ready to start", "No blockers")
        ),
        "phase4": (
            ("✅ Complete", "FastAPI + Streamlit dashboard")
            if phase4_complete
            else (
                "📋 Planned",
                "Pending Phase 3" if not phase3_complete else "Ready to start",
            )
        ),
    }


async def get_db_stats() -> dict:
    """Query actual database state."""
    pool = await asyncpg.create_pool(
        host="localhost",
        port=5432,
        database="research_kb",
        user="postgres",
        password="postgres",
        min_size=1,
        max_size=2,
    )

    try:
        async with pool.acquire() as conn:
            stats = {}

            # Core counts
            stats["chunks"] = await conn.fetchval("SELECT COUNT(*) FROM chunks")
            stats["concepts"] = await conn.fetchval("SELECT COUNT(*) FROM concepts")
            stats["sources"] = await conn.fetchval("SELECT COUNT(*) FROM sources")
            stats["relationships"] = await conn.fetchval(
                "SELECT COUNT(*) FROM concept_relationships"
            )
            stats["chunk_concepts"] = await conn.fetchval("SELECT COUNT(*) FROM chunk_concepts")
            stats["citations"] = await conn.fetchval("SELECT COUNT(*) FROM citations")

            # Enrichment tables
            stats["methods"] = await conn.fetchval("SELECT COUNT(*) FROM methods")
            stats["assumptions"] = await conn.fetchval("SELECT COUNT(*) FROM assumptions")

            # Enrichment coverage (methods with attributes populated)
            stats["methods_with_assumptions"] = await conn.fetchval(
                "SELECT COUNT(*) FROM methods WHERE array_length(required_assumptions, 1) > 0"
            )
            stats["assumptions_with_math"] = await conn.fetchval(
                "SELECT COUNT(*) FROM assumptions WHERE mathematical_statement IS NOT NULL"
            )

            # Embedding coverage
            stats["chunks_with_embeddings"] = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
            )
            stats["concepts_with_embeddings"] = await conn.fetchval(
                "SELECT COUNT(*) FROM concepts WHERE embedding IS NOT NULL"
            )

            # Source breakdown
            source_types = await conn.fetch(
                "SELECT source_type, COUNT(*) as count FROM sources GROUP BY source_type ORDER BY count DESC"
            )
            stats["source_types"] = {row["source_type"]: row["count"] for row in source_types}

            # Concept type breakdown
            concept_types = await conn.fetch(
                "SELECT concept_type, COUNT(*) as count FROM concepts GROUP BY concept_type ORDER BY count DESC"
            )
            stats["concept_types"] = {row["concept_type"]: row["count"] for row in concept_types}

            # Relationship type breakdown
            rel_types = await conn.fetch(
                "SELECT relationship_type, COUNT(*) as count FROM concept_relationships GROUP BY relationship_type ORDER BY count DESC"
            )
            stats["relationship_types"] = {
                row["relationship_type"]: row["count"] for row in rel_types
            }

            # Domain breakdown (canonical: domain_id column, not metadata JSON)
            domain_rows = await conn.fetch("""SELECT s.domain_id AS domain, COUNT(*) AS source_count
                   FROM sources s
                   GROUP BY s.domain_id
                   ORDER BY source_count DESC""")
            stats["domains"] = {row["domain"]: row["source_count"] for row in domain_rows}

            # Date ranges
            stats["sources_date_range"] = await conn.fetchrow(
                "SELECT MIN(created_at) as min, MAX(created_at) as max FROM sources"
            )
            stats["concepts_date_range"] = await conn.fetchrow(
                "SELECT MIN(created_at) as min, MAX(created_at) as max FROM concepts"
            )

            return stats

    finally:
        await pool.close()


def generate_status_md(stats: dict) -> str:
    """Generate markdown status document from stats."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get dynamic Phase status
    phase_status = get_phase_status(stats)

    # Calculate percentages
    chunk_embed_pct = (
        100 * stats["chunks_with_embeddings"] / stats["chunks"] if stats["chunks"] > 0 else 0
    )
    concept_embed_pct = (
        100 * stats["concepts_with_embeddings"] / stats["concepts"] if stats["concepts"] > 0 else 0
    )

    # Format source types
    source_lines = "\n".join(
        f"| {stype} | {count:,} |" for stype, count in stats["source_types"].items()
    )

    # Format concept types
    concept_lines = "\n".join(
        f"| {ctype} | {count:,} |" for ctype, count in stats["concept_types"].items()
    )

    # Format relationship types
    rel_lines = "\n".join(
        f"| {rtype} | {count:,} |" for rtype, count in stats["relationship_types"].items()
    )

    # Format domain breakdown
    domain_lines = "\n".join(
        f"| {domain} | {count:,} |" for domain, count in stats.get("domains", {}).items()
    )

    return f"""# Current Status

**Auto-generated**: {timestamp}
**DO NOT EDIT MANUALLY** - Run `python scripts/generate_status.py` to update

---

## Database State

| Table | Count |
|-------|------:|
| sources | {stats['sources']:,} |
| chunks | {stats['chunks']:,} |
| concepts | {stats['concepts']:,} |
| concept_relationships | {stats['relationships']:,} |
| chunk_concepts | {stats['chunk_concepts']:,} |
| citations | {stats['citations']:,} |
| methods | {stats['methods']:,} |
| assumptions | {stats['assumptions']:,} |

---

## Enrichment Coverage

| Entity | Enriched | Total | Coverage |
|--------|----------:|------:|---------:|
| Methods (with assumptions) | {stats['methods_with_assumptions']:,} | {stats['methods']:,} | {100 * stats['methods_with_assumptions'] / max(stats['methods'], 1):.1f}% |
| Assumptions (with math) | {stats['assumptions_with_math']:,} | {stats['assumptions']:,} | {100 * stats['assumptions_with_math'] / max(stats['assumptions'], 1):.1f}% |

---

## Embedding Coverage

| Entity | With Embeddings | Total | Coverage |
|--------|----------------:|------:|---------:|
| Chunks | {stats['chunks_with_embeddings']:,} | {stats['chunks']:,} | {chunk_embed_pct:.1f}% |
| Concepts | {stats['concepts_with_embeddings']:,} | {stats['concepts']:,} | {concept_embed_pct:.1f}% |

---

## Source Breakdown

| Type | Count |
|------|------:|
{source_lines}

---

## Domain Breakdown

| Domain | Sources |
|--------|--------:|
{domain_lines}

---

## Concept Type Distribution

| Type | Count |
|------|------:|
{concept_lines}

---

## Relationship Type Distribution

| Type | Count |
|------|------:|
{rel_lines}

---

## Phase Status

Based on database population and feature detection:

| Phase | Status | Evidence |
|-------|--------|----------|
| Phase 1: Foundation | {phase_status['phase1'][0]} | {phase_status['phase1'][1]} |
| Phase 1.5: PDF Ingestion | {phase_status['phase15'][0]} | {phase_status['phase15'][1]} |
| Phase 2: Knowledge Graph | {phase_status['phase2'][0]} | {phase_status['phase2'][1]} |
| Phase 3: Enhanced Retrieval | {phase_status['phase3'][0]} | {phase_status['phase3'][1]} |
| Phase 4: API & Dashboard | {phase_status['phase4'][0]} | {phase_status['phase4'][1]} |

---

## Quick Commands

```bash
# Regenerate this file
python scripts/generate_status.py

# View live stats
research-kb stats

# Run quality checks
python scripts/run_quality_checks.py
```

---

*Generated by `scripts/generate_status.py` - database is the source of truth*
"""


def main():
    parser = argparse.ArgumentParser(description="Generate status documentation from database")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if current docs match reality (exit 1 if different)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print to stdout instead of writing file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/status/CURRENT_STATUS.md",
        help="Output file path (default: docs/status/CURRENT_STATUS.md)",
    )
    args = parser.parse_args()

    # Get stats from database
    print("Querying database...")
    stats = asyncio.run(get_db_stats())

    # Generate markdown
    content = generate_status_md(stats)

    if args.stdout:
        print(content)
        return

    output_path = Path(__file__).parent.parent / args.output

    if args.check:
        # Check mode: compare with existing file
        if output_path.exists():
            existing = output_path.read_text()
            # Compare ignoring timestamp line
            existing_lines = [
                l for l in existing.split("\n") if not l.startswith("**Auto-generated**")
            ]
            new_lines = [l for l in content.split("\n") if not l.startswith("**Auto-generated**")]
            if existing_lines != new_lines:
                print(f"ERROR: {args.output} is out of date!")
                print("Run 'python scripts/generate_status.py' to update")
                sys.exit(1)
            else:
                print(f"OK: {args.output} matches database state")
        else:
            print(f"ERROR: {args.output} does not exist!")
            sys.exit(1)
    else:
        # Write mode
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        print(f"✓ Updated {args.output}")
        print(f"  Sources: {stats['sources']:,}")
        print(f"  Chunks: {stats['chunks']:,}")
        print(f"  Concepts: {stats['concepts']:,}")
        print(f"  Relationships: {stats['relationships']:,}")


if __name__ == "__main__":
    main()
