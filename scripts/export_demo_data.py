#!/usr/bin/env python3
"""Export demo corpus data from the live database to JSON fixtures.

Exports sources, chunks (text only, no embeddings), concepts, relationships,
chunk_concepts, and citations for the demo paper set.

Usage:
    python scripts/export_demo_data.py                  # Export all demo papers
    python scripts/export_demo_data.py --output-dir fixtures/demo/data
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import UUID

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "storage" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "common" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "contracts" / "src"))

# arXiv IDs matching download_demo_corpus.sh
DEMO_ARXIV_IDS = [
    "1608.00060",
    "1510.04342",
    "1712.09988",
    "1504.01132",
    "1803.09015",
    "2108.02196",
    "1011.1079",
    "1607.00699",
    "1903.10075",
    "2005.11401",
    "2312.10997",
    "1706.03762",
    "2203.02155",
    "2104.08691",
    "2306.08302",
    "2404.16130",
    "2308.14522",
    "2005.14165",
    "1810.04805",
    "2307.09288",
    "2201.11903",
    "2112.10752",
    "2305.10601",
    "2310.06825",
    "2309.01219",
]


class UUIDEncoder(json.JSONEncoder):
    """JSON encoder that handles UUIDs and datetimes."""

    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return super().default(obj)


async def export_demo_data(output_dir: Path) -> dict:
    """Export demo paper data from live database."""
    import asyncpg

    output_dir.mkdir(parents=True, exist_ok=True)

    pool = await asyncpg.create_pool(
        "postgresql://postgres:postgres@localhost:5432/research_kb",
        min_size=1,
        max_size=3,
    )

    stats = {}

    try:
        async with pool.acquire() as conn:
            # 1. Find demo source IDs
            source_rows = await conn.fetch(
                """SELECT id, source_type, title, authors, year,
                          metadata, citation_authority, domain_id
                   FROM sources
                   WHERE metadata->>'arxiv_id' = ANY($1::text[])
                   ORDER BY year, title""",
                DEMO_ARXIV_IDS,
            )
            source_ids = [r["id"] for r in source_rows]
            print(f"Found {len(source_ids)} demo sources in database")

            if not source_ids:
                print("No demo sources found. Run setup_demo.py first.")
                await pool.close()
                return {}

            # 2. Export sources (strip file_path for privacy)
            sources = []
            for r in source_rows:
                sources.append(
                    {
                        "id": str(r["id"]),
                        "source_type": r["source_type"],
                        "title": r["title"],
                        "authors": r["authors"],
                        "year": r["year"],
                        "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                        "citation_authority": r["citation_authority"],
                        "domain_id": r["domain_id"],
                    }
                )
            _write_json(output_dir / "sources.json", sources)
            stats["sources"] = len(sources)
            print(f"  Exported {len(sources)} sources")

            # 3. Export chunks (text only, skip embeddings)
            chunk_rows = await conn.fetch(
                """SELECT id, source_id, content, content_hash, location,
                          page_start, page_end, metadata, domain_id
                   FROM chunks
                   WHERE source_id = ANY($1::uuid[])
                   ORDER BY source_id, page_start""",
                source_ids,
            )
            chunks = []
            for r in chunk_rows:
                chunks.append(
                    {
                        "id": str(r["id"]),
                        "source_id": str(r["source_id"]),
                        "content": r["content"],
                        "content_hash": r["content_hash"],
                        "location": r["location"],
                        "page_start": r["page_start"],
                        "page_end": r["page_end"],
                        "metadata": json.loads(r["metadata"]) if r["metadata"] else None,
                        "domain_id": r["domain_id"],
                    }
                )
            _write_json(output_dir / "chunks.json", chunks)
            stats["chunks"] = len(chunks)
            print(f"  Exported {len(chunks)} chunks")

            # 4. Export concepts linked to demo chunks
            chunk_ids = [r["id"] for r in chunk_rows]
            concept_rows = await conn.fetch(
                """SELECT DISTINCT c.id, c.name, c.canonical_name, c.aliases,
                          c.concept_type, c.category, c.definition,
                          c.extraction_method, c.confidence_score, c.validated,
                          c.metadata, c.domain_id
                   FROM concepts c
                   JOIN chunk_concepts cc ON cc.concept_id = c.id
                   WHERE cc.chunk_id = ANY($1::uuid[])
                   ORDER BY c.canonical_name""",
                chunk_ids,
            )
            concept_ids = {r["id"] for r in concept_rows}
            concepts = []
            for r in concept_rows:
                concepts.append(
                    {
                        "id": str(r["id"]),
                        "name": r["name"],
                        "canonical_name": r["canonical_name"],
                        "aliases": r["aliases"],
                        "concept_type": r["concept_type"],
                        "category": r["category"],
                        "definition": r["definition"],
                        "extraction_method": r["extraction_method"],
                        "confidence_score": r["confidence_score"],
                        "validated": r["validated"],
                        "metadata": json.loads(r["metadata"]) if r["metadata"] else None,
                        "domain_id": r["domain_id"],
                    }
                )
            _write_json(output_dir / "concepts.json", concepts)
            stats["concepts"] = len(concepts)
            print(f"  Exported {len(concepts)} concepts")

            # 5. Export chunk_concepts links
            cc_rows = await conn.fetch(
                """SELECT chunk_id, concept_id, mention_type, relevance_score
                   FROM chunk_concepts
                   WHERE chunk_id = ANY($1::uuid[])""",
                chunk_ids,
            )
            chunk_concepts = [
                {
                    "chunk_id": str(r["chunk_id"]),
                    "concept_id": str(r["concept_id"]),
                    "mention_type": r["mention_type"],
                    "relevance_score": r["relevance_score"],
                }
                for r in cc_rows
            ]
            _write_json(output_dir / "chunk_concepts.json", chunk_concepts)
            stats["chunk_concepts"] = len(chunk_concepts)
            print(f"  Exported {len(chunk_concepts)} chunk-concept links")

            # 6. Export relationships between demo concepts
            rel_rows = await conn.fetch(
                """SELECT id, source_concept_id, target_concept_id,
                          relationship_type, is_directed, strength,
                          evidence_chunk_ids, confidence_score
                   FROM concept_relationships
                   WHERE source_concept_id = ANY($1::uuid[])
                     AND target_concept_id = ANY($1::uuid[])""",
                list(concept_ids),
            )
            relationships = []
            for r in rel_rows:
                relationships.append(
                    {
                        "id": str(r["id"]),
                        "source_concept_id": str(r["source_concept_id"]),
                        "target_concept_id": str(r["target_concept_id"]),
                        "relationship_type": r["relationship_type"],
                        "is_directed": r["is_directed"],
                        "strength": r["strength"],
                        "evidence_chunk_ids": (
                            [str(x) for x in r["evidence_chunk_ids"]]
                            if r["evidence_chunk_ids"]
                            else None
                        ),
                        "confidence_score": r["confidence_score"],
                    }
                )
            _write_json(output_dir / "relationships.json", relationships)
            stats["relationships"] = len(relationships)
            print(f"  Exported {len(relationships)} relationships")

            # 7. Export citations from demo sources
            cit_rows = await conn.fetch(
                """SELECT id, source_id, authors, title, year, venue, doi,
                          arxiv_id, raw_string, extraction_method,
                          confidence_score, metadata, s2_paper_id, context
                   FROM citations
                   WHERE source_id = ANY($1::uuid[])""",
                source_ids,
            )
            citations = []
            for r in cit_rows:
                citations.append(
                    {
                        "id": str(r["id"]),
                        "source_id": str(r["source_id"]),
                        "authors": r["authors"],
                        "title": r["title"],
                        "year": r["year"],
                        "venue": r["venue"],
                        "doi": r["doi"],
                        "arxiv_id": r["arxiv_id"],
                        "raw_string": r["raw_string"],
                        "extraction_method": r["extraction_method"],
                        "confidence_score": r["confidence_score"],
                        "metadata": json.loads(r["metadata"]) if r["metadata"] else None,
                        "s2_paper_id": r["s2_paper_id"],
                        "context": r["context"],
                    }
                )
            _write_json(output_dir / "citations.json", citations)
            stats["citations"] = len(citations)
            print(f"  Exported {len(citations)} citations")

    finally:
        await pool.close()

    # Write manifest
    _write_json(output_dir / "manifest.json", stats)
    return stats


def _write_json(path: Path, data: list | dict) -> None:
    """Write JSON with compact formatting."""
    with open(path, "w") as f:
        json.dump(data, f, cls=UUIDEncoder, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Export demo data to JSON fixtures")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "fixtures" / "demo" / "data",
        help="Output directory for JSON fixtures",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Exporting demo corpus data")
    print("=" * 60)

    stats = asyncio.run(export_demo_data(args.output_dir))

    if stats:
        print(f"\nExported to: {args.output_dir}")
        print(f"  Sources:       {stats.get('sources', 0):>8}")
        print(f"  Chunks:        {stats.get('chunks', 0):>8}")
        print(f"  Concepts:      {stats.get('concepts', 0):>8}")
        print(f"  Chunk-Concept: {stats.get('chunk_concepts', 0):>8}")
        print(f"  Relationships: {stats.get('relationships', 0):>8}")
        print(f"  Citations:     {stats.get('citations', 0):>8}")
        print("\nNote: Embeddings are NOT exported (generate with embed server).")
        print("Load with: python scripts/load_demo_data.py")


if __name__ == "__main__":
    main()
