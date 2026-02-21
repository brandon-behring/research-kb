#!/usr/bin/env python3
"""Load demo corpus from pre-exported JSON fixtures into PostgreSQL.

This is the fast-path setup: skip PDF download, ingestion, and concept
extraction by loading pre-extracted data directly.

Usage:
    python scripts/load_demo_data.py                    # Load all demo data
    python scripts/load_demo_data.py --data-dir fixtures/demo/data
    python scripts/load_demo_data.py --domain software_engineering  # Filter by domain
    python scripts/load_demo_data.py --skip-concepts     # Sources + chunks only
    python scripts/load_demo_data.py --embed             # Also generate embeddings

After loading, generate embeddings if not using --embed:
    python -m research_kb_pdf.embed_server &
    python scripts/embed_missing.py
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from uuid import UUID

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "storage" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "common" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "contracts" / "src"))

DEFAULT_DATA_DIR = PROJECT_ROOT / "fixtures" / "demo" / "data"


def _load_json(path: Path) -> list:
    """Load JSON fixture file."""
    if not path.exists():
        print(f"  [skip] {path.name} (not found)")
        return []
    with open(path) as f:
        return json.load(f)


async def load_demo_data(
    data_dir: Path,
    skip_concepts: bool = False,
    generate_embeddings: bool = False,
    domain_filter: str | None = None,
) -> dict:
    """Load demo data from JSON fixtures into PostgreSQL."""
    import asyncpg

    from research_kb_storage.connection import DatabaseConfig

    config = DatabaseConfig(min_pool_size=1, max_pool_size=3)
    pool = await asyncpg.create_pool(config.get_dsn(), min_size=1, max_size=3)

    stats = {"inserted": {}, "skipped": {}}

    if domain_filter:
        print(f"  Filtering to domain: {domain_filter}")

    try:
        async with pool.acquire() as conn:
            # 1. Load sources
            sources = _load_json(data_dir / "sources.json")
            if domain_filter:
                sources = [s for s in sources if s.get("domain_id", "default") == domain_filter]
            s_inserted, s_skipped = 0, 0
            for s in sources:
                existing = await conn.fetchval(
                    "SELECT id FROM sources WHERE id = $1",
                    UUID(s["id"]),
                )
                if existing:
                    s_skipped += 1
                    continue

                file_hash = (
                    s.get("file_hash") or hashlib.sha256(s["title"].encode()).hexdigest()[:16]
                )
                await conn.execute(
                    """INSERT INTO sources
                       (id, source_type, title, authors, year, file_hash,
                        metadata, citation_authority, domain_id)
                       VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9)
                       ON CONFLICT (id) DO NOTHING""",
                    UUID(s["id"]),
                    s["source_type"],
                    s["title"],
                    s.get("authors"),
                    s.get("year"),
                    file_hash,
                    json.dumps(s.get("metadata", {})),
                    s.get("citation_authority"),
                    s.get("domain_id", "default"),
                )
                s_inserted += 1
            stats["inserted"]["sources"] = s_inserted
            stats["skipped"]["sources"] = s_skipped
            print(f"  Sources: {s_inserted} inserted, {s_skipped} skipped")

            # 2. Load chunks
            chunks = _load_json(data_dir / "chunks.json")
            if domain_filter:
                chunks = [ch for ch in chunks if ch.get("domain_id", "default") == domain_filter]
            c_inserted, c_skipped = 0, 0
            # Batch insert for performance
            batch_size = 500
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                for ch in batch:
                    existing = await conn.fetchval(
                        "SELECT id FROM chunks WHERE id = $1",
                        UUID(ch["id"]),
                    )
                    if existing:
                        c_skipped += 1
                        continue

                    await conn.execute(
                        """INSERT INTO chunks
                           (id, source_id, content, content_hash, location,
                            page_start, page_end, metadata, domain_id)
                           VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)
                           ON CONFLICT (id) DO NOTHING""",
                        UUID(ch["id"]),
                        UUID(ch["source_id"]),
                        ch["content"],
                        ch.get("content_hash")
                        or hashlib.sha256(ch["content"].encode()).hexdigest()[:16],
                        ch.get("location"),
                        ch.get("page_start"),
                        ch.get("page_end"),
                        json.dumps(ch["metadata"]) if ch.get("metadata") else None,
                        ch.get("domain_id", "default"),
                    )
                    c_inserted += 1
                if (i + batch_size) % 2000 == 0 and i > 0:
                    print(f"    ... {i + batch_size}/{len(chunks)} chunks processed")
            stats["inserted"]["chunks"] = c_inserted
            stats["skipped"]["chunks"] = c_skipped
            print(f"  Chunks: {c_inserted} inserted, {c_skipped} skipped")

            if skip_concepts:
                print("  (Skipping concepts, relationships, chunk_concepts)")
            else:
                # 3. Load concepts
                concepts = _load_json(data_dir / "concepts.json")
                co_inserted, co_skipped = 0, 0
                for co in concepts:
                    existing = await conn.fetchval(
                        "SELECT id FROM concepts WHERE id = $1",
                        UUID(co["id"]),
                    )
                    if existing:
                        co_skipped += 1
                        continue

                    await conn.execute(
                        """INSERT INTO concepts
                           (id, name, canonical_name, aliases, concept_type,
                            category, definition, extraction_method,
                            confidence_score, validated, metadata, domain_id)
                           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12)
                           ON CONFLICT (id) DO NOTHING""",
                        UUID(co["id"]),
                        co["name"],
                        co["canonical_name"],
                        co.get("aliases"),
                        co["concept_type"],
                        co.get("category"),
                        co.get("definition"),
                        co.get("extraction_method"),
                        co.get("confidence_score"),
                        co.get("validated"),
                        json.dumps(co["metadata"]) if co.get("metadata") else None,
                        co.get("domain_id", "default"),
                    )
                    co_inserted += 1
                stats["inserted"]["concepts"] = co_inserted
                stats["skipped"]["concepts"] = co_skipped
                print(f"  Concepts: {co_inserted} inserted, {co_skipped} skipped")

                # 4. Load chunk_concepts
                chunk_concepts = _load_json(data_dir / "chunk_concepts.json")
                cc_inserted, cc_skipped = 0, 0
                for cc in chunk_concepts:
                    try:
                        await conn.execute(
                            """INSERT INTO chunk_concepts
                               (chunk_id, concept_id, mention_type, relevance_score)
                               VALUES ($1, $2, $3, $4)
                               ON CONFLICT DO NOTHING""",
                            UUID(cc["chunk_id"]),
                            UUID(cc["concept_id"]),
                            cc["mention_type"],
                            cc.get("relevance_score"),
                        )
                        cc_inserted += 1
                    except Exception:
                        cc_skipped += 1
                stats["inserted"]["chunk_concepts"] = cc_inserted
                stats["skipped"]["chunk_concepts"] = cc_skipped
                print(f"  Chunk-concepts: {cc_inserted} inserted, {cc_skipped} skipped")

                # 5. Load relationships
                relationships = _load_json(data_dir / "relationships.json")
                r_inserted, r_skipped = 0, 0
                for rel in relationships:
                    try:
                        await conn.execute(
                            """INSERT INTO concept_relationships
                               (id, source_concept_id, target_concept_id,
                                relationship_type, is_directed, strength,
                                evidence_chunk_ids, confidence_score)
                               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                               ON CONFLICT (id) DO NOTHING""",
                            UUID(rel["id"]),
                            UUID(rel["source_concept_id"]),
                            UUID(rel["target_concept_id"]),
                            rel["relationship_type"],
                            rel.get("is_directed", True),
                            rel.get("strength"),
                            (
                                [UUID(x) for x in rel["evidence_chunk_ids"]]
                                if rel.get("evidence_chunk_ids")
                                else None
                            ),
                            rel.get("confidence_score"),
                        )
                        r_inserted += 1
                    except Exception:
                        r_skipped += 1
                stats["inserted"]["relationships"] = r_inserted
                stats["skipped"]["relationships"] = r_skipped
                print(f"  Relationships: {r_inserted} inserted, {r_skipped} skipped")

            # 6. Load citations
            citations = _load_json(data_dir / "citations.json")
            ci_inserted, ci_skipped = 0, 0
            for ci in citations:
                existing = await conn.fetchval(
                    "SELECT id FROM citations WHERE id = $1",
                    UUID(ci["id"]),
                )
                if existing:
                    ci_skipped += 1
                    continue

                try:
                    await conn.execute(
                        """INSERT INTO citations
                           (id, source_id, authors, title, year, venue, doi,
                            arxiv_id, raw_string, extraction_method,
                            confidence_score, metadata, s2_paper_id, context)
                           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                                   $12::jsonb, $13, $14)
                           ON CONFLICT (id) DO NOTHING""",
                        UUID(ci["id"]),
                        UUID(ci["source_id"]),
                        ci.get("authors"),
                        ci.get("title"),
                        ci.get("year"),
                        ci.get("venue"),
                        ci.get("doi"),
                        ci.get("arxiv_id"),
                        ci["raw_string"],
                        ci.get("extraction_method"),
                        ci.get("confidence_score"),
                        json.dumps(ci["metadata"]) if ci.get("metadata") else None,
                        ci.get("s2_paper_id"),
                        ci.get("context"),
                    )
                    ci_inserted += 1
                except Exception:
                    ci_skipped += 1
            stats["inserted"]["citations"] = ci_inserted
            stats["skipped"]["citations"] = ci_skipped
            print(f"  Citations: {ci_inserted} inserted, {ci_skipped} skipped")

    finally:
        await pool.close()

    # Optional: generate embeddings
    if generate_embeddings:
        print("\n--- Generating embeddings ---")
        print("  Connecting to embed server...")
        try:
            from research_kb_pdf.embed_client import EmbeddingClient

            client = EmbeddingClient()
            client.ping()
            print("  Embed server available. Run: python scripts/embed_missing.py")
        except Exception as e:
            print(f"  Embed server not available: {e}")
            print("  Start with: python -m research_kb_pdf.embed_server &")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Load demo data from JSON fixtures")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing JSON fixture files",
    )
    parser.add_argument(
        "--skip-concepts",
        action="store_true",
        help="Skip loading concepts/relationships (sources + chunks only)",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Also generate embeddings (requires embed server)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter fixtures to a specific domain_id (e.g., 'software_engineering')",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Data directory not found: {args.data_dir}")
        print("Run 'python scripts/export_demo_data.py' first to generate fixtures,")
        print("or use 'python scripts/setup_demo.py' for full pipeline setup.")
        sys.exit(1)

    manifest = args.data_dir / "manifest.json"
    if manifest.exists():
        with open(manifest) as f:
            m = json.load(f)
        print("Demo data manifest:")
        for k, v in m.items():
            print(f"  {k}: {v}")
        print()

    print("=" * 60)
    print("Loading demo corpus from fixtures")
    print("=" * 60)

    stats = asyncio.run(
        load_demo_data(
            args.data_dir,
            skip_concepts=args.skip_concepts,
            generate_embeddings=args.embed,
            domain_filter=args.domain,
        )
    )

    total_inserted = sum(stats.get("inserted", {}).values())
    print(f"\nTotal records inserted: {total_inserted}")

    if total_inserted > 0:
        print("\nNext steps:")
        print("  1. Generate embeddings (required for search):")
        print("     python -m research_kb_pdf.embed_server &")
        print("     python scripts/embed_missing.py")
        print("  2. Sync to KuzuDB (required for graph queries):")
        print("     python scripts/sync_kuzu.py")
        print("  3. Try a search:")
        print('     research-kb query "instrumental variables"')


if __name__ == "__main__":
    main()
