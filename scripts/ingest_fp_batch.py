#!/usr/bin/env python3
"""Ingest 5 FP + SE-with-FP books directly via ingest_textbook().

Bypasses the catalog (mass_ingest_catalog.py) because:
- Granin FDA and Sfregola Scala are in catalog_books_r2.json as domain="skip"
- Effective Python and Refactoring are not in the catalog at all
- Fluent Python's catalog path points to an unmounted external drive

Uses Docling extraction (--no-embed default). MinerU re-extraction happens
in the next phase via reextract_with_mineru.py once these sources exist.

Usage:
    # Stop embed_server + rerank_server + ollama first (GPU contention)
    python scripts/ingest_fp_batch.py --no-embed
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ingest_missing_textbooks import ingest_textbook

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import configure_logging, get_logger
from research_kb_storage import DatabaseConfig, get_connection_pool

logger = get_logger(__name__)

LIB = Path("/home/brandon_behring/Claude/research-kb/fixtures/library_books")
ACQ = Path("/home/brandon_behring/Downloads/acqusition_books/acqusition_books")

BOOKS: list[dict] = [
    {
        "path": LIB / "Functional_Design_and_Architecture.pdf",
        "title": "Functional Design and Architecture",
        "authors": ["Alexander Granin"],
        "year": 2024,
        "domain_id": "functional_programming",
    },
    {
        "path": LIB / "Get_Programming_with_Scala.pdf",
        "title": "Get Programming with Scala",
        "authors": ["Daniela Sfregola"],
        "year": 2021,
        "domain_id": "functional_programming",
    },
    {
        "path": ACQ
        / "Fluent Python_ Clear, Concise, and Effective Programming, -- Luciano Ramalho -- 2nd, 2022 -- Beijing _ O'Reilly Media, Inc -- 9781492056355 -- 6b8f1e751c6d6b82a49cc155099f9949 -- Anna\u2019s Archive.pdf",
        "title": "Fluent Python: Clear, Concise, and Effective Programming (2nd ed)",
        "authors": ["Luciano Ramalho"],
        "year": 2022,
        "domain_id": "software_engineering",
    },
    {
        "path": ACQ
        / "Effective Python_ 125 Specific Ways to Write Better Python_ -- Brett Slatkin -- 3, 2024 nov 30 -- Pearson Education, Limited -- 9780138172183 -- 859e2b0eab74efb9626b6b79a353f72a -- Anna\u2019s Archive.pdf",
        "title": "Effective Python: 125 Specific Ways to Write Better Python (3rd ed)",
        "authors": ["Brett Slatkin"],
        "year": 2024,
        "domain_id": "software_engineering",
    },
    {
        "path": ACQ
        / "Refactoring_ Improving the Design of Existing Code -- Martin Fowler, Kent Beck -- Addison-Wesley Signature Series (Fowler) Ser, 2nd ed, -- 9780134757599 -- f4fc75eb1eee5537400908e6f59db631 -- Anna\u2019s Archive.pdf",
        "title": "Refactoring: Improving the Design of Existing Code (2nd ed)",
        "authors": ["Martin Fowler", "Kent Beck"],
        "year": 2018,
        "domain_id": "software_engineering",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct ingest of FP + SE-with-FP books")
    parser.add_argument("--no-embed", action="store_true", default=True)
    parser.add_argument("--with-embed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--source-ids-out",
        type=str,
        default=None,
        help="Write newly-created source_ids to this file, one per line, with format: domain_id source_id title",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    configure_logging(level="INFO")

    if args.with_embed:
        args.no_embed = False

    # Pre-flight: verify all files exist
    missing = [b for b in BOOKS if not b["path"].exists()]
    if missing:
        for b in missing:
            print(f"MISSING: {b['path']}")
        sys.exit(1)

    if args.dry_run:
        print(f"Would ingest {len(BOOKS)} books:")
        for b in BOOKS:
            size_mb = b["path"].stat().st_size / 1024 / 1024
            print(f"  [{b['domain_id']}] {b['title']} ({size_mb:.1f} MB)")
        return

    await get_connection_pool(DatabaseConfig())

    results = []
    for i, book in enumerate(BOOKS, 1):
        title = book["title"]
        print(f"\n[{i}/{len(BOOKS)}] {title}")
        try:
            source_id, chunks, headings = await ingest_textbook(
                pdf_path=str(book["path"]),
                title=title,
                authors=book["authors"],
                year=book["year"],
                domain_id=book["domain_id"],
                no_embed=args.no_embed,
            )
            print(f"  OK: source_id={source_id} chunks={chunks} headings={headings}")
            results.append(
                {
                    "title": title,
                    "domain_id": book["domain_id"],
                    "source_id": source_id,
                    "chunks": chunks,
                    "status": "ok",
                }
            )
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
            results.append(
                {
                    "title": title,
                    "domain_id": book["domain_id"],
                    "status": "failed",
                    "error": str(e)[:200],
                }
            )

    ok = sum(1 for r in results if r["status"] == "ok")
    failed = sum(1 for r in results if r["status"] == "failed")
    total_chunks = sum(r.get("chunks", 0) for r in results if r["status"] == "ok")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {ok}/{len(BOOKS)} ok, {failed} failed, {total_chunks} chunks created")

    if args.source_ids_out:
        out_path = Path(args.source_ids_out)
        lines = [
            f"{r['domain_id']}\t{r['source_id']}\t{r['title']}"
            for r in results
            if r["status"] == "ok"
        ]
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"Wrote {len(lines)} source_ids to {out_path}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
