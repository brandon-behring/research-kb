#!/usr/bin/env python3
"""Ingest blog PDFs into research-kb.

Reads the manifest at fixtures/papers/blogs/switchback_blogs.json, then
delegates each entry to ``PDFDispatcher.ingest_pdf`` with
``source_type=SourceType.BLOG``. Embedding is skipped here — run
``scripts/embed_missing.py`` afterward once the embed_server is back up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from research_kb_contracts import SourceType
from research_kb_pdf.dispatcher import PDFDispatcher

MANIFEST = Path("fixtures/papers/blogs/switchback_blogs.json")
BLOGS_DIR = Path("fixtures/papers/blogs")


async def ingest_one(
    dispatcher: PDFDispatcher, blog: dict, skip_embedding: bool
) -> tuple[bool, str]:
    """Ingest a single blog entry; return (success, message)."""
    slug = blog["slug"]
    pdf_path = BLOGS_DIR / f"{slug}.pdf"
    if not pdf_path.exists():
        return False, f"PDF not found: {pdf_path}"

    metadata = {
        "url": blog["url"],
        "venue": blog.get("venue"),
        "blog_slug": slug,
    }

    try:
        result = await dispatcher.ingest_pdf(
            pdf_path=str(pdf_path),
            source_type=SourceType.BLOG,
            title=blog["title"],
            domain_id="causal_inference",
            authors=blog["authors"],
            year=blog["year"],
            metadata=metadata,
            skip_embedding=skip_embedding,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"exception: {exc}"

    if result.source is None:
        return False, "no source returned"
    return True, f"source_id={result.source.id} chunks={result.chunk_count}"


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--slug", help="Ingest only this slug")
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        default=True,
        help="Skip embedding (default True — run embed_missing.py after)",
    )
    parser.add_argument(
        "--with-embedding",
        action="store_false",
        dest="skip_embedding",
        help="Embed during ingestion (requires embed_server running)",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    blogs: list[dict] = manifest["blogs"]
    if args.slug:
        blogs = [b for b in blogs if b["slug"] == args.slug]
        if not blogs:
            print(f"no blog with slug {args.slug!r}", file=sys.stderr)
            return 2

    dispatcher = PDFDispatcher()

    succeeded: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []
    for blog in blogs:
        print(f"[{blog['slug']}] ingesting {blog['title']}")
        ok, msg = await ingest_one(dispatcher, blog, args.skip_embedding)
        if ok:
            succeeded.append((blog["slug"], msg))
            print(f"  ✓ {msg}")
        else:
            failed.append((blog["slug"], msg))
            print(f"  ✗ {msg}")

    print(f"\nIngested {len(succeeded)}/{len(blogs)}")
    if failed:
        print("Failures:")
        for slug, msg in failed:
            print(f"  - {slug}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
