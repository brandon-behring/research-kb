#!/usr/bin/env python3
"""Render blog URLs to PDF via headless Chrome.

Reads a manifest at fixtures/papers/blogs/switchback_blogs.json, renders
each URL to ``{slug}.pdf`` + writes a ``{slug}.json`` sidecar for
downstream ingestion.

Strategy
--------
1. Try direct URL via ``google-chrome-stable --headless --print-to-pdf``.
2. On non-zero output size (0-byte PDF = blocked / 403) or any failure,
   retry via the Wayback Machine. Flag ``wayback: true`` in the manifest
   prefers Wayback as the primary for flaky domains (Lyft, DoorDash).

Usage
-----
    .venv/bin/python scripts/fetch_blog_as_pdf.py
    .venv/bin/python scripts/fetch_blog_as_pdf.py --slug statsig_switchback_overview
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

CHROME = "/usr/bin/google-chrome-stable"
MANIFEST = Path("fixtures/papers/blogs/switchback_blogs.json")
BLOGS_DIR = Path("fixtures/papers/blogs")
RENDER_TIMEOUT_SECONDS = 60
MIN_PDF_BYTES = 20_000  # <20 kB implies blocked / empty


def wayback_url(url: str) -> str:
    """Return a Wayback Machine latest-snapshot URL for ``url``."""
    return f"https://web.archive.org/web/2024/{url}"


def render_pdf(target_url: str, pdf_path: Path) -> bool:
    """Render ``target_url`` to ``pdf_path`` via headless Chrome.

    Returns True if output file is larger than ``MIN_PDF_BYTES``.
    """
    cmd = [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--hide-scrollbars",
        "--virtual-time-budget=15000",
        f"--print-to-pdf={pdf_path}",
        target_url,
    ]
    try:
        subprocess.run(
            cmd,
            timeout=RENDER_TIMEOUT_SECONDS,
            capture_output=True,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False

    if not pdf_path.exists():
        return False
    return pdf_path.stat().st_size >= MIN_PDF_BYTES


def fetch_one(blog: dict, blogs_dir: Path) -> tuple[bool, str]:
    """Fetch a single blog entry; return (success, notes)."""
    slug = blog["slug"]
    url = blog["url"]
    prefer_wayback = blog.get("wayback", False)
    pdf_path = blogs_dir / f"{slug}.pdf"

    attempts: list[str] = []
    if prefer_wayback:
        attempts.extend([wayback_url(url), url])
    else:
        attempts.extend([url, wayback_url(url)])

    for attempt_url in attempts:
        if render_pdf(attempt_url, pdf_path):
            return True, f"rendered via {attempt_url}"
        if pdf_path.exists():
            pdf_path.unlink()

    return False, f"all attempts failed for {slug}"


def write_sidecar(blog: dict, blogs_dir: Path) -> None:
    """Write companion metadata sidecar for ingestion."""
    slug = blog["slug"]
    sidecar_path = blogs_dir / f"{slug}.json"
    sidecar = {
        "title": blog["title"],
        "authors": blog["authors"],
        "year": blog["year"],
        "url": blog["url"],
        "venue": blog.get("venue"),
        "source_type": "blog",
        "domain_id": "causal_inference",
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", help="Fetch only the matching slug")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--pace", type=float, default=1.0, help="Seconds between requests")
    args = parser.parse_args()

    if not Path(CHROME).exists():
        print(f"ERROR: chrome binary not found at {CHROME}", file=sys.stderr)
        return 2

    manifest = json.loads(args.manifest.read_text())
    blogs: list[dict] = manifest["blogs"]
    if args.slug:
        blogs = [b for b in blogs if b["slug"] == args.slug]
        if not blogs:
            print(f"no blog with slug {args.slug!r}", file=sys.stderr)
            return 2

    BLOGS_DIR.mkdir(parents=True, exist_ok=True)

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []
    for blog in blogs:
        print(f"[{blog['slug']}] fetching {blog['url']}")
        ok, notes = fetch_one(blog, BLOGS_DIR)
        if ok:
            write_sidecar(blog, BLOGS_DIR)
            succeeded.append(blog["slug"])
            print(f"  ✓ {notes}")
        else:
            failed.append((blog["slug"], notes))
            print(f"  ✗ {notes}")
        time.sleep(args.pace)

    print(f"\nFetched {len(succeeded)}/{len(blogs)}")
    if failed:
        print("Failures:")
        for slug, notes in failed:
            print(f"  - {slug}: {notes}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
