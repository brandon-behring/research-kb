"""CLI entry point: python -m scripts.manning {catalog|audit|cleanup|ingest}.

Each subcommand maps to a module in the scripts/manning/ package.
All DB-mutating operations are dry-run by default.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.manning",
        description="Manning Library ingestion system. Catalog-centric architecture.",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- catalog ---
    cat = sub.add_parser("catalog", help="Generate/update manning_catalog.yaml")
    cat.add_argument(
        "--generate",
        action="store_true",
        help="Generate catalog from Documents/ directory",
    )
    cat.add_argument(
        "--books-dir",
        type=Path,
        default=None,
        help="Override Documents/ directory path",
    )
    cat.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override catalog output path",
    )
    cat.add_argument(
        "--no-merge",
        action="store_true",
        help="Don't merge with existing catalog (overwrite)",
    )

    # --- audit ---
    sub.add_parser("audit", help="Cross-reference catalog/disk/DB")

    # --- cleanup ---
    clean = sub.add_parser("cleanup", help="DB hygiene operations (dry-run by default)")
    clean.add_argument("--fix-domains", action="store_true", help="Normalize domain tags")
    clean.add_argument(
        "--assign-domains",
        action="store_true",
        help="Assign domains to 'none' sources from catalog",
    )
    clean.add_argument(
        "--find-duplicates",
        action="store_true",
        help="Detect and report duplicate sources",
    )
    clean.add_argument(
        "--merge-duplicates",
        nargs=2,
        metavar=("KEEP_ID", "DELETE_ID"),
        help="Merge two duplicate sources (keep first, delete second)",
    )
    clean.add_argument(
        "--auto-merge",
        action="store_true",
        help="Auto-merge all duplicate groups (keep highest chunk count)",
    )
    clean.add_argument(
        "--report-untagged",
        action="store_true",
        help="Report non-Manning 'none' domain sources",
    )
    clean.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply changes (default is dry-run)",
    )

    # --- ingest ---
    ing = sub.add_parser("ingest", help="Ingest Manning books from Documents/")
    ing.add_argument("--tier", type=int, choices=[1, 2, 3], help="Filter by tier")
    ing.add_argument("--domain", type=str, help="Filter by domain_id")
    ing.add_argument("--title", type=str, help="Filter by title (exact or substring)")
    ing.add_argument("--dry-run", action="store_true", help="Preview without ingesting")
    ing.add_argument(
        "--upgrade-meaps",
        action="store_true",
        help="Replace old MEAP versions with newer ones",
    )
    ing.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Minimal output (one line per book + summary)",
    )
    ing.add_argument(
        "--json",
        action="store_true",
        help="JSON output for programmatic parsing",
    )

    return parser


async def cmd_catalog(args: argparse.Namespace) -> None:
    """Handle 'catalog' subcommand."""
    from .catalog import DEFAULT_BOOKS_DIR, CATALOG_PATH, generate_catalog

    if not args.generate:
        print("Use --generate to create/update the catalog.")
        print(f"  Catalog path: {CATALOG_PATH}")
        return

    books_dir = args.books_dir or DEFAULT_BOOKS_DIR
    output = args.output or CATALOG_PATH

    print(f"Scanning: {books_dir}")
    books = generate_catalog(
        books_dir=books_dir,
        output_path=output,
        merge_existing=not args.no_merge,
    )

    # Summary stats
    classified = sum(1 for b in books if b.domain_id != "unclassified")
    meap_count = sum(1 for b in books if b.meap_version is not None)
    no_pdf = sum(1 for b in books if b.best_pdf is None)

    print(f"\nCatalog written: {output}")
    print(f"  Total books: {len(books)}")
    print(f"  Classified: {classified}")
    print(f"  Unclassified: {len(books) - classified}")
    print(f"  MEAP versions: {meap_count}")
    if no_pdf:
        print(f"  No PDF found: {no_pdf}")

    if classified < len(books):
        print(f"\nNext: Edit {output} to assign domain_id and tier for unclassified books.")


async def cmd_audit(args: argparse.Namespace) -> None:
    """Handle 'audit' subcommand."""
    from .audit import format_audit_report, run_audit

    print("Running audit: catalog ↔ disk ↔ DB ...")
    rows, summary = await run_audit()
    print(format_audit_report(rows, summary))


async def cmd_cleanup(args: argparse.Namespace) -> None:
    """Handle 'cleanup' subcommand."""
    from .cleanup import (
        auto_merge_duplicates,
        find_duplicates,
        fix_domains,
        assign_domains,
        merge_duplicates,
        report_untagged,
        write_duplicate_report,
        write_untagged_report,
    )

    # Need at least one operation
    ops = [
        args.fix_domains,
        args.assign_domains,
        args.find_duplicates,
        args.merge_duplicates,
        args.auto_merge,
        args.report_untagged,
    ]
    if not any(ops):
        print("Specify at least one operation:")
        print("  --fix-domains        Normalize domain tags")
        print("  --assign-domains     Assign domains from catalog")
        print("  --find-duplicates    Detect duplicates")
        print("  --merge-duplicates   Merge two sources")
        print("  --auto-merge         Auto-merge all duplicate groups")
        print("  --report-untagged    Report unclassified non-Manning sources")
        return

    # Add packages to path
    _project_root = Path(__file__).parent.parent.parent
    for pkg in ("storage", "contracts", "common"):
        sys.path.insert(0, str(_project_root / "packages" / pkg / "src"))

    from research_kb_storage import DatabaseConfig, get_connection_pool

    config = DatabaseConfig()
    pool = await get_connection_pool(config)

    mode = "APPLY" if args.apply else "DRY RUN"

    if args.fix_domains:
        print(f"\n--- Fix Domains ({mode}) ---")
        changes = await fix_domains(pool, apply=args.apply)
        if changes:
            for c in changes:
                print(f"  {c['title'][:50]}: '{c['old_domain']}' → '{c['new_domain']}'")
            print(f"\n  {len(changes)} domain(s) {'fixed' if args.apply else 'would be fixed'}.")
        else:
            print("  No domain normalization needed.")

    if args.assign_domains:
        print(f"\n--- Assign Domains ({mode}) ---")
        proposals = await assign_domains(pool, apply=args.apply)
        if proposals:
            for p in proposals:
                print(
                    f"  {p['db_title'][:40]} → domain='{p['proposed_domain']}' "
                    f"(matched: {p['catalog_title']})"
                )
            print(
                f"\n  {len(proposals)} domain(s) "
                f"{'assigned' if args.apply else 'would be assigned'}."
            )
        else:
            print("  No untagged Manning sources found.")

    if args.find_duplicates:
        print("\n--- Find Duplicates ---")
        groups = await find_duplicates(pool)
        if groups:
            for i, g in enumerate(groups):
                print(f"\n  Group {i + 1} ({g['match_type']}):")
                for s in g["sources"]:
                    print(
                        f"    {s['id'][:8]}... | {s['title'][:40]} | "
                        f"domain={s['domain']} | {s['chunk_count']} chunks"
                    )

            report_path = write_duplicate_report(groups)
            print(f"\n  {len(groups)} duplicate group(s) found.")
            print(f"  Report: {report_path}")
            print(
                f"  To merge: python -m scripts.manning cleanup "
                f"--merge-duplicates <KEEP_ID> <DELETE_ID> --apply"
            )
        else:
            print("  No duplicates found.")

    if args.auto_merge:
        print(f"\n--- Auto-Merge Duplicates ({mode}) ---")
        results = await auto_merge_duplicates(pool, apply=args.apply)
        if results:
            errors = [r for r in results if "error" in r]
            skipped = [r for r in results if r.get("skipped")]
            merges = [
                r for r in results if "error" not in r and not r.get("skipped")
            ]
            for r in merges:
                status = "MERGED" if r.get("applied") else "would merge"
                print(
                    f"  [{r['match_type']}] Keep: {r['keep']['title'][:40]} "
                    f"← Delete: {r['delete']['title'][:40]} "
                    f"({r['chunks_to_reassign']} chunks) [{status}]"
                )
            for r in errors:
                print(f"  ERROR: {r['error']}")
            if skipped:
                print(f"\n  Skipped {len(skipped)} group(s) with mixed titles:")
                for s in skipped:
                    print(f"    Titles: {s['titles']}")
            print(
                f"\n  {len(merges)} merge(s) {'applied' if args.apply else 'would apply'}, "
                f"{len(errors)} error(s), {len(skipped)} group(s) skipped."
            )
        else:
            print("  No duplicates found.")

    if args.merge_duplicates:
        keep_id, delete_id = args.merge_duplicates
        print(f"\n--- Merge Duplicates ({mode}) ---")
        result = await merge_duplicates(pool, keep_id, delete_id, apply=args.apply)

        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Keep: {result['keep']['title']} ({result['keep']['id'][:8]}...)")
            print(f"  Delete: {result['delete']['title']} ({result['delete']['id'][:8]}...)")
            print(f"  Chunks to reassign: {result['chunks_to_reassign']}")
            print(f"  Citations affected: {result['citations_affected']}")
            print(f"  Status: {result['status']}")

    if args.report_untagged:
        print("\n--- Untagged Non-Manning Sources ---")
        untagged = await report_untagged(pool)
        if untagged:
            for u in untagged:
                print(f"  {u['source_id'][:8]}... | {u['title'][:50]} | {u['chunk_count']} chunks")
            report_path = write_untagged_report(untagged)
            print(f"\n  {len(untagged)} untagged non-Manning source(s).")
            print(f"  Report: {report_path}")
        else:
            print("  All non-Manning sources have domain tags.")


async def cmd_ingest(args: argparse.Namespace) -> None:
    """Handle 'ingest' subcommand."""
    from .ingest import run_ingest

    await run_ingest(
        tier=args.tier,
        domain=args.domain,
        title=args.title,
        dry_run=args.dry_run,
        upgrade_meaps=args.upgrade_meaps,
        quiet=args.quiet,
        json_output=args.json,
    )


async def amain() -> None:
    """Async entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    handlers = {
        "catalog": cmd_catalog,
        "audit": cmd_audit,
        "cleanup": cmd_cleanup,
        "ingest": cmd_ingest,
    }

    handler = handlers.get(args.command)
    if handler:
        await handler(args)
    else:
        parser.print_help()


def main() -> None:
    """Sync entry point."""
    asyncio.run(amain())


if __name__ == "__main__":
    main()
