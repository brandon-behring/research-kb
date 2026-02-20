#!/usr/bin/env python3
"""
Retry Dead Letter Queue (DLQ) entries.

Provides utilities to inspect and retry failed PDF ingestions from the DLQ.

Usage:
    # List all failed entries
    python scripts/retry_dlq.py list

    # List entries by error type
    python scripts/retry_dlq.py list --error-type GROBIDError

    # Show details for specific entry
    python scripts/retry_dlq.py show <entry_id>

    # Retry specific entry
    python scripts/retry_dlq.py retry <entry_id>

    # Retry all entries
    python scripts/retry_dlq.py retry-all

    # Clear all entries (after inspection)
    python scripts/retry_dlq.py clear

Created: 2025-12-10
Part of: U1 Track B (PDF Improvements)
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_pdf.dlq import DeadLetterQueue, DLQEntry
from research_kb_common import get_logger

logger = get_logger(__name__)

DEFAULT_DLQ_PATH = Path(__file__).parent.parent / "data" / "dlq" / "failed_pdfs.jsonl"


def cmd_list(dlq: DeadLetterQueue, error_type: Optional[str] = None) -> int:
    """List DLQ entries."""
    entries = dlq.list(error_type=error_type)

    if not entries:
        print("No DLQ entries found.")
        return 0

    print(f"\n{'=' * 60}")
    print(f"DEAD LETTER QUEUE ({len(entries)} entries)")
    print(f"{'=' * 60}\n")

    # Group by error type
    by_error = {}
    for entry in entries:
        by_error.setdefault(entry.error_type, []).append(entry)

    for error_type, type_entries in sorted(by_error.items()):
        print(f"[{error_type}] ({len(type_entries)} entries)")
        for entry in type_entries[:5]:  # Show first 5 per type
            file_name = Path(entry.file_path).name
            print(f"  • {entry.id[:8]}... | {file_name[:50]}")
            print(f"    {entry.error_message[:60]}...")
        if len(type_entries) > 5:
            print(f"  ... and {len(type_entries) - 5} more")
        print()

    return 0


def cmd_show(dlq: DeadLetterQueue, entry_id: str) -> int:
    """Show details for a specific entry."""
    # Support partial ID matching
    entries = dlq.list()
    matches = [e for e in entries if e.id.startswith(entry_id)]

    if not matches:
        print(f"No entry found matching: {entry_id}")
        return 1

    if len(matches) > 1:
        print(f"Multiple entries match '{entry_id}':")
        for e in matches:
            print(f"  • {e.id}")
        return 1

    entry = matches[0]

    print(f"\n{'=' * 60}")
    print(f"DLQ ENTRY: {entry.id}")
    print(f"{'=' * 60}")
    print(f"File: {entry.file_path}")
    print(f"Error Type: {entry.error_type}")
    print(f"Error Message: {entry.error_message}")
    print(f"Timestamp: {entry.timestamp}")
    print(f"Retry Count: {entry.retry_count}")
    print(f"Metadata: {entry.metadata}")
    print(f"\nTraceback:\n{entry.traceback}")

    return 0


async def retry_entry(dlq: DeadLetterQueue, entry: DLQEntry) -> bool:
    """Retry a single DLQ entry.

    Returns True if successful, False otherwise.
    """
    from research_kb_pdf.dispatcher import PDFDispatcher

    print(f"Retrying: {Path(entry.file_path).name}")

    # Check if file exists
    pdf_path = Path(entry.file_path)
    if not pdf_path.exists():
        print(f"  ❌ File not found: {entry.file_path}")
        return False

    try:
        # Create dispatcher and retry
        dispatcher = PDFDispatcher()
        result = await dispatcher.ingest_pdf(pdf_path)

        if result.status == "success":
            print("  ✅ Success!")
            dlq.remove(entry.id)
            return True
        else:
            print(f"  ❌ Failed: {result.error}")
            return False

    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return False


async def cmd_retry(dlq: DeadLetterQueue, entry_id: str) -> int:
    """Retry a specific entry."""
    # Support partial ID matching
    entries = dlq.list()
    matches = [e for e in entries if e.id.startswith(entry_id)]

    if not matches:
        print(f"No entry found matching: {entry_id}")
        return 1

    if len(matches) > 1:
        print(f"Multiple entries match '{entry_id}':")
        for e in matches:
            print(f"  • {e.id}")
        return 1

    entry = matches[0]
    success = await retry_entry(dlq, entry)

    return 0 if success else 1


async def cmd_retry_all(dlq: DeadLetterQueue, error_type: Optional[str] = None) -> int:
    """Retry all DLQ entries."""
    entries = dlq.list(error_type=error_type)

    if not entries:
        print("No DLQ entries to retry.")
        return 0

    print(f"Retrying {len(entries)} entries...\n")

    success_count = 0
    fail_count = 0

    for entry in entries:
        success = await retry_entry(dlq, entry)
        if success:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {success_count} success, {fail_count} failed")

    return 0 if fail_count == 0 else 1


def cmd_clear(dlq: DeadLetterQueue) -> int:
    """Clear all DLQ entries."""
    count = dlq.count()

    if count == 0:
        print("DLQ is already empty.")
        return 0

    print(f"This will delete {count} DLQ entries.")
    confirm = input("Are you sure? (yes/no): ")

    if confirm.lower() != "yes":
        print("Aborted.")
        return 1

    cleared = dlq.clear_all()
    print(f"Cleared {cleared} entries.")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Retry Dead Letter Queue entries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all failed entries
  python scripts/retry_dlq.py list

  # List only GROBID errors
  python scripts/retry_dlq.py list --error-type GROBIDError

  # Show entry details
  python scripts/retry_dlq.py show abc123

  # Retry specific entry
  python scripts/retry_dlq.py retry abc123

  # Retry all entries
  python scripts/retry_dlq.py retry-all
        """,
    )
    parser.add_argument(
        "--dlq-path",
        type=Path,
        default=DEFAULT_DLQ_PATH,
        help="Path to DLQ JSONL file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    list_parser = subparsers.add_parser("list", help="List DLQ entries")
    list_parser.add_argument("--error-type", help="Filter by error type")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show entry details")
    show_parser.add_argument("entry_id", help="Entry ID (or prefix)")

    # Retry command
    retry_parser = subparsers.add_parser("retry", help="Retry specific entry")
    retry_parser.add_argument("entry_id", help="Entry ID (or prefix)")

    # Retry-all command
    retry_all_parser = subparsers.add_parser("retry-all", help="Retry all entries")
    retry_all_parser.add_argument("--error-type", help="Filter by error type")

    # Clear command
    subparsers.add_parser("clear", help="Clear all entries")

    # Stats command
    subparsers.add_parser("stats", help="Show DLQ statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize DLQ
    dlq = DeadLetterQueue(args.dlq_path)

    # Dispatch to command handler
    if args.command == "list":
        return cmd_list(dlq, args.error_type)
    elif args.command == "show":
        return cmd_show(dlq, args.entry_id)
    elif args.command == "retry":
        return asyncio.run(cmd_retry(dlq, args.entry_id))
    elif args.command == "retry-all":
        return asyncio.run(cmd_retry_all(dlq, getattr(args, "error_type", None)))
    elif args.command == "clear":
        return cmd_clear(dlq)
    elif args.command == "stats":
        entries = dlq.list()
        print(f"Total entries: {len(entries)}")
        by_error = {}
        for e in entries:
            by_error[e.error_type] = by_error.get(e.error_type, 0) + 1
        for error_type, count in sorted(by_error.items()):
            print(f"  {error_type}: {count}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
