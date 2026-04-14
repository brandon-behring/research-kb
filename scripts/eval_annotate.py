#!/usr/bin/env python3
"""Interactive CLI annotation tool for evaluation ground truth.

Presents pre-labeled candidates with color-coded panels for human review.
Supports incremental save, resume, and session-based annotation batches.

Controls:
  Enter     Accept suggested grade
  0-3       Override with specific grade
  e         Edit evidence text (grade >= 2)
  s         Skip this chunk
  q         Save progress and quit
  ?         Show help

Usage:
    python scripts/eval_annotate.py --input fixtures/eval/v2_pool_candidates.yaml
    python scripts/eval_annotate.py --input pool.yaml --resume
    python scripts/eval_annotate.py --input pool.yaml --query-range 1-20
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("ERROR: 'rich' library required. Install with: uv add --dev rich", file=sys.stderr)
    sys.exit(1)


# Add packages to path for DB access (full text fetch)
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))


GRADE_COLORS = {
    0: "red",
    1: "yellow",
    2: "green",
    3: "bright_green",
}

GRADE_LABELS = {
    0: "irrelevant",
    1: "marginal",
    2: "relevant",
    3: "highly relevant",
}

console = Console()


def show_help() -> None:
    """Print help panel."""
    help_table = Table(title="Annotation Controls", show_header=True)
    help_table.add_column("Key", style="bold cyan", width=8)
    help_table.add_column("Action")
    help_table.add_row("Enter", "Accept suggested grade")
    help_table.add_row("0-3", "Override with specific grade")
    help_table.add_row("e", "Edit evidence text (for grade >= 2)")
    help_table.add_row("f", "Fetch full chunk text from DB")
    help_table.add_row("s", "Skip this chunk (leave unannotated)")
    help_table.add_row("q", "Save progress and quit")
    help_table.add_row("?", "Show this help")
    console.print(help_table)
    console.print()


def fetch_full_chunk_text(chunk_id: str) -> Optional[str]:
    """Fetch full chunk content from DB.

    Parameters
    ----------
    chunk_id : str
        UUID of the chunk.

    Returns
    -------
    str or None
        Full chunk content, or None on error.
    """
    import asyncio
    from uuid import UUID

    async def _fetch() -> Optional[str]:
        try:
            from research_kb_storage.connection import get_connection_pool

            pool = await get_connection_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT content FROM chunks WHERE id = $1",
                    UUID(chunk_id),
                )
                return row["content"] if row else None
        except Exception as e:
            return f"[Error fetching: {e}]"

    return asyncio.run(_fetch())


def format_chunk_panel(
    candidate: dict[str, Any],
    chunk_idx: int,
    total_chunks: int,
    query_text: str,
) -> Panel:
    """Build a Rich panel for a candidate chunk.

    Parameters
    ----------
    candidate : dict
        Candidate with text_snippet, suggested_grade, similarity_score, etc.
    chunk_idx : int
        1-based index of this chunk.
    total_chunks : int
        Total chunks for this query.
    query_text : str
        The query being annotated.

    Returns
    -------
    Panel
        Formatted panel for display.
    """
    suggested = candidate.get("suggested_grade", 0)
    sim = candidate.get("similarity_score")
    sim_str = f"{sim:.3f}" if sim is not None else "N/A"
    color = GRADE_COLORS.get(suggested, "white")

    # Build snippet text
    text = candidate.get("text_snippet", "(no text)")
    # Show up to 400 chars for annotation context
    if len(text) > 400:
        text = text[:400] + "..."

    # Source info
    source_title = candidate.get("source_title", "Unknown")
    page_start = candidate.get("page_start")
    page_end = candidate.get("page_end")
    page_str = ""
    if page_start is not None:
        page_str = f" p.{page_start}"
        if page_end is not None and page_end != page_start:
            page_str = f" p.{page_start}-{page_end}"

    # Config info
    configs = candidate.get("found_in_configs", [])
    config_str = ", ".join(configs)

    subtitle = f"{source_title}{page_str} | configs: {config_str}"

    return Panel(
        text,
        title=f"Chunk {chunk_idx}/{total_chunks} | Suggested: {suggested} ({GRADE_LABELS.get(suggested, '?')}) | sim={sim_str}",
        subtitle=subtitle,
        border_style=color,
        width=min(console.width, 100),
    )


def annotate_chunk(
    candidate: dict[str, Any],
    chunk_idx: int,
    total_chunks: int,
    query_text: str,
) -> Optional[str]:
    """Present a chunk and get annotation from user.

    Parameters
    ----------
    candidate : dict
        Candidate to annotate.
    chunk_idx : int
        1-based index.
    total_chunks : int
        Total for this query.
    query_text : str
        The query.

    Returns
    -------
    str or None
        "quit" if user wants to quit, None otherwise.
    """
    panel = format_chunk_panel(candidate, chunk_idx, total_chunks, query_text)
    console.print(panel)

    suggested = candidate.get("suggested_grade", 0)

    while True:
        response = Prompt.ask(
            f"  Grade [Enter={suggested}, 0-3, e=evidence, f=full text, s=skip, q=quit, ?=help]",
            default=str(suggested),
        )

        response = response.strip().lower()

        if response == "q":
            return "quit"

        if response == "?":
            show_help()
            continue

        if response == "s":
            # Skip — mark so resume mode doesn't re-present
            candidate["skipped"] = True
            candidate["annotated_at"] = datetime.now(timezone.utc).isoformat()
            console.print("  [dim]Skipped[/dim]")
            return None

        if response == "f":
            # Fetch full chunk text from DB
            console.print("  [dim]Fetching full text...[/dim]")
            full_text = fetch_full_chunk_text(candidate["chunk_id"])
            if full_text:
                text_panel = Panel(
                    full_text[:2000],
                    title=f"Full text ({len(full_text)} chars)",
                    border_style="cyan",
                    width=100,
                )
                console.print(text_panel)
            else:
                console.print("  [red]Could not fetch chunk text[/red]")
            continue

        if response == "e":
            # Edit evidence text
            current = candidate.get("evidence_text", "")
            new_evidence = Prompt.ask("  Evidence text", default=current or "")
            candidate["evidence_text"] = new_evidence if new_evidence else None
            console.print(f"  [dim]Evidence updated[/dim]")
            continue

        if response in ("0", "1", "2", "3"):
            grade = int(response)
            human_override = grade != suggested
            candidate["relevance"] = grade
            candidate["human_override"] = human_override
            candidate["annotator"] = "claude_code+human"
            candidate["annotated_at"] = datetime.now(timezone.utc).isoformat()

            # For grade >= 2, prompt for evidence if not already set
            if grade >= 2 and not candidate.get("evidence_text"):
                snippet = candidate.get("text_snippet", "")[:150]
                candidate["evidence_text"] = snippet

            grade_label = GRADE_LABELS.get(grade, "?")
            override_str = " [bold](override)[/bold]" if human_override else ""
            console.print(
                f"  [{GRADE_COLORS.get(grade, 'white')}]{grade} ({grade_label}){override_str}[/]"
            )
            return None

        # Default: treat as accepting the suggestion
        if response == str(suggested) or response == "":
            candidate["relevance"] = suggested
            candidate["human_override"] = False
            candidate["annotator"] = "claude_code+human"
            candidate["annotated_at"] = datetime.now(timezone.utc).isoformat()

            if suggested >= 2 and not candidate.get("evidence_text"):
                snippet = candidate.get("text_snippet", "")[:150]
                candidate["evidence_text"] = snippet

            grade_label = GRADE_LABELS.get(suggested, "?")
            console.print(
                f"  [{GRADE_COLORS.get(suggested, 'white')}]{suggested} ({grade_label})[/]"
            )
            return None

        console.print("  [red]Invalid input. Type 0-3, Enter, e, s, q, or ?[/red]")


def save_progress(data: dict[str, Any], output_path: Path) -> None:
    """Write current annotation state to disk.

    Parameters
    ----------
    data : dict
        Full pooling data with annotations.
    output_path : Path
        Where to write.
    """
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def count_annotated(data: dict[str, Any]) -> tuple[int, int, int]:
    """Count annotation progress.

    Returns
    -------
    tuple[int, int, int]
        (annotated, skipped, total) chunk counts.
    """
    annotated = 0
    skipped = 0
    total = 0
    for q in data.get("queries", []):
        for c in q.get("candidates", []):
            total += 1
            if c.get("relevance") is not None:
                annotated += 1
            elif c.get("annotated_at") is not None:
                skipped += 1
    return annotated, skipped, total


def print_summary(data: dict[str, Any]) -> None:
    """Print annotation summary statistics."""
    annotated, skipped, total = count_annotated(data)
    remaining = total - annotated - skipped

    # Grade distribution
    grade_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    override_count = 0
    for q in data.get("queries", []):
        for c in q.get("candidates", []):
            if c.get("relevance") is not None:
                grade_counts[c["relevance"]] = grade_counts.get(c["relevance"], 0) + 1
                if c.get("human_override"):
                    override_count += 1

    table = Table(title="Annotation Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Annotated", f"{annotated}/{total}")
    table.add_row("Remaining", str(remaining))
    table.add_row("Skipped", str(skipped))
    table.add_row("Override rate", f"{override_count}/{annotated}" if annotated else "N/A")
    table.add_row("---", "---")
    for grade in [3, 2, 1, 0]:
        color = GRADE_COLORS.get(grade, "white")
        table.add_row(
            f"Grade {grade} ({GRADE_LABELS[grade]})",
            f"[{color}]{grade_counts.get(grade, 0)}[/]",
        )

    console.print()
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive CLI annotation tool for evaluation ground truth"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Pre-labeled candidates YAML (from eval_prelabel.py)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output YAML path (default: overwrite input file)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-annotated chunks",
    )
    parser.add_argument(
        "--query-range",
        type=str,
        default=None,
        help="Annotate only queries N-M (1-indexed, e.g., '1-20')",
    )
    args = parser.parse_args()

    output_path = args.output or args.input

    # Load data
    with open(args.input) as f:
        data = yaml.safe_load(f)

    queries = data.get("queries", [])
    if not queries:
        console.print("[red]ERROR: No queries found in input file[/red]")
        sys.exit(1)

    # Parse query range
    start_idx = 0
    end_idx = len(queries)
    if args.query_range:
        parts = args.query_range.split("-")
        if len(parts) == 2:
            start_idx = int(parts[0]) - 1  # 1-indexed to 0-indexed
            end_idx = int(parts[1])
        elif len(parts) == 1:
            start_idx = int(parts[0]) - 1
            end_idx = start_idx + 1
        else:
            console.print(f"[red]Invalid query range: {args.query_range}. Use N-M format.[/red]")
            sys.exit(1)

    # Show initial status
    annotated, skipped, total = count_annotated(data)
    console.print(f"[bold]Annotation session[/bold]")
    console.print(f"  Queries: {start_idx + 1}-{end_idx} of {len(queries)}")
    console.print(
        f"  Progress: {annotated}/{total} annotated, {total - annotated - skipped} remaining"
    )
    console.print()
    show_help()

    quit_requested = False

    for qi in range(start_idx, end_idx):
        if quit_requested:
            break

        q = queries[qi]
        query_text = q["query_text"]
        domain = q.get("domain", "?")
        candidates = q.get("candidates", [])

        # Count already annotated for this query
        already_done = sum(1 for c in candidates if c.get("relevance") is not None)
        if args.resume and already_done == len(candidates):
            continue

        console.rule(f'Query {qi + 1}/{len(queries)}: "{query_text}" [{domain}]')

        for ci, candidate in enumerate(candidates):
            if quit_requested:
                break

            # Skip already-annotated or explicitly skipped chunks in resume mode
            if args.resume and (candidate.get("relevance") is not None or candidate.get("skipped")):
                continue

            result = annotate_chunk(
                candidate=candidate,
                chunk_idx=ci + 1,
                total_chunks=len(candidates),
                query_text=query_text,
            )

            if result == "quit":
                quit_requested = True
                break

        # Incremental save after each query
        save_progress(data, output_path)

        # Per-query summary
        query_annotated = sum(1 for c in candidates if c.get("relevance") is not None)
        query_relevant = sum(
            1 for c in candidates if c.get("relevance") is not None and c["relevance"] >= 2
        )
        console.print(
            f"  [dim]Query done: {query_annotated}/{len(candidates)} annotated, "
            f"{query_relevant} relevant (grade >= 2)[/dim]"
        )

    # Final save
    save_progress(data, output_path)

    # Summary
    print_summary(data)
    console.print(f"\n  Saved to: {output_path}")


if __name__ == "__main__":
    main()
