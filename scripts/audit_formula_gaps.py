#!/usr/bin/env python3
"""Audit Docling formula-extraction gaps across the corpus.

Per issue #9: Docling drops LaTeX math to ``<!-- formula-not-decoded -->``
placeholders, degrading retrieval quality for math-heavy textbooks. This
script surveys the corpus for affected sources, groups them into
remediation tiers by domain math-weight, and emits a prioritized work
list consumable by ``scripts/reextract_with_mineru.py``.

Tier grouping (from issue #9):

- **Tier 1** — core math (mathematics, linear_algebra, optimization,
  probability_theory, analysis, topology_geometry) — remediate ≥10% gap.
- **Tier 2** — quantitative sciences (statistics, machine_learning,
  reinforcement_learning, deep_learning, causal_inference, time_series,
  econometrics) — remediate ≥20%.
- **Tier 3** — applied (physics, engineering, finance, nlp, cs,
  information_theory, rag_llm, algorithms, computer_science) —
  remediate ≥30%.
- **Tier 4** — prose-heavy (software_engineering, functional_programming,
  business, economics, neuroscience, interview_prep, general) — defer
  unless user-reported.

Usage::

    python scripts/audit_formula_gaps.py
    python scripts/audit_formula_gaps.py --min-gap-pct 20 --min-chunks 100
    python scripts/audit_formula_gaps.py --domain mathematics --format json
    python scripts/audit_formula_gaps.py --tier-breakdown
    python scripts/audit_formula_gaps.py --tier 1 --format json \\
        > data/tier1_worklist.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Make packages importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import configure_logging, get_logger  # noqa: E402
from research_kb_storage import get_connection_pool  # noqa: E402

# Redirect logs to stderr so JSON/Markdown output on stdout stays clean.
configure_logging(level="WARNING")
logger = get_logger(__name__)


# Tier definitions — keep in sync with issue #9 and docs/runbooks/.
TIER_1_DOMAINS = frozenset(
    {
        "mathematics",
        "linear_algebra",
        "optimization",
        "probability_theory",
        "analysis",
        "topology_geometry",
    }
)
TIER_2_DOMAINS = frozenset(
    {
        "statistics",
        "machine_learning",
        "reinforcement_learning",
        "deep_learning",
        "causal_inference",
        "time_series",
        "econometrics",
    }
)
TIER_3_DOMAINS = frozenset(
    {
        "physics",
        "engineering",
        "finance",
        "nlp",
        "information_theory",
        "rag_llm",
        "algorithms",
        "computer_science",
    }
)
TIER_4_DOMAINS = frozenset(
    {
        "software_engineering",
        "functional_programming",
        "business",
        "economics",
        "neuroscience",
        "interview_prep",
        "general",
    }
)

TIER_MIN_GAP_PCT = {
    1: 10.0,
    2: 20.0,
    3: 30.0,
    4: 100.0,  # Effectively "defer" — only remediate if someone complains.
}


def classify_tier(domain_id: str) -> int:
    """Return the remediation tier (1-4) for a domain.

    Unknown domains default to tier 3 (conservative remediation threshold).
    """
    if domain_id in TIER_1_DOMAINS:
        return 1
    if domain_id in TIER_2_DOMAINS:
        return 2
    if domain_id in TIER_3_DOMAINS:
        return 3
    if domain_id in TIER_4_DOMAINS:
        return 4
    return 3


@dataclass
class GapRow:
    """One source's formula-gap audit row."""

    source_id: str
    title: str
    domain_id: str
    tier: int
    total_chunks: int
    latex_chunks: int
    gap_chunks: int
    gap_pct: float


async def fetch_gap_rows(
    pool,
    min_chunks: int,
    min_gap_pct: float,
    domain: Optional[str] = None,
    tier: Optional[int] = None,
) -> list[GapRow]:
    """Query sources with formula gaps above the given thresholds.

    Parameters
    ----------
    pool : asyncpg.Pool
        Connection pool.
    min_chunks : int
        Only include sources with at least this many total chunks.
    min_gap_pct : float
        Only include sources whose gap % is at least this value.
    domain : str, optional
        Restrict to a single domain.
    tier : int, optional
        Restrict to a single tier (1-4). Applies the tier's min_gap_pct floor
        unless ``min_gap_pct`` is stricter.

    Returns
    -------
    list[GapRow]
        Rows sorted by ``gap_chunks`` descending.
    """
    query = """
        SELECT
            s.id::text AS source_id,
            s.title,
            s.domain_id,
            COUNT(c.id) AS total_chunks,
            COUNT(c.id) FILTER (WHERE c.content LIKE '%$$%') AS latex_chunks,
            COUNT(c.id) FILTER (WHERE c.content LIKE '%formula-not-decoded%')
                AS gap_chunks
        FROM sources s
        JOIN chunks c ON c.source_id = s.id
        GROUP BY s.id, s.title, s.domain_id
        HAVING COUNT(c.id) >= $1
           AND COUNT(c.id) FILTER (WHERE c.content LIKE '%formula-not-decoded%') > 0
        ORDER BY gap_chunks DESC
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, min_chunks)

    effective_floor = min_gap_pct
    if tier is not None:
        effective_floor = max(effective_floor, TIER_MIN_GAP_PCT[tier])

    out: list[GapRow] = []
    for r in rows:
        total = int(r["total_chunks"])
        gaps = int(r["gap_chunks"])
        pct = round(100.0 * gaps / total, 1) if total else 0.0
        if pct < effective_floor:
            continue
        if domain is not None and r["domain_id"] != domain:
            continue
        row_tier = classify_tier(r["domain_id"])
        if tier is not None and row_tier != tier:
            continue
        out.append(
            GapRow(
                source_id=r["source_id"],
                title=r["title"],
                domain_id=r["domain_id"],
                tier=row_tier,
                total_chunks=total,
                latex_chunks=int(r["latex_chunks"]),
                gap_chunks=gaps,
                gap_pct=pct,
            )
        )
    return out


def render_markdown(rows: list[GapRow], limit: Optional[int] = None) -> str:
    """Render rows as a Markdown table, sorted by gap_chunks desc."""
    show = rows if limit is None else rows[:limit]
    if not show:
        return "_No sources match the filter._\n"
    lines = [
        "| # | Title | Domain | Tier | Chunks | LaTeX | Gaps | Gap% |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(show, 1):
        title = r.title[:70].replace("|", "\\|")
        lines.append(
            f"| {i} | {title} | {r.domain_id} | {r.tier} | "
            f"{r.total_chunks} | {r.latex_chunks} | {r.gap_chunks} | "
            f"{r.gap_pct:.1f}% |"
        )
    return "\n".join(lines) + "\n"


def render_tier_breakdown(rows: list[GapRow]) -> str:
    """Render a tier-level summary with counts and absolute gap totals."""
    tiers: dict[int, dict[str, int]] = {}
    for r in rows:
        t = tiers.setdefault(r.tier, {"sources": 0, "chunks": 0, "latex": 0, "gaps": 0})
        t["sources"] += 1
        t["chunks"] += r.total_chunks
        t["latex"] += r.latex_chunks
        t["gaps"] += r.gap_chunks

    lines = [
        "| Tier | Floor | Sources | Total chunks | LaTeX chunks | Gap chunks |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for t in sorted(tiers):
        tinfo = tiers[t]
        lines.append(
            f"| {t} | {TIER_MIN_GAP_PCT[t]:.0f}% | {tinfo['sources']} | "
            f"{tinfo['chunks']:,} | {tinfo['latex']:,} | {tinfo['gaps']:,} |"
        )
    return "\n".join(lines) + "\n"


def render_json(rows: list[GapRow]) -> str:
    """Render rows as a JSON document for consumption by downstream tooling."""
    payload = {
        "total_sources": len(rows),
        "total_gap_chunks": sum(r.gap_chunks for r in rows),
        "total_latex_chunks": sum(r.latex_chunks for r in rows),
        "rows": [asdict(r) for r in rows],
    }
    return json.dumps(payload, indent=2) + "\n"


async def run(args: argparse.Namespace) -> int:
    pool = await get_connection_pool()
    try:
        rows = await fetch_gap_rows(
            pool,
            min_chunks=args.min_chunks,
            min_gap_pct=args.min_gap_pct,
            domain=args.domain,
            tier=args.tier,
        )
    finally:
        from research_kb_storage import close_connection_pool

        await close_connection_pool()

    if args.format == "json":
        sys.stdout.write(render_json(rows))
    else:
        sys.stdout.write(
            f"# Formula-gap audit\n\n"
            f"- Sources: **{len(rows)}**\n"
            f"- Total gap chunks: **{sum(r.gap_chunks for r in rows):,}**\n"
            f"- Total LaTeX chunks: **{sum(r.latex_chunks for r in rows):,}**\n"
            f"- Filter: min_gap_pct={args.min_gap_pct}%, "
            f"min_chunks={args.min_chunks}"
            + (f", domain={args.domain}" if args.domain else "")
            + (f", tier={args.tier}" if args.tier else "")
            + "\n\n"
        )

        if args.tier_breakdown and rows:
            sys.stdout.write("## Tier breakdown\n\n")
            sys.stdout.write(render_tier_breakdown(rows))
            sys.stdout.write("\n")

        sys.stdout.write("## Top offenders\n\n")
        sys.stdout.write(render_markdown(rows, limit=args.limit))

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Docling formula-extraction gaps across the corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--min-gap-pct",
        type=float,
        default=10.0,
        help="Minimum gap %% to include (default: 10.0).",
    )
    parser.add_argument(
        "--min-chunks",
        type=int,
        default=100,
        help="Minimum total chunks to include (default: 100).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter to a single domain (e.g., mathematics).",
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Filter to a single remediation tier (1-4).",
    )
    parser.add_argument(
        "--format",
        choices=["md", "json"],
        default="md",
        help="Output format (default: md).",
    )
    parser.add_argument(
        "--tier-breakdown",
        action="store_true",
        help="Include a tier-level summary table (md format only).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Show top N offenders only (md format; JSON always includes all).",
    )
    args = parser.parse_args()
    rc = asyncio.run(run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
