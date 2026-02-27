#!/usr/bin/env python3
"""Semantic concept deduplication — Phase AF-2.

Six deduplication passes:
  Prune passes (DELETE + CASCADE):
    prune-code     — Code-like names (snake_case, dot-notation)
    prune-numeric  — Numbered references (theorem 1029, lemma 418)
    prune-long     — Sentence-length names (>60 chars, extraction artifacts)

  Merge passes (FK re-pointing + DELETE):
    merge-hyphen   — Hyphen/space variants (cross-validation → cross validation)
    merge-article  — Article prefixes (the central limit theorem → central limit theorem)
    merge-plural   — Morphological -es plurals (processes → process)

Usage:
    # Dry run (default) — show what would change
    python scripts/deduplicate_concepts_v2.py --pass all

    # Execute a single pass
    python scripts/deduplicate_concepts_v2.py --pass prune-code --execute

    # Limit + verbose for testing
    python scripts/deduplicate_concepts_v2.py --pass merge-hyphen --limit 10 --verbose --execute
"""

import argparse
import asyncio
from dataclasses import dataclass, field
from uuid import UUID

import asyncpg

from research_kb_common import get_logger

logger = get_logger(__name__)

# Tables referencing concepts(id) — all 10 FK columns
FK_TABLES = [
    ("chunk_concepts", "concept_id"),
    ("concept_relationships", "source_concept_id"),
    ("concept_relationships", "target_concept_id"),
    ("methods", "concept_id"),
    ("assumptions", "concept_id"),
    ("method_assumption_cache", "method_concept_id"),
    ("method_assumption_cache", "assumption_concept_id"),
    ("method_aliases", "method_concept_id"),
    ("cross_domain_links", "source_concept_id"),
    ("cross_domain_links", "target_concept_id"),
]

# Tables with unique constraints that need duplicate removal before FK update.
# Format: (table, fk_column, unique_columns_excluding_fk)
# When re-pointing keep_id, if the same (unique_columns) row already exists
# for keep_id, delete the remove_id row instead of updating.
UNIQUE_CONSTRAINT_TABLES = [
    # chunk_concepts: unique(chunk_id, concept_id)
    ("chunk_concepts", "concept_id", ["chunk_id"]),
    # concept_relationships: unique(source, target, type)
    ("concept_relationships", "source_concept_id", ["target_concept_id", "relationship_type"]),
    ("concept_relationships", "target_concept_id", ["source_concept_id", "relationship_type"]),
    # methods: unique(concept_id) — one method per concept
    ("methods", "concept_id", []),
    # assumptions: unique(concept_id)
    ("assumptions", "concept_id", []),
    # method_assumption_cache: unique(method_concept_id, assumption_name)
    ("method_assumption_cache", "method_concept_id", ["assumption_name"]),
    # method_aliases: unique(alias)
    ("method_aliases", "method_concept_id", []),
    # cross_domain_links: unique(source_concept_id, target_concept_id)
    ("cross_domain_links", "source_concept_id", ["target_concept_id"]),
    ("cross_domain_links", "target_concept_id", ["source_concept_id"]),
]

# Safety: concepts in these tables should never be pruned
SAFETY_TABLES = ["methods", "assumptions", "method_assumption_cache"]


@dataclass
class PassStats:
    """Accumulated statistics for a single pass."""

    pass_name: str
    candidates: int = 0
    skipped_safety: int = 0
    processed: int = 0
    fk_updates: int = 0
    aliases_merged: int = 0
    concepts_deleted: int = 0
    self_loops_removed: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prune helpers
# ---------------------------------------------------------------------------


async def check_safety(conn: asyncpg.Connection, concept_id: UUID, max_chunk_links: int) -> bool:
    """Return True if concept is safe to prune (not in safety tables, below chunk threshold).

    Args:
        conn: Database connection
        concept_id: Concept to check
        max_chunk_links: Maximum chunk_concepts rows allowed for pruning

    Returns:
        True if safe to prune, False if protected
    """
    for table in SAFETY_TABLES:
        col = "method_concept_id" if "method" in table else "concept_id"
        count = await conn.fetchval(
            f"SELECT COUNT(*) FROM {table} WHERE {col} = $1", concept_id  # noqa: S608
        )
        if count > 0:
            return False

    # Also check assumption_concept_id in method_assumption_cache
    count = await conn.fetchval(
        "SELECT COUNT(*) FROM method_assumption_cache WHERE assumption_concept_id = $1",
        concept_id,
    )
    if count > 0:
        return False

    chunk_count = await conn.fetchval(
        "SELECT COUNT(*) FROM chunk_concepts WHERE concept_id = $1", concept_id
    )
    if chunk_count > max_chunk_links:
        return False

    return True


async def prune_concept(conn: asyncpg.Connection, concept_id: UUID) -> None:
    """Delete a concept. ON DELETE CASCADE handles most FKs.

    Some FKs use ON DELETE SET NULL (method_assumption_cache.assumption_concept_id),
    so we explicitly delete those references first to keep things clean.
    """
    # Clean up SET NULL references explicitly
    await conn.execute(
        "DELETE FROM method_assumption_cache WHERE assumption_concept_id = $1",
        concept_id,
    )

    # CASCADE handles: chunk_concepts, concept_relationships, methods, assumptions,
    # method_assumption_cache(method_concept_id), method_aliases, cross_domain_links
    await conn.execute("DELETE FROM concepts WHERE id = $1", concept_id)


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


async def merge_concept(
    conn: asyncpg.Connection,
    keep_id: UUID,
    keep_name: str,
    remove_id: UUID,
    remove_name: str,
) -> dict[str, int]:
    """Merge remove_id into keep_id across all 10 FK references.

    Steps:
    1. For each FK table, delete rows that would create duplicate unique keys.
    2. Update remaining rows to point to keep_id.
    3. Merge aliases from remove into keep.
    4. Delete the remove concept.
    5. Remove self-loop relationships.

    Returns:
        Dict of update counts per table.
    """
    update_counts: dict[str, int] = {}

    # Re-point all FK references
    for table, fk_col, unique_cols in UNIQUE_CONSTRAINT_TABLES:
        # Step 1: Delete rows that would conflict after re-pointing
        if unique_cols:
            unique_clause = " AND ".join(
                f"dup.{col} = existing.{col}" for col in unique_cols
            )
            await conn.execute(
                f"""
                DELETE FROM {table} dup
                WHERE dup.{fk_col} = $1
                AND EXISTS (
                    SELECT 1 FROM {table} existing
                    WHERE existing.{fk_col} = $2
                    AND {unique_clause}
                )
                """,  # noqa: S608
                remove_id,
                keep_id,
            )
        else:
            # Table has unique(concept_id) — delete if keep already exists
            await conn.execute(
                f"""
                DELETE FROM {table}
                WHERE {fk_col} = $1
                AND EXISTS (SELECT 1 FROM {table} WHERE {fk_col} = $2)
                """,  # noqa: S608
                remove_id,
                keep_id,
            )

        # Step 2: Update remaining rows
        result = await conn.execute(
            f"UPDATE {table} SET {fk_col} = $1 WHERE {fk_col} = $2",  # noqa: S608
            keep_id,
            remove_id,
        )
        count = int(result.split()[-1])
        key = f"{table}.{fk_col}"
        update_counts[key] = update_counts.get(key, 0) + count

    # Step 3: Merge aliases
    remove_aliases = await conn.fetchval(
        "SELECT aliases FROM concepts WHERE id = $1", remove_id
    )
    aliases_to_add = [remove_name]
    if remove_aliases:
        aliases_to_add.extend(remove_aliases)

    await conn.execute(
        """
        UPDATE concepts
        SET aliases = array_cat(COALESCE(aliases, ARRAY[]::text[]), $1::text[])
        WHERE id = $2
        """,
        aliases_to_add,
        keep_id,
    )
    update_counts["aliases_merged"] = len(aliases_to_add)

    # Step 4: Delete the remove concept
    await conn.execute("DELETE FROM concepts WHERE id = $1", remove_id)

    # Step 5: Remove self-loop relationships created by merge
    result = await conn.execute(
        """
        DELETE FROM concept_relationships
        WHERE source_concept_id = $1 AND target_concept_id = $1
        """,
        keep_id,
    )
    update_counts["self_loops_removed"] = int(result.split()[-1])

    return update_counts


# ---------------------------------------------------------------------------
# Pass implementations
# ---------------------------------------------------------------------------


async def pass_prune_code(
    conn: asyncpg.Connection,
    dry_run: bool,
    limit: int | None,
    verbose: bool,
) -> PassStats:
    """Prune code-like concept names (snake_case, dot.notation)."""
    stats = PassStats(pass_name="prune-code")

    sql = """
    SELECT id, canonical_name FROM concepts
    WHERE canonical_name ~ '[a-z]_[a-z]'
       OR canonical_name ~ '^\\w+\\.\\w+'
    ORDER BY canonical_name
    """
    if limit:
        sql += f" LIMIT {limit}"

    rows = await conn.fetch(sql)
    stats.candidates = len(rows)

    for row in rows:
        cid, name = row["id"], row["canonical_name"]
        safe = await check_safety(conn, cid, max_chunk_links=5)
        if not safe:
            stats.skipped_safety += 1
            if verbose:
                print(f"  SKIP (safety): {name}")
            continue

        if verbose:
            print(f"  PRUNE: {name}")

        if not dry_run:
            try:
                async with conn.transaction():
                    await prune_concept(conn, cid)
                stats.concepts_deleted += 1
            except Exception as e:
                stats.errors.append(f"{name}: {e}")
        else:
            stats.concepts_deleted += 1

        stats.processed += 1

    return stats


async def pass_prune_numeric(
    conn: asyncpg.Connection,
    dry_run: bool,
    limit: int | None,
    verbose: bool,
) -> PassStats:
    """Prune numbered references (theorem 1029, lemma 418, etc.)."""
    stats = PassStats(pass_name="prune-numeric")

    sql = """
    SELECT id, canonical_name FROM concepts
    WHERE canonical_name ~ '^(theorem|lemma|proposition|corollary|definition|example|equation|figure|table|section|chapter|exercise|problem|remark|property|result|step|case|claim|condition|rule|algorithm|assumption|axiom|conjecture|fact|formula|hypothesis|inequality|observation|postulate|principle|proof) \\d+(\\.\\d+)*$'
       OR canonical_name ~ '^\\d+(\\.\\d+)+$'
    ORDER BY canonical_name
    """
    if limit:
        sql += f" LIMIT {limit}"

    rows = await conn.fetch(sql)
    stats.candidates = len(rows)

    for row in rows:
        cid, name = row["id"], row["canonical_name"]
        safe = await check_safety(conn, cid, max_chunk_links=5)
        if not safe:
            stats.skipped_safety += 1
            if verbose:
                print(f"  SKIP (safety): {name}")
            continue

        if verbose:
            print(f"  PRUNE: {name}")

        if not dry_run:
            try:
                async with conn.transaction():
                    await prune_concept(conn, cid)
                stats.concepts_deleted += 1
            except Exception as e:
                stats.errors.append(f"{name}: {e}")
        else:
            stats.concepts_deleted += 1

        stats.processed += 1

    return stats


async def pass_prune_long(
    conn: asyncpg.Connection,
    dry_run: bool,
    limit: int | None,
    verbose: bool,
) -> PassStats:
    """Prune sentence-length concept names (>60 chars, extraction artifacts)."""
    stats = PassStats(pass_name="prune-long")

    sql = """
    SELECT id, canonical_name FROM concepts
    WHERE LENGTH(canonical_name) > 60
    ORDER BY canonical_name
    """
    if limit:
        sql += f" LIMIT {limit}"

    rows = await conn.fetch(sql)
    stats.candidates = len(rows)

    for row in rows:
        cid, name = row["id"], row["canonical_name"]
        safe = await check_safety(conn, cid, max_chunk_links=3)
        if not safe:
            stats.skipped_safety += 1
            if verbose:
                print(f"  SKIP (safety): {name[:60]}...")
            continue

        if verbose:
            print(f"  PRUNE: {name[:80]}...")

        if not dry_run:
            try:
                async with conn.transaction():
                    await prune_concept(conn, cid)
                stats.concepts_deleted += 1
            except Exception as e:
                stats.errors.append(f"{name[:40]}: {e}")
        else:
            stats.concepts_deleted += 1

        stats.processed += 1

    return stats


async def pass_merge_hyphen(
    conn: asyncpg.Connection,
    dry_run: bool,
    limit: int | None,
    verbose: bool,
) -> PassStats:
    """Merge hyphen/space variants (keep space-form, matches new normalization)."""
    stats = PassStats(pass_name="merge-hyphen")

    sql = """
    SELECT
        c_space.id AS keep_id,
        c_space.canonical_name AS keep_name,
        c_hyphen.id AS remove_id,
        c_hyphen.canonical_name AS remove_name
    FROM concepts c_hyphen
    JOIN concepts c_space
        ON c_space.canonical_name = REPLACE(c_hyphen.canonical_name, '-', ' ')
    WHERE c_hyphen.canonical_name LIKE '%-%'
      AND c_hyphen.id != c_space.id
    ORDER BY c_hyphen.canonical_name
    """
    if limit:
        sql += f" LIMIT {limit}"

    rows = await conn.fetch(sql)
    stats.candidates = len(rows)

    for row in rows:
        keep_id, keep_name = row["keep_id"], row["keep_name"]
        remove_id, remove_name = row["remove_id"], row["remove_name"]

        if verbose:
            print(f"  MERGE: {remove_name} -> {keep_name}")

        if not dry_run:
            try:
                async with conn.transaction():
                    counts = await merge_concept(
                        conn, keep_id, keep_name, remove_id, remove_name
                    )
                    stats.fk_updates += sum(
                        v for k, v in counts.items()
                        if k not in ("aliases_merged", "self_loops_removed")
                    )
                    stats.aliases_merged += counts.get("aliases_merged", 0)
                    stats.self_loops_removed += counts.get("self_loops_removed", 0)
                stats.concepts_deleted += 1
            except Exception as e:
                stats.errors.append(f"{remove_name}: {e}")
        else:
            stats.concepts_deleted += 1

        stats.processed += 1

    return stats


async def pass_merge_article(
    conn: asyncpg.Connection,
    dry_run: bool,
    limit: int | None,
    verbose: bool,
) -> PassStats:
    """Merge article-prefixed variants (keep form without article)."""
    stats = PassStats(pass_name="merge-article")

    # Match concepts starting with "the ", "a ", "an " that have a counterpart without
    sql = """
    SELECT
        c_bare.id AS keep_id,
        c_bare.canonical_name AS keep_name,
        c_article.id AS remove_id,
        c_article.canonical_name AS remove_name
    FROM concepts c_article
    JOIN concepts c_bare
        ON c_bare.canonical_name = REGEXP_REPLACE(c_article.canonical_name, '^(the|a|an) ', '')
    WHERE c_article.canonical_name ~ '^(the|a|an) '
      AND c_article.id != c_bare.id
    ORDER BY c_article.canonical_name
    """
    if limit:
        sql += f" LIMIT {limit}"

    rows = await conn.fetch(sql)
    stats.candidates = len(rows)

    for row in rows:
        keep_id, keep_name = row["keep_id"], row["keep_name"]
        remove_id, remove_name = row["remove_id"], row["remove_name"]

        if verbose:
            print(f"  MERGE: {remove_name} -> {keep_name}")

        if not dry_run:
            try:
                async with conn.transaction():
                    counts = await merge_concept(
                        conn, keep_id, keep_name, remove_id, remove_name
                    )
                    stats.fk_updates += sum(
                        v for k, v in counts.items()
                        if k not in ("aliases_merged", "self_loops_removed")
                    )
                    stats.aliases_merged += counts.get("aliases_merged", 0)
                    stats.self_loops_removed += counts.get("self_loops_removed", 0)
                stats.concepts_deleted += 1
            except Exception as e:
                stats.errors.append(f"{remove_name}: {e}")
        else:
            stats.concepts_deleted += 1

        stats.processed += 1

    return stats


async def pass_merge_plural(
    conn: asyncpg.Connection,
    dry_run: bool,
    limit: int | None,
    verbose: bool,
) -> PassStats:
    """Merge morphological -es plurals (keep singular).

    Covers:
    - +es: process/processes, index/indexes
    - -sis -> -ses: hypothesis/hypotheses, analysis/analyses
    - -ix -> -ices: matrix/matrices, index/indices
    """
    stats = PassStats(pass_name="merge-plural")

    # Pattern 1: plain +es (where +s form doesn't exist — those were handled in v1)
    sql_es = """
    SELECT
        c_sing.id AS keep_id,
        c_sing.canonical_name AS keep_name,
        c_plur.id AS remove_id,
        c_plur.canonical_name AS remove_name
    FROM concepts c_plur
    JOIN concepts c_sing
        ON c_sing.canonical_name = LEFT(c_plur.canonical_name, LENGTH(c_plur.canonical_name) - 2)
    WHERE c_plur.canonical_name LIKE '%es'
      AND LENGTH(c_plur.canonical_name) > 4
      AND c_plur.id != c_sing.id
      -- Exclude pairs already handled by v1 (+s dedup)
      AND NOT EXISTS (
          SELECT 1 FROM concepts c_s
          WHERE c_s.canonical_name = c_sing.canonical_name || 's'
            AND c_s.id = c_plur.id
      )
    ORDER BY c_plur.canonical_name
    """

    # Pattern 2: -sis -> -ses (hypothesis -> hypotheses)
    sql_ses = """
    SELECT
        c_sing.id AS keep_id,
        c_sing.canonical_name AS keep_name,
        c_plur.id AS remove_id,
        c_plur.canonical_name AS remove_name
    FROM concepts c_plur
    JOIN concepts c_sing
        ON c_sing.canonical_name = LEFT(c_plur.canonical_name, LENGTH(c_plur.canonical_name) - 3) || 'is'
    WHERE c_plur.canonical_name LIKE '%ses'
      AND LENGTH(c_plur.canonical_name) > 5
      AND c_plur.id != c_sing.id
    ORDER BY c_plur.canonical_name
    """

    # Pattern 3: -ix -> -ices (matrix -> matrices)
    sql_ices = """
    SELECT
        c_sing.id AS keep_id,
        c_sing.canonical_name AS keep_name,
        c_plur.id AS remove_id,
        c_plur.canonical_name AS remove_name
    FROM concepts c_plur
    JOIN concepts c_sing
        ON c_sing.canonical_name = LEFT(c_plur.canonical_name, LENGTH(c_plur.canonical_name) - 4) || 'ix'
    WHERE c_plur.canonical_name LIKE '%ices'
      AND LENGTH(c_plur.canonical_name) > 5
      AND c_plur.id != c_sing.id
    ORDER BY c_plur.canonical_name
    """

    # Collect all pairs, dedup by remove_id to avoid double-processing
    seen_remove_ids: set[UUID] = set()
    all_rows = []

    for sql in [sql_es, sql_ses, sql_ices]:
        query = sql
        if limit:
            query += f" LIMIT {limit}"
        rows = await conn.fetch(query)
        for row in rows:
            if row["remove_id"] not in seen_remove_ids:
                seen_remove_ids.add(row["remove_id"])
                all_rows.append(row)

    stats.candidates = len(all_rows)

    for row in all_rows:
        keep_id, keep_name = row["keep_id"], row["keep_name"]
        remove_id, remove_name = row["remove_id"], row["remove_name"]

        if verbose:
            print(f"  MERGE: {remove_name} -> {keep_name}")

        if not dry_run:
            try:
                async with conn.transaction():
                    counts = await merge_concept(
                        conn, keep_id, keep_name, remove_id, remove_name
                    )
                    stats.fk_updates += sum(
                        v for k, v in counts.items()
                        if k not in ("aliases_merged", "self_loops_removed")
                    )
                    stats.aliases_merged += counts.get("aliases_merged", 0)
                    stats.self_loops_removed += counts.get("self_loops_removed", 0)
                stats.concepts_deleted += 1
            except Exception as e:
                stats.errors.append(f"{remove_name}: {e}")
        else:
            stats.concepts_deleted += 1

        stats.processed += 1

    return stats


# ---------------------------------------------------------------------------
# Pass registry
# ---------------------------------------------------------------------------

PASS_REGISTRY: dict[str, callable] = {
    "prune-code": pass_prune_code,
    "prune-numeric": pass_prune_numeric,
    "prune-long": pass_prune_long,
    "merge-hyphen": pass_merge_hyphen,
    "merge-article": pass_merge_article,
    "merge-plural": pass_merge_plural,
}

PASS_ORDER = [
    "prune-code",
    "prune-numeric",
    "prune-long",
    "merge-hyphen",
    "merge-article",
    "merge-plural",
]


def print_stats(stats: PassStats) -> None:
    """Print summary statistics for a pass."""
    print(f"\n--- {stats.pass_name} ---")
    print(f"  Candidates:          {stats.candidates:,}")
    print(f"  Skipped (safety):    {stats.skipped_safety:,}")
    print(f"  Processed:           {stats.processed:,}")
    if stats.fk_updates:
        print(f"  FK updates:          {stats.fk_updates:,}")
    if stats.aliases_merged:
        print(f"  Aliases merged:      {stats.aliases_merged:,}")
    print(f"  Concepts deleted:    {stats.concepts_deleted:,}")
    if stats.self_loops_removed:
        print(f"  Self-loops removed:  {stats.self_loops_removed:,}")
    if stats.errors:
        print(f"  Errors:              {len(stats.errors)}")
        for err in stats.errors[:5]:
            print(f"    - {err}")
        if len(stats.errors) > 5:
            print(f"    ... and {len(stats.errors) - 5} more")


async def main():
    parser = argparse.ArgumentParser(
        description="Semantic concept deduplication (Phase AF-2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pass names:
  prune-code     Code-like names (snake_case, dot.notation)
  prune-numeric  Numbered references (theorem 1029, lemma 418)
  prune-long     Sentence-length names (>60 chars)
  merge-hyphen   Hyphen/space variants (cross-validation -> cross validation)
  merge-article  Article prefixes (the CLT -> CLT)
  merge-plural   Morphological -es plurals (processes -> process)
  all            Run all passes in order
""",
    )
    parser.add_argument(
        "--pass",
        dest="pass_name",
        required=True,
        choices=[*PASS_ORDER, "all"],
        help="Which dedup pass to run",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute changes (default: dry run)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit to first N items (for testing)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show details for each item",
    )
    args = parser.parse_args()

    dry_run = not args.execute
    passes = PASS_ORDER if args.pass_name == "all" else [args.pass_name]

    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        database="research_kb",
        user="postgres",
        password="postgres",
    )

    try:
        # Snapshot before
        before_count = await conn.fetchval("SELECT COUNT(*) FROM concepts")
        print(f"{'DRY RUN' if dry_run else 'EXECUTING'}: Semantic Concept Deduplication (Phase AF-2)")
        print(f"Concepts before: {before_count:,}")
        print(f"Passes: {', '.join(passes)}")
        print("=" * 60)

        all_stats: list[PassStats] = []

        for pass_name in passes:
            print(f"\nRunning pass: {pass_name}...")
            fn = PASS_REGISTRY[pass_name]
            stats = await fn(conn, dry_run, args.limit, args.verbose)
            all_stats.append(stats)
            print_stats(stats)

        # Grand summary
        total_deleted = sum(s.concepts_deleted for s in all_stats)
        total_errors = sum(len(s.errors) for s in all_stats)

        print("\n" + "=" * 60)
        print("GRAND SUMMARY")
        print("=" * 60)
        print(f"Total concepts deleted: {total_deleted:,}")
        print(f"Total errors:           {total_errors}")

        if not dry_run:
            after_count = await conn.fetchval("SELECT COUNT(*) FROM concepts")
            print(f"\nConcepts before: {before_count:,}")
            print(f"Concepts after:  {after_count:,}")
            print(f"Reduction:       {before_count - after_count:,} ({(before_count - after_count) / before_count * 100:.1f}%)")
        else:
            estimated = before_count - total_deleted
            print(f"\nEstimated after: ~{estimated:,} ({total_deleted / before_count * 100:.1f}% reduction)")
            print("\nThis was a DRY RUN. Use --execute to apply changes.")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
