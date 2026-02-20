"""DB hygiene: normalize domains, assign domains, detect/merge duplicates.

All operations are dry-run by default. Pass --apply to execute mutations.

Operations:
A. --fix-domains: Normalize "causal inference" → "causal_inference" etc.
B. --assign-domains: Match "none" domain sources to catalog, propose assignments.
C. --find-duplicates: Group by file_hash or fuzzy title, produce report.
D. --merge-duplicates SRC_KEEP SRC_DELETE: Reassign chunks, delete empty source.
E. --report-untagged: Report non-Manning "none" domain sources.
"""

import json
import sys
import uuid
from pathlib import Path

from .audit import fuzzy_title_match, normalize_for_match
from .catalog import load_catalog

import re


def _normalize_strict(title: str) -> str:
    """Strict normalizer that preserves edition info.

    Unlike normalize_for_match (which strips editions for catalog matching),
    this keeps "second edition", "2nd edition", etc. so that different
    editions are NOT merged as duplicates.
    """
    t = title.lower().strip()
    # Remove punctuation except spaces (but keep words intact)
    t = re.sub(r"[^\w\s]", "", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


_project_root = Path(__file__).parent.parent.parent
for pkg in ("storage", "contracts", "common"):
    sys.path.insert(0, str(_project_root / "packages" / pkg / "src"))


# Domain normalization map: messy → canonical
DOMAIN_NORMALIZATION: dict[str, str] = {
    "causal inference": "causal_inference",
    "causal_inference/ml": "causal_inference",
    "causal inference/ml": "causal_inference",
    "deep learning": "deep_learning",
    "time series": "time_series",
    "time_series/finance": "time_series",
    "machine learning": "ml_engineering",
    "data science": "data_science",
    "software engineering": "software_engineering",
    "functional programming": "functional_programming",
    "rag llm": "rag_llm",
    "rag/llm": "rag_llm",
    "interview prep": "interview_prep",
}


async def fix_domains(pool, apply: bool = False) -> list[dict]:
    """Normalize domain tags in sources.metadata->>'domain'.

    Pure string normalization — safe operation.

    Returns:
        List of {source_id, title, old_domain, new_domain} for changed sources.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, title, COALESCE(metadata->>'domain', 'none') AS domain, metadata
            FROM sources
            ORDER BY title
            """
        )

    changes: list[dict] = []
    for row in rows:
        old_domain = row["domain"]
        canonical = DOMAIN_NORMALIZATION.get(old_domain.lower().strip())
        if canonical and canonical != old_domain:
            changes.append(
                {
                    "source_id": str(row["id"]),
                    "title": row["title"],
                    "old_domain": old_domain,
                    "new_domain": canonical,
                }
            )

    if apply and changes:
        async with pool.acquire() as conn:
            for change in changes:
                await conn.execute(
                    """
                    UPDATE sources
                    SET metadata = jsonb_set(
                        COALESCE(metadata, '{}'::jsonb),
                        '{domain}',
                        to_jsonb($2::text)
                    )
                    WHERE id = $1
                    """,
                    uuid.UUID(change["source_id"]),
                    change["new_domain"],
                )

    return changes


async def assign_domains(pool, apply: bool = False) -> list[dict]:
    """Match 'none' domain sources to catalog, propose domain assignments.

    Uses fuzzy title matching against the catalog. Only proposes assignments
    with high confidence (threshold 0.6 for fuzzy_title_match).

    Returns:
        List of {source_id, title, proposed_domain, catalog_title, confidence}.
    """
    catalog = load_catalog()
    catalog_map = {book.title: book for book in catalog}

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, title, file_path, metadata
            FROM sources
            WHERE COALESCE(metadata->>'domain', 'none') = 'none'
               OR metadata->>'domain' IS NULL
            ORDER BY title
            """
        )

    proposals: list[dict] = []
    for row in rows:
        db_title = row["title"]
        best_match: dict | None = None  # noqa: F841
        matched_catalog_title: str | None = None

        for catalog_title, book in catalog_map.items():
            if fuzzy_title_match(catalog_title, db_title):
                best_match = book  # noqa: F841
                matched_catalog_title = catalog_title
                break

        if matched_catalog_title and catalog_map[matched_catalog_title].domain_id != "unclassified":
            book = catalog_map[matched_catalog_title]
            proposals.append(
                {
                    "source_id": str(row["id"]),
                    "db_title": db_title,
                    "catalog_title": matched_catalog_title,
                    "proposed_domain": book.domain_id,
                    "proposed_title": matched_catalog_title,
                }
            )

    if apply and proposals:
        async with pool.acquire() as conn:
            for p in proposals:
                # Update domain
                await conn.execute(
                    """
                    UPDATE sources
                    SET metadata = jsonb_set(
                        COALESCE(metadata, '{}'::jsonb),
                        '{domain}',
                        to_jsonb($2::text)
                    ),
                    title = $3
                    WHERE id = $1
                    """,
                    uuid.UUID(p["source_id"]),
                    p["proposed_domain"],
                    p["proposed_title"],
                )

    return proposals


async def find_duplicates(pool) -> list[dict]:
    """Detect duplicate sources by file_hash and fuzzy title matching.

    Groups:
    1. Same file_hash → definite duplicates
    2. Fuzzy title match (>0.85 Jaccard) → probable duplicates

    Returns:
        List of duplicate groups, each with source IDs, titles, chunk counts.
    """
    async with pool.acquire() as conn:
        # Group by file_hash first
        hash_groups = await conn.fetch(
            """
            SELECT file_hash, array_agg(id) AS ids, array_agg(title) AS titles
            FROM sources
            WHERE file_hash IS NOT NULL
            GROUP BY file_hash
            HAVING COUNT(*) > 1
            """
        )

        # Get all sources for title-based matching
        all_sources = await conn.fetch(
            """
            SELECT s.id, s.title, s.file_hash,
                   COALESCE(s.metadata->>'domain', 'none') AS domain,
                   COUNT(c.id) AS chunk_count
            FROM sources s
            LEFT JOIN chunks c ON c.source_id = s.id
            GROUP BY s.id, s.title, s.file_hash, domain
            ORDER BY s.title
            """
        )

    groups: list[dict] = []
    seen_ids: set[str] = set()

    # File hash duplicates
    for group in hash_groups:
        ids = [str(uid) for uid in group["ids"]]
        group_sources = [s for s in all_sources if str(s["id"]) in ids]

        groups.append(
            {
                "match_type": "file_hash",
                "file_hash": group["file_hash"],
                "sources": [
                    {
                        "id": str(s["id"]),
                        "title": s["title"],
                        "domain": s["domain"],
                        "chunk_count": s["chunk_count"],
                    }
                    for s in group_sources
                ],
            }
        )
        for uid in ids:
            seen_ids.add(uid)

    # Title-based duplicates (higher threshold for title-only matching)
    sources_list = [dict(s) for s in all_sources]
    for i, src_a in enumerate(sources_list):
        if str(src_a["id"]) in seen_ids:
            continue
        title_group = [src_a]

        for j in range(i + 1, len(sources_list)):
            src_b = sources_list[j]
            if str(src_b["id"]) in seen_ids:
                continue

            norm_a = normalize_for_match(src_a["title"])
            norm_b = normalize_for_match(src_b["title"])
            words_a = set(norm_a.split())
            words_b = set(norm_b.split())
            if not words_a or not words_b:
                continue

            jaccard = len(words_a & words_b) / len(words_a | words_b)
            if jaccard >= 0.85:
                title_group.append(src_b)
                seen_ids.add(str(src_b["id"]))

        if len(title_group) > 1:
            seen_ids.add(str(src_a["id"]))
            groups.append(
                {
                    "match_type": "fuzzy_title",
                    "sources": [
                        {
                            "id": str(s["id"]),
                            "title": s["title"],
                            "domain": s["domain"],
                            "chunk_count": s["chunk_count"],
                        }
                        for s in title_group
                    ],
                }
            )

    return groups


async def merge_duplicates(
    pool,
    keep_id: str,
    delete_id: str,
    apply: bool = False,
) -> dict:
    """Merge two duplicate sources: keep one, reassign chunks from the other.

    Pre-flight checks:
    1. Both source IDs exist
    2. Verify FK constraints (chunks, chunk_concepts, citations)
    3. Reassign chunks from delete_id to keep_id
    4. Delete the empty source record

    Args:
        pool: Database connection pool.
        keep_id: UUID of source to keep.
        delete_id: UUID of source to delete.
        apply: If False, just report what would happen.

    Returns:
        Dict with merge details and status.
    """
    keep_uuid = uuid.UUID(keep_id)
    delete_uuid = uuid.UUID(delete_id)

    async with pool.acquire() as conn:
        # Verify both sources exist
        keep_src = await conn.fetchrow("SELECT id, title FROM sources WHERE id = $1", keep_uuid)
        delete_src = await conn.fetchrow("SELECT id, title FROM sources WHERE id = $1", delete_uuid)

        if not keep_src:
            return {"error": f"Source to keep not found: {keep_id}"}
        if not delete_src:
            return {"error": f"Source to delete not found: {delete_id}"}

        # Count chunks to reassign
        delete_chunks = await conn.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE source_id = $1", delete_uuid
        )

        # Check citations referencing the source to delete
        citing_count = await conn.fetchval(
            "SELECT COUNT(*) FROM citations WHERE source_id = $1",
            delete_uuid,
        )

        result = {
            "keep": {"id": keep_id, "title": keep_src["title"]},
            "delete": {"id": delete_id, "title": delete_src["title"]},
            "chunks_to_reassign": delete_chunks,
            "citations_affected": citing_count,
            "applied": apply,
        }

        if apply:
            # Reassign chunks
            await conn.execute(
                "UPDATE chunks SET source_id = $1 WHERE source_id = $2",
                keep_uuid,
                delete_uuid,
            )

            # Reassign citations
            await conn.execute(
                "UPDATE citations SET source_id = $1 WHERE source_id = $2",
                keep_uuid,
                delete_uuid,
            )

            # Delete the empty source
            await conn.execute("DELETE FROM sources WHERE id = $1", delete_uuid)
            result["status"] = "merged"
        else:
            result["status"] = "dry_run"

    return result


async def report_untagged(pool) -> list[dict]:
    """Report all 'none' domain sources that don't match the Manning catalog.

    These are non-Manning books (or unrecognized Manning books) that need
    manual domain classification.

    Returns:
        List of {source_id, title, file_path, chunk_count} for untagged sources.
    """
    catalog = load_catalog()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT s.id, s.title, s.file_path,
                   COUNT(c.id) AS chunk_count
            FROM sources s
            LEFT JOIN chunks c ON c.source_id = s.id
            WHERE COALESCE(s.metadata->>'domain', 'none') = 'none'
               OR s.metadata->>'domain' IS NULL
            GROUP BY s.id, s.title, s.file_path
            ORDER BY s.title
            """
        )

    untagged: list[dict] = []
    for row in rows:
        # Check if it matches any catalog entry
        matched = False
        for book in catalog:
            if fuzzy_title_match(book.title, row["title"]):
                matched = True
                break

        if not matched:
            untagged.append(
                {
                    "source_id": str(row["id"]),
                    "title": row["title"],
                    "file_path": row["file_path"],
                    "chunk_count": row["chunk_count"],
                }
            )

    return untagged


async def auto_merge_duplicates(pool, apply: bool = False) -> list[dict]:
    """For each duplicate group, keep highest-chunk-count entry, merge others.

    Batch version of merge_duplicates() — processes all duplicate groups
    automatically based on the simple heuristic: more chunks = better ingestion.

    Safety: For fuzzy_title groups, only merges entries whose normalized titles
    match exactly. This prevents merging different editions or different books
    that share a long prefix (e.g. "Deep Learning with Python" vs
    "Deep Learning with Python, Second Edition").

    Args:
        pool: Database connection pool.
        apply: If False, just report what would happen.

    Returns:
        List of merge result dicts (one per merge operation).
        Entries with "skipped" key are false-positive groups that were not merged.
    """
    groups = await find_duplicates(pool)
    if not groups:
        return []

    results: list[dict] = []
    for group in groups:
        if group["match_type"] == "file_hash":
            # File hash match — always safe to merge
            sources = sorted(group["sources"], key=lambda s: s["chunk_count"], reverse=True)
            keep = sources[0]
            for delete in sources[1:]:
                result = await merge_duplicates(pool, keep["id"], delete["id"], apply=apply)
                result["match_type"] = group["match_type"]
                results.append(result)
        else:
            # Fuzzy title match — sub-group by normalized title for safety
            by_norm: dict[str, list[dict]] = {}
            for s in group["sources"]:
                norm = _normalize_strict(s["title"])
                by_norm.setdefault(norm, []).append(s)

            if len(by_norm) == 1:
                # All titles normalize identically — safe to merge
                sources = sorted(
                    group["sources"],
                    key=lambda s: s["chunk_count"],
                    reverse=True,
                )
                keep = sources[0]
                for delete in sources[1:]:
                    result = await merge_duplicates(pool, keep["id"], delete["id"], apply=apply)
                    result["match_type"] = group["match_type"]
                    results.append(result)
            else:
                # Mixed titles — merge within each sub-group, skip singletons
                for norm_title, sub_sources in by_norm.items():
                    if len(sub_sources) > 1:
                        sub_sorted = sorted(
                            sub_sources,
                            key=lambda s: s["chunk_count"],
                            reverse=True,
                        )
                        keep = sub_sorted[0]
                        for delete in sub_sorted[1:]:
                            result = await merge_duplicates(
                                pool, keep["id"], delete["id"], apply=apply
                            )
                            result["match_type"] = "fuzzy_title_subgroup"
                            results.append(result)

                # Report the skipped entries
                skipped_titles = [norm for norm, subs in by_norm.items() if len(subs) == 1]
                if skipped_titles:
                    results.append(
                        {
                            "skipped": True,
                            "reason": "mixed_titles_in_group",
                            "titles": [s["title"] for s in group["sources"]],
                            "distinct_normalized": list(by_norm.keys()),
                        }
                    )

    return results


def write_duplicate_report(groups: list[dict], output_path: Path | None = None) -> Path:
    """Write duplicate groups to a JSON report file."""
    if output_path is None:
        output_path = Path(__file__).parent / "duplicate_report.json"

    with open(output_path, "w") as f:
        json.dump(groups, f, indent=2, default=str)

    return output_path


def write_untagged_report(untagged: list[dict], output_path: Path | None = None) -> Path:
    """Write untagged sources to a JSON report file."""
    if output_path is None:
        output_path = Path(__file__).parent / "untagged_sources_report.json"

    with open(output_path, "w") as f:
        json.dump(untagged, f, indent=2, default=str)

    return output_path
