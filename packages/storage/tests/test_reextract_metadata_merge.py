"""Regression test for the reextract_with_mineru metadata-merge bug.

Issue: #8 §C. The bug: ``scripts/reextract_with_mineru.py`` called
``json.dumps({...})`` with a ``$1::jsonb`` cast on a connection that already
had the asyncpg jsonb codec registered (``encoder=json.dumps``). The
codec re-encoded the str, Postgres stored the result as a JSONB scalar
string, and ``metadata || <string_scalar>`` coerced both sides to an
array, yielding ``[dict, "stringified_json"]``.

The fix: pass the dict directly and drop the ``::jsonb`` cast. The codec
serializes it once. Pattern matches ``SourceStore.update_metadata``
at ``packages/storage/src/research_kb_storage/source_store.py:385``.

This test exercises the exact asyncpg pattern the script uses on a
connection configured the same way. Pre-fix, ``jsonb_typeof`` was
``'array'`` (see manual reproduction in plan PR). Post-fix, it is
``'object'``.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

pytestmark = pytest.mark.integration


async def test_mineru_metadata_merge_preserves_dict(db_pool) -> None:
    """After merge, ``sources.metadata`` remains a JSONB object (not array).

    Mirrors ``scripts/reextract_with_mineru.py`` connection setup (jsonb
    codec at lines 288-293) and the fixed UPDATE pattern.
    """
    source_id = uuid4()

    async with db_pool.acquire() as conn:
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )

        await conn.execute(
            """
            INSERT INTO sources (
                id, source_type, title, file_hash, metadata, domain_id
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            source_id,
            "paper",
            "test-fixture",
            f"metadata-merge-hash-{source_id}",
            {"orig_key": "orig_value"},
            "causal_inference",
        )

        # Fixed pattern: dict directly, no ::jsonb cast.
        # Pre-fix pattern used json.dumps({...}) + $1::jsonb, producing
        # [{dict}, "stringified_json"] via double-encoding.
        await conn.execute(
            "UPDATE sources SET metadata = metadata || $1 WHERE id = $2",
            {
                "extraction_method": "mineru",
                "total_pages": 100,
                "has_equations": True,
            },
            source_id,
        )

        row = await conn.fetchrow(
            "SELECT jsonb_typeof(metadata) AS t, metadata FROM sources WHERE id = $1",
            source_id,
        )

    assert row["t"] == "object", (
        f"metadata must be a JSONB object after merge, got {row['t']!r}. "
        "Double-encoding regression from issue #8 §C has returned."
    )
    assert row["metadata"]["orig_key"] == "orig_value"
    assert row["metadata"]["extraction_method"] == "mineru"
    assert row["metadata"]["total_pages"] == 100
    assert row["metadata"]["has_equations"] is True


async def test_mineru_metadata_merge_broken_pattern_produces_array(db_pool) -> None:
    """Negative guard: the broken pattern (json.dumps + ::jsonb) yields an array.

    Documents the bug's failure mode. If this test starts returning ``'object'``,
    either (a) asyncpg changed its codec behavior, or (b) the jsonb codec is no
    longer registered and this pattern now works by accident. Either way, the
    fixed test above is the authoritative regression guard.
    """
    source_id = uuid4()

    async with db_pool.acquire() as conn:
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )

        await conn.execute(
            """
            INSERT INTO sources (
                id, source_type, title, file_hash, metadata, domain_id
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            source_id,
            "paper",
            "test-fixture-broken-pattern",
            f"metadata-merge-broken-hash-{source_id}",
            {"orig_key": "orig_value"},
            "causal_inference",
        )

        # Intentionally the broken pre-fix pattern.
        await conn.execute(
            "UPDATE sources SET metadata = metadata || $1::jsonb WHERE id = $2",
            json.dumps({"extraction_method": "mineru"}),
            source_id,
        )

        row = await conn.fetchrow(
            "SELECT jsonb_typeof(metadata) AS t FROM sources WHERE id = $1",
            source_id,
        )

    assert row["t"] == "array", (
        "Broken pattern should still reproduce the original bug; if this asserts "
        "'object' now, asyncpg's codec semantics changed and the fix may be moot."
    )
