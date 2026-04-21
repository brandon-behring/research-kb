"""Static regression guard for the reextract_with_mineru metadata-merge bug.

Issue: #8 §C. A previous fix comment claimed "fixed 2026-04-11" while the
code still produced the bug. This test is a static check that the buggy
pattern is not present in ``scripts/reextract_with_mineru.py``:

1. ``json.dumps(`` must NOT appear inside the metadata UPDATE block — the
   asyncpg jsonb codec (``encoder=json.dumps``) is already registered on
   the connection, so passing a dict directly is the correct contract.
2. ``$1::jsonb`` must NOT appear in the same block — the cast combined
   with a double-encoded str is what coerced the merge to an array.

Paired with ``test_reextract_metadata_merge.py`` (integration test that
exercises the runtime behavior). Together they guard against someone
accidentally reintroducing the bug by re-adding ``json.dumps``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "reextract_with_mineru.py"


def _extract_metadata_update_block(text: str) -> str:
    """Return the substring around the sources.metadata UPDATE statement."""
    match = re.search(
        r"UPDATE\s+sources\s+SET\s+metadata[\s\S]{0,600}?\)",
        text,
        re.IGNORECASE,
    )
    if match is None:
        raise AssertionError(
            "Could not locate 'UPDATE sources SET metadata ...' block in "
            f"{SCRIPT}. Structure changed; update this test's locator."
        )
    return match.group(0)


def test_script_exists() -> None:
    """Sanity check: the script exists at the expected path."""
    assert SCRIPT.exists(), f"Expected script at {SCRIPT}"


def test_metadata_update_does_not_use_json_dumps() -> None:
    """The metadata UPDATE block must not call ``json.dumps``.

    If this fails, the double-encoding bug has been reintroduced. See
    ``test_reextract_metadata_merge.py`` for the runtime-level assertion.
    """
    text = SCRIPT.read_text()
    block = _extract_metadata_update_block(text)

    assert "json.dumps" not in block, (
        "json.dumps found in the sources.metadata UPDATE block. The asyncpg "
        "jsonb codec already runs json.dumps; calling it explicitly causes "
        "double-encoding and the [dict, stringified_json] array bug from "
        "issue #8 §C. Pass the dict directly."
    )


def test_metadata_update_does_not_use_explicit_jsonb_cast() -> None:
    """The metadata UPDATE block must not cast ``$1`` to ``jsonb``.

    With the jsonb codec registered and a dict passed directly, the
    parameter already arrives as jsonb. The explicit ``::jsonb`` cast
    combined with a double-encoded str was what produced the string
    scalar that triggered array coercion.
    """
    text = SCRIPT.read_text()
    block = _extract_metadata_update_block(text)

    assert "$1::jsonb" not in block, (
        "'$1::jsonb' cast found in the sources.metadata UPDATE. Drop the "
        "explicit cast; the jsonb codec handles serialization. See issue "
        "#8 §C."
    )
