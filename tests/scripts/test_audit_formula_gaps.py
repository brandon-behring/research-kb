"""Tests for ``scripts/audit_formula_gaps.py``.

Unit tests cover the pure functions (tier classification, rendering).
Integration tests seed a small fixture in the test database, run
``fetch_gap_rows``, and assert the filter logic (min_chunks,
min_gap_pct, domain, tier) produces the expected rows.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit_formula_gaps.py"
_spec = importlib.util.spec_from_file_location("audit_formula_gaps", _SCRIPT_PATH)
audit = importlib.util.module_from_spec(_spec)
# Register in sys.modules BEFORE exec so dataclasses / type resolution works.
sys.modules["audit_formula_gaps"] = audit
_spec.loader.exec_module(audit)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Unit tests — pure functions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClassifyTier:
    def test_tier_1_math_domains(self):
        for d in [
            "mathematics",
            "linear_algebra",
            "optimization",
            "probability_theory",
            "analysis",
            "topology_geometry",
        ]:
            assert audit.classify_tier(d) == 1, d

    def test_tier_2_quant_domains(self):
        for d in [
            "machine_learning",
            "statistics",
            "deep_learning",
            "causal_inference",
            "reinforcement_learning",
            "time_series",
            "econometrics",
        ]:
            assert audit.classify_tier(d) == 2, d

    def test_tier_3_applied_domains(self):
        for d in ["physics", "engineering", "nlp", "rag_llm"]:
            assert audit.classify_tier(d) == 3, d

    def test_tier_4_prose_domains(self):
        for d in ["software_engineering", "business", "interview_prep"]:
            assert audit.classify_tier(d) == 4, d

    def test_unknown_domain_defaults_to_tier_3(self):
        """Conservative default — neither over- nor under-remediate."""
        assert audit.classify_tier("this_domain_does_not_exist") == 3


@pytest.mark.unit
class TestRendering:
    def _row(self, **kw):
        defaults = dict(
            source_id=str(uuid4()),
            title="Example Title",
            domain_id="mathematics",
            tier=1,
            total_chunks=100,
            latex_chunks=5,
            gap_chunks=50,
            gap_pct=50.0,
        )
        defaults.update(kw)
        return audit.GapRow(**defaults)

    def test_markdown_empty_rows(self):
        out = audit.render_markdown([])
        assert "No sources" in out

    def test_markdown_includes_title_and_counts(self):
        out = audit.render_markdown([self._row(title="My Book", gap_chunks=123)])
        assert "My Book" in out
        assert "123" in out
        assert "50.0%" in out

    def test_markdown_pipe_in_title_is_escaped(self):
        out = audit.render_markdown([self._row(title="Weird | Title")])
        assert "Weird \\| Title" in out

    def test_markdown_respects_limit(self):
        rows = [self._row(title=f"Row {i}") for i in range(5)]
        out = audit.render_markdown(rows, limit=2)
        assert "Row 0" in out
        assert "Row 1" in out
        assert "Row 2" not in out

    def test_tier_breakdown_aggregates_per_tier(self):
        rows = [
            self._row(tier=1, total_chunks=100, latex_chunks=2, gap_chunks=60),
            self._row(tier=1, total_chunks=200, latex_chunks=0, gap_chunks=100),
            self._row(tier=2, total_chunks=50, latex_chunks=5, gap_chunks=20),
        ]
        out = audit.render_tier_breakdown(rows)
        # Tier 1 aggregates: 2 sources, 300 chunks, 2 latex, 160 gaps
        assert "| 1 | 10% | 2 | 300 | 2 | 160 |" in out
        # Tier 2 aggregates: 1 source, 50 chunks, 5 latex, 20 gaps
        assert "| 2 | 20% | 1 | 50 | 5 | 20 |" in out

    def test_json_round_trip(self):
        rows = [self._row(title="a", gap_chunks=10), self._row(title="b", gap_chunks=20)]
        parsed = json.loads(audit.render_json(rows))
        assert parsed["total_sources"] == 2
        assert parsed["total_gap_chunks"] == 30
        assert {r["title"] for r in parsed["rows"]} == {"a", "b"}


# ---------------------------------------------------------------------------
# Integration tests — seeded DB, real query
# ---------------------------------------------------------------------------


_REQUIRED_DOMAINS = (
    "mathematics",
    "linear_algebra",
    "optimization",
    "probability_theory",
    "analysis",
    "topology_geometry",
    "machine_learning",
)


@pytest_asyncio.fixture
async def audit_pool(test_db):
    """Yield the pool with required domain FKs pre-seeded.

    The root-level ``test_db`` fixture TRUNCATEs sources CASCADE between
    tests, but only three domains (causal_inference / time_series / rag_llm)
    exist by default. The audit tier logic needs Tier 1 + Tier 2 domains,
    which we seed here with ``ON CONFLICT DO NOTHING`` so reruns are safe.
    """
    async with test_db.acquire() as conn:
        for d in _REQUIRED_DOMAINS:
            await conn.execute(
                "INSERT INTO domains (id, name) VALUES ($1, $2) " "ON CONFLICT DO NOTHING",
                d,
                d.replace("_", " ").title(),
            )
    yield test_db


async def _register_jsonb(conn) -> None:
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


async def _seed(conn, *, title: str, domain: str, chunks: list[dict]) -> str:
    """Seed one source + its chunks. Returns source id."""
    await _register_jsonb(conn)
    sid = uuid4()
    await conn.execute(
        """
        INSERT INTO sources (id, source_type, title, file_hash, metadata, domain_id)
        VALUES ($1, 'paper', $2, $3, $4, $5)
        """,
        sid,
        title,
        f"hash-audit-{sid}",
        {},
        domain,
    )
    for i, c in enumerate(chunks):
        await conn.execute(
            """
            INSERT INTO chunks (
                source_id, content, content_hash, page_start, page_end,
                embedding, metadata, domain_id
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            sid,
            c["content"],
            f"chunk-{sid}-{i}",
            i,
            i,
            None,
            {},
            domain,
        )
    return str(sid)


@pytest.mark.integration
class TestFetchGapRowsAgainstDB:
    """Integration tests with seeded rows in ``research_kb_test``."""

    async def test_basic_filter_and_sort_descending_by_gaps(self, audit_pool):
        async with audit_pool.acquire() as conn:
            # Source A: 100 chunks, 30 gaps (30%), mathematics (tier 1)
            await _seed(
                conn,
                title="Source A",
                domain="mathematics",
                chunks=(
                    [{"content": "formula-not-decoded pretend"} for _ in range(30)]
                    + [{"content": "regular chunk"} for _ in range(70)]
                ),
            )
            # Source B: 200 chunks, 100 gaps (50%), linear_algebra (tier 1)
            await _seed(
                conn,
                title="Source B",
                domain="linear_algebra",
                chunks=(
                    [{"content": "formula-not-decoded pretend"} for _ in range(100)]
                    + [{"content": "regular chunk"} for _ in range(100)]
                ),
            )
            # Source C: 50 chunks (below default min_chunks), should be filtered
            await _seed(
                conn,
                title="Source C",
                domain="mathematics",
                chunks=[{"content": "formula-not-decoded pretend"} for _ in range(50)],
            )

        rows = await audit.fetch_gap_rows(audit_pool, min_chunks=100, min_gap_pct=10.0)

        titles = [r.title for r in rows]
        assert "Source C" not in titles, "Sources below min_chunks should be excluded"
        # Sorted by gap_chunks desc: B (100 gaps) before A (30 gaps)
        idx_a, idx_b = titles.index("Source A"), titles.index("Source B")
        assert idx_b < idx_a, f"expected B before A by gap_chunks desc, got {titles}"

    async def test_tier_filter_applies_tier_floor(self, audit_pool):
        async with audit_pool.acquire() as conn:
            # mathematics, 10% gaps — passes tier-1 floor (10%)
            await _seed(
                conn,
                title="At tier-1 floor",
                domain="mathematics",
                chunks=(
                    [{"content": "formula-not-decoded"} for _ in range(12)]
                    + [{"content": "regular"} for _ in range(108)]
                ),
            )
            # mathematics, 5% gaps — below tier-1 floor
            await _seed(
                conn,
                title="Below tier-1 floor",
                domain="mathematics",
                chunks=(
                    [{"content": "formula-not-decoded"} for _ in range(6)]
                    + [{"content": "regular"} for _ in range(114)]
                ),
            )

        rows = await audit.fetch_gap_rows(audit_pool, min_chunks=100, min_gap_pct=0.0, tier=1)
        titles = [r.title for r in rows]
        assert "At tier-1 floor" in titles
        assert "Below tier-1 floor" not in titles

    async def test_domain_filter(self, audit_pool):
        async with audit_pool.acquire() as conn:
            await _seed(
                conn,
                title="Math source",
                domain="mathematics",
                chunks=(
                    [{"content": "formula-not-decoded"} for _ in range(20)]
                    + [{"content": "regular"} for _ in range(100)]
                ),
            )
            await _seed(
                conn,
                title="ML source",
                domain="machine_learning",
                chunks=(
                    [{"content": "formula-not-decoded"} for _ in range(20)]
                    + [{"content": "regular"} for _ in range(100)]
                ),
            )

        rows = await audit.fetch_gap_rows(
            audit_pool, min_chunks=100, min_gap_pct=10.0, domain="mathematics"
        )
        titles = [r.title for r in rows]
        assert titles == ["Math source"]

    async def test_counts_latex_and_gaps_separately(self, audit_pool):
        async with audit_pool.acquire() as conn:
            await _seed(
                conn,
                title="Mixed source",
                domain="mathematics",
                chunks=[
                    {"content": "formula-not-decoded"},  # gap
                    {"content": "$$x = y$$ has LaTeX"},  # latex
                    {"content": "$$ e^{i\\pi} = -1$$"},  # latex
                    {"content": "plain prose"},
                ]
                * 30,  # 120 chunks total: 30 gaps, 60 latex
            )

        rows = await audit.fetch_gap_rows(audit_pool, min_chunks=100, min_gap_pct=10.0)
        mine = next(r for r in rows if r.title == "Mixed source")
        assert mine.total_chunks == 120
        assert mine.gap_chunks == 30
        assert mine.latex_chunks == 60
        assert mine.gap_pct == 25.0
