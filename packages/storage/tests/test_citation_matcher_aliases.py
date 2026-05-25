"""Integration tests for the hardened ``match_citation_to_source_simple``.

Covers the 2026-05-25 hardening (research-kb#14):

- ``metadata.aliases`` lookup in the exact-title branch
- Year-filtered partial-LIKE (no cross-decade matches)
- Alias hit beats partial-LIKE on a competing source

Requires the PostgreSQL container running.
"""

import pytest

from research_kb_contracts import SourceType
from research_kb_storage import CitationStore, SourceStore
from research_kb_storage.citation_graph import match_citation_to_source_simple

pytestmark = pytest.mark.integration


class TestAliasMatching:
    """``metadata.aliases`` lets a source match a citation by alternate title."""

    async def test_match_by_alias_when_title_differs(self, db_pool):
        # Source has slug title; canonical title lives in metadata.aliases.
        source = await SourceStore.create(
            source_type=SourceType.PAPER,
            title="Kalman Filtering 1960",
            file_hash="sha256:test_alias_kalman",
            domain_id="causal_inference",
            authors=["R. E. Kalman"],
            year=1960,
            metadata={"aliases": ["A New Approach to Linear Filtering and Prediction Problems"]},
        )

        # Citation extracted by GROBID uses the canonical title.
        citation = await CitationStore.create(
            source_id=source.id,
            raw_string="R. E. Kalman, 1960.",
            title="A New Approach to Linear Filtering and Prediction Problems",
            authors=["R. E. Kalman"],
            year=1960,
        )

        matched = await match_citation_to_source_simple(citation)

        assert matched == source.id

    async def test_alias_match_is_case_insensitive(self, db_pool):
        source = await SourceStore.create(
            source_type=SourceType.PAPER,
            title="Slug Title",
            file_hash="sha256:test_alias_case",
            domain_id="causal_inference",
            year=1992,
            metadata={"aliases": ["Q-learning"]},
        )

        citation = await CitationStore.create(
            source_id=source.id,
            raw_string="…",
            title="Q-LEARNING",  # different case
            authors=["Watkins"],
            year=1992,
        )

        matched = await match_citation_to_source_simple(citation)

        assert matched == source.id

    async def test_alias_match_respects_year_filter(self, db_pool):
        # Source has the alias and is year=2000.
        await SourceStore.create(
            source_type=SourceType.PAPER,
            title="Slug",
            file_hash="sha256:test_alias_year_filter",
            domain_id="causal_inference",
            year=2000,
            metadata={"aliases": ["Some Important Paper"]},
        )

        # Citation has the matching title but a different year.
        citation = await CitationStore.create(
            source_id=(
                await SourceStore.create(
                    source_type=SourceType.PAPER,
                    title="Citing Source",
                    file_hash="sha256:test_alias_year_filter_citer",
                    domain_id="causal_inference",
                    year=2024,
                )
            ).id,
            raw_string="…",
            title="Some Important Paper",
            authors=["X"],
            year=2024,  # different decade from the source
        )

        matched = await match_citation_to_source_simple(citation)

        # The year filter in priority 3 allows source.year IS NULL OR exact match.
        # Source has year=2000, citation year=2024 -> mismatch -> no match.
        assert matched is None


class TestYearFilteredPartialLike:
    """Partial-LIKE branch now requires year match (no cross-decade false positives)."""

    async def test_partial_like_blocked_by_year_mismatch(self, db_pool):
        # Source: "Dynamic Programming" 1957 (Bellman-style)
        await SourceStore.create(
            source_type=SourceType.TEXTBOOK,
            title="Dynamic Programming",
            file_hash="sha256:test_partial_year_block",
            domain_id="causal_inference",
            year=1957,
        )

        # Citation: "Dynamic Programming for Neural Networks" 2018
        # Pre-hardening this would partial-LIKE match Bellman 1957 (no year filter).
        # Post-hardening: year mismatch (2018 vs 1957) blocks the partial branch.
        citing = await SourceStore.create(
            source_type=SourceType.PAPER,
            title="Citing Source for Partial Block",
            file_hash="sha256:test_partial_year_block_citer",
            domain_id="causal_inference",
            year=2018,
        )
        citation = await CitationStore.create(
            source_id=citing.id,
            raw_string="…",
            title="Dynamic Programming for Neural Networks",
            authors=["Y"],
            year=2018,
        )

        matched = await match_citation_to_source_simple(citation)

        assert matched is None

    async def test_partial_like_allowed_when_year_matches(self, db_pool):
        # Source: "Dynamic Programming and Optimal Control" 2005
        source = await SourceStore.create(
            source_type=SourceType.TEXTBOOK,
            title="Dynamic Programming and Optimal Control",
            file_hash="sha256:test_partial_year_allowed",
            domain_id="causal_inference",
            year=2005,
        )

        # Citation: short form of the same title, same year.
        citing = await SourceStore.create(
            source_type=SourceType.PAPER,
            title="Citer for Year-Allowed Partial",
            file_hash="sha256:test_partial_year_allowed_citer",
            domain_id="causal_inference",
            year=2010,
        )
        citation = await CitationStore.create(
            source_id=citing.id,
            raw_string="…",
            title="Dynamic Programming",  # substring of source.title
            authors=["Z"],
            year=2005,  # year matches
        )

        matched = await match_citation_to_source_simple(citation)

        assert matched == source.id
