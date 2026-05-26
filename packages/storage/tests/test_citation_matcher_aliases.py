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


class TestLikeMetacharacterSafety:
    """Partial-substring branch is metacharacter-safe (regression for 2026-05-25 code review)."""

    async def test_literal_percent_in_citation_title_does_not_wildcard_match(self, db_pool):
        # Source title does NOT actually contain "100% nlp" as a substring,
        # but its first 3 chars are "100" and it ends with " nlp". Under the
        # buggy LIKE-based matcher, the pattern '%100% nlp%' would match
        # because the literal '%' in the citation title acts as a wildcard
        # between "100" and " nlp" in the LIKE pattern.
        source = await SourceStore.create(
            source_type=SourceType.PAPER,
            title="100 papers on building scalable nlp systems",
            file_hash="sha256:test_pct_wildcard_source",
            domain_id="causal_inference",
            year=2020,
        )

        # Citation with a literal '%' in the title.
        citing = await SourceStore.create(
            source_type=SourceType.PAPER,
            title="Citer for percent wildcard test",
            file_hash="sha256:test_pct_wildcard_citer",
            domain_id="causal_inference",
            year=2020,
        )
        citation = await CitationStore.create(
            source_id=citing.id,
            raw_string="…",
            title="100% NLP",
            authors=["Author"],
            year=2020,
        )

        matched = await match_citation_to_source_simple(citation)

        # Post-fix: '%' is treated as a literal character, not a wildcard.
        # Neither substring contains the other → no match.
        assert matched is None, (
            f"Expected None (no real substring overlap), got {matched} — "
            f"if this fails, the partial-substring branch may have "
            f"regressed to LIKE semantics where '%' is wildcard."
        )

    async def test_literal_underscore_in_citation_title_does_not_wildcard_match(self, db_pool):
        # '_' is the single-char LIKE wildcard. Source and citation differ
        # only in the position where '_' sits in the citation, but if '_'
        # were treated as a wildcard, the citation would match the source.
        source = await SourceStore.create(
            source_type=SourceType.PAPER,
            title="abc xyz def",
            file_hash="sha256:test_underscore_wildcard_source",
            domain_id="causal_inference",
            year=2021,
        )

        citing = await SourceStore.create(
            source_type=SourceType.PAPER,
            title="Citer for underscore wildcard test",
            file_hash="sha256:test_underscore_wildcard_citer",
            domain_id="causal_inference",
            year=2021,
        )
        citation = await CitationStore.create(
            source_id=citing.id,
            raw_string="…",
            title="abc_xyz_def",  # underscore-separated (no actual substring overlap)
            authors=["Author"],
            year=2021,
        )

        matched = await match_citation_to_source_simple(citation)

        # Post-fix: '_' is literal. Citation 'abc_xyz_def' isn't a substring
        # of source 'abc xyz def' (different chars at positions 3 and 7),
        # and source isn't a substring of citation either → no match.
        assert matched is None


class TestMalformedAliasesGuard:
    """Malformed `metadata.aliases` (non-array) doesn't poison matching for other sources.

    Regression for 2026-05-25 code review: without the
    `jsonb_typeof(metadata->'aliases') = 'array'` guard, a single source
    with `aliases: "foo"` (a string, not a list) raises Postgres
    invalid_parameter_value when the EXISTS subquery evaluates
    jsonb_array_elements_text on it. That error then propagates, and the
    matcher can't resolve ANY citation in that SELECT scope.
    """

    async def test_malformed_aliases_string_does_not_break_other_matches(self, db_pool):
        # Source A: malformed aliases (string instead of array). Loaded
        # via SourceStore.create which permits arbitrary JSONB metadata.
        await SourceStore.create(
            source_type=SourceType.PAPER,
            title="Source with bad aliases shape",
            file_hash="sha256:test_malformed_aliases_a",
            domain_id="causal_inference",
            year=2019,
            metadata={"aliases": "this-is-a-string-not-an-array"},
        )

        # Source B: clean metadata + canonical title for the citation
        # to resolve to.
        source_b = await SourceStore.create(
            source_type=SourceType.PAPER,
            title="Clean Target Title",
            file_hash="sha256:test_malformed_aliases_b",
            domain_id="causal_inference",
            year=2019,
        )

        # Citation that should resolve to source_b via exact-title match
        # in priority 3 — but only if the EXISTS subquery doesn't blow up
        # on source A's malformed aliases.
        citing = await SourceStore.create(
            source_type=SourceType.PAPER,
            title="Citer for malformed-aliases test",
            file_hash="sha256:test_malformed_aliases_citer",
            domain_id="causal_inference",
            year=2019,
        )
        citation = await CitationStore.create(
            source_id=citing.id,
            raw_string="…",
            title="Clean Target Title",
            authors=["Author"],
            year=2019,
        )

        matched = await match_citation_to_source_simple(citation)

        # Post-fix: jsonb_typeof guard short-circuits source A's row,
        # source B matches via exact title.
        assert matched == source_b.id

    async def test_malformed_aliases_object_does_not_break_other_matches(self, db_pool):
        # Variation: aliases is a JSON object instead of an array.
        await SourceStore.create(
            source_type=SourceType.PAPER,
            title="Source with object-shape aliases",
            file_hash="sha256:test_malformed_aliases_obj_a",
            domain_id="causal_inference",
            year=2020,
            metadata={"aliases": {"some_key": "some_value"}},
        )

        source_b = await SourceStore.create(
            source_type=SourceType.PAPER,
            title="Another Clean Target",
            file_hash="sha256:test_malformed_aliases_obj_b",
            domain_id="causal_inference",
            year=2020,
        )

        citing = await SourceStore.create(
            source_type=SourceType.PAPER,
            title="Citer for object-aliases test",
            file_hash="sha256:test_malformed_aliases_obj_citer",
            domain_id="causal_inference",
            year=2020,
        )
        citation = await CitationStore.create(
            source_id=citing.id,
            raw_string="…",
            title="Another Clean Target",
            authors=["Author"],
            year=2020,
        )

        matched = await match_citation_to_source_simple(citation)
        assert matched == source_b.id
