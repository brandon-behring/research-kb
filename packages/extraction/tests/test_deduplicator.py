"""Tests for concept deduplication."""

import pytest
from uuid import uuid4

from research_kb_extraction.deduplicator import Deduplicator, ABBREVIATION_MAP
from research_kb_extraction.models import ExtractedConcept

pytestmark = pytest.mark.unit


class TestCanonicalName:
    """Tests for canonical name normalization."""

    def test_abbreviation_expansion(self):
        """Test abbreviation expansion."""
        dedup = Deduplicator(domain_id="causal_inference")

        assert dedup.to_canonical_name("IV") == "instrumental variables"
        assert dedup.to_canonical_name("iv") == "instrumental variables"
        # Hyphens in abbreviation expansions normalize to spaces
        assert dedup.to_canonical_name("DiD") == "difference in differences"
        assert dedup.to_canonical_name("DID") == "difference in differences"
        assert dedup.to_canonical_name("2SLS") == "two stage least squares"
        assert dedup.to_canonical_name("ATE") == "average treatment effect"
        assert dedup.to_canonical_name("LATE") == "local average treatment effect"
        assert dedup.to_canonical_name("DML") == "double machine learning"

    def test_whitespace_normalization(self):
        """Test whitespace handling."""
        dedup = Deduplicator(domain_id="causal_inference")

        assert dedup.to_canonical_name("  instrumental   variables  ") == "instrumental variables"
        assert dedup.to_canonical_name("difference\tin\tdifferences") == "difference in differences"

    def test_parenthetical_removal(self):
        """Test parenthetical content removal."""
        dedup = Deduplicator(domain_id="causal_inference")

        result = dedup.to_canonical_name("instrumental variables (IV)")
        assert result == "instrumental variables"

        result = dedup.to_canonical_name("two-stage least squares (2SLS)")
        assert result == "two stage least squares"

    def test_special_character_removal(self):
        """Test special character handling."""
        dedup = Deduplicator(domain_id="causal_inference")

        # Hyphens now normalize to spaces
        assert dedup.to_canonical_name("difference-in-differences") == "difference in differences"

        # Other special chars removed
        result = dedup.to_canonical_name("concept: name!")
        assert ":" not in result
        assert "!" not in result

    def test_passthrough_unknown(self):
        """Test unknown terms pass through normalized."""
        dedup = Deduplicator(domain_id="causal_inference")

        assert dedup.to_canonical_name("propensity score matching") == "propensity score matching"
        assert dedup.to_canonical_name("Causal Forest") == "causal forest"

    def test_hyphen_normalization(self):
        """Test hyphens normalize to spaces (Phase AF-2 prevention)."""
        dedup = Deduplicator(domain_id="causal_inference")

        assert dedup.to_canonical_name("cross-validation") == "cross validation"
        assert dedup.to_canonical_name("cross validation") == "cross validation"
        assert dedup.to_canonical_name("k-fold cross-validation") == "k fold cross validation"
        assert dedup.to_canonical_name("out-of-sample") == "out of sample"

    def test_article_stripping(self):
        """Test leading articles are stripped (Phase AF-2 prevention)."""
        dedup = Deduplicator(domain_id="causal_inference")

        assert dedup.to_canonical_name("the central limit theorem") == "central limit theorem"
        assert dedup.to_canonical_name("The Central Limit Theorem") == "central limit theorem"
        assert dedup.to_canonical_name("a priori assumption") == "priori assumption"
        assert dedup.to_canonical_name("an estimator") == "estimator"

    def test_article_not_stripped_mid_word(self):
        """Test articles only stripped at word boundary, not mid-word."""
        dedup = Deduplicator(domain_id="causal_inference")

        # "the" inside a word should not be stripped
        assert dedup.to_canonical_name("theorem") == "theorem"
        assert dedup.to_canonical_name("analysis") == "analysis"
        assert dedup.to_canonical_name("another method") == "another method"

    def test_hyphen_and_article_combined(self):
        """Test combined hyphen normalization + article stripping."""
        dedup = Deduplicator(domain_id="causal_inference")

        assert dedup.to_canonical_name("the difference-in-differences") == "difference in differences"
        assert dedup.to_canonical_name("a well-known method") == "well known method"

    def test_hyphen_space_equivalence(self):
        """Test that hyphenated and space forms produce identical canonical names."""
        dedup = Deduplicator(domain_id="causal_inference")

        pairs = [
            ("cross-validation", "cross validation"),
            ("difference-in-differences", "difference in differences"),
            ("out-of-sample", "out of sample"),
            ("pre-treatment", "pre treatment"),
            ("double-robust", "double robust"),
        ]
        for hyphenated, spaced in pairs:
            assert dedup.to_canonical_name(hyphenated) == dedup.to_canonical_name(spaced), (
                f"Mismatch: {hyphenated!r} vs {spaced!r}"
            )

    def test_article_equivalence(self):
        """Test that article-prefixed and bare forms produce identical canonical names."""
        dedup = Deduplicator(domain_id="causal_inference")

        pairs = [
            ("the central limit theorem", "central limit theorem"),
            ("the law of large numbers", "law of large numbers"),
            ("a causal effect", "causal effect"),
            ("an instrumental variable", "instrumental variable"),
        ]
        for with_article, bare in pairs:
            assert dedup.to_canonical_name(with_article) == dedup.to_canonical_name(bare), (
                f"Mismatch: {with_article!r} vs {bare!r}"
            )


class TestKnownConceptManagement:
    """Tests for known concept registration."""

    def test_register_concept(self):
        """Test registering a known concept."""
        dedup = Deduplicator(domain_id="causal_inference")
        concept_id = uuid4()

        dedup.register_known_concept("instrumental variables", concept_id)

        assert dedup.find_existing_concept("instrumental variables") == concept_id
        assert dedup.find_existing_concept("IV") == concept_id  # Via abbreviation

    def test_load_multiple_concepts(self):
        """Test loading multiple concepts."""
        dedup = Deduplicator(domain_id="causal_inference")
        concepts = {
            "instrumental variables": uuid4(),
            "difference-in-differences": uuid4(),
            "propensity score matching": uuid4(),
        }

        dedup.load_known_concepts(concepts)

        for name, id in concepts.items():
            assert dedup.find_existing_concept(name) == id

    def test_find_nonexistent(self):
        """Test finding non-existent concept."""
        dedup = Deduplicator(domain_id="causal_inference")

        assert dedup.find_existing_concept("unknown concept") is None

    def test_register_with_hyphen_finds_via_space(self):
        """Test that registering with hyphens allows lookup via space form."""
        dedup = Deduplicator(domain_id="causal_inference")
        concept_id = uuid4()

        dedup.register_known_concept("cross-validation", concept_id)

        assert dedup.find_existing_concept("cross validation") == concept_id
        assert dedup.find_existing_concept("cross-validation") == concept_id

    def test_register_with_article_finds_without(self):
        """Test that registering with article allows lookup without it."""
        dedup = Deduplicator(domain_id="causal_inference")
        concept_id = uuid4()

        dedup.register_known_concept("the central limit theorem", concept_id)

        assert dedup.find_existing_concept("central limit theorem") == concept_id
        assert dedup.find_existing_concept("the central limit theorem") == concept_id


class TestDeduplication:
    """Tests for batch deduplication."""

    async def test_deduplicate_unique_concepts(self):
        """Test deduplicating unique concepts."""
        dedup = Deduplicator(domain_id="causal_inference")

        concepts = [
            ExtractedConcept(name="instrumental variables", concept_type="method"),
            ExtractedConcept(name="difference-in-differences", concept_type="method"),
            ExtractedConcept(name="parallel trends", concept_type="assumption"),
        ]

        results = await dedup.deduplicate_batch(concepts)

        assert len(results) == 3
        assert all(r.is_new for r in results)

    async def test_deduplicate_with_abbreviations(self):
        """Test deduplicating concepts with abbreviations."""
        dedup = Deduplicator(domain_id="causal_inference")

        concepts = [
            ExtractedConcept(name="instrumental variables", concept_type="method"),
            ExtractedConcept(name="IV", concept_type="method"),  # Same as above
            ExtractedConcept(name="2SLS", concept_type="method"),  # Also IV-related
        ]

        results = await dedup.deduplicate_batch(concepts)

        # IV should match instrumental variables
        iv_results = [r for r in results if r.matched_canonical_name == "instrumental variables"]
        assert len(iv_results) == 2  # Original + IV abbreviation

        # First one is new, second matches first
        new_count = sum(1 for r in results if r.is_new)
        # instrumental variables + two-stage least squares are new
        # IV matches instrumental variables
        assert new_count == 2

    async def test_deduplicate_against_known(self):
        """Test deduplicating against pre-registered concepts."""
        dedup = Deduplicator(domain_id="causal_inference")
        existing_id = uuid4()
        dedup.register_known_concept("instrumental variables", existing_id)

        concepts = [
            ExtractedConcept(name="IV", concept_type="method"),
        ]

        results = await dedup.deduplicate_batch(concepts)

        assert len(results) == 1
        assert not results[0].is_new
        assert results[0].matched_concept_id == existing_id

    async def test_deduplicate_empty(self):
        """Test deduplicating empty list."""
        dedup = Deduplicator(domain_id="causal_inference")

        results = await dedup.deduplicate_batch([])

        assert results == []

    async def test_deduplicate_hyphen_space_variants(self):
        """Test that hyphen and space variants deduplicate to same concept."""
        dedup = Deduplicator(domain_id="causal_inference")

        concepts = [
            ExtractedConcept(name="cross-validation", concept_type="method"),
            ExtractedConcept(name="cross validation", concept_type="method"),
        ]

        results = await dedup.deduplicate_batch(concepts)

        assert len(results) == 2
        new_count = sum(1 for r in results if r.is_new)
        assert new_count == 1  # Second is a duplicate

    async def test_deduplicate_article_variants(self):
        """Test that article-prefixed and bare forms deduplicate."""
        dedup = Deduplicator(domain_id="causal_inference")

        concepts = [
            ExtractedConcept(name="central limit theorem", concept_type="theorem"),
            ExtractedConcept(name="the central limit theorem", concept_type="theorem"),
        ]

        results = await dedup.deduplicate_batch(concepts)

        assert len(results) == 2
        new_count = sum(1 for r in results if r.is_new)
        assert new_count == 1  # Second is a duplicate


class TestSimilarity:
    """Tests for similarity computation."""

    async def test_exact_canonical_match(self):
        """Test exact canonical name match."""
        dedup = Deduplicator(domain_id="causal_inference")

        c1 = ExtractedConcept(name="instrumental variables", concept_type="method")
        c2 = ExtractedConcept(name="Instrumental Variables", concept_type="method")

        similarity = await dedup.compute_similarity(c1, c2)
        assert similarity == 1.0

    async def test_alias_match(self):
        """Test alias matching gives high similarity."""
        dedup = Deduplicator(domain_id="causal_inference")

        c1 = ExtractedConcept(
            name="instrumental variables",
            concept_type="method",
            aliases=["IV", "2SLS"],
        )
        c2 = ExtractedConcept(name="IV", concept_type="method")

        similarity = await dedup.compute_similarity(c1, c2)
        assert similarity >= 0.95

    async def test_different_concepts(self):
        """Test different concepts have low similarity."""
        dedup = Deduplicator(domain_id="causal_inference")

        c1 = ExtractedConcept(name="instrumental variables", concept_type="method")
        c2 = ExtractedConcept(name="propensity score matching", concept_type="method")

        similarity = await dedup.compute_similarity(c1, c2)
        assert similarity < 0.5

    async def test_hyphen_space_similarity(self):
        """Test that hyphen and space variants are identical."""
        dedup = Deduplicator(domain_id="causal_inference")

        c1 = ExtractedConcept(name="cross-validation", concept_type="method")
        c2 = ExtractedConcept(name="cross validation", concept_type="method")

        similarity = await dedup.compute_similarity(c1, c2)
        assert similarity == 1.0


class TestAliases:
    """Tests for alias generation."""

    def test_get_all_aliases(self):
        """Test getting all aliases including abbreviations."""
        dedup = Deduplicator(domain_id="causal_inference")

        concept = ExtractedConcept(
            name="instrumental variables",
            concept_type="method",
            aliases=["IV estimation"],
        )

        aliases = dedup.get_all_aliases(concept)

        assert "instrumental variables" in aliases
        assert "iv estimation" in aliases
        # Check reverse abbreviation lookup
        assert "iv" in aliases  # From ABBREVIATION_MAP


class TestAbbreviationMap:
    """Tests for the abbreviation map."""

    def test_common_abbreviations_exist(self):
        """Test common abbreviations are in the map."""
        expected = {
            "iv": "instrumental variables",
            "did": "difference-in-differences",
            "ate": "average treatment effect",
            "rdd": "regression discontinuity design",
            "dml": "double machine learning",
        }

        for abbrev, expansion in expected.items():
            assert abbrev in ABBREVIATION_MAP
            assert ABBREVIATION_MAP[abbrev] == expansion

    def test_abbreviation_map_lowercase(self):
        """Test all abbreviations are lowercase."""
        for abbrev in ABBREVIATION_MAP.keys():
            assert abbrev == abbrev.lower()
