"""Integration tests for seed concept validation.

Tests the complete validation pipeline:
- Seed concept loading from YAML
- Concept matching strategies
- Metric calculations
- Report generation
"""

import sys
from pathlib import Path

import pytest
import yaml

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "extraction" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "contracts" / "src"))

from validate_seed_concepts import (
    ConceptMatcher,
    ReportGenerator,
    SeedConceptLoader,
    ValidationMetrics,
    ValidationReport,
)
from research_kb_extraction import Deduplicator


SEED_FILE = Path(__file__).parent.parent.parent / "fixtures/concepts/seed_concepts.yaml"


# ============================================================
# YAML STRUCTURE TESTS
# ============================================================


def test_seed_concepts_file_exists():
    """Test that seed concepts YAML file exists."""
    assert SEED_FILE.exists(), f"Seed concepts file not found: {SEED_FILE}"


def test_seed_yaml_valid_structure():
    """Test that seed YAML has valid structure."""
    with open(SEED_FILE) as f:
        data = yaml.safe_load(f)

    # Check metadata
    assert "metadata" in data
    assert "version" in data["metadata"]
    assert "total_concepts" in data["metadata"]

    # Check concepts sections
    for section in ["methods", "assumptions", "problems", "definitions"]:
        assert section in data, f"Missing section: {section}"
        assert isinstance(data[section], list), f"{section} should be a list"

    # Check metrics
    assert "metrics" in data
    assert "recall_targets" in data["metrics"]
    assert "precision_targets" in data["metrics"]


def test_seed_concepts_count():
    """Test that seed concepts match expected count."""
    loader = SeedConceptLoader(SEED_FILE)
    concepts = loader.load()

    # Should be 48 concepts total (15+10+6+7+10) - updated in Phase 2 Step 9
    assert len(concepts) == 48, f"Expected 48 concepts, got {len(concepts)}"

    # Check distribution
    methods = loader.get_by_type("method")
    assumptions = loader.get_by_type("assumption")
    problems = loader.get_by_type("problem")
    definitions = loader.get_by_type("definition")
    theorems = loader.get_by_type("theorem")

    assert len(methods) == 15, f"Expected 15 methods, got {len(methods)}"
    assert len(assumptions) == 10, f"Expected 10 assumptions, got {len(assumptions)}"
    assert len(problems) == 6, f"Expected 6 problems, got {len(problems)}"
    assert len(definitions) == 7, f"Expected 7 definitions, got {len(definitions)}"
    assert len(theorems) == 10, f"Expected 10 theorems, got {len(theorems)}"


def test_seed_concepts_have_required_fields():
    """Test that all seed concepts have required fields."""
    loader = SeedConceptLoader(SEED_FILE)
    concepts = loader.load()

    for concept in concepts:
        # Required fields
        assert concept.name, f"Concept missing name: {concept}"
        assert concept.canonical_name, f"Concept missing canonical_name: {concept.name}"
        assert concept.concept_type, f"Concept missing type: {concept.name}"

        # Type should be valid
        assert concept.concept_type in [
            "method",
            "assumption",
            "problem",
            "definition",
            "theorem",
        ], f"Invalid type for {concept.name}: {concept.concept_type}"

        # Should have definition
        assert concept.definition, f"Concept missing definition: {concept.name}"


# ============================================================
# ABBREVIATION COVERAGE TESTS
# ============================================================


def test_abbreviation_map_covers_seed_aliases():
    """Test that deduplicator covers common abbreviations in seed concepts."""
    loader = SeedConceptLoader(SEED_FILE)
    concepts = loader.load()

    dedup = Deduplicator()

    missing_abbrevs = []
    for concept in concepts:
        for alias in concept.aliases:
            # Check if abbreviation (short and uppercase/mixed case)
            if len(alias) <= 5 and not alias.islower():
                # Try to canonicalize
                canonical = dedup.to_canonical_name(alias)

                # Check if it expanded (not just lowercased)
                if canonical == alias.lower():
                    missing_abbrevs.append((concept.name, alias))

    # Some abbreviations may not be in ABBREVIATION_MAP yet (acceptable)
    # Just log them, don't fail
    if missing_abbrevs:
        print(f"\nAbbreviations not in ABBREVIATION_MAP: {missing_abbrevs}")

    # With 48 seed concepts (expanded in Phase 2 Step 9), more abbreviations expected
    # Allow up to 15 missing (many theorems have standard abbreviations like CLT, LLN)
    assert len(missing_abbrevs) < 15, f"Too many missing abbreviations: {len(missing_abbrevs)}"


# ============================================================
# MATCHING TESTS
# ============================================================


def test_exact_match_strategy():
    """Test exact canonical name matching."""
    from research_kb_contracts import Concept, ConceptType
    from uuid import uuid4

    # Create mock extracted concepts
    extracted = [
        Concept(
            id=uuid4(),
            name="Instrumental Variables",
            canonical_name="instrumental variables",
            aliases=["IV"],
            concept_type=ConceptType.METHOD,
            confidence_score=0.9,
            validated=False,
            metadata={},
            created_at="2025-12-02T00:00:00",
        )
    ]

    # Load seed concepts
    loader = SeedConceptLoader(SEED_FILE)
    seeds = loader.load()
    iv_seed = next(s for s in seeds if s.canonical_name == "instrumental variables")

    # Test matching
    matcher = ConceptMatcher(extracted, confidence_threshold=0.7)
    match = matcher.match_concept(iv_seed)

    assert match.found, "IV should be matched"
    assert match.strategy == "exact_canonical"
    assert match.score == 1.0


def test_fuzzy_match_strategy():
    """Test fuzzy alias matching."""
    from research_kb_contracts import Concept, ConceptType
    from uuid import uuid4

    # Create mock with DIFFERENT canonical name but matching alias
    # This tests that fuzzy alias matching works when exact canonical doesn't match
    extracted = [
        Concept(
            id=uuid4(),
            name="DiD Estimator",  # Different name
            canonical_name="did estimator",  # Different canonical (won't match exactly)
            aliases=["DiD", "DD"],  # But has matching alias
            concept_type=ConceptType.METHOD,
            confidence_score=0.85,
            validated=False,
            metadata={},
            created_at="2025-12-02T00:00:00",
        )
    ]

    loader = SeedConceptLoader(SEED_FILE)
    seeds = loader.load()
    did_seed = next(s for s in seeds if "DiD" in s.aliases)

    matcher = ConceptMatcher(extracted, confidence_threshold=0.7)
    match = matcher.match_concept(did_seed)

    assert match.found, "DiD should be matched by alias"
    assert match.strategy == "fuzzy_alias", f"Expected fuzzy_alias but got {match.strategy}"


# ============================================================
# METRICS TESTS
# ============================================================


def test_recall_calculation():
    """Test recall metrics calculation."""
    from research_kb_contracts import Concept, ConceptType
    from uuid import uuid4
    from validate_seed_concepts import ConceptMatch, SeedConcept

    # Create mock data
    seed1 = SeedConcept(
        name="instrumental variables",
        canonical_name="instrumental variables",
        aliases=["IV"],
        concept_type="method",
        difficulty="easy",
    )
    seed2 = SeedConcept(
        name="unconfoundedness",
        canonical_name="unconfoundedness",
        aliases=["CIA"],
        concept_type="assumption",
        difficulty="easy",
    )
    seed3 = SeedConcept(
        name="endogeneity",
        canonical_name="endogeneity",
        aliases=[],
        concept_type="problem",
        difficulty="easy",
    )

    extracted1 = Concept(
        id=uuid4(),
        name="IV",
        canonical_name="instrumental variables",
        aliases=[],
        concept_type=ConceptType.METHOD,
        confidence_score=0.9,
        validated=False,
        metadata={},
        created_at="2025-12-02T00:00:00",
    )

    # Create matches (2 found, 1 missing)
    matches = {
        "instrumental variables": ConceptMatch(
            seed=seed1, extracted=extracted1, strategy="exact", score=1.0, found=True
        ),
        "unconfoundedness": ConceptMatch(
            seed=seed2, extracted=None, strategy="not_found", score=0.0, found=False
        ),
        "endogeneity": ConceptMatch(
            seed=seed3, extracted=None, strategy="not_found", score=0.0, found=False
        ),
    }

    # Calculate recall
    metrics = ValidationMetrics.calculate_recall([seed1, seed2, seed3], matches)

    # Should be 1/3 = 0.333
    assert abs(metrics.overall - 0.333) < 0.01
    assert metrics.by_type["method"] == 1.0  # 1/1
    assert metrics.by_type["assumption"] == 0.0  # 0/1
    assert metrics.by_type["problem"] == 0.0  # 0/1


def test_precision_calculation():
    """Test precision metrics calculation."""
    from research_kb_contracts import Concept, ConceptType
    from uuid import uuid4
    from validate_seed_concepts import ConceptMatch, SeedConcept

    # Create extracted concepts (2 true positives, 1 false positive)
    extracted = [
        Concept(
            id=uuid4(),
            name="IV",
            canonical_name="instrumental variables",
            aliases=[],
            concept_type=ConceptType.METHOD,
            confidence_score=0.9,
            validated=False,
            metadata={},
            created_at="2025-12-02T00:00:00",
        ),
        Concept(
            id=uuid4(),
            name="Unconfoundedness",
            canonical_name="unconfoundedness",
            aliases=[],
            concept_type=ConceptType.ASSUMPTION,
            confidence_score=0.85,
            validated=False,
            metadata={},
            created_at="2025-12-02T00:00:00",
        ),
        Concept(
            id=uuid4(),
            name="Spurious Concept",
            canonical_name="spurious concept",
            aliases=[],
            concept_type=ConceptType.METHOD,
            confidence_score=0.75,
            validated=False,
            metadata={},
            created_at="2025-12-02T00:00:00",
        ),
    ]

    # Create matches (2 found, ignoring false positive)
    seed1 = SeedConcept(
        name="instrumental variables",
        canonical_name="instrumental variables",
        aliases=[],
        concept_type="method",
    )
    seed2 = SeedConcept(
        name="unconfoundedness",
        canonical_name="unconfoundedness",
        aliases=[],
        concept_type="assumption",
    )

    matches = {
        "instrumental variables": ConceptMatch(
            seed=seed1, extracted=extracted[0], strategy="exact", score=1.0, found=True
        ),
        "unconfoundedness": ConceptMatch(
            seed=seed2, extracted=extracted[1], strategy="exact", score=1.0, found=True
        ),
    }

    # Calculate precision
    metrics = ValidationMetrics.calculate_precision(extracted, matches)

    # Should be 2/3 = 0.667
    assert abs(metrics.precision - 0.667) < 0.01
    assert len(metrics.false_positives) == 1
    assert metrics.false_positives[0].name == "Spurious Concept"


# ============================================================
# REPORT GENERATION TESTS
# ============================================================


def test_text_report_generation():
    """Test text report generation."""
    from datetime import datetime
    from validate_seed_concepts import RecallMetrics, PrecisionMetrics, RelationshipMetrics

    report = ValidationReport(
        seed_count=25,
        extracted_count=30,
        matched_count=20,
        recall_metrics=RecallMetrics(
            overall=0.80,
            by_type={"method": 0.85, "assumption": 0.80},
            by_difficulty={"easy": 0.90, "medium": 0.75},
            found_concepts=["IV", "DiD"],
            missing_concepts=["Theorem 1"],
        ),
        precision_metrics=PrecisionMetrics(
            precision=0.75,
            false_positives=[],
            by_confidence={0.7: 0.80, 0.8: 0.85},
        ),
        relationship_metrics=RelationshipMetrics(found=12, expected=20, recall=0.60),
        timestamp=datetime.now(),
        confidence_threshold=0.7,
        semantic_threshold=0.95,
    )

    text = ReportGenerator.generate_text(report)

    assert "SEED CONCEPT VALIDATION REPORT" in text
    assert "Overall recall: 0.800" in text
    assert "Overall precision: 0.750" in text
    assert "method: 0.850" in text


def test_json_report_generation():
    """Test JSON report generation."""
    from datetime import datetime
    from validate_seed_concepts import RecallMetrics, PrecisionMetrics, RelationshipMetrics
    import json

    report = ValidationReport(
        seed_count=25,
        extracted_count=30,
        matched_count=20,
        recall_metrics=RecallMetrics(overall=0.80),
        precision_metrics=PrecisionMetrics(precision=0.75),
        relationship_metrics=RelationshipMetrics(found=12, expected=20, recall=0.60),
        timestamp=datetime.now(),
        confidence_threshold=0.7,
        semantic_threshold=0.95,
    )

    json_output = ReportGenerator.generate_json(report)
    data = json.loads(json_output)

    assert data["summary"]["seed_count"] == 25
    assert data["recall"]["overall"] == 0.80
    assert data["precision"]["overall"] == 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
