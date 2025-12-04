#!/usr/bin/env python3
"""Validate concept extraction quality against seed concepts.

This script validates Ollama-based concept extraction by:
1. Loading curated seed concepts from YAML
2. Matching extracted concepts using 3 strategies (exact, fuzzy, semantic)
3. Calculating recall, precision, and relationship metrics
4. Generating reports in multiple formats (text, JSON, Markdown)

Usage:
    # Basic validation with terminal output
    python scripts/validate_seed_concepts.py

    # With filtering
    python scripts/validate_seed_concepts.py --type method --confidence 0.8

    # Different output formats
    python scripts/validate_seed_concepts.py --output json > report.json
    python scripts/validate_seed_concepts.py --output markdown > report.md

    # Control matching strategies
    python scripts/validate_seed_concepts.py --no-semantic
    python scripts/validate_seed_concepts.py --semantic-threshold 0.90
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import get_logger
from research_kb_contracts import Concept, ConceptRelationship, ConceptType
from research_kb_storage import (
    ConceptStore,
    RelationshipStore,
    get_connection_pool,
)

logger = get_logger(__name__)


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class SeedConcept:
    """Seed concept from YAML for validation."""

    name: str
    canonical_name: str
    aliases: list[str]
    concept_type: str
    category: Optional[str] = None
    difficulty: Optional[str] = None
    source: Optional[str] = None
    definition: Optional[str] = None
    expected_relationships: list[dict] = field(default_factory=list)


@dataclass
class ConceptMatch:
    """Result of matching seed concept to extracted concept."""

    seed: SeedConcept
    extracted: Optional[Concept]
    strategy: str  # exact_canonical, fuzzy_alias, semantic_similarity, not_found
    score: float  # 0.0-1.0
    found: bool


@dataclass
class RecallMetrics:
    """Recall metrics by type and difficulty."""

    overall: float = 0.0
    by_type: dict[str, float] = field(default_factory=dict)
    by_difficulty: dict[str, float] = field(default_factory=dict)
    found_concepts: list[str] = field(default_factory=list)
    missing_concepts: list[str] = field(default_factory=list)


@dataclass
class PrecisionMetrics:
    """Precision metrics with false positive analysis."""

    precision: float = 0.0
    false_positives: list[Concept] = field(default_factory=list)
    by_confidence: dict[float, float] = field(default_factory=dict)


@dataclass
class RelationshipMetrics:
    """Relationship validation metrics."""

    found: int = 0
    expected: int = 0
    recall: float = 0.0
    missing_relationships: list[dict] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Complete validation report."""

    seed_count: int
    extracted_count: int
    matched_count: int
    recall_metrics: RecallMetrics
    precision_metrics: PrecisionMetrics
    relationship_metrics: RelationshipMetrics
    timestamp: datetime
    confidence_threshold: float
    semantic_threshold: float


# ============================================================
# SEED CONCEPT LOADER
# ============================================================


class SeedConceptLoader:
    """Load and parse seed concepts from YAML."""

    def __init__(self, yaml_path: Path):
        self.yaml_path = yaml_path
        self.concepts: list[SeedConcept] = []
        self.relationships: list[dict] = []

    def load(self) -> list[SeedConcept]:
        """Load seed concepts from YAML file."""
        logger.info("loading_seed_concepts", path=str(self.yaml_path))

        with open(self.yaml_path) as f:
            data = yaml.safe_load(f)

        # Parse concepts by type
        for concept_type in ["methods", "assumptions", "problems", "definitions", "theorems"]:
            if concept_type not in data:
                continue

            for concept_data in data[concept_type]:
                # Extract expected relationships
                expected_rels = []
                if "expected_relationships" in concept_data:
                    expected_rels = concept_data["expected_relationships"]

                seed = SeedConcept(
                    name=concept_data["name"],
                    canonical_name=concept_data["canonical_name"],
                    aliases=concept_data.get("aliases", []),
                    concept_type=concept_data["concept_type"],
                    category=concept_data.get("category"),
                    difficulty=concept_data.get("difficulty"),
                    source=concept_data.get("source"),
                    definition=concept_data.get("definition"),
                    expected_relationships=expected_rels,
                )
                self.concepts.append(seed)

        # Parse relationships
        if "relationships" in data:
            self.relationships = data["relationships"]

        logger.info(
            "seed_concepts_loaded",
            count=len(self.concepts),
            relationships=len(self.relationships),
        )
        return self.concepts

    def get_by_type(self, concept_type: str) -> list[SeedConcept]:
        """Get seed concepts filtered by type."""
        return [c for c in self.concepts if c.concept_type == concept_type]

    def get_by_difficulty(self, difficulty: str) -> list[SeedConcept]:
        """Get seed concepts filtered by difficulty."""
        return [c for c in self.concepts if c.difficulty == difficulty]


# ============================================================
# CONCEPT MATCHER
# ============================================================


class ConceptMatcher:
    """Match seed concepts to extracted concepts using multiple strategies."""

    def __init__(
        self,
        extracted_concepts: list[Concept],
        confidence_threshold: float = 0.7,
        semantic_threshold: float = 0.95,
        use_semantic: bool = True,
    ):
        self.extracted_concepts = extracted_concepts
        self.confidence_threshold = confidence_threshold
        self.semantic_threshold = semantic_threshold
        self.use_semantic = use_semantic

    def match_concept(self, seed: SeedConcept) -> ConceptMatch:
        """Match seed concept using all strategies."""

        # Strategy 1: Exact canonical name match
        match = self._exact_match(seed)
        if match:
            return match

        # Strategy 2: Fuzzy alias match
        match = self._fuzzy_match(seed)
        if match:
            return match

        # Strategy 3: Semantic similarity (if enabled)
        if self.use_semantic:
            match = self._semantic_match(seed)
            if match:
                return match

        # No match found
        return ConceptMatch(
            seed=seed,
            extracted=None,
            strategy="not_found",
            score=0.0,
            found=False,
        )

    def _exact_match(self, seed: SeedConcept) -> Optional[ConceptMatch]:
        """Match by exact canonical name."""
        seed_canonical = seed.canonical_name.lower().strip()

        for extracted in self.extracted_concepts:
            if extracted.canonical_name.lower().strip() == seed_canonical:
                confidence = extracted.confidence_score or 0.0
                if confidence >= self.confidence_threshold:
                    return ConceptMatch(
                        seed=seed,
                        extracted=extracted,
                        strategy="exact_canonical",
                        score=1.0,
                        found=True,
                    )
        return None

    def _fuzzy_match(self, seed: SeedConcept) -> Optional[ConceptMatch]:
        """Match by alias overlap."""
        # Build set of all seed aliases (including canonical name)
        seed_aliases = {seed.canonical_name.lower().strip()}
        seed_aliases.update({a.lower().strip() for a in seed.aliases})

        for extracted in self.extracted_concepts:
            # Build set of extracted aliases
            extracted_aliases = {extracted.canonical_name.lower().strip()}
            extracted_aliases.update({a.lower().strip() for a in extracted.aliases})

            # Check for intersection
            if seed_aliases & extracted_aliases:
                confidence = extracted.confidence_score or 0.0
                if confidence >= self.confidence_threshold:
                    return ConceptMatch(
                        seed=seed,
                        extracted=extracted,
                        strategy="fuzzy_alias",
                        score=0.95,
                        found=True,
                    )
        return None

    def _semantic_match(self, seed: SeedConcept) -> Optional[ConceptMatch]:
        """Match by embedding cosine similarity.

        Note: Requires extracted concepts to have embeddings.
        Gracefully degrades if embeddings unavailable.
        """
        # TODO: Implement embedding-based matching when embedding client available
        # For now, this is a placeholder that returns None
        #
        # Implementation would:
        # 1. Compute seed embedding: embed(f"{seed.name}: {seed.definition}")
        # 2. For each extracted concept with embedding:
        #    - Compute cosine similarity
        #    - If similarity > semantic_threshold and confidence > threshold:
        #      - Return ConceptMatch with strategy="semantic_similarity"
        #
        # Graceful degradation: If embedding client unavailable or
        # extracted concepts lack embeddings, return None

        return None


# ============================================================
# VALIDATION METRICS
# ============================================================


class ValidationMetrics:
    """Calculate validation metrics."""

    @staticmethod
    def calculate_recall(
        seed_concepts: list[SeedConcept],
        matches: dict[str, ConceptMatch],
    ) -> RecallMetrics:
        """Calculate recall metrics."""
        metrics = RecallMetrics()

        # Overall recall
        found = [m for m in matches.values() if m.found]
        metrics.overall = len(found) / len(seed_concepts) if seed_concepts else 0.0

        # Track found and missing
        metrics.found_concepts = [m.seed.name for m in found]
        metrics.missing_concepts = [
            m.seed.name for m in matches.values() if not m.found
        ]

        # Per-type recall
        by_type = defaultdict(lambda: {"found": 0, "total": 0})
        for seed in seed_concepts:
            concept_type = seed.concept_type
            by_type[concept_type]["total"] += 1
            if matches[seed.name].found:
                by_type[concept_type]["found"] += 1

        metrics.by_type = {
            t: stats["found"] / stats["total"] if stats["total"] > 0 else 0.0
            for t, stats in by_type.items()
        }

        # Per-difficulty recall
        by_difficulty = defaultdict(lambda: {"found": 0, "total": 0})
        for seed in seed_concepts:
            if seed.difficulty:
                by_difficulty[seed.difficulty]["total"] += 1
                if matches[seed.name].found:
                    by_difficulty[seed.difficulty]["found"] += 1

        metrics.by_difficulty = {
            d: stats["found"] / stats["total"] if stats["total"] > 0 else 0.0
            for d, stats in by_difficulty.items()
        }

        return metrics

    @staticmethod
    def calculate_precision(
        extracted_concepts: list[Concept],
        matches: dict[str, ConceptMatch],
    ) -> PrecisionMetrics:
        """Calculate precision metrics."""
        metrics = PrecisionMetrics()

        # Build set of extracted IDs that matched seed
        matched_ids = {
            m.extracted.id for m in matches.values() if m.found and m.extracted
        }

        # Count false positives
        false_positives = [c for c in extracted_concepts if c.id not in matched_ids]
        metrics.false_positives = false_positives

        # Overall precision
        metrics.precision = (
            len(matched_ids) / len(extracted_concepts) if extracted_concepts else 0.0
        )

        # Precision by confidence threshold
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            high_conf = [
                c for c in extracted_concepts if (c.confidence_score or 0.0) >= threshold
            ]
            in_seed = len([c for c in high_conf if c.id in matched_ids])
            metrics.by_confidence[threshold] = (
                in_seed / len(high_conf) if high_conf else 0.0
            )

        return metrics

    @staticmethod
    async def validate_relationships(
        matches: dict[str, ConceptMatch],
    ) -> RelationshipMetrics:
        """Validate expected relationships in database.

        Note: Queries PostgreSQL RelationshipStore (schema-agnostic).
        """
        metrics = RelationshipMetrics()

        for match in matches.values():
            if not match.found or not match.seed.expected_relationships:
                continue

            for expected_rel in match.seed.expected_relationships:
                metrics.expected += 1

                # Find target concept match
                target_match = next(
                    (
                        m
                        for m in matches.values()
                        if m.seed.name == expected_rel["target"]
                    ),
                    None,
                )

                if not target_match or not target_match.found:
                    metrics.missing_relationships.append(
                        {
                            "source": match.seed.name,
                            "target": expected_rel["target"],
                            "type": expected_rel["type"],
                            "reason": "target_not_found",
                        }
                    )
                    continue

                # Check if relationship exists in database
                rels = await RelationshipStore.list_from_concept(match.extracted.id)
                rel_exists = any(
                    r.target_concept_id == target_match.extracted.id
                    and r.relationship_type.value == expected_rel["type"]
                    for r in rels
                )

                if rel_exists:
                    metrics.found += 1
                else:
                    metrics.missing_relationships.append(
                        {
                            "source": match.seed.name,
                            "target": expected_rel["target"],
                            "type": expected_rel["type"],
                            "reason": "relationship_not_extracted",
                        }
                    )

        metrics.recall = (
            metrics.found / metrics.expected if metrics.expected > 0 else 0.0
        )

        return metrics


# ============================================================
# VALIDATION REPORT GENERATOR
# ============================================================


class ReportGenerator:
    """Generate validation reports in multiple formats."""

    @staticmethod
    def generate_text(report: ValidationReport) -> str:
        """Generate rich terminal output with colors."""
        lines = []

        # Header
        lines.append("=" * 60)
        lines.append("SEED CONCEPT VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Timestamp: {report.timestamp.isoformat()}")
        lines.append(f"Confidence threshold: {report.confidence_threshold:.2f}")
        lines.append(f"Semantic threshold: {report.semantic_threshold:.2f}")
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 60)
        lines.append(f"Seed concepts: {report.seed_count}")
        lines.append(f"Extracted concepts: {report.extracted_count}")
        lines.append(f"Matched: {report.matched_count} ({report.matched_count/report.seed_count*100:.1f}%)")
        lines.append("")

        # Recall
        lines.append("RECALL METRICS")
        lines.append("-" * 60)
        lines.append(f"Overall recall: {report.recall_metrics.overall:.3f}")
        lines.append("\nBy concept type:")
        for ctype, recall in sorted(report.recall_metrics.by_type.items()):
            status = "✓" if recall >= 0.75 else "⚠"
            lines.append(f"  {status} {ctype}: {recall:.3f}")

        if report.recall_metrics.by_difficulty:
            lines.append("\nBy difficulty:")
            for difficulty, recall in sorted(report.recall_metrics.by_difficulty.items()):
                status = "✓" if recall >= 0.75 else "⚠"
                lines.append(f"  {status} {difficulty}: {recall:.3f}")

        # Precision
        lines.append("\nPRECISION METRICS")
        lines.append("-" * 60)
        lines.append(f"Overall precision: {report.precision_metrics.precision:.3f}")
        lines.append(f"False positives: {len(report.precision_metrics.false_positives)}")
        lines.append("\nBy confidence threshold:")
        for threshold, precision in sorted(report.precision_metrics.by_confidence.items()):
            lines.append(f"  >= {threshold:.1f}: {precision:.3f}")

        # Relationships
        lines.append("\nRELATIONSHIP VALIDATION")
        lines.append("-" * 60)
        lines.append(
            f"Found: {report.relationship_metrics.found}/{report.relationship_metrics.expected} "
            f"({report.relationship_metrics.recall:.3f})"
        )

        # Missing concepts
        if report.recall_metrics.missing_concepts:
            lines.append("\nMISSING CONCEPTS")
            lines.append("-" * 60)
            for name in sorted(report.recall_metrics.missing_concepts)[:10]:
                lines.append(f"  • {name}")
            if len(report.recall_metrics.missing_concepts) > 10:
                lines.append(
                    f"  ... and {len(report.recall_metrics.missing_concepts) - 10} more"
                )

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    @staticmethod
    def generate_json(report: ValidationReport) -> str:
        """Generate JSON output for CI/CD."""
        data = {
            "timestamp": report.timestamp.isoformat(),
            "thresholds": {
                "confidence": report.confidence_threshold,
                "semantic": report.semantic_threshold,
            },
            "summary": {
                "seed_count": report.seed_count,
                "extracted_count": report.extracted_count,
                "matched_count": report.matched_count,
            },
            "recall": {
                "overall": report.recall_metrics.overall,
                "by_type": report.recall_metrics.by_type,
                "by_difficulty": report.recall_metrics.by_difficulty,
                "missing_concepts": report.recall_metrics.missing_concepts,
            },
            "precision": {
                "overall": report.precision_metrics.precision,
                "false_positive_count": len(report.precision_metrics.false_positives),
                "by_confidence": report.precision_metrics.by_confidence,
            },
            "relationships": {
                "found": report.relationship_metrics.found,
                "expected": report.relationship_metrics.expected,
                "recall": report.relationship_metrics.recall,
            },
        }
        return json.dumps(data, indent=2)

    @staticmethod
    def generate_markdown(report: ValidationReport) -> str:
        """Generate Markdown report for documentation."""
        lines = []

        lines.append("# Seed Concept Validation Report")
        lines.append(f"\n**Timestamp**: {report.timestamp.isoformat()}")
        lines.append(f"**Confidence threshold**: {report.confidence_threshold:.2f}")
        lines.append(f"**Semantic threshold**: {report.semantic_threshold:.2f}")

        lines.append("\n## Summary\n")
        lines.append(f"- Seed concepts: {report.seed_count}")
        lines.append(f"- Extracted concepts: {report.extracted_count}")
        lines.append(
            f"- Matched: {report.matched_count} ({report.matched_count/report.seed_count*100:.1f}%)"
        )

        lines.append("\n## Recall Metrics\n")
        lines.append(f"**Overall recall**: {report.recall_metrics.overall:.3f}\n")
        lines.append("### By Concept Type\n")
        lines.append("| Type | Recall | Status |")
        lines.append("|------|--------|--------|")
        for ctype, recall in sorted(report.recall_metrics.by_type.items()):
            status = "✓ Pass" if recall >= 0.75 else "⚠ Warning"
            lines.append(f"| {ctype} | {recall:.3f} | {status} |")

        lines.append("\n## Precision Metrics\n")
        lines.append(f"**Overall precision**: {report.precision_metrics.precision:.3f}")
        lines.append(f"\n**False positives**: {len(report.precision_metrics.false_positives)}")

        lines.append("\n## Relationship Validation\n")
        lines.append(
            f"- Found: {report.relationship_metrics.found}/{report.relationship_metrics.expected}"
        )
        lines.append(f"- Recall: {report.relationship_metrics.recall:.3f}")

        if report.recall_metrics.missing_concepts:
            lines.append("\n## Missing Concepts\n")
            for name in sorted(report.recall_metrics.missing_concepts)[:20]:
                lines.append(f"- {name}")

        return "\n".join(lines)


# ============================================================
# MAIN VALIDATION ORCHESTRATOR
# ============================================================


async def validate_seed_concepts(
    seed_yaml_path: Path,
    concept_type_filter: Optional[str] = None,
    confidence_threshold: float = 0.7,
    semantic_threshold: float = 0.95,
    use_semantic: bool = True,
    output_format: str = "text",
) -> ValidationReport:
    """Main validation orchestrator."""

    # Load seed concepts
    loader = SeedConceptLoader(seed_yaml_path)
    seed_concepts = loader.load()

    # Filter by type if requested
    if concept_type_filter:
        seed_concepts = loader.get_by_type(concept_type_filter)
        logger.info("filtered_by_type", type=concept_type_filter, count=len(seed_concepts))

    # Load extracted concepts from database
    logger.info("loading_extracted_concepts")
    extracted_concepts = await ConceptStore.list_all()
    logger.info("extracted_concepts_loaded", count=len(extracted_concepts))

    # Match concepts
    matcher = ConceptMatcher(
        extracted_concepts,
        confidence_threshold,
        semantic_threshold,
        use_semantic,
    )

    matches = {}
    for seed in seed_concepts:
        match = matcher.match_concept(seed)
        matches[seed.name] = match
        if match.found:
            logger.debug(
                "concept_matched",
                seed=seed.name,
                strategy=match.strategy,
                score=match.score,
            )

    # Calculate metrics
    recall_metrics = ValidationMetrics.calculate_recall(seed_concepts, matches)
    precision_metrics = ValidationMetrics.calculate_precision(extracted_concepts, matches)
    relationship_metrics = await ValidationMetrics.validate_relationships(matches)

    # Generate report
    report = ValidationReport(
        seed_count=len(seed_concepts),
        extracted_count=len(extracted_concepts),
        matched_count=len([m for m in matches.values() if m.found]),
        recall_metrics=recall_metrics,
        precision_metrics=precision_metrics,
        relationship_metrics=relationship_metrics,
        timestamp=datetime.now(),
        confidence_threshold=confidence_threshold,
        semantic_threshold=semantic_threshold,
    )

    # Output report
    if output_format == "json":
        print(ReportGenerator.generate_json(report))
    elif output_format == "markdown":
        print(ReportGenerator.generate_markdown(report))
    else:  # text
        print(ReportGenerator.generate_text(report))

    return report


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Validate concept extraction quality against seed concepts"
    )
    parser.add_argument(
        "--seed-file",
        type=Path,
        default=Path(__file__).parent.parent / "fixtures/concepts/seed_concepts.yaml",
        help="Path to seed concepts YAML file",
    )
    parser.add_argument(
        "--type",
        choices=["method", "assumption", "problem", "definition", "theorem"],
        help="Filter by concept type",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Confidence threshold for matching (default: 0.7)",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.95,
        help="Semantic similarity threshold (default: 0.95)",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic matching",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Run validation
    try:
        report = asyncio.run(
            validate_seed_concepts(
                seed_yaml_path=args.seed_file,
                concept_type_filter=args.type,
                confidence_threshold=args.confidence,
                semantic_threshold=args.semantic_threshold,
                use_semantic=not args.no_semantic,
                output_format=args.output,
            )
        )

        # Exit with appropriate code
        if report.recall_metrics.overall >= 0.80 and report.precision_metrics.precision >= 0.75:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Validation failed

    except Exception as e:
        logger.error("validation_failed", error=str(e))
        sys.exit(2)


if __name__ == "__main__":
    main()
