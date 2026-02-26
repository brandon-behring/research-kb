"""Concept deduplication using canonical names and embeddings.

Provides:
- Canonical name normalization (e.g., "IV" -> "instrumental variables")
- Domain-specific abbreviation expansion
- Embedding-based similarity matching
- Merge decisions for near-duplicate concepts

Domain support:
- Pass domain_id to Deduplicator to use domain-specific abbreviations
- domain_id is required (no default)
"""

import re
from typing import Optional
from uuid import UUID

from research_kb_common import get_logger

from research_kb_extraction.models import ConceptMatch, ExtractedConcept

logger = get_logger(__name__)


# Legacy abbreviation map for backward compatibility (causal inference)
# New code should use Deduplicator(domain_id="...") instead
ABBREVIATION_MAP = {
    "iv": "instrumental variables",
    "ivs": "instrumental variables",
    "2sls": "two-stage least squares",
    "tsls": "two-stage least squares",
    "did": "difference-in-differences",
    "dd": "difference-in-differences",
    "diff-in-diff": "difference-in-differences",
    "rdd": "regression discontinuity design",
    "rd": "regression discontinuity",
    "psm": "propensity score matching",
    "ate": "average treatment effect",
    "att": "average treatment effect on the treated",
    "atc": "average treatment effect on the controls",
    "atu": "average treatment effect on the untreated",
    "late": "local average treatment effect",
    "cate": "conditional average treatment effect",
    "itt": "intention to treat",
    "toa": "treatment on the treated",
    "ols": "ordinary least squares",
    "gls": "generalized least squares",
    "gmm": "generalized method of moments",
    "ml": "machine learning",
    "dml": "double machine learning",
    "lasso": "least absolute shrinkage and selection operator",
    "rf": "random forest",
    "gbm": "gradient boosting machine",
    "xgboost": "extreme gradient boosting",
    "dag": "directed acyclic graph",
    "scm": "structural causal model",
    "sem": "structural equation model",
    "rct": "randomized controlled trial",
    "fe": "fixed effects",
    "re": "random effects",
    "cia": "conditional independence assumption",
    "sutva": "stable unit treatment value assumption",
    "nuc": "no unmeasured confounding",
}


class Deduplicator:
    """Deduplicate concepts using canonical names and embeddings.

    Strategy:
    1. Normalize to canonical name (expand abbreviations, lowercase)
    2. Check exact canonical name match
    3. Check embedding similarity > threshold
    4. Return match decision

    Example:
        >>> dedup = Deduplicator()
        >>> canonical = dedup.to_canonical_name("IV")
        >>> print(canonical)  # "instrumental variables"

        >>> # Domain-specific deduplication
        >>> dedup = Deduplicator(domain_id="time_series")
        >>> canonical = dedup.to_canonical_name("ARIMA")
        >>> print(canonical)  # "autoregressive integrated moving average"
    """

    def __init__(
        self,
        domain_id: str,
        similarity_threshold: float = 0.95,
        embed_client: Optional[object] = None,
    ):
        """Initialize deduplicator.

        Args:
            similarity_threshold: Embedding similarity threshold for merge
            embed_client: Optional embedding client for similarity
            domain_id: Domain for abbreviation expansion (required)
        """
        self.similarity_threshold = similarity_threshold
        self.embed_client = embed_client
        self.domain_id = domain_id
        self._known_concepts: dict[str, UUID] = {}  # canonical_name -> id

        # Load domain-specific abbreviations
        try:
            from research_kb_extraction.domain_prompts import get_domain_abbreviations

            self._abbreviation_map = get_domain_abbreviations(domain_id)
        except ImportError:
            # Fallback to legacy map if domain_prompts not available
            self._abbreviation_map = ABBREVIATION_MAP
            logger.warning("domain_prompts_import_failed", using_legacy=True)

    def to_canonical_name(self, name: str) -> str:
        """Convert a concept name to its canonical form.

        Steps:
        1. Lowercase and strip
        2. Expand known abbreviations (domain-specific)
        3. Normalize whitespace
        4. Remove special characters

        Args:
            name: Raw concept name

        Returns:
            Canonical name string
        """
        # Lowercase and strip
        canonical = name.lower().strip()

        # Check for direct abbreviation match (using domain-specific map)
        if canonical in self._abbreviation_map:
            canonical = self._abbreviation_map[canonical]

        # Normalize whitespace
        canonical = re.sub(r"\s+", " ", canonical)

        # Remove parenthetical content (often abbreviation definitions)
        canonical = re.sub(r"\s*\([^)]*\)\s*", " ", canonical).strip()

        # Remove special characters except hyphens
        canonical = re.sub(r"[^\w\s-]", "", canonical)

        # Final whitespace normalization
        canonical = re.sub(r"\s+", " ", canonical).strip()

        return canonical

    def register_known_concept(self, canonical_name: str, concept_id: UUID) -> None:
        """Register a known concept for deduplication.

        Args:
            canonical_name: Canonical name of the concept
            concept_id: Database ID of the concept
        """
        self._known_concepts[canonical_name.lower()] = concept_id

    def load_known_concepts(self, concepts: dict[str, UUID]) -> None:
        """Load multiple known concepts.

        Args:
            concepts: Dict mapping canonical_name -> UUID
        """
        for name, id in concepts.items():
            self._known_concepts[name.lower()] = id

    def find_existing_concept(self, name: str) -> Optional[UUID]:
        """Check if a concept already exists by canonical name.

        Args:
            name: Concept name to check

        Returns:
            UUID if found, None otherwise
        """
        canonical = self.to_canonical_name(name)
        return self._known_concepts.get(canonical)

    async def deduplicate_batch(
        self,
        concepts: list[ExtractedConcept],
    ) -> list[ConceptMatch]:
        """Deduplicate a batch of extracted concepts.

        Args:
            concepts: List of extracted concepts

        Returns:
            List of ConceptMatch with deduplication decisions
        """
        results = []
        seen_canonical: dict[str, ConceptMatch] = {}

        for concept in concepts:
            canonical = self.to_canonical_name(concept.name)

            # Check if we've seen this canonical name in this batch
            if canonical in seen_canonical:
                # Merge with existing
                existing = seen_canonical[canonical]
                results.append(
                    ConceptMatch(
                        extracted=concept,
                        matched_concept_id=existing.matched_concept_id,
                        matched_canonical_name=canonical,
                        similarity_score=1.0,  # Exact canonical match
                        is_new=False,
                    )
                )
                continue

            # Check against known concepts
            existing_id = self.find_existing_concept(concept.name)
            if existing_id:
                match = ConceptMatch(
                    extracted=concept,
                    matched_concept_id=existing_id,
                    matched_canonical_name=canonical,
                    similarity_score=1.0,
                    is_new=False,
                )
            else:
                # New concept
                match = ConceptMatch(
                    extracted=concept,
                    matched_concept_id=None,
                    matched_canonical_name=canonical,
                    similarity_score=0.0,
                    is_new=True,
                )

            seen_canonical[canonical] = match
            results.append(match)

        # Log deduplication stats
        new_count = sum(1 for m in results if m.is_new)
        merged_count = len(results) - new_count

        logger.info(
            "deduplication_complete",
            total=len(concepts),
            new=new_count,
            merged=merged_count,
        )

        return results

    async def compute_similarity(
        self,
        concept1: ExtractedConcept,
        concept2: ExtractedConcept,
    ) -> float:
        """Compute semantic similarity between two concepts.

        Uses embedding similarity if embed_client available,
        otherwise falls back to canonical name comparison.

        Args:
            concept1: First concept
            concept2: Second concept

        Returns:
            Similarity score 0.0-1.0
        """
        # If same canonical name, they're identical
        c1_canonical = self.to_canonical_name(concept1.name)
        c2_canonical = self.to_canonical_name(concept2.name)

        if c1_canonical == c2_canonical:
            return 1.0

        # Check if one is alias of other
        c1_aliases = {self.to_canonical_name(a) for a in concept1.aliases}
        c2_aliases = {self.to_canonical_name(a) for a in concept2.aliases}

        if c1_canonical in c2_aliases or c2_canonical in c1_aliases:
            return 0.95

        # Use embeddings if available
        if self.embed_client:
            try:
                # Create text for embedding
                text1 = f"{concept1.name}: {concept1.definition or ''}"
                text2 = f"{concept2.name}: {concept2.definition or ''}"

                emb1 = self.embed_client.embed(text1)  # type: ignore[attr-defined]  # embed_client typed as object, actual type has .embed()
                emb2 = self.embed_client.embed(text2)  # type: ignore[attr-defined]

                # Cosine similarity
                return self._cosine_similarity(emb1, emb2)
            except Exception as e:
                logger.warning("embedding_similarity_failed", error=str(e))

        # Fallback: simple string similarity
        return self._jaccard_similarity(c1_canonical, c2_canonical)

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        """Compute Jaccard similarity between two strings (word-level)."""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def get_all_aliases(self, concept: ExtractedConcept) -> set[str]:
        """Get all known aliases for a concept including abbreviations.

        Args:
            concept: Concept to get aliases for

        Returns:
            Set of all canonical aliases
        """
        aliases = {self.to_canonical_name(concept.name)}
        aliases.update(self.to_canonical_name(a) for a in concept.aliases)

        # Check reverse abbreviation map (using domain-specific map)
        canonical = self.to_canonical_name(concept.name)
        for abbrev, expansion in self._abbreviation_map.items():
            if expansion == canonical:
                aliases.add(abbrev)

        return aliases
