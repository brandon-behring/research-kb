"""Automated literature review generation via graph traversal + search + LLM synthesis.

Provides:
- Topic exploration: graph neighborhood + search to find related concepts and sources
- Section organization: structured review (overview, methods, assumptions, applications)
- Evidence assembly: per-section chunk hydration with source citations
- LLM composition: multi-call Anthropic synthesis with style control
- Quality metrics: concept coverage, citation count, source diversity

Architecture:
- Uses get_neighborhood() from graph_queries for concept exploration
- Uses search_hybrid() for evidence retrieval
- Uses Anthropic Haiku for synthesis (same pattern as synthesis.py)
- Graceful degradation: graph_only if no LLM, search_only if no graph

See CLAUDE.md "MCP Server" section for tool documentation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from research_kb_common import get_logger

logger = get_logger(__name__)

# ── Content limits ───────────────────────────────────────────────────────────

MAX_EVIDENCE_PER_SECTION = 8
"""Maximum evidence chunks per review section."""

MAX_CONCEPTS_EXPLORED = 30
"""Maximum concepts to explore from graph neighborhood."""

EVIDENCE_CONTENT_MAX_CHARS = 400
"""Maximum characters per evidence chunk (truncated with ellipsis)."""

# ── Section definitions ──────────────────────────────────────────────────────

REVIEW_SECTIONS = [
    {
        "id": "overview",
        "title": "Overview",
        "description": "Core definitions, scope, and foundational ideas",
        "concept_types": ["concept", "definition", "problem"],
        "search_suffix": "definition overview introduction",
    },
    {
        "id": "methods",
        "title": "Methods & Techniques",
        "description": "Algorithms, estimators, and analytical approaches",
        "concept_types": ["method", "technique", "model"],
        "search_suffix": "method algorithm estimator technique",
    },
    {
        "id": "assumptions",
        "title": "Assumptions & Conditions",
        "description": "Required assumptions, identification conditions, and validity requirements",
        "concept_types": ["assumption", "principle", "theorem"],
        "search_suffix": "assumption condition requirement identification",
    },
    {
        "id": "applications",
        "title": "Applications & Extensions",
        "description": "Practical applications, extensions, and related work",
        "concept_types": ["technique", "model", "concept"],
        "search_suffix": "application example empirical extension",
    },
]

# ── LLM Prompts ──────────────────────────────────────────────────────────────

SECTION_SYNTHESIS_PROMPT = """You are an expert academic reviewer writing a section of a literature review about "{topic}".

SECTION: {section_title}
PURPOSE: {section_description}

RELATED CONCEPTS (from knowledge graph):
{concept_list}

EVIDENCE FROM SOURCES:
{evidence_text}

INSTRUCTIONS:
1. Write a coherent 2-4 paragraph section synthesizing the evidence
2. Cite sources inline as (Author/Title, Year, p. X) based on the provided evidence
3. Cover the key concepts listed above, prioritizing those with strongest evidence
4. Note gaps where evidence is sparse
5. Do NOT invent information not in the provided evidence
6. Maintain {style} tone throughout

Write the section directly. No heading needed (it will be added automatically)."""

INTRODUCTION_PROMPT = """You are an expert academic reviewer. Write a brief introduction (1-2 paragraphs) for a literature review about "{topic}".

The review covers these sections:
{section_summaries}

Based on evidence from {source_count} sources spanning {domain_count} domain(s).

KEY CONCEPTS: {concept_list}

Write a focused introduction that:
1. States the topic and its significance
2. Previews the review structure
3. Notes the scope and coverage

Keep it concise. No heading needed."""

CONCLUSION_PROMPT = """You are an expert academic reviewer. Write a brief conclusion (1-2 paragraphs) for a literature review about "{topic}".

KEY FINDINGS FROM EACH SECTION:
{section_summaries}

COVERAGE METRICS:
- Sources referenced: {source_count}
- Concepts covered: {concept_count}
- Domains spanned: {domain_list}

Write a conclusion that:
1. Summarizes key themes and findings
2. Identifies gaps in the current literature
3. Suggests directions for future research

Keep it concise. No heading needed."""


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class ReviewEvidence:
    """A piece of evidence for a literature review section."""

    chunk_id: UUID
    source_id: UUID
    source_title: str
    content: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    relevance_score: float = 0.0
    domain: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        return {
            "chunk_id": str(self.chunk_id),
            "source_id": str(self.source_id),
            "source_title": self.source_title,
            "content": self.content[:EVIDENCE_CONTENT_MAX_CHARS],
            "page_start": self.page_start,
            "page_end": self.page_end,
            "relevance_score": round(self.relevance_score, 3),
            "domain": self.domain,
        }


@dataclass
class ReviewSection:
    """A section of the literature review."""

    section_id: str
    title: str
    description: str
    concepts: list[dict] = field(default_factory=list)
    evidence: list[ReviewEvidence] = field(default_factory=list)
    synthesis: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        return {
            "section_id": self.section_id,
            "title": self.title,
            "description": self.description,
            "concepts": self.concepts,
            "evidence": [e.to_dict() for e in self.evidence],
            "synthesis": self.synthesis,
        }


@dataclass
class LiteratureReview:
    """Complete automated literature review."""

    topic: str
    introduction: Optional[str] = None
    sections: list[ReviewSection] = field(default_factory=list)
    conclusion: Optional[str] = None
    quality_metrics: dict = field(default_factory=dict)
    style: str = "educational"
    source: str = "graph+search+llm"

    def to_dict(self) -> dict:
        """Full serialization for JSON output."""
        return {
            "topic": self.topic,
            "introduction": self.introduction,
            "sections": [s.to_dict() for s in self.sections],
            "conclusion": self.conclusion,
            "quality_metrics": self.quality_metrics,
            "style": self.style,
            "source": self.source,
        }

    def to_markdown(self) -> str:
        """Render as markdown document."""
        parts = [f"# Literature Review: {self.topic}\n"]

        if self.introduction:
            parts.append(self.introduction)
            parts.append("")

        for section in self.sections:
            parts.append(f"## {section.title}\n")
            if section.synthesis:
                parts.append(section.synthesis)
            else:
                parts.append(f"*{section.description}*")
                if section.concepts:
                    concept_names = [
                        c["name"] if isinstance(c, dict) else c for c in section.concepts[:10]
                    ]
                    parts.append(f"\nKey concepts: {', '.join(concept_names)}")
            parts.append("")

        if self.conclusion:
            parts.append("## Conclusion\n")
            parts.append(self.conclusion)
            parts.append("")

        # Quality footer
        qm = self.quality_metrics
        if qm:
            parts.append("---")
            parts.append(
                f"*Generated from {qm.get('sources_cited', 0)} sources, "
                f"{qm.get('concepts_covered', 0)} concepts, "
                f"{qm.get('evidence_chunks', 0)} evidence passages. "
                f"Source: {self.source}.*"
            )

        return "\n".join(parts)


# ── Core implementation ──────────────────────────────────────────────────────


def _truncate(text: str, max_chars: int = EVIDENCE_CONTENT_MAX_CHARS) -> str:
    """Truncate text at the last sentence boundary within max_chars.

    Prefers cutting at '. ', '? ', or '! ' boundaries. Falls back to
    word boundary if no sentence end found in the first 60% of text.
    """
    if len(text) <= max_chars:
        return text

    # Search for last sentence boundary within limit
    search_region = text[: max_chars - 3]
    for sep in [". ", "? ", "! "]:
        idx = search_region.rfind(sep)
        # Only use if we keep at least 60% of the allowed length
        if idx >= max_chars * 0.6:
            return text[: idx + 1]

    # Fall back to word boundary
    space_idx = search_region.rfind(" ")
    if space_idx >= max_chars * 0.6:
        return text[:space_idx] + "..."

    return text[: max_chars - 3] + "..."


async def _explore_topic_concepts(
    topic: str,
    max_concepts: int = MAX_CONCEPTS_EXPLORED,
) -> list[dict]:
    """Find concepts related to a topic via graph neighborhood + search.

    Strategy:
    1. Search for concepts matching the topic by name
    2. Expand via graph neighborhood (1-2 hops)
    3. Deduplicate and rank by relevance

    Args:
        topic: Topic string to explore
        max_concepts: Maximum concepts to return

    Returns:
        List of concept dicts with name, type, definition, relevance
    """
    from research_kb_storage.concept_store import ConceptStore
    from research_kb_storage.graph_queries import get_neighborhood

    # Find seed concepts matching the topic
    seed_concepts = await ConceptStore.search(topic, limit=5)

    if not seed_concepts:
        # Try individual words from topic
        for word in topic.split():
            if len(word) >= 3:
                results = await ConceptStore.search(word, limit=3)
                seed_concepts.extend(results)
        # Deduplicate
        seen_ids = set()
        unique = []
        for c in seed_concepts:
            if c.id not in seen_ids:
                seen_ids.add(c.id)
                unique.append(c)
        seed_concepts = unique[:5]

    if not seed_concepts:
        logger.warning("no_seed_concepts_found", topic=topic)
        return []

    # Expand via neighborhood
    all_concepts: dict[UUID, dict] = {}

    # Add seed concepts (highest relevance)
    for c in seed_concepts:
        ctype = c.concept_type.value if hasattr(c.concept_type, "value") else str(c.concept_type)
        all_concepts[c.id] = {
            "id": str(c.id),
            "name": c.canonical_name or c.name,
            "type": ctype,
            "definition": c.definition,
            "relevance": 1.0,
        }

    # 1-hop neighborhood for each seed
    for seed in seed_concepts[:3]:  # Limit to top 3 seeds to control explosion
        try:
            neighborhood = await get_neighborhood(seed.id, hops=1)
            for n in neighborhood.get("concepts", []):
                if n.id not in all_concepts and len(all_concepts) < max_concepts:
                    ntype = (
                        n.concept_type.value
                        if hasattr(n.concept_type, "value")
                        else str(n.concept_type)
                    )
                    all_concepts[n.id] = {
                        "id": str(n.id),
                        "name": n.canonical_name or n.name,
                        "type": ntype,
                        "definition": n.definition,
                        "relevance": 0.5,
                    }
        except Exception as e:
            logger.debug("neighborhood_expansion_failed", seed=seed.name, error=str(e))

    # Sort by relevance (seeds first, then neighbors)
    return sorted(all_concepts.values(), key=lambda c: -c["relevance"])[:max_concepts]


async def _search_evidence_for_section(
    topic: str,
    section_def: dict,
    concepts: list[dict],
    max_evidence: int = MAX_EVIDENCE_PER_SECTION,
    exclude_chunk_ids: set[UUID] | None = None,
) -> list[ReviewEvidence]:
    """Search for evidence chunks relevant to a review section.

    Uses hybrid search with section-specific query augmentation.

    Args:
        topic: Main topic
        section_def: Section definition from REVIEW_SECTIONS
        concepts: Related concepts for this section
        max_evidence: Maximum evidence chunks to collect
        exclude_chunk_ids: Chunk IDs already used in other sections (for dedup)

    Returns:
        List of ReviewEvidence, deduplicated by source and cross-section
    """
    from research_kb_pdf import EmbeddingClient
    from research_kb_storage.search import SearchQuery, search_hybrid

    embed_client = EmbeddingClient()

    # Build section-specific query
    section_concepts = [c["name"] for c in concepts if c["type"] in section_def["concept_types"]][
        :5
    ]

    if section_concepts:
        query_text = f"{topic} {' '.join(section_concepts)} {section_def['search_suffix']}"
    else:
        query_text = f"{topic} {section_def['search_suffix']}"

    try:
        query_embedding = embed_client.embed_query(query_text)
        query = SearchQuery(
            text=query_text,
            embedding=query_embedding,
            fts_weight=0.3,
            vector_weight=0.7,
            limit=max_evidence * 2,  # Fetch extra for diversity filtering
        )
        results = await search_hybrid(query)
    except Exception as e:
        logger.warning("section_search_failed", section=section_def["id"], error=str(e))
        return []

    # Convert to ReviewEvidence with source diversity
    evidence: list[ReviewEvidence] = []
    seen_sources: set[UUID] = set()

    _exclude = exclude_chunk_ids or set()

    for r in results:
        # Limit to max_evidence, preferring diverse sources
        if len(evidence) >= max_evidence:
            break

        # Skip chunks already used in other sections
        if r.chunk.id in _exclude:
            continue

        # Skip duplicate sources after first 2 per source
        source_count = sum(1 for e in evidence if e.source_id == r.source.id)
        if source_count >= 2:
            continue

        evidence.append(
            ReviewEvidence(
                chunk_id=r.chunk.id,
                source_id=r.source.id,
                source_title=r.source.title,
                content=_truncate(r.chunk.content),
                page_start=r.chunk.page_start,
                page_end=r.chunk.page_end,
                relevance_score=r.combined_score,
                domain=getattr(r.chunk, "domain_id", None),
            )
        )
        seen_sources.add(r.source.id)

    return evidence


async def _synthesize_section(
    topic: str,
    section: ReviewSection,
    style: str = "educational",
) -> Optional[str]:
    """Generate section synthesis via Anthropic LLM.

    Args:
        topic: Main topic
        section: ReviewSection with concepts and evidence populated
        style: Writing style (educational/research/implementation)

    Returns:
        Synthesized section text, or None if LLM unavailable
    """
    if not section.evidence:
        logger.debug("no_evidence_for_synthesis", section=section.section_id)
        return None

    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic_not_installed", section=section.section_id)
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("anthropic_api_key_missing", section=section.section_id)
        return None

    # Build concept list
    concept_list = (
        "\n".join(f"- {c['name']} ({c['type']})" for c in section.concepts[:10])
        if section.concepts
        else "No specific concepts identified."
    )

    # Build evidence text
    evidence_parts = []
    for i, ev in enumerate(section.evidence):
        citation = f"{ev.source_title}"
        if ev.page_start:
            citation += f", p. {ev.page_start}"
            if ev.page_end and ev.page_end != ev.page_start:
                citation += f"-{ev.page_end}"
        evidence_parts.append(f"[{i + 1}] ({citation})\n{ev.content}")
    evidence_text = "\n\n".join(evidence_parts)

    prompt = SECTION_SYNTHESIS_PROMPT.format(
        topic=topic,
        section_title=section.title,
        section_description=section.description,
        concept_list=concept_list,
        evidence_text=evidence_text,
        style=style,
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        logger.error("section_synthesis_failed", section=section.section_id, error=str(e))
        return None


async def _synthesize_intro_conclusion(
    topic: str,
    sections: list[ReviewSection],
    style: str = "educational",
    part: str = "introduction",
) -> Optional[str]:
    """Generate introduction or conclusion via Anthropic LLM.

    Args:
        topic: Main topic
        sections: Completed review sections
        style: Writing style
        part: "introduction" or "conclusion"

    Returns:
        Generated text, or None if LLM unavailable
    """
    try:
        import anthropic
    except ImportError:
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    # Build section summaries
    section_summaries = "\n".join(
        f"- {s.title}: {len(s.evidence)} evidence sources, "
        f"{len(s.concepts)} concepts" + (f" — {s.synthesis[:100]}..." if s.synthesis else "")
        for s in sections
    )

    # Collect metrics
    all_sources = set()
    all_concepts = set()
    all_domains = set()
    for s in sections:
        for e in s.evidence:
            all_sources.add(e.source_title)
            if e.domain:
                all_domains.add(e.domain)
        all_concepts.update(c["name"] if isinstance(c, dict) else c for c in s.concepts)

    if part == "introduction":
        prompt = INTRODUCTION_PROMPT.format(
            topic=topic,
            section_summaries=section_summaries,
            source_count=len(all_sources),
            domain_count=len(all_domains),
            concept_list=", ".join(list(all_concepts)[:15]),
        )
    else:
        prompt = CONCLUSION_PROMPT.format(
            topic=topic,
            section_summaries=section_summaries,
            source_count=len(all_sources),
            concept_count=len(all_concepts),
            domain_list=", ".join(sorted(all_domains)) if all_domains else "single domain",
        )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"{part}_synthesis_failed", error=str(e))
        return None


# ── Main orchestrator ────────────────────────────────────────────────────────


async def generate_literature_review(
    topic: str,
    style: str = "educational",
    use_llm: bool = True,
    max_concepts: int = MAX_CONCEPTS_EXPLORED,
    max_evidence_per_section: int = MAX_EVIDENCE_PER_SECTION,
) -> LiteratureReview:
    """Generate a structured literature review for a topic.

    Orchestration:
    1. Explore topic via graph neighborhood to find related concepts
    2. Organize concepts into review sections by type
    3. Search for evidence per section (hybrid FTS + vector)
    4. Synthesize each section via LLM (if enabled)
    5. Generate introduction and conclusion
    6. Compute quality metrics

    Args:
        topic: Topic to review (e.g., "double machine learning")
        style: Writing style — "educational", "research", "implementation"
        use_llm: Enable LLM synthesis (requires ANTHROPIC_API_KEY)
        max_concepts: Maximum concepts to explore
        max_evidence_per_section: Evidence chunks per section

    Returns:
        LiteratureReview with sections, evidence, and optional synthesis

    Example:
        >>> review = await generate_literature_review(
        ...     "instrumental variables",
        ...     style="research",
        ...     use_llm=True,
        ... )
        >>> print(review.to_markdown())
    """
    logger.info("literature_review_start", topic=topic, style=style, use_llm=use_llm)

    # 1. Explore concepts
    concepts = await _explore_topic_concepts(topic, max_concepts=max_concepts)
    logger.info("concepts_explored", topic=topic, count=len(concepts))

    if not concepts:
        # Fallback: search-only mode (no graph data)
        return await _generate_search_only_review(topic, style, use_llm, max_evidence_per_section)

    # 2. Organize into sections
    sections: list[ReviewSection] = []
    for section_def in REVIEW_SECTIONS:
        section_concepts = [
            {"name": c["name"], "type": c["type"]}
            for c in concepts
            if c["type"] in section_def["concept_types"]
        ]

        section = ReviewSection(
            section_id=section_def["id"],
            title=section_def["title"],
            description=section_def["description"],
            concepts=section_concepts,
        )
        sections.append(section)

    # 3. Search for evidence per section (with cross-section dedup)
    used_chunk_ids: set[UUID] = set()
    for section in sections:
        section.evidence = await _search_evidence_for_section(
            topic,
            next(sd for sd in REVIEW_SECTIONS if sd["id"] == section.section_id),
            concepts,
            max_evidence=max_evidence_per_section,
            exclude_chunk_ids=used_chunk_ids,
        )
        used_chunk_ids.update(e.chunk_id for e in section.evidence)
        logger.debug(
            "section_evidence_gathered",
            section=section.section_id,
            evidence_count=len(section.evidence),
        )

    # 4. LLM synthesis per section
    source_mode = "graph+search"
    if use_llm:
        for section in sections:
            section.synthesis = await _synthesize_section(topic, section, style)
            if section.synthesis:
                source_mode = "graph+search+llm"

    # 5. Introduction and conclusion
    introduction = None
    conclusion = None
    if use_llm:
        introduction = await _synthesize_intro_conclusion(topic, sections, style, "introduction")
        conclusion = await _synthesize_intro_conclusion(topic, sections, style, "conclusion")

    # 6. Quality metrics
    all_sources = set()
    all_chunk_ids: list[set] = []
    all_evidence_count = 0
    for s in sections:
        section_chunks = set()
        for e in s.evidence:
            all_sources.add(str(e.source_id))
            section_chunks.add(str(e.chunk_id))
        all_chunk_ids.append(section_chunks)
        all_evidence_count += len(s.evidence)

    # Evidence overlap: what fraction of slots are duplicates across sections
    total_slots = sum(len(s) for s in all_chunk_ids)
    unique_chunks = len(set().union(*all_chunk_ids)) if all_chunk_ids else 0
    overlap_ratio = round(1 - unique_chunks / total_slots, 3) if total_slots > 0 else 0.0

    all_concepts = set()
    for s in sections:
        all_concepts.update(c["name"] if isinstance(c, dict) else c for c in s.concepts)

    sections_with_synthesis = sum(1 for s in sections if s.synthesis)

    quality_metrics = {
        "concepts_explored": len(concepts),
        "concepts_covered": len(all_concepts),
        "sources_cited": len(all_sources),
        "evidence_chunks": all_evidence_count,
        "unique_evidence_chunks": unique_chunks,
        "evidence_overlap_ratio": overlap_ratio,
        "sections_total": len(sections),
        "sections_synthesized": sections_with_synthesis,
        "has_introduction": introduction is not None,
        "has_conclusion": conclusion is not None,
    }

    review = LiteratureReview(
        topic=topic,
        introduction=introduction,
        sections=sections,
        conclusion=conclusion,
        quality_metrics=quality_metrics,
        style=style,
        source=source_mode,
    )

    logger.info(
        "literature_review_complete",
        topic=topic,
        sources=len(all_sources),
        concepts=len(all_concepts),
        evidence=all_evidence_count,
        synthesized=sections_with_synthesis,
    )

    return review


async def _generate_search_only_review(
    topic: str,
    style: str,
    use_llm: bool,
    max_evidence: int,
) -> LiteratureReview:
    """Fallback: generate review using search only (no graph concepts).

    Used when no concepts are found in the graph for the topic.

    Args:
        topic: Topic to review
        style: Writing style
        use_llm: Enable LLM synthesis
        max_evidence: Evidence per section

    Returns:
        LiteratureReview with search-based evidence
    """
    logger.info("literature_review_search_only_fallback", topic=topic)

    sections = []
    for section_def in REVIEW_SECTIONS:
        section = ReviewSection(
            section_id=section_def["id"],
            title=section_def["title"],
            description=section_def["description"],
        )

        # Direct search without concept guidance
        section.evidence = await _search_evidence_for_section(
            topic, section_def, [], max_evidence=max_evidence
        )

        if use_llm and section.evidence:
            section.synthesis = await _synthesize_section(topic, section, style)

        sections.append(section)

    all_sources = set()
    all_evidence_count = 0
    for s in sections:
        for e in s.evidence:
            all_sources.add(str(e.source_id))
        all_evidence_count += len(s.evidence)

    quality_metrics = {
        "concepts_explored": 0,
        "concepts_covered": 0,
        "sources_cited": len(all_sources),
        "evidence_chunks": all_evidence_count,
        "sections_total": len(sections),
        "sections_synthesized": sum(1 for s in sections if s.synthesis),
        "has_introduction": False,
        "has_conclusion": False,
        "mode": "search_only",
    }

    return LiteratureReview(
        topic=topic,
        sections=sections,
        quality_metrics=quality_metrics,
        style=style,
        source="search+llm" if use_llm else "search_only",
    )
