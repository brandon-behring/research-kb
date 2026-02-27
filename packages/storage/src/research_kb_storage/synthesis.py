"""Connection explanation and synthesis via graph traversal + evidence hydration.

Provides:
- Path hydration: attach evidence chunks to each concept in a graph path
- LLM synthesis: generate natural language explanations via Anthropic API
- explain_connection(): main orchestrator combining path + evidence + synthesis

Architecture:
- Uses find_shortest_path() from graph_queries for path discovery
- Waterfall evidence strategy: relationship evidence > "defines" chunks > any chunk
- Anthropic Haiku for synthesis (same pattern as assumption_audit.py)
- Graceful degradation: graph_only, no_evidence, no_path, error modes

See CLAUDE.md "MCP Server" section for tool documentation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from research_kb_common import get_logger
from research_kb_contracts import Concept, ConceptRelationship

logger = get_logger(__name__)

# ── Content limits ───────────────────────────────────────────────────────────

EVIDENCE_CONTENT_MAX_CHARS = 300
"""Maximum characters per evidence chunk content (truncated with ellipsis)."""

# ── LLM Prompt ───────────────────────────────────────────────────────────────

CONNECTION_SYNTHESIS_PROMPT = """You are an expert in causal inference, econometrics, and statistical methodology.

Explain how "{concept_a}" relates to "{concept_b}" by synthesizing the path and evidence below.

PATH: {path_explanation}

{evidence_sections}

INSTRUCTIONS:
1. Explain the connection step by step, following the path
2. Ground each step in the provided evidence - cite sources by title and page
3. Highlight key assumptions or conditions at each step
4. Note any practical implications
5. If any step lacks evidence, note the gap explicitly

{style_instructions}

Write 2-4 paragraphs. Cite sources inline as (Author, Year, p. X).
Do NOT invent information not supported by the evidence."""

STYLE_INSTRUCTIONS = {
    "educational": (
        "STYLE: Educational - explain for a graduate student learning these methods. "
        "Focus on intuition and understanding. Define technical terms."
    ),
    "research": (
        "STYLE: Research - explain for a methodologist. "
        "Focus on assumptions, identification conditions, and methodological rigor. "
        "Be precise about what each connection requires."
    ),
    "implementation": (
        "STYLE: Implementation - explain for a practitioner writing code. "
        "Focus on practical steps, parameter choices, and common pitfalls. "
        "Mention relevant packages or algorithms where applicable."
    ),
}


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class EvidenceChunk:
    """A chunk of text providing evidence for a concept in the path."""

    chunk_id: UUID
    source_id: UUID
    source_title: str
    content: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    concept_name: str = ""
    mention_type: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        return {
            "chunk_id": str(self.chunk_id),
            "source_id": str(self.source_id),
            "source_title": self.source_title,
            "content": self.content,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "concept_name": self.concept_name,
            "mention_type": self.mention_type,
        }


@dataclass
class PathStep:
    """A single concept in the explanation path, with optional evidence."""

    concept_id: UUID
    concept_name: str
    concept_type: str
    definition: Optional[str] = None
    relationship_to_next: Optional[str] = None
    evidence: list[EvidenceChunk] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        return {
            "concept_id": str(self.concept_id),
            "concept_name": self.concept_name,
            "concept_type": self.concept_type,
            "definition": self.definition,
            "relationship_to_next": self.relationship_to_next,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass
class ConnectionExplanation:
    """Complete explanation of a connection between two concepts."""

    concept_a: str
    concept_b: str
    path: list[PathStep]
    path_length: int
    path_explanation: str
    synthesis: Optional[str] = None
    synthesis_style: str = "educational"
    source: str = "graph_only"
    confidence: float = 1.0
    evidence_count: int = 0

    def to_dict(self) -> dict:
        """Full serialization for JSON output."""
        return {
            "concept_a": self.concept_a,
            "concept_b": self.concept_b,
            "path": [s.to_dict() for s in self.path],
            "path_length": self.path_length,
            "path_explanation": self.path_explanation,
            "synthesis": self.synthesis,
            "synthesis_style": self.synthesis_style,
            "source": self.source,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
        }


# ── Evidence hydration ───────────────────────────────────────────────────────


def _truncate_content(text: str, max_chars: int = EVIDENCE_CONTENT_MAX_CHARS) -> str:
    """Truncate text content to max_chars with ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


async def hydrate_path_with_evidence(
    path: list[tuple[Concept, Optional[ConceptRelationship]]],
    max_evidence_per_step: int = 2,
) -> list[PathStep]:
    """Attach evidence chunks to each concept in a graph path.

    Waterfall strategy per step:
    1. evidence_chunk_ids from the relationship edge (strongest signal)
    2. "defines" mention chunks from ChunkConceptStore
    3. Any mention chunk (fallback)

    Args:
        path: List of (Concept, Optional[Relationship]) from find_shortest_path
        max_evidence_per_step: Maximum evidence chunks per concept (1-3)

    Returns:
        List of PathStep with hydrated evidence
    """
    from research_kb_storage.chunk_concept_store import ChunkConceptStore
    from research_kb_storage.chunk_store import ChunkStore
    from research_kb_storage.source_store import SourceStore

    max_evidence_per_step = max(1, min(3, max_evidence_per_step))

    # Source title cache to avoid repeated DB lookups
    source_cache: dict[UUID, str] = {}

    async def _get_source_title(source_id: UUID) -> str:
        if source_id in source_cache:
            return source_cache[source_id]
        source = await SourceStore.get_by_id(source_id)
        title = source.title if source else "Unknown Source"
        source_cache[source_id] = title
        return title

    async def _chunk_to_evidence(
        chunk_id: UUID, concept_name: str, mention_type: Optional[str] = None
    ) -> Optional[EvidenceChunk]:
        """Fetch a chunk and convert to EvidenceChunk."""
        chunk = await ChunkStore.get_by_id(chunk_id)
        if not chunk:
            return None
        title = await _get_source_title(chunk.source_id)
        return EvidenceChunk(
            chunk_id=chunk.id,
            source_id=chunk.source_id,
            source_title=title,
            content=_truncate_content(chunk.content),
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            concept_name=concept_name,
            mention_type=mention_type,
        )

    steps: list[PathStep] = []

    for i, (concept, relationship) in enumerate(path):
        concept_name = concept.canonical_name or concept.name
        concept_type = (
            concept.concept_type.value
            if hasattr(concept.concept_type, "value")
            else str(concept.concept_type)
        )

        # Determine relationship to next concept
        rel_to_next = None
        if i < len(path) - 1:
            next_rel = path[i + 1][1]
            if next_rel:
                rel_to_next = (
                    next_rel.relationship_type.value
                    if hasattr(next_rel.relationship_type, "value")
                    else str(next_rel.relationship_type)
                )

        step = PathStep(
            concept_id=concept.id,
            concept_name=concept_name,
            concept_type=concept_type,
            definition=concept.definition,
            relationship_to_next=rel_to_next,
        )

        # ── Waterfall evidence collection ────────────────────────────
        seen_chunk_ids: set[UUID] = set()

        # (1) evidence_chunk_ids from the incoming relationship
        if relationship and relationship.evidence_chunk_ids:
            for eid in relationship.evidence_chunk_ids:
                if len(step.evidence) >= max_evidence_per_step:
                    break
                if eid in seen_chunk_ids:
                    continue
                ev = await _chunk_to_evidence(eid, concept_name, "relationship_evidence")
                if ev:
                    step.evidence.append(ev)
                    seen_chunk_ids.add(eid)

        # (2) "defines" mention chunks
        if len(step.evidence) < max_evidence_per_step:
            try:
                defines_chunks = await ChunkConceptStore.list_chunks_for_concept(
                    concept.id, mention_type="defines", limit=1
                )
                for cc in defines_chunks:
                    if len(step.evidence) >= max_evidence_per_step:
                        break
                    if cc.chunk_id in seen_chunk_ids:
                        continue
                    ev = await _chunk_to_evidence(cc.chunk_id, concept_name, "defines")
                    if ev:
                        step.evidence.append(ev)
                        seen_chunk_ids.add(cc.chunk_id)
            except Exception as e:
                logger.debug("evidence_defines_lookup_failed", concept=concept_name, error=str(e))

        # (3) Any mention chunk (fallback)
        if len(step.evidence) < max_evidence_per_step:
            try:
                any_chunks = await ChunkConceptStore.list_chunks_for_concept(concept.id, limit=1)
                for cc in any_chunks:
                    if len(step.evidence) >= max_evidence_per_step:
                        break
                    if cc.chunk_id in seen_chunk_ids:
                        continue
                    ev = await _chunk_to_evidence(cc.chunk_id, concept_name, cc.mention_type)
                    if ev:
                        step.evidence.append(ev)
                        seen_chunk_ids.add(cc.chunk_id)
            except Exception as e:
                logger.debug("evidence_any_lookup_failed", concept=concept_name, error=str(e))

        steps.append(step)

    return steps


# ── LLM synthesis ────────────────────────────────────────────────────────────


def _build_evidence_sections(steps: list[PathStep]) -> str:
    """Build evidence text sections for the synthesis prompt."""
    sections = []
    for i, step in enumerate(steps, 1):
        section_lines = [f"STEP {i}: {step.concept_name} ({step.concept_type})"]
        if step.definition:
            section_lines.append(f"  Definition: {step.definition[:200]}")
        if step.evidence:
            section_lines.append("  Evidence:")
            for ev in step.evidence:
                page_ref = ""
                if ev.page_start:
                    if ev.page_end and ev.page_end != ev.page_start:
                        page_ref = f", pp. {ev.page_start}-{ev.page_end}"
                    else:
                        page_ref = f", p. {ev.page_start}"
                section_lines.append(f"    - [{ev.source_title}{page_ref}]: {ev.content}")
        else:
            section_lines.append("  Evidence: (none available)")
        if step.relationship_to_next:
            section_lines.append(f"  -> ({step.relationship_to_next}) ->")
        sections.append("\n".join(section_lines))
    return "\n\n".join(sections)


async def synthesize_connection(
    path_steps: list[PathStep],
    concept_a: str,
    concept_b: str,
    path_explanation: str,
    style: str = "educational",
    model: str = "claude-haiku-4-5-20251001",
) -> Optional[str]:
    """Call Anthropic to generate a synthesis explanation.

    Args:
        path_steps: Hydrated path steps with evidence
        concept_a: Starting concept name
        concept_b: Ending concept name
        path_explanation: Human-readable path from explain_path()
        style: One of "educational", "research", "implementation"
        model: Anthropic model name

    Returns:
        Synthesis text, or None on failure
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("synthesis_no_api_key")
        return None

    style_text = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["educational"])
    evidence_text = _build_evidence_sections(path_steps)

    prompt = CONNECTION_SYNTHESIS_PROMPT.format(
        concept_a=concept_a,
        concept_b=concept_b,
        path_explanation=path_explanation,
        evidence_sections=evidence_text,
        style_instructions=style_text,
    )

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
            system="You are an expert in causal inference, econometrics, and statistical methodology. Write clear, evidence-grounded explanations.",
        )

        block = message.content[0]
        if not hasattr(block, "text"):
            raise ValueError(f"Expected text block, got {type(block).__name__}")

        return block.text.strip()

    except ImportError:
        logger.warning("synthesis_anthropic_import_failed", message="pip install anthropic")
        return None
    except Exception as e:
        logger.error("synthesis_llm_failed", error=str(e))
        return None


# ── Main orchestrator ────────────────────────────────────────────────────────


async def explain_connection(
    concept_a: str,
    concept_b: str,
    style: str = "educational",
    max_evidence_per_step: int = 2,
    use_llm: bool = True,
) -> ConnectionExplanation:
    """Explain how two concepts connect through the knowledge graph.

    Main entry point: find path -> hydrate evidence -> LLM synthesize.

    Args:
        concept_a: Name of starting concept (fuzzy matched)
        concept_b: Name of ending concept (fuzzy matched)
        style: Synthesis style - "educational", "research", "implementation"
        max_evidence_per_step: Evidence chunks per path step (1-3)
        use_llm: Whether to call Anthropic for synthesis (False = graph_only)

    Returns:
        ConnectionExplanation with path, evidence, and optional synthesis
    """
    from research_kb_storage.concept_store import ConceptStore
    from research_kb_storage.graph_queries import explain_path, find_shortest_path

    # Validate style
    if style not in STYLE_INSTRUCTIONS:
        style = "educational"

    # ── Resolve concepts ─────────────────────────────────────────
    concepts_a = await ConceptStore.search(query=concept_a, limit=1)
    if not concepts_a:
        return ConnectionExplanation(
            concept_a=concept_a,
            concept_b=concept_b,
            path=[],
            path_length=0,
            path_explanation=f"Concept not found: {concept_a}",
            source="error",
            confidence=0.0,
        )

    concepts_b = await ConceptStore.search(query=concept_b, limit=1)
    if not concepts_b:
        return ConnectionExplanation(
            concept_a=concept_a,
            concept_b=concept_b,
            path=[],
            path_length=0,
            path_explanation=f"Concept not found: {concept_b}",
            source="error",
            confidence=0.0,
        )

    start = concepts_a[0]
    end = concepts_b[0]

    # Use resolved names for display
    start_name = start.canonical_name or start.name
    end_name = end.canonical_name or end.name

    # ── Find path ────────────────────────────────────────────────
    raw_path = await find_shortest_path(start.id, end.id)

    if not raw_path:
        return ConnectionExplanation(
            concept_a=start_name,
            concept_b=end_name,
            path=[],
            path_length=0,
            path_explanation=f"No path found between {start_name} and {end_name}",
            source="no_path",
            confidence=0.0,
        )

    path_explanation = explain_path(raw_path)

    # ── Hydrate evidence ─────────────────────────────────────────
    path_steps = await hydrate_path_with_evidence(raw_path, max_evidence_per_step)
    total_evidence = sum(len(s.evidence) for s in path_steps)
    has_evidence = total_evidence > 0

    # ── Synthesize ───────────────────────────────────────────────
    synthesis = None
    if use_llm:
        synthesis = await synthesize_connection(
            path_steps=path_steps,
            concept_a=start_name,
            concept_b=end_name,
            path_explanation=path_explanation,
            style=style,
        )

    # Determine source label
    if synthesis and has_evidence:
        source_label = "graph+llm"
        confidence = 0.85
    elif synthesis and not has_evidence:
        source_label = "graph+llm_no_evidence"
        confidence = 0.7
    elif not use_llm:
        source_label = "graph_only"
        confidence = 1.0
    else:
        # LLM was requested but failed
        source_label = "graph_only"
        confidence = 1.0

    return ConnectionExplanation(
        concept_a=start_name,
        concept_b=end_name,
        path=path_steps,
        path_length=len(raw_path) - 1,
        path_explanation=path_explanation,
        synthesis=synthesis,
        synthesis_style=style,
        source=source_label,
        confidence=confidence,
        evidence_count=total_evidence,
    )
