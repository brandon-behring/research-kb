"""AssumptionStore - Query method assumptions for auditing.

Phase 4.1a: Foundation for Assumption Auditing (North Star feature)
Phase 4.1b: Ollama fallback when graph returns <3 assumptions
Phase 4.1c: Anthropic backend for higher-quality assumption extraction

Provides:
- Find method concept by name/alias (case-insensitive)
- Query knowledge graph for METHOD → REQUIRES/USES → ASSUMPTION
- LLM extraction fallback when graph insufficient (Ollama or Anthropic)
- Cache LLM-extracted assumptions for future queries
- Return structured Claude-first output format

The goal: Transform research-kb from "filing cabinet" to "PhD collaborator"
by surfacing required assumptions when implementing methods.
"""

from dataclasses import dataclass, field
import json
from typing import Optional
from uuid import UUID

from research_kb_common import StorageError, get_logger
from research_kb_contracts import Concept, ConceptType

from research_kb_storage.connection import get_connection_pool

logger = get_logger(__name__)

# Minimum assumptions from graph before triggering Ollama fallback
MIN_ASSUMPTIONS_THRESHOLD = 3

# Specialized prompt for extracting method assumptions
ASSUMPTION_EXTRACTION_PROMPT = """You are an expert in causal inference and statistical methods.

Given a method name and optional context, extract the key assumptions required for this method to be valid.

METHOD: {method_name}
DEFINITION: {definition}

For each assumption, provide:
1. name: The assumption name (e.g., "unconfoundedness", "parallel trends")
2. formal_statement: Mathematical notation if applicable (e.g., "Y(t) ⊥ T | X")
3. plain_english: Simple explanation (e.g., "No unmeasured confounders")
4. importance: "critical" (identification fails if violated), "standard" (recommended), or "technical" (mathematical regularity)
5. violation_consequence: What goes wrong if violated
6. verification_approaches: How to check this assumption (list of strings)

OUTPUT FORMAT (JSON):
{{
  "assumptions": [
    {{
      "name": "assumption name",
      "formal_statement": "mathematical notation or null",
      "plain_english": "simple explanation",
      "importance": "critical|standard|technical",
      "violation_consequence": "what goes wrong",
      "verification_approaches": ["approach 1", "approach 2"]
    }}
  ]
}}

Focus on assumptions that are:
- Required for causal identification (most important)
- Required for valid inference/estimation
- Technical regularity conditions (if commonly discussed)

Return ONLY valid JSON, no additional text."""


@dataclass
class AssumptionDetail:
    """Structured assumption for Claude-first output.

    Designed for MCP tool consumption and docstring generation.
    """

    name: str
    concept_id: Optional[UUID] = None
    formal_statement: Optional[str] = None  # "Y(t) ⊥ T | X for all t"
    plain_english: Optional[str] = None  # "No unmeasured confounders"
    importance: str = "standard"  # critical, standard, technical
    violation_consequence: Optional[str] = None
    verification_approaches: list[str] = field(default_factory=list)
    source_citation: Optional[str] = None
    relationship_type: Optional[str] = None  # REQUIRES or USES
    confidence: Optional[float] = None


@dataclass
class MethodAssumptions:
    """Complete assumption audit result for a method.

    Claude-first output format for MCP reasoning.
    """

    method: str
    method_id: Optional[UUID] = None
    method_aliases: list[str] = field(default_factory=list)
    definition: Optional[str] = None
    assumptions: list[AssumptionDetail] = field(default_factory=list)
    source: str = "graph"  # graph, cache, ollama
    code_docstring_snippet: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "method": self.method,
            "method_id": str(self.method_id) if self.method_id else None,
            "method_aliases": self.method_aliases,
            "definition": self.definition,
            "assumptions": [
                {
                    "name": a.name,
                    "concept_id": str(a.concept_id) if a.concept_id else None,
                    "formal_statement": a.formal_statement,
                    "plain_english": a.plain_english,
                    "importance": a.importance,
                    "violation_consequence": a.violation_consequence,
                    "verification_approaches": a.verification_approaches,
                    "source_citation": a.source_citation,
                    "relationship_type": a.relationship_type,
                    "confidence": a.confidence,
                }
                for a in self.assumptions
            ],
            "source": self.source,
            "code_docstring_snippet": self.code_docstring_snippet,
        }


class MethodAssumptionAuditor:
    """Storage and query operations for method assumptions.

    Primary query flow:
    1. Find method concept by name/alias
    2. Query graph for REQUIRES/USES → ASSUMPTION relationships
    3. Enrich with cached detailed data if available
    4. Return structured Claude-first format

    Phase 4.1b adds: Ollama fallback when graph returns <3 assumptions.
    """

    @staticmethod
    async def find_method(method_name: str) -> Optional[Concept]:
        """Find method concept by name, canonical name, or alias.

        Case-insensitive matching. Checks:
        1. method_aliases table
        2. concepts.canonical_name
        3. concepts.name
        4. concepts.aliases array

        Args:
            method_name: Query string (e.g., "DML", "double machine learning")

        Returns:
            Matching method Concept or None
        """
        from pgvector.asyncpg import register_vector
        import json

        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await register_vector(conn)
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                # Use the helper function from migration
                row = await conn.fetchrow(
                    """
                    SELECT
                        c.id,
                        c.name,
                        c.canonical_name,
                        c.aliases,
                        c.concept_type,
                        c.definition,
                        c.domain_id,
                        c.category,
                        c.embedding,
                        c.extraction_method,
                        c.confidence_score,
                        c.validated,
                        c.metadata,
                        c.created_at
                    FROM concepts c
                    WHERE c.concept_type = 'method'
                      AND (
                          LOWER(c.canonical_name) = LOWER($1)
                          OR LOWER(c.name) = LOWER($1)
                          OR LOWER($1) = ANY(SELECT LOWER(unnest(c.aliases)))
                      )
                    LIMIT 1
                    """,
                    method_name,
                )

                if row is None:
                    # Try method_aliases table if it exists
                    alias_row = await conn.fetchrow(
                        """
                        SELECT c.id, c.name, c.canonical_name, c.aliases,
                               c.concept_type, c.definition, c.domain_id,
                               c.category, c.embedding, c.extraction_method,
                               c.confidence_score, c.validated, c.metadata,
                               c.created_at
                        FROM method_aliases ma
                        JOIN concepts c ON c.id = ma.method_concept_id
                        WHERE LOWER(ma.alias) = LOWER($1)
                        LIMIT 1
                        """,
                        method_name,
                    )
                    if alias_row:
                        row = alias_row

                if row is None:
                    logger.debug(
                        "method_not_found",
                        query=method_name,
                    )
                    return None

                return _row_to_concept(row)

        except Exception as e:
            # Table might not exist yet, fall back to direct concept search
            if "method_aliases" in str(e):
                logger.debug("method_aliases_table_not_found", falling_back="concepts")
                return await MethodAssumptionAuditor._find_method_fallback(method_name)
            logger.error("find_method_failed", query=method_name, error=str(e))
            raise StorageError(f"Failed to find method: {e}") from e

    @staticmethod
    async def _find_method_fallback(method_name: str) -> Optional[Concept]:
        """Fallback method search without method_aliases table."""
        from pgvector.asyncpg import register_vector
        import json

        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                await register_vector(conn)
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )

                row = await conn.fetchrow(
                    """
                    SELECT *
                    FROM concepts c
                    WHERE c.concept_type = 'method'
                      AND (
                          LOWER(c.canonical_name) = LOWER($1)
                          OR LOWER(c.name) = LOWER($1)
                          OR LOWER($1) = ANY(SELECT LOWER(unnest(c.aliases)))
                      )
                    LIMIT 1
                    """,
                    method_name,
                )

                if row is None:
                    return None

                return _row_to_concept(row)

        except Exception as e:
            logger.error("find_method_fallback_failed", error=str(e))
            raise StorageError(f"Failed to find method: {e}") from e

    @staticmethod
    async def get_assumptions_from_graph(
        method_id: UUID,
        filter_by_domain: bool = True,
    ) -> list[AssumptionDetail]:
        """Query knowledge graph for method assumptions.

        Finds concepts connected via REQUIRES or USES relationships
        where the target concept has type 'assumption'.

        Args:
            method_id: UUID of the method concept
            filter_by_domain: If True, only return assumptions from the same domain
                            as the method (prevents cross-domain contamination).
                            Default True to avoid physics concepts in causal methods.

        Returns:
            List of AssumptionDetail objects from graph
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                # Build query with optional domain filtering
                # This prevents cross-domain contamination (e.g., physics in DML)
                if filter_by_domain:
                    query = """
                    SELECT
                        c.id AS assumption_id,
                        c.name AS assumption_name,
                        c.canonical_name,
                        c.definition,
                        cr.relationship_type,
                        cr.strength,
                        cr.confidence_score,
                        cr.evidence_chunk_ids
                    FROM concept_relationships cr
                    JOIN concepts c ON c.id = cr.target_concept_id
                    JOIN concepts method ON method.id = cr.source_concept_id
                    WHERE cr.source_concept_id = $1
                      AND cr.relationship_type IN ('REQUIRES', 'USES')
                      AND c.concept_type = 'assumption'
                      AND (c.domain_id IS NULL OR method.domain_id IS NULL OR c.domain_id = method.domain_id)
                    ORDER BY
                        CASE cr.relationship_type
                            WHEN 'REQUIRES' THEN 1
                            WHEN 'USES' THEN 2
                        END,
                        cr.strength DESC
                    """
                else:
                    query = """
                    SELECT
                        c.id AS assumption_id,
                        c.name AS assumption_name,
                        c.canonical_name,
                        c.definition,
                        cr.relationship_type,
                        cr.strength,
                        cr.confidence_score,
                        cr.evidence_chunk_ids
                    FROM concept_relationships cr
                    JOIN concepts c ON c.id = cr.target_concept_id
                    WHERE cr.source_concept_id = $1
                      AND cr.relationship_type IN ('REQUIRES', 'USES')
                      AND c.concept_type = 'assumption'
                    ORDER BY
                        CASE cr.relationship_type
                            WHEN 'REQUIRES' THEN 1
                            WHEN 'USES' THEN 2
                        END,
                        cr.strength DESC
                    """
                rows = await conn.fetch(query, method_id)

                assumptions = []
                for row in rows:
                    # Determine importance based on relationship type
                    importance = (
                        "critical" if row["relationship_type"] == "REQUIRES" else "standard"
                    )

                    assumptions.append(
                        AssumptionDetail(
                            name=row["assumption_name"],
                            concept_id=row["assumption_id"],
                            plain_english=row["definition"],
                            importance=importance,
                            relationship_type=row["relationship_type"],
                            confidence=row["confidence_score"],
                        )
                    )

                logger.info(
                    "graph_assumptions_retrieved",
                    method_id=str(method_id),
                    count=len(assumptions),
                )

                return assumptions

        except Exception as e:
            logger.error(
                "get_assumptions_from_graph_failed",
                method_id=str(method_id),
                error=str(e),
            )
            raise StorageError(f"Failed to get assumptions from graph: {e}") from e

    @staticmethod
    async def get_cached_assumptions(
        method_id: UUID,
    ) -> list[AssumptionDetail]:
        """Get cached LLM-extracted assumption details.

        Returns enriched assumption data from method_assumption_cache table
        if available. Includes formal statements, verification approaches, etc.

        Args:
            method_id: UUID of the method concept

        Returns:
            List of cached AssumptionDetail objects
        """
        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                # Check if table exists
                table_exists = await conn.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'method_assumption_cache'
                    )
                    """
                )

                if not table_exists:
                    logger.debug("method_assumption_cache_not_found")
                    return []

                rows = await conn.fetch(
                    """
                    SELECT
                        assumption_name,
                        assumption_concept_id,
                        formal_statement,
                        plain_english,
                        importance,
                        violation_consequence,
                        verification_approaches,
                        source_citation,
                        confidence_score
                    FROM method_assumption_cache
                    WHERE method_concept_id = $1
                    ORDER BY
                        CASE importance
                            WHEN 'critical' THEN 1
                            WHEN 'standard' THEN 2
                            WHEN 'technical' THEN 3
                        END
                    """,
                    method_id,
                )

                assumptions = []
                for row in rows:
                    assumptions.append(
                        AssumptionDetail(
                            name=row["assumption_name"],
                            concept_id=row["assumption_concept_id"],
                            formal_statement=row["formal_statement"],
                            plain_english=row["plain_english"],
                            importance=row["importance"] or "standard",
                            violation_consequence=row["violation_consequence"],
                            verification_approaches=row["verification_approaches"] or [],
                            source_citation=row["source_citation"],
                            confidence=row["confidence_score"],
                        )
                    )

                return assumptions

        except Exception as e:
            logger.error(
                "get_cached_assumptions_failed",
                method_id=str(method_id),
                error=str(e),
            )
            # Don't raise - cache is optional enhancement
            return []

    @staticmethod
    async def extract_assumptions_with_ollama(
        method_name: str,
        definition: Optional[str] = None,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
    ) -> list[AssumptionDetail]:
        """Extract assumptions using Ollama LLM.

        Called when graph returns fewer than MIN_ASSUMPTIONS_THRESHOLD results.
        Uses specialized prompt to extract assumptions for the method.

        Args:
            method_name: Name of the method
            definition: Optional definition to provide context
            model: Ollama model name
            base_url: Ollama server URL

        Returns:
            List of AssumptionDetail objects from LLM
        """
        import httpx

        prompt = ASSUMPTION_EXTRACTION_PROMPT.format(
            method_name=method_name,
            definition=definition or "Not provided",
        )

        try:
            async with httpx.AsyncClient(
                base_url=base_url,
                timeout=httpx.Timeout(60.0),
            ) as client:
                # Check if Ollama is available
                try:
                    tags_response = await client.get("/api/tags")
                    if tags_response.status_code != 200:
                        logger.warning("ollama_unavailable", status=tags_response.status_code)
                        return []
                except Exception as e:
                    logger.warning("ollama_connection_failed", error=str(e))
                    return []

                # Generate assumptions
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 4096,
                    },
                }

                response = await client.post("/api/generate", json=payload)
                response.raise_for_status()

                data = response.json()
                response_text = data.get("response", "")

                # Parse JSON response
                parsed = json.loads(response_text)
                raw_assumptions = parsed.get("assumptions", [])

                assumptions = []
                for raw in raw_assumptions:
                    # Validate importance
                    importance = raw.get("importance", "standard")
                    if importance not in ("critical", "standard", "technical"):
                        importance = "standard"

                    assumptions.append(
                        AssumptionDetail(
                            name=raw.get("name", "unknown"),
                            formal_statement=raw.get("formal_statement"),
                            plain_english=raw.get("plain_english"),
                            importance=importance,
                            violation_consequence=raw.get("violation_consequence"),
                            verification_approaches=raw.get("verification_approaches", []),
                            confidence=0.7,  # LLM-extracted, moderate confidence
                        )
                    )

                logger.info(
                    "ollama_assumptions_extracted",
                    method=method_name,
                    count=len(assumptions),
                )

                return assumptions

        except json.JSONDecodeError as e:
            logger.error("ollama_json_parse_error", error=str(e))
            return []
        except Exception as e:
            logger.error("ollama_extraction_failed", error=str(e))
            return []

    @staticmethod
    async def extract_assumptions_with_anthropic(
        method_name: str,
        definition: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
    ) -> list[AssumptionDetail]:
        """Extract assumptions using Anthropic API (Haiku).

        Higher quality than Ollama, especially for structured JSON output.
        Used as LLM fallback when graph returns fewer than MIN_ASSUMPTIONS_THRESHOLD.

        Args:
            method_name: Name of the method
            definition: Optional definition to provide context
            model: Anthropic model name (default: Haiku 4.5)

        Returns:
            List of AssumptionDetail objects from LLM
        """
        prompt = ASSUMPTION_EXTRACTION_PROMPT.format(
            method_name=method_name,
            definition=definition or "Not provided",
        )

        try:
            import os

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("anthropic_no_api_key")
                return []

            import anthropic

            client = anthropic.Anthropic(api_key=api_key)

            message = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
                system="You are an expert in causal inference and econometrics. Return ONLY valid JSON.",
            )

            response_text = message.content[0].text

            # Strip markdown code fences if present (Haiku often wraps JSON)
            response_text = response_text.strip()
            if response_text.startswith("```"):
                # Remove opening fence (```json or ```)
                first_newline = response_text.index("\n")
                response_text = response_text[first_newline + 1 :]
                # Remove closing fence
                if response_text.endswith("```"):
                    response_text = response_text[:-3].strip()

            # Parse JSON response
            parsed = json.loads(response_text)
            raw_assumptions = parsed.get("assumptions", [])

            assumptions = []
            for raw in raw_assumptions:
                # Validate importance
                importance = raw.get("importance", "standard")
                if importance not in ("critical", "standard", "technical"):
                    importance = "standard"

                assumptions.append(
                    AssumptionDetail(
                        name=raw.get("name", "unknown"),
                        formal_statement=raw.get("formal_statement"),
                        plain_english=raw.get("plain_english"),
                        importance=importance,
                        violation_consequence=raw.get("violation_consequence"),
                        verification_approaches=raw.get("verification_approaches", []),
                        confidence=0.85,  # Higher than Ollama (0.7) — Haiku is more reliable
                    )
                )

            logger.info(
                "anthropic_assumptions_extracted",
                method=method_name,
                model=model,
                count=len(assumptions),
            )

            return assumptions

        except ImportError:
            logger.warning(
                "anthropic_import_failed",
                message="pip install anthropic",
            )
            return []
        except json.JSONDecodeError as e:
            logger.error("anthropic_json_parse_error", error=str(e))
            return []
        except Exception as e:
            logger.error("anthropic_extraction_failed", error=str(e))
            return []

    @staticmethod
    async def cache_assumptions(
        method_id: UUID,
        assumptions: list[AssumptionDetail],
        extraction_method: str = "ollama:llama3.1:8b",
    ) -> int:
        """Cache extracted assumptions to database.

        Stores assumptions in method_assumption_cache table for future queries.
        Uses INSERT ON CONFLICT to update existing entries.

        Args:
            method_id: UUID of the method concept
            assumptions: List of assumptions to cache
            extraction_method: Source of extraction (e.g., "ollama:llama3.1:8b")

        Returns:
            Number of assumptions cached
        """
        if not assumptions:
            return 0

        pool = await get_connection_pool()

        try:
            async with pool.acquire() as conn:
                # Check if table exists
                table_exists = await conn.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'method_assumption_cache'
                    )
                    """
                )

                if not table_exists:
                    logger.warning("method_assumption_cache_table_not_found")
                    return 0

                cached_count = 0
                for a in assumptions:
                    await conn.execute(
                        """
                        INSERT INTO method_assumption_cache (
                            method_concept_id,
                            assumption_name,
                            assumption_concept_id,
                            formal_statement,
                            plain_english,
                            importance,
                            violation_consequence,
                            verification_approaches,
                            source_citation,
                            extraction_method,
                            confidence_score
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        ON CONFLICT (method_concept_id, assumption_name)
                        DO UPDATE SET
                            formal_statement = COALESCE(EXCLUDED.formal_statement, method_assumption_cache.formal_statement),
                            plain_english = COALESCE(EXCLUDED.plain_english, method_assumption_cache.plain_english),
                            importance = EXCLUDED.importance,
                            violation_consequence = COALESCE(EXCLUDED.violation_consequence, method_assumption_cache.violation_consequence),
                            verification_approaches = COALESCE(EXCLUDED.verification_approaches, method_assumption_cache.verification_approaches),
                            extraction_method = EXCLUDED.extraction_method,
                            confidence_score = EXCLUDED.confidence_score,
                            updated_at = NOW()
                        """,
                        method_id,
                        a.name,
                        a.concept_id,
                        a.formal_statement,
                        a.plain_english,
                        a.importance,
                        a.violation_consequence,
                        (a.verification_approaches if a.verification_approaches else None),
                        a.source_citation,
                        extraction_method,
                        a.confidence,
                    )
                    cached_count += 1

                logger.info(
                    "assumptions_cached",
                    method_id=str(method_id),
                    count=cached_count,
                )

                return cached_count

        except Exception as e:
            logger.error(
                "cache_assumptions_failed",
                method_id=str(method_id),
                error=str(e),
            )
            # Don't raise - caching is optional
            return 0

    @staticmethod
    async def audit_assumptions(
        method_name: str,
        use_ollama_fallback: bool = True,
        use_llm_fallback: bool = True,
        llm_backend: str = "ollama",
        filter_by_domain: bool = True,
    ) -> MethodAssumptions:
        """Main entry point: Get assumptions for a method.

        Query flow:
        1. Find method concept by name/alias
        2. Query graph for REQUIRES/USES → ASSUMPTION (with domain filtering)
        3. Enrich with cached details if available
        4. If <3 assumptions and LLM fallback enabled: extract via chosen backend + cache
        5. Generate code_docstring_snippet

        Args:
            method_name: Method name, abbreviation, or alias
            use_ollama_fallback: Legacy parameter — enables Ollama extraction for sparse results.
                               Kept for backward compatibility. Equivalent to
                               use_llm_fallback=True, llm_backend="ollama".
            use_llm_fallback: Enable LLM extraction for sparse results (default True)
            llm_backend: LLM backend to use: "ollama" or "anthropic" (default "ollama")
            filter_by_domain: If True (default), only return assumptions from the
                            same domain as the method. This prevents cross-domain
                            contamination (e.g., physics concepts in causal methods).

        Returns:
            MethodAssumptions with full audit data

        Raises:
            StorageError: If method not found or query fails
        """
        # Step 1: Find the method
        method = await MethodAssumptionAuditor.find_method(method_name)

        if method is None:
            logger.warning("audit_assumptions_method_not_found", query=method_name)
            return MethodAssumptions(
                method=method_name,
                assumptions=[],
                source="not_found",
                code_docstring_snippet=f"# Method '{method_name}' not found in knowledge base",
            )

        # Step 2: Get assumptions from graph (with domain filtering)
        graph_assumptions = await MethodAssumptionAuditor.get_assumptions_from_graph(
            method.id,
            filter_by_domain=filter_by_domain,
        )

        # Step 3: Try to enrich with cached details
        cached_assumptions = await MethodAssumptionAuditor.get_cached_assumptions(method.id)

        # Merge: cached data takes precedence for matching names
        cached_by_name = {a.name.lower(): a for a in cached_assumptions}
        final_assumptions = []

        for ga in graph_assumptions:
            cached = cached_by_name.get(ga.name.lower())
            if cached:
                # Merge: keep graph's concept_id and relationship_type,
                # use cached's enriched fields
                final_assumptions.append(
                    AssumptionDetail(
                        name=ga.name,
                        concept_id=ga.concept_id,
                        formal_statement=cached.formal_statement,
                        plain_english=cached.plain_english or ga.plain_english,
                        importance=cached.importance,
                        violation_consequence=cached.violation_consequence,
                        verification_approaches=cached.verification_approaches,
                        source_citation=cached.source_citation,
                        relationship_type=ga.relationship_type,
                        confidence=ga.confidence,
                    )
                )
            else:
                final_assumptions.append(ga)

        # Add any cached assumptions not in graph (manually curated)
        graph_names = {a.name.lower() for a in graph_assumptions}
        for cached in cached_assumptions:
            if cached.name.lower() not in graph_names:
                final_assumptions.append(cached)

        # Determine source
        source = "graph" if graph_assumptions else ("cache" if cached_assumptions else "empty")

        # Step 4: LLM fallback if insufficient assumptions
        # Resolve effective settings: use_ollama_fallback is legacy shorthand
        effective_llm_fallback = use_llm_fallback and (
            use_ollama_fallback or llm_backend != "ollama"
        )
        effective_backend = llm_backend

        if effective_llm_fallback and len(final_assumptions) < MIN_ASSUMPTIONS_THRESHOLD:
            logger.info(
                "triggering_llm_fallback",
                method=method.canonical_name,
                backend=effective_backend,
                current_count=len(final_assumptions),
                threshold=MIN_ASSUMPTIONS_THRESHOLD,
            )

            # Dispatch to the chosen backend
            if effective_backend == "anthropic":
                llm_assumptions = await MethodAssumptionAuditor.extract_assumptions_with_anthropic(
                    method_name=method.canonical_name or method.name,
                    definition=method.definition,
                )
                extraction_method_str = "anthropic:claude-haiku-4-5-20251001"
            else:
                llm_assumptions = await MethodAssumptionAuditor.extract_assumptions_with_ollama(
                    method_name=method.canonical_name or method.name,
                    definition=method.definition,
                )
                extraction_method_str = "ollama:llama3.1:8b"

            if llm_assumptions:
                # Cache the LLM results for future queries
                await MethodAssumptionAuditor.cache_assumptions(
                    method_id=method.id,
                    assumptions=llm_assumptions,
                    extraction_method=extraction_method_str,
                )

                # Merge LLM assumptions (don't duplicate by name)
                existing_names = {a.name.lower() for a in final_assumptions}
                for la in llm_assumptions:
                    if la.name.lower() not in existing_names:
                        final_assumptions.append(la)
                        existing_names.add(la.name.lower())

                backend_label = effective_backend
                source = f"graph+{backend_label}" if graph_assumptions else backend_label

        # Step 5: Generate code docstring snippet
        docstring = _generate_docstring_snippet(
            method.canonical_name or method.name, final_assumptions
        )

        result = MethodAssumptions(
            method=method.canonical_name or method.name,
            method_id=method.id,
            method_aliases=method.aliases or [],
            definition=method.definition,
            assumptions=final_assumptions,
            source=source,
            code_docstring_snippet=docstring,
        )

        logger.info(
            "audit_assumptions_complete",
            method=method.canonical_name,
            assumption_count=len(final_assumptions),
            source=source,
        )

        return result


def _generate_docstring_snippet(method_name: str, assumptions: list[AssumptionDetail]) -> str:
    """Generate a code docstring snippet for Claude to use.

    Creates a ready-to-paste docstring section listing assumptions
    with their importance levels.

    Args:
        method_name: Canonical method name
        assumptions: List of assumptions to include

    Returns:
        Docstring snippet string
    """
    if not assumptions:
        return f"# No assumptions documented for {method_name}"

    lines = ["Assumptions:"]

    # Group by importance
    critical = [a for a in assumptions if a.importance == "critical"]
    standard = [a for a in assumptions if a.importance == "standard"]
    technical = [a for a in assumptions if a.importance == "technical"]

    for group, label in [
        (critical, "[CRITICAL]"),
        (standard, ""),
        (technical, "[technical]"),
    ]:
        for a in group:
            prefix = f"    {label} " if label else "    "
            name_part = a.name
            if a.plain_english:
                lines.append(f"{prefix}- {name_part}: {a.plain_english}")
            else:
                lines.append(f"{prefix}- {name_part}")

    return "\n".join(lines)


def _row_to_concept(row) -> Concept:
    """Convert database row to Concept model."""
    from research_kb_contracts import Concept

    return Concept(
        id=row["id"],
        name=row["name"],
        canonical_name=row["canonical_name"],
        aliases=row["aliases"] or [],
        concept_type=ConceptType(row["concept_type"]),
        domain_id=row.get("domain_id", "causal_inference"),
        category=row.get("category"),
        definition=row.get("definition"),
        embedding=list(row["embedding"]) if row.get("embedding") is not None else None,
        extraction_method=row.get("extraction_method"),
        confidence_score=row.get("confidence_score"),
        validated=row.get("validated", False),
        metadata=row.get("metadata") or {},
        created_at=row.get("created_at"),
    )
