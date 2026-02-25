"""Query Expansion - Expand user queries for improved recall.

Provides four expansion strategies:
1. Synonym expansion - Deterministic lookup from domain-specific synonym map
2. Graph expansion - Leverage knowledge graph for related concepts (1-hop)
3. LLM expansion - Optional semantic expansion via Ollama
4. HyDE (Hypothetical Document Embeddings) - Generate hypothetical answer for better vector search

Design decisions:
- Original terms boosted 2x in FTS query (precision preserved, recall added)
- LLM expansion gracefully falls back if Ollama unavailable
- Graph expansion limited to 1-hop to avoid noise
- HyDE improves terse queries by 5-10% (configurable backend: Ollama/Anthropic)

Master Plan Reference: Phase 3 Enhanced Retrieval
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import yaml

from research_kb_common import get_logger

if TYPE_CHECKING:
    from research_kb_extraction.ollama_client import OllamaClient

logger = get_logger(__name__)

# Default path for synonym map
DEFAULT_SYNONYM_MAP_PATH = (
    Path(__file__).parent.parent.parent.parent.parent / "fixtures" / "concepts" / "synonym_map.yaml"
)


@dataclass
class ExpandedQuery:
    """Container for expanded query information.

    Attributes:
        original: Original user query text
        expanded_terms: List of all expansion terms (without original)
        fts_query: PostgreSQL FTS query string with boosting
        expansion_sources: Dict mapping source -> list of terms added
    """

    original: str
    expanded_terms: list[str] = field(default_factory=list)
    fts_query: str = ""
    expansion_sources: dict[str, list[str]] = field(default_factory=dict)

    @property
    def all_terms(self) -> list[str]:
        """Get all terms (original + expanded)."""
        return [self.original] + self.expanded_terms

    @property
    def expansion_count(self) -> int:
        """Total number of expansion terms added."""
        return len(self.expanded_terms)


class QueryExpander:
    """Expand queries using synonyms, graph, and optional LLM.

    Example:
        >>> expander = QueryExpander.from_yaml("fixtures/concepts/synonym_map.yaml")
        >>> result = await expander.expand("IV estimation")
        >>> print(result.fts_query)
        'IV:A | estimation:A | instrumental:B | variables:B'
    """

    def __init__(
        self,
        synonym_map: Optional[dict[str, list[str]]] = None,
        ollama_client: Optional["OllamaClient"] = None,
    ):
        """Initialize query expander.

        Args:
            synonym_map: Dict mapping terms to synonyms (case-insensitive keys)
            ollama_client: Optional OllamaClient for LLM expansion
        """
        # Normalize synonym map keys to lowercase
        self.synonym_map: dict[str, list[str]] = {}
        if synonym_map:
            for key, values in synonym_map.items():
                self.synonym_map[key.lower()] = [v.lower() for v in values]

        self.ollama_client = ollama_client

    @classmethod
    def from_yaml(
        cls,
        yaml_path: Optional[Path] = None,
        ollama_client: Optional["OllamaClient"] = None,
    ) -> "QueryExpander":
        """Create expander from YAML synonym file.

        Args:
            yaml_path: Path to synonym_map.yaml (defaults to fixtures location)
            ollama_client: Optional OllamaClient for LLM expansion

        Returns:
            Configured QueryExpander instance

        Example:
            >>> expander = QueryExpander.from_yaml()
            >>> expander.synonym_map.get("iv")
            ['instrumental variables', 'instrumental variable', '2sls', 'two-stage least squares']
        """
        if yaml_path is None:
            yaml_path = DEFAULT_SYNONYM_MAP_PATH

        synonym_map: dict[str, Any] = {}
        if yaml_path.exists():
            try:
                with open(yaml_path, "r") as f:
                    data = yaml.safe_load(f) or {}
                    synonym_map = data
                    logger.info(
                        "synonym_map_loaded",
                        path=str(yaml_path),
                        entry_count=len(synonym_map),
                    )
            except Exception as e:
                logger.warning(
                    "synonym_map_load_failed",
                    path=str(yaml_path),
                    error=str(e),
                )
        else:
            logger.warning(
                "synonym_map_not_found",
                path=str(yaml_path),
            )

        return cls(synonym_map=synonym_map, ollama_client=ollama_client)

    def expand_with_synonyms(self, query: str) -> list[str]:
        """Expand query using synonym map.

        Performs case-insensitive matching against synonym map keys.
        Returns synonyms for all matching terms in query.

        Args:
            query: User query text

        Returns:
            List of synonym terms (not including original query terms)

        Example:
            >>> expander.expand_with_synonyms("IV for endogeneity")
            ['instrumental variables', 'instrumental variable', '2sls', ...]
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        expansions = []
        matched_keys = set()

        # Check full query against synonyms (for multi-word matches)
        if query_lower in self.synonym_map:
            expansions.extend(self.synonym_map[query_lower])
            matched_keys.add(query_lower)

        # Check individual words
        for word in query_words:
            if word in self.synonym_map and word not in matched_keys:
                for synonym in self.synonym_map[word]:
                    if synonym.lower() not in query_lower:  # Avoid duplicating query terms
                        expansions.append(synonym)
                matched_keys.add(word)

        # Also check if query contains any synonym map keys (partial match)
        for key, synonyms in self.synonym_map.items():
            if key in query_lower and key not in matched_keys:
                for synonym in synonyms:
                    if synonym.lower() not in query_lower:
                        expansions.append(synonym)

        # Deduplicate while preserving order
        seen = set()
        unique_expansions = []
        for term in expansions:
            term_lower = term.lower()
            if term_lower not in seen and term_lower not in query_lower:
                seen.add(term_lower)
                unique_expansions.append(term)

        return unique_expansions

    async def expand_with_graph(
        self,
        query: str,
        max_concepts: int = 3,
    ) -> list[str]:
        """Expand query using knowledge graph relationships.

        Finds concepts mentioned in query and retrieves 1-hop neighbors.

        Args:
            query: User query text
            max_concepts: Maximum number of concept names to add

        Returns:
            List of related concept names from graph
        """
        try:
            from research_kb_storage.query_extractor import extract_query_concepts
            from research_kb_storage.graph_queries import get_neighborhood

            # Extract concepts from query
            concept_ids = await extract_query_concepts(query, max_concepts=3)

            if not concept_ids:
                return []

            expansions = []
            query_lower = query.lower()

            for concept_id in concept_ids[:2]:  # Limit to 2 concepts
                try:
                    # Get 1-hop neighbors
                    neighborhood = await get_neighborhood(concept_id, hops=1)

                    for concept in neighborhood.get("concepts", []):
                        name = concept.canonical_name or concept.name
                        # Avoid adding terms already in query
                        if name and name.lower() not in query_lower:
                            expansions.append(name)

                        if len(expansions) >= max_concepts:
                            break

                except Exception as e:
                    logger.debug(
                        "graph_expansion_concept_failed",
                        concept_id=str(concept_id),
                        error=str(e),
                    )
                    continue

                if len(expansions) >= max_concepts:
                    break

            # Deduplicate
            seen = set()
            unique = []
            for term in expansions:
                if term.lower() not in seen:
                    seen.add(term.lower())
                    unique.append(term)

            return unique[:max_concepts]

        except Exception as e:
            logger.warning(
                "graph_expansion_failed",
                query=query[:100],
                error=str(e),
            )
            return []

    async def expand_with_llm(
        self,
        query: str,
        max_terms: int = 3,
        domain: str | None = None,
    ) -> list[str]:
        """Expand query using LLM (Ollama).

        Generates semantically related terms for complex queries.
        Gracefully returns empty list if Ollama unavailable.

        Args:
            query: User query text
            max_terms: Maximum terms to generate
            domain: Knowledge domain for context (e.g. "causal inference")

        Returns:
            List of LLM-generated expansion terms
        """
        if self.ollama_client is None:
            return []

        try:
            # Check if Ollama is available
            if not await self.ollama_client.is_available():
                logger.debug("ollama_unavailable_for_expansion")
                return []

            domain_context = f" about {domain}" if domain else ""
            prompt = f"""You are expanding a search query{domain_context}.

Given the query: "{query}"

List up to {max_terms} related technical terms or synonyms that would help find relevant documents.
Return ONLY a JSON array of strings, no explanation.

Example: ["term1", "term2", "term3"]
"""

            response = await self.ollama_client.generate(
                prompt=prompt,
                system="You are a helpful assistant that returns only valid JSON arrays.",
                json_mode=True,
            )

            import json

            terms = json.loads(response)

            if isinstance(terms, list):
                # Filter out terms already in query
                query_lower = query.lower()
                filtered = [t for t in terms if isinstance(t, str) and t.lower() not in query_lower]
                return filtered[:max_terms]

            return []

        except Exception as e:
            logger.warning(
                "llm_expansion_failed",
                query=query[:100],
                error=str(e),
            )
            return []

    def build_fts_query(
        self,
        original: str,
        expansions: list[str],
        original_weight: str = "A",
        expansion_weight: str = "B",
    ) -> str:
        """Build PostgreSQL FTS query with term weighting.

        Original terms get weight A (highest), expansions get weight B.
        This implements the 2x boost for original terms.

        Args:
            original: Original query text
            expansions: Expansion terms
            original_weight: FTS weight for original (A=highest)
            expansion_weight: FTS weight for expansions

        Returns:
            PostgreSQL FTS query string

        Example:
            >>> expander.build_fts_query("IV estimation", ["instrumental variables"])
            'IV:A | estimation:A | instrumental:B | variables:B'
        """
        parts = []

        # Add original query words with highest weight
        for word in original.split():
            # Escape special FTS characters
            word_clean = self._escape_fts(word)
            if word_clean:
                parts.append(f"{word_clean}:{original_weight}")

        # Add expansion terms with lower weight
        for term in expansions:
            for word in term.split():
                word_clean = self._escape_fts(word)
                if word_clean:
                    parts.append(f"{word_clean}:{expansion_weight}")

        return " | ".join(parts) if parts else ""

    def _escape_fts(self, term: str) -> str:
        """Escape term for PostgreSQL FTS.

        Removes special characters that would break FTS parsing.
        """
        # Remove common problematic characters
        import re

        cleaned = re.sub(r"[^\w\s-]", "", term)
        return cleaned.strip()

    async def expand(
        self,
        query: str,
        use_synonyms: bool = True,
        use_graph: bool = True,
        use_llm: bool = False,
    ) -> ExpandedQuery:
        """Expand query using configured strategies.

        Args:
            query: User query text
            use_synonyms: Enable synonym expansion (fast, deterministic)
            use_graph: Enable graph expansion (~10ms)
            use_llm: Enable LLM expansion (optional, slower)

        Returns:
            ExpandedQuery with all expansion information

        Example:
            >>> result = await expander.expand("IV", use_synonyms=True, use_graph=True)
            >>> print(result.expansion_sources)
            {'synonyms': ['instrumental variables', '2sls'], 'graph': ['endogeneity']}
        """
        if not query or not query.strip():
            return ExpandedQuery(original=query)

        query = query.strip()
        all_expansions = []
        expansion_sources: dict[str, list[str]] = {}

        # 1. Synonym expansion (instant)
        if use_synonyms:
            synonym_terms = self.expand_with_synonyms(query)
            if synonym_terms:
                all_expansions.extend(synonym_terms)
                expansion_sources["synonyms"] = synonym_terms
                logger.debug(
                    "synonym_expansion",
                    query=query,
                    terms=synonym_terms,
                )

        # 2. Graph expansion (~10ms)
        if use_graph:
            graph_terms = await self.expand_with_graph(query)
            if graph_terms:
                # Avoid duplicates with synonyms
                new_terms = [
                    t for t in graph_terms if t.lower() not in {e.lower() for e in all_expansions}
                ]
                if new_terms:
                    all_expansions.extend(new_terms)
                    expansion_sources["graph"] = new_terms
                    logger.debug(
                        "graph_expansion",
                        query=query,
                        terms=new_terms,
                    )

        # 3. LLM expansion (optional, ~200-500ms)
        if use_llm:
            llm_terms = await self.expand_with_llm(query)
            if llm_terms:
                # Avoid duplicates
                new_terms = [
                    t for t in llm_terms if t.lower() not in {e.lower() for e in all_expansions}
                ]
                if new_terms:
                    all_expansions.extend(new_terms)
                    expansion_sources["llm"] = new_terms
                    logger.debug(
                        "llm_expansion",
                        query=query,
                        terms=new_terms,
                    )

        # Build FTS query with boosting
        fts_query = self.build_fts_query(query, all_expansions)

        result = ExpandedQuery(
            original=query,
            expanded_terms=all_expansions,
            fts_query=fts_query,
            expansion_sources=expansion_sources,
        )

        logger.info(
            "query_expanded",
            original=query,
            expansion_count=len(all_expansions),
            sources=list(expansion_sources.keys()),
        )

        return result


# Module-level convenience function
async def expand_query(
    query: str,
    use_synonyms: bool = True,
    use_graph: bool = True,
    use_llm: bool = False,
    synonym_map_path: Optional[Path] = None,
) -> ExpandedQuery:
    """Convenience function to expand a query.

    Creates a QueryExpander instance and expands the query.
    For repeated expansions, prefer creating a QueryExpander instance directly.

    Args:
        query: User query text
        use_synonyms: Enable synonym expansion
        use_graph: Enable graph expansion
        use_llm: Enable LLM expansion
        synonym_map_path: Optional path to synonym YAML

    Returns:
        ExpandedQuery with expansion information
    """
    expander = QueryExpander.from_yaml(synonym_map_path)

    # Add Ollama client if LLM requested
    if use_llm:
        try:
            from research_kb_extraction.ollama_client import OllamaClient

            expander.ollama_client = OllamaClient()
        except ImportError:
            logger.warning("ollama_client_import_failed")

    return await expander.expand(
        query,
        use_synonyms=use_synonyms,
        use_graph=use_graph,
        use_llm=use_llm,
    )


# ============================================================================
# HyDE (Hypothetical Document Embeddings)
# ============================================================================


@dataclass
class HydeConfig:
    """Configuration for HyDE query expansion.

    HyDE generates a hypothetical document that would answer the query,
    then embeds that document instead of the raw query. This improves
    retrieval for terse queries like "IV" or "DML".

    Attributes:
        enabled: Enable HyDE expansion
        backend: LLM backend ('ollama' or 'anthropic')
        model: Model name (backend-specific)
        max_length: Maximum hypothetical document length
    """

    enabled: bool = False
    backend: str = "ollama"  # 'ollama' or 'anthropic'
    model: str = "llama3.1:8b"  # For Ollama; 'claude-3-5-haiku-20241022' for Anthropic
    max_length: int = 200  # Words in hypothetical document


HYDE_PROMPT_TEMPLATE = """You are a research assistant{domain_context}.

Given a search query, generate a hypothetical paragraph from a textbook or paper that would directly answer this query. Write as if you are quoting from an authoritative source.

Query: {query}

Write a {max_length}-word paragraph that would appear in a research textbook answering this query. Be technical and precise. Do not include any preamble or explanation - just write the hypothetical passage."""


async def generate_hyde_document(
    query: str,
    config: Optional[HydeConfig] = None,
    domain: str | None = None,
) -> Optional[str]:
    """Generate hypothetical document for HyDE query expansion.

    Args:
        query: User search query
        config: HyDE configuration (defaults to Ollama backend)
        domain: Knowledge domain for context (e.g. "causal inference")

    Returns:
        Hypothetical document text, or None if generation fails

    Example:
        >>> doc = await generate_hyde_document("instrumental variables", domain="causal inference")
        >>> print(doc[:100])
        'Instrumental variables (IV) estimation is a technique used to...'
    """
    if config is None:
        config = HydeConfig(enabled=True)

    if not config.enabled:
        return None

    domain_context = f" specializing in {domain}" if domain else ""
    prompt = HYDE_PROMPT_TEMPLATE.format(
        query=query,
        max_length=config.max_length,
        domain_context=domain_context,
    )

    try:
        if config.backend == "ollama":
            return await _generate_hyde_ollama(prompt, config.model)
        elif config.backend == "anthropic":
            return await _generate_hyde_anthropic(prompt, config.model)
        else:
            logger.warning("hyde_unknown_backend", backend=config.backend)
            return None

    except Exception as e:
        logger.warning(
            "hyde_generation_failed",
            query=query[:100],
            backend=config.backend,
            error=str(e),
        )
        return None


async def _generate_hyde_ollama(prompt: str, model: str) -> Optional[str]:
    """Generate hypothetical document using Ollama.

    Args:
        prompt: Full prompt for generation
        model: Ollama model name

    Returns:
        Generated text or None
    """
    try:
        from research_kb_extraction.ollama_client import OllamaClient

        client = OllamaClient(model=model)

        if not await client.is_available():
            logger.debug("ollama_unavailable_for_hyde")
            return None

        response = await client.generate(
            prompt=prompt,
            system="You are a technical writer for causal inference research.",
        )

        logger.debug(
            "hyde_ollama_generated",
            model=model,
            length=len(response),
        )

        return response.strip()

    except ImportError:
        logger.warning("hyde_ollama_import_failed")
        return None
    except Exception as e:
        logger.warning("hyde_ollama_error", error=str(e))
        return None


async def _generate_hyde_anthropic(prompt: str, model: str) -> Optional[str]:
    """Generate hypothetical document using Anthropic API.

    Args:
        prompt: Full prompt for generation
        model: Anthropic model name

    Returns:
        Generated text or None
    """
    try:
        import os
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("hyde_anthropic_no_api_key")
            return None

        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
            system="You are a technical writer for causal inference research.",
        )

        block = message.content[0]
        if not hasattr(block, "text"):
            raise ValueError(f"Expected text block, got {type(block).__name__}")
        response = block.text

        logger.debug(
            "hyde_anthropic_generated",
            model=model,
            length=len(response),
        )

        return response.strip()

    except ImportError:
        logger.warning("hyde_anthropic_import_failed", message="pip install anthropic")
        return None
    except Exception as e:
        logger.warning("hyde_anthropic_error", error=str(e))
        return None


async def get_hyde_embedding(
    query: str,
    config: Optional[HydeConfig] = None,
) -> Optional[list[float]]:
    """Get embedding for query using HyDE.

    Generates hypothetical document and embeds it instead of raw query.
    Falls back to None if HyDE fails (caller should use regular query embedding).

    Args:
        query: User search query
        config: HyDE configuration

    Returns:
        1024-dimensional embedding vector, or None if HyDE fails

    Example:
        >>> embedding = await get_hyde_embedding("IV assumptions")
        >>> if embedding is None:
        ...     embedding = embed_client.embed_query(query)  # Fallback
    """
    # Generate hypothetical document
    hyde_doc = await generate_hyde_document(query, config)

    if not hyde_doc:
        return None

    try:
        # Import embedding client
        from research_kb_pdf import EmbeddingClient

        client = EmbeddingClient()

        # Embed the hypothetical document (not as query - no instruction prefix)
        # This is key: HyDE document is embedded as a passage, not a query
        embedding = client.embed(hyde_doc)

        logger.info(
            "hyde_embedding_generated",
            query=query[:50],
            hyde_length=len(hyde_doc),
        )

        return embedding

    except Exception as e:
        logger.warning("hyde_embedding_failed", error=str(e))
        return None
