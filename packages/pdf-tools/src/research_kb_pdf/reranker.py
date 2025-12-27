"""Cross-encoder reranking for research-kb search results.

Provides:
- CrossEncoderReranker class using BAAI/bge-reranker-v2-m3
- Two-stage retrieval: fast bi-encoder retrieval → accurate cross-encoder reranking

Model: BAAI/bge-reranker-v2-m3 (278M params, NDCG@10: 0.52)
Alternative: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M params, faster but less accurate)

Usage:
    >>> from research_kb_pdf.reranker import CrossEncoderReranker
    >>> reranker = CrossEncoderReranker()
    >>> reranked = reranker.rerank("query", search_results, top_k=10)
"""

from dataclasses import dataclass
from typing import Optional

import torch
from sentence_transformers import CrossEncoder

from research_kb_common import get_logger

logger = get_logger(__name__)

# Configuration
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"  # Pin for reproducibility
FALLBACK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_TOP_K = 10
MAX_BATCH_SIZE = 50  # Typical reranking window


@dataclass
class RerankResult:
    """Result from cross-encoder reranking.

    Attributes:
        content: The text content that was reranked
        original_rank: Position before reranking (1-based)
        rerank_score: Cross-encoder relevance score
        new_rank: Position after reranking (1-based)
        metadata: Optional additional metadata from original result
    """

    content: str
    original_rank: int
    rerank_score: float
    new_rank: int
    metadata: Optional[dict] = None


class CrossEncoderReranker:
    """Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

    Two-stage retrieval pattern:
    1. Fast bi-encoder retrieval returns top-50 candidates
    2. Cross-encoder accurately reranks to top-10

    The cross-encoder jointly encodes query+document pairs, enabling
    fine-grained semantic matching that bi-encoders cannot achieve.

    Attributes:
        model: SentenceTransformers CrossEncoder instance
        model_name: Name of the loaded model
        device: 'cuda' or 'cpu'

    Example:
        >>> reranker = CrossEncoderReranker()
        >>> results = [{"content": "doc1"}, {"content": "doc2"}]
        >>> reranked = reranker.rerank_dicts("query", results, top_k=5)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        use_fast: bool = False,
    ):
        """Initialize cross-encoder reranker.

        Args:
            model_name: CrossEncoder model name (default: BGE-reranker-v2-m3)
            device: 'cuda' or 'cpu' (auto-detected if None)
            use_fast: Use faster MiniLM model instead of BGE

        Raises:
            RuntimeError: If model fails to load
        """
        revision = None
        if use_fast:
            model_name = FALLBACK_MODEL
        else:
            revision = DEFAULT_REVISION

        self.device = device or DEVICE
        self.model_name = model_name
        self.revision = revision

        logger.info("loading_reranker", model=model_name, device=self.device, revision=revision)

        try:
            self.model = CrossEncoder(model_name, device=self.device, revision=revision)
        except Exception as e:
            logger.error("reranker_load_failed", model=model_name, error=str(e))
            raise RuntimeError(f"Failed to load reranker model: {e}") from e

        # Warmup with representative pairs
        warmup_pairs = [
            ["What is instrumental variables?", "IV is an econometric method."],
            ["causal inference methods", "Double machine learning combines ML."],
        ]
        _ = self.model.predict(warmup_pairs)

        logger.info("reranker_loaded", model=model_name, device=self.device)

    def predict_scores(self, query: str, documents: list[str]) -> list[float]:
        """Compute relevance scores for query-document pairs.

        Args:
            query: Search query
            documents: List of document texts

        Returns:
            List of relevance scores (higher = more relevant)

        Example:
            >>> reranker = CrossEncoderReranker()
            >>> scores = reranker.predict_scores("IV", ["doc1", "doc2"])
            >>> len(scores)
            2
        """
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs, show_progress_bar=False)

        return [float(s) for s in scores]

    def rerank_texts(
        self,
        query: str,
        documents: list[str],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[RerankResult]:
        """Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of results to return

        Returns:
            Top-k RerankResult objects sorted by rerank_score

        Example:
            >>> reranker = CrossEncoderReranker()
            >>> docs = ["relevant doc", "less relevant"]
            >>> results = reranker.rerank_texts("query", docs, top_k=1)
            >>> results[0].new_rank
            1
        """
        if not documents:
            return []

        scores = self.predict_scores(query, documents)

        # Create results with original ranks
        results = [
            RerankResult(
                content=doc,
                original_rank=i + 1,
                rerank_score=score,
                new_rank=0,  # Will be set after sorting
            )
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]

        # Sort by score descending
        results.sort(key=lambda r: r.rerank_score, reverse=True)

        # Assign new ranks and truncate to top_k
        for new_rank, result in enumerate(results[:top_k], start=1):
            result.new_rank = new_rank

        return results[:top_k]

    def rerank_dicts(
        self,
        query: str,
        results: list[dict],
        content_key: str = "content",
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """Rerank dictionary results (e.g., from search API).

        Args:
            query: Search query
            results: List of result dictionaries
            content_key: Key containing text content (default: 'content')
            top_k: Number of results to return

        Returns:
            Reranked list of dictionaries with added 'rerank_score' key

        Example:
            >>> results = [{"content": "doc1", "id": 1}, {"content": "doc2", "id": 2}]
            >>> reranked = reranker.rerank_dicts("query", results)
            >>> "rerank_score" in reranked[0]
            True
        """
        if not results:
            return []

        documents = [r.get(content_key, "") for r in results]
        scores = self.predict_scores(query, documents)

        # Add scores to results
        scored_results = [
            {**result, "rerank_score": score, "original_rank": i + 1}
            for i, (result, score) in enumerate(zip(results, scores))
        ]

        # Sort by rerank_score descending
        scored_results.sort(key=lambda r: r["rerank_score"], reverse=True)

        # Add new rank
        for new_rank, result in enumerate(scored_results[:top_k], start=1):
            result["new_rank"] = new_rank

        return scored_results[:top_k]
