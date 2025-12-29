"""Tests for cross-encoder reranking server and client.

Tests both the core CrossEncoderReranker class and the server/client IPC.
"""

import pytest

from research_kb_pdf.reranker import (
    CrossEncoderReranker,
    RerankResult,
    FALLBACK_MODEL,
)


@pytest.mark.requires_reranker
class TestCrossEncoderRerankerUnit:
    """Unit tests for CrossEncoderReranker (no socket needed).

    Requires cross-encoder model to be loadable. Skip with:
        pytest -m "not requires_reranker"
    """

    @pytest.fixture(scope="class")
    def reranker(self):
        """Create reranker instance.

        Uses class scope to avoid reloading the model for each test.
        Skips all tests in this class if model loading fails.
        """
        try:
            # Use faster model for tests to reduce CI time
            return CrossEncoderReranker(use_fast=True)
        except RuntimeError as e:
            pytest.skip(f"Reranker model not available: {e}")

    def test_reranker_initialization(self, reranker):
        """Test reranker initializes with correct model."""
        assert reranker.model_name == FALLBACK_MODEL
        assert reranker.device in ["cuda", "cpu"]

    def test_predict_scores_basic(self, reranker):
        """Test predicting scores for query-document pairs."""
        query = "instrumental variables estimation"
        documents = [
            "Instrumental variables (IV) is a method for causal inference.",
            "The weather is sunny today.",
        ]

        scores = reranker.predict_scores(query, documents)

        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
        # IV doc should score higher than weather doc
        assert scores[0] > scores[1]

    def test_predict_scores_empty(self, reranker):
        """Test predicting scores with empty documents."""
        scores = reranker.predict_scores("test query", [])
        assert scores == []

    def test_rerank_texts_basic(self, reranker):
        """Test reranking documents by relevance."""
        query = "difference in differences parallel trends"
        documents = [
            "The sun rises in the east.",  # Irrelevant
            "DiD requires parallel trends assumption for causal identification.",  # Relevant
            "Machine learning models can predict outcomes.",  # Somewhat related
        ]

        results = reranker.rerank_texts(query, documents, top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, RerankResult) for r in results)

        # Check ranking - DiD document should be first
        assert results[0].new_rank == 1
        assert results[1].new_rank == 2
        assert results[2].new_rank == 3

        # DiD document (originally index 1) should now be first
        assert "DiD" in results[0].content or "parallel trends" in results[0].content

    def test_rerank_texts_top_k(self, reranker):
        """Test top_k limits number of results."""
        query = "test query"
        documents = [f"Document {i}" for i in range(10)]

        results = reranker.rerank_texts(query, documents, top_k=3)

        assert len(results) == 3
        assert results[0].new_rank == 1
        assert results[2].new_rank == 3

    def test_rerank_texts_empty(self, reranker):
        """Test reranking with empty documents."""
        results = reranker.rerank_texts("query", [], top_k=5)
        assert results == []

    def test_rerank_preserves_original_rank(self, reranker):
        """Test that original rank is preserved in results."""
        query = "causal inference"
        documents = ["First doc", "Second doc about causal inference", "Third doc"]

        results = reranker.rerank_texts(query, documents, top_k=3)

        # Check original ranks are correct
        original_ranks = sorted([r.original_rank for r in results])
        assert original_ranks == [1, 2, 3]

    def test_rerank_dicts_basic(self, reranker):
        """Test reranking dictionary results."""
        query = "propensity score matching"
        results = [
            {"id": 1, "content": "Propensity score matching is a method for bias reduction."},
            {"id": 2, "content": "The quick brown fox jumps over the lazy dog."},
            {"id": 3, "content": "PSM balances covariates between treatment groups."},
        ]

        reranked = reranker.rerank_dicts(query, results, content_key="content", top_k=3)

        assert len(reranked) == 3
        assert all("rerank_score" in r for r in reranked)
        assert all("original_rank" in r for r in reranked)
        assert all("new_rank" in r for r in reranked)

        # Check that original IDs are preserved
        ids = [r["id"] for r in reranked]
        assert set(ids) == {1, 2, 3}

    def test_rerank_dicts_custom_content_key(self, reranker):
        """Test reranking with custom content key."""
        query = "regression"
        results = [
            {"id": 1, "text": "Linear regression is a statistical method."},
            {"id": 2, "text": "Cats are fluffy animals."},
        ]

        reranked = reranker.rerank_dicts(query, results, content_key="text", top_k=2)

        assert len(reranked) == 2
        # Regression doc should be ranked higher
        assert reranked[0]["text"].startswith("Linear regression")

    def test_semantic_relevance_ordering(self, reranker):
        """Test that semantically relevant documents are ranked higher."""
        query = "What is the backdoor criterion in causal inference?"
        documents = [
            "The backdoor criterion is a graphical condition for identifying causal effects.",
            "Recipe for chocolate cake: mix flour, sugar, and eggs.",
            "Pearl introduced the backdoor criterion in his causal inference framework.",
            "Stock prices fluctuated today due to market conditions.",
        ]

        results = reranker.rerank_texts(query, documents, top_k=4)

        # The two backdoor-related documents should be in top 2
        top_2_content = [r.content for r in results[:2]]
        assert any("backdoor" in c for c in top_2_content)

        # Irrelevant documents should be at bottom
        bottom_2_content = [r.content for r in results[2:]]
        assert any("cake" in c or "Stock" in c for c in bottom_2_content)


class TestRerankResultDataclass:
    """Tests for RerankResult dataclass."""

    def test_rerank_result_creation(self):
        """Test creating RerankResult."""
        result = RerankResult(
            content="Test content",
            original_rank=5,
            rerank_score=0.85,
            new_rank=1,
            metadata={"source": "test"},
        )

        assert result.content == "Test content"
        assert result.original_rank == 5
        assert result.rerank_score == 0.85
        assert result.new_rank == 1
        assert result.metadata == {"source": "test"}

    def test_rerank_result_default_metadata(self):
        """Test RerankResult with default metadata."""
        result = RerankResult(
            content="Test",
            original_rank=1,
            rerank_score=0.5,
            new_rank=1,
        )

        assert result.metadata is None


@pytest.mark.requires_reranker
class TestRerankServerUnit:
    """Unit tests for RerankServer (no socket needed).

    Requires cross-encoder model to be loadable. Skip with:
        pytest -m "not requires_reranker"
    """

    @pytest.fixture(scope="class")
    def server(self):
        """Create rerank server instance.

        Skips all tests in this class if model loading fails.
        """
        from research_kb_pdf.rerank_server import RerankServer

        try:
            return RerankServer(model_name=FALLBACK_MODEL)
        except RuntimeError as e:
            pytest.skip(f"Reranker model not available: {e}")

    def test_server_initialization(self, server):
        """Test server initializes correctly."""
        assert server.model_name == FALLBACK_MODEL
        assert server.device in ["cuda", "cpu"]

    def test_handle_request_ping(self, server):
        """Test handle_request with ping action."""
        request = {"action": "ping"}
        response = server.handle_request(request)

        assert response["status"] == "ok"
        assert "device" in response
        assert "model" in response

    def test_handle_request_rerank(self, server):
        """Test handle_request with rerank action."""
        request = {
            "action": "rerank",
            "query": "causal inference",
            "documents": ["Doc about causal inference", "Doc about cooking"],
            "top_k": 2,
        }
        response = server.handle_request(request)

        assert "results" in response
        assert response["count"] == 2
        assert len(response["results"]) == 2

    def test_handle_request_predict(self, server):
        """Test handle_request with predict action."""
        request = {
            "action": "predict",
            "query": "test",
            "documents": ["doc1", "doc2"],
        }
        response = server.handle_request(request)

        assert "scores" in response
        assert response["count"] == 2
        assert len(response["scores"]) == 2

    def test_handle_request_missing_query(self, server):
        """Test handle_request with missing query."""
        request = {"action": "rerank", "documents": ["doc"]}
        response = server.handle_request(request)

        assert "error" in response

    def test_handle_request_missing_documents(self, server):
        """Test handle_request with missing documents."""
        request = {"action": "rerank", "query": "test"}
        response = server.handle_request(request)

        assert "error" in response

    def test_handle_request_unknown_action(self, server):
        """Test handle_request with unknown action."""
        request = {"action": "unknown"}
        response = server.handle_request(request)

        assert "error" in response
        assert "Unknown action" in response["error"]


@pytest.mark.integration
@pytest.mark.requires_reranker
class TestRerankClientIntegration:
    """Integration tests for RerankClient (requires running server).

    Skip these tests if the rerank server is not running.
    """

    @pytest.fixture
    def client(self):
        """Create rerank client."""
        from research_kb_pdf.rerank_client import RerankClient

        client = RerankClient()

        # Skip if server not running
        if not client.is_available():
            pytest.skip("Rerank server not running")

        return client

    def test_client_ping(self, client):
        """Test client ping."""
        status = client.ping()
        assert status["status"] == "ok"

    def test_client_rerank(self, client):
        """Test client rerank."""
        results = client.rerank(
            query="instrumental variables",
            documents=["IV estimation", "cooking recipe"],
            top_k=2,
        )

        assert len(results) == 2
        # IV doc should be first
        assert "IV" in results[0].content

    def test_client_predict_scores(self, client):
        """Test client predict_scores."""
        scores = client.predict_scores(
            query="machine learning",
            documents=["ML models", "random text"],
        )

        assert len(scores) == 2
        assert scores[0] > scores[1]  # ML doc should score higher
