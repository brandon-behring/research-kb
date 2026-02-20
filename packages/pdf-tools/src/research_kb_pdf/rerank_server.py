#!/usr/bin/env python3
"""Reranking server for research-kb search enhancement.

Long-running daemon providing cross-encoder reranking via Unix socket.
Model: BAAI/bge-reranker-v2-m3 (278M params, ~1GB model size)

Usage:
    # Start server
    python -m research_kb_pdf.rerank_server

    # Test mode
    python -m research_kb_pdf.rerank_server --test

    # Use faster model
    python -m research_kb_pdf.rerank_server --fast

Architecture:
    - Unix domain socket for IPC
    - Batch processing support (up to 50 query-doc pairs)
    - GPU acceleration if available
    - Warmup on startup for consistent latency
"""

import json
import os
import socket

import torch

from research_kb_common import get_logger
from research_kb_pdf.reranker import CrossEncoderReranker, DEFAULT_MODEL, FALLBACK_MODEL

# Configuration
SOCKET_PATH = "/tmp/research_kb_rerank.sock"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BUFFER_SIZE = 262144  # 256KB for larger batch requests (query + 50 docs)
MAX_BATCH_SIZE = 50  # Typical reranking window

logger = get_logger(__name__)


class RerankServer:
    """Long-running cross-encoder reranking server for research-kb."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = DEVICE):
        """Initialize rerank server with cross-encoder model.

        Args:
            model_name: CrossEncoder model name
            device: 'cuda' or 'cpu'
        """
        logger.info("initializing_rerank_server", model=model_name, device=device)
        self.reranker = CrossEncoderReranker(model_name=model_name, device=device)
        self.device = device
        self.model_name = model_name
        logger.info("rerank_server_ready", model=model_name, device=device)

    def handle_request(self, data: dict) -> dict:
        """Handle JSON request and return JSON response.

        Supported actions:
            - rerank: Rerank documents for a query
            - predict: Get raw relevance scores
            - ping: Health check
            - shutdown: Graceful shutdown

        Args:
            data: Request dictionary with 'action' field

        Returns:
            Response dictionary
        """
        try:
            action = data.get("action", "rerank")

            if action == "rerank":
                query = data.get("query", "")
                documents = data.get("documents", [])
                top_k = data.get("top_k", 10)

                if not query:
                    return {"error": "Missing 'query' field"}
                if not documents:
                    return {"error": "Missing 'documents' field"}

                results = self.reranker.rerank_texts(query, documents, top_k=top_k)

                return {
                    "results": [
                        {
                            "content": r.content,
                            "original_rank": r.original_rank,
                            "rerank_score": r.rerank_score,
                            "new_rank": r.new_rank,
                        }
                        for r in results
                    ],
                    "count": len(results),
                }

            elif action == "predict":
                query = data.get("query", "")
                documents = data.get("documents", [])

                if not query:
                    return {"error": "Missing 'query' field"}
                if not documents:
                    return {"error": "Missing 'documents' field"}

                scores = self.reranker.predict_scores(query, documents)
                return {"scores": scores, "count": len(scores)}

            elif action == "ping":
                return {
                    "status": "ok",
                    "device": self.device,
                    "model": self.model_name,
                }

            elif action == "shutdown":
                logger.info("shutdown_requested")
                return {"status": "shutting_down"}

            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            logger.error("request_error", error=str(e))
            return {"error": str(e)}

    def run_server(self, socket_path: str = SOCKET_PATH):
        """Run Unix socket server.

        Args:
            socket_path: Path to Unix domain socket

        Note:
            Runs indefinitely until shutdown request or interrupt
        """
        # Remove existing socket
        if os.path.exists(socket_path):
            os.remove(socket_path)

        # Create socket
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(socket_path)
        server.listen(5)
        os.chmod(socket_path, 0o666)  # Allow other processes to connect

        logger.info("rerank_server_listening", socket=socket_path)

        try:
            while True:
                conn, _ = server.accept()
                try:
                    data = b""
                    while True:
                        chunk = conn.recv(BUFFER_SIZE)
                        if not chunk:
                            break
                        data += chunk
                        # Check for complete JSON
                        try:
                            request = json.loads(data.decode("utf-8"))
                            break
                        except json.JSONDecodeError:
                            continue

                    if data:
                        request = json.loads(data.decode("utf-8"))
                        response = self.handle_request(request)
                        conn.sendall(json.dumps(response).encode("utf-8"))

                        if request.get("action") == "shutdown":
                            break

                except Exception as e:
                    logger.error("connection_error", error=str(e))
                    error_response = json.dumps({"error": str(e)})
                    try:
                        conn.sendall(error_response.encode("utf-8"))
                    except Exception:
                        pass
                finally:
                    conn.close()

        finally:
            server.close()
            if os.path.exists(socket_path):
                os.remove(socket_path)
            logger.info("rerank_server_stopped")


def main():
    """Run the reranking server."""
    import argparse

    parser = argparse.ArgumentParser(description="Research-KB Reranking Server")
    parser.add_argument("--socket", default=SOCKET_PATH, help="Unix socket path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster MiniLM model instead of BGE",
    )
    parser.add_argument("--test", action="store_true", help="Test mode: rerank sample and exit")
    args = parser.parse_args()

    model_name = FALLBACK_MODEL if args.fast else args.model

    if args.test:
        # Test mode: load model and test reranking
        reranker = CrossEncoderReranker(model_name=model_name)
        test_query = "instrumental variables for causal inference"
        test_docs = [
            "Instrumental variables (IV) is a method for estimating causal effects.",
            "Machine learning models can predict outcomes.",
            "2SLS is a common IV estimator that addresses endogeneity.",
        ]
        results = reranker.rerank_texts(test_query, test_docs, top_k=3)
        print(f"Test reranking for {len(test_docs)} documents")
        print(f"Query: {test_query}")
        print("Results:")
        for r in results:
            print(f"  Rank {r.new_rank} (was {r.original_rank}): score={r.rerank_score:.4f}")
            print(f"    {r.content[:60]}...")
        return

    # Run server
    server = RerankServer(model_name=model_name)
    server.run_server(args.socket)


if __name__ == "__main__":
    main()
