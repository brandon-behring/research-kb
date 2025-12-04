#!/usr/bin/env python3
"""Embedding server for research-kb PDF processing.

Long-running daemon providing fast embeddings via Unix socket.
Model: BAAI/bge-large-en-v1.5 (1024 dimensions, ~1.3GB model size)

Usage:
    # Start server
    python -m research_kb_pdf.embed_server

    # Test mode
    python -m research_kb_pdf.embed_server --test

Architecture:
    - Unix domain socket for IPC
    - Batch processing support (up to 32 texts)
    - GPU acceleration if available
    - Warmup on startup for consistent latency
"""

import json
import os
import socket

import torch
from sentence_transformers import SentenceTransformer

from research_kb_common import get_logger

# Configuration
SOCKET_PATH = "/tmp/research_kb_embed.sock"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BUFFER_SIZE = 131072  # 128KB for larger batch requests
MAX_BATCH_SIZE = 32  # Recommended batch size for BGE

logger = get_logger(__name__)


class EmbeddingServer:
    """Long-running embedding server for research-kb."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        """Initialize embedding server with BGE model.

        Args:
            model_name: SentenceTransformer model name
            device: 'cuda' or 'cpu'
        """
        logger.info("loading_model", model=model_name, device=device)
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.model_name = model_name

        # Warmup with representative text
        warmup_texts = [
            "Introduction to machine learning",
            "This is a warmup query for the embedding model",
        ]
        _ = self.model.encode(warmup_texts, convert_to_numpy=True)

        logger.info(
            "model_loaded",
            dim=self.model.get_sentence_embedding_dimension(),
            device=device,
        )

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            1024-dimensional embedding vector

        Example:
            >>> server = EmbeddingServer()
            >>> embedding = server.embed("Hello world")
            >>> len(embedding)
            1024
        """
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()

    def embed_batch(
        self, texts: list[str], batch_size: int = MAX_BATCH_SIZE
    ) -> list[list[float]]:
        """Embed multiple texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Maximum batch size (default 32)

        Returns:
            List of 1024-dimensional embeddings

        Example:
            >>> server = EmbeddingServer()
            >>> embeddings = server.embed_batch(["Text 1", "Text 2"])
            >>> len(embeddings)
            2
        """
        if len(texts) > batch_size:
            # Process in smaller batches to avoid memory issues
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                all_embeddings.extend(batch_embeddings.tolist())
            return all_embeddings
        else:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()

    def handle_request(self, data: dict) -> dict:
        """Handle JSON request and return JSON response.

        Supported actions:
            - embed: Single text embedding
            - embed_batch: Multiple text embeddings
            - ping: Health check
            - shutdown: Graceful shutdown

        Args:
            data: Request dictionary with 'action' field

        Returns:
            Response dictionary
        """
        try:
            action = data.get("action", "embed")

            if action == "embed":
                text = data.get("text", "")
                if not text:
                    return {"error": "Missing 'text' field"}
                embedding = self.embed(text)
                return {"embedding": embedding, "dim": len(embedding)}

            elif action == "embed_batch":
                texts = data.get("texts", [])
                if not texts:
                    return {"error": "Missing 'texts' field"}
                batch_size = data.get("batch_size", MAX_BATCH_SIZE)
                embeddings = self.embed_batch(texts, batch_size)
                return {
                    "embeddings": embeddings,
                    "count": len(embeddings),
                    "dim": len(embeddings[0]) if embeddings else 0,
                }

            elif action == "ping":
                return {
                    "status": "ok",
                    "device": self.device,
                    "model": self.model_name,
                    "dim": self.model.get_sentence_embedding_dimension(),
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

        logger.info("server_listening", socket=socket_path)

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
            logger.info("server_stopped")


def main():
    """Run the embedding server."""
    import argparse

    parser = argparse.ArgumentParser(description="Research-KB Embedding Server")
    parser.add_argument("--socket", default=SOCKET_PATH, help="Unix socket path")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name")
    parser.add_argument(
        "--test", action="store_true", help="Test mode: embed sample and exit"
    )
    args = parser.parse_args()

    if args.test:
        # Test mode: load model and test embedding
        server = EmbeddingServer(args.model)
        test_texts = [
            "Introduction to causal inference",
            "Methods for estimating treatment effects",
        ]
        embeddings = server.embed_batch(test_texts)
        print(f"✅ Test embeddings for {len(test_texts)} texts")
        print(f"Dimension: {len(embeddings[0])}")
        print(f"First embedding preview: {embeddings[0][:5]}")
        return

    # Run server
    server = EmbeddingServer(args.model)
    server.run_server(args.socket)


if __name__ == "__main__":
    main()
