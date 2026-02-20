"""Test realistic MCP-style queries through the daemon."""

import json
import socket
import time


SOCKET_PATH = "/tmp/research_kb_daemon_$USER.sock"

QUERIES = [
    ("assumptions of instrumental variables", "causal_inference"),
    ("attention mechanism in transformers", "deep_learning"),
    ("difference between RCT and natural experiment", "causal_inference"),
    ("Black-Scholes option pricing formula", "finance"),
    ("API design best practices RESTful", "software_engineering"),
    ("GARCH volatility model estimation", "time_series"),
    ("Bayesian posterior inference MCMC", "statistics"),
    ("gradient boosting vs random forest", "machine_learning"),
    ("retrieval augmented generation pipeline", "rag_llm"),
    ("difference in differences parallel trends", "econometrics"),
]


def query_daemon(query: str, request_id: int) -> dict:
    """Send a search query to the daemon via Unix socket."""
    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "search",
            "params": {"query": query},
            "id": request_id,
        }
    )

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect(SOCKET_PATH)
    sock.sendall(request.encode() + b"\n")

    response = b""
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        response += chunk

    sock.close()
    return json.loads(response.decode())


def main():
    print("MCP Query Quality Exploration")
    print("=" * 70)

    for i, (query, expected_domain) in enumerate(QUERIES):
        start = time.time()
        try:
            resp = query_daemon(query, i + 1)
        except Exception as e:
            print(f"\n  ERROR: {query} → {e}")
            continue

        latency_ms = (time.time() - start) * 1000
        result = resp.get("result", [])

        # Handle both list and dict result formats
        if isinstance(result, dict):
            items = result.get("results", [])
        elif isinstance(result, list):
            items = result
        else:
            items = []

        print(f"\n  Query: {query}")
        print(
            f"  Expected domain: {expected_domain} | Latency: {latency_ms:.0f}ms | Results: {len(items)}"
        )

        for j, item in enumerate(items[:5]):
            if isinstance(item, dict):
                src = item.get("source", {})
                title = src.get("title", "?")[:50] if isinstance(src, dict) else str(src)[:50]
                score = item.get("score", item.get("combined_score", 0))
                content = ""
                if "chunk" in item and isinstance(item["chunk"], dict):
                    content = item["chunk"].get("content", "")[:80].replace("\n", " ")
                elif "content" in item:
                    content = item["content"][:80].replace("\n", " ")
                print(f"    {j+1}. [{score:.3f}] {title}")
                if content:
                    print(f"       {content}")
            else:
                print(f"    {j+1}. {str(item)[:100]}")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
