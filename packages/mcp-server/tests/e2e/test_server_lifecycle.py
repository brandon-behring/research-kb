import subprocess
import json
import pytest
import sys
import os
from pathlib import Path

# Path to the package root (where pyproject.toml is)
PACKAGE_ROOT = Path(__file__).parents[3]

@pytest.mark.e2e
def test_server_starts_and_initializes():
    """
    E2E Test: Starts the MCP server process and performs the initialization handshake.
    Verifies that the server connects to dependencies and speaks JSON-RPC.
    """
    
    # Ensure PYTHONPATH includes the src directories of all packages
    # This mimics an installed environment or 'poetry run'
    env = os.environ.copy()
    src_paths = [
        str(PACKAGE_ROOT / "src"),
        str(PACKAGE_ROOT.parent / "api" / "src"),
        str(PACKAGE_ROOT.parent / "storage" / "src"),
        str(PACKAGE_ROOT.parent / "contracts" / "src"),
        str(PACKAGE_ROOT.parent / "common" / "src"),
    ]
    env["PYTHONPATH"] = os.pathsep.join(src_paths + [env.get("PYTHONPATH", "")])

    # Run the server module
    process = subprocess.Popen(
        [sys.executable, "-m", "research_kb_mcp.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(PACKAGE_ROOT)
    )

    try:
        # 1. Send Initialize
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-harness", "version": "0.1"}
            }
        }
        
        print(f"Sending: {json.dumps(init_req)}")
        process.stdin.write(json.dumps(init_req) + "\n")
        process.stdin.flush()

        # 2. Read Response
        # FastMCP might log to stderr, so we only read stdout for protocol
        # But if logs leak to stdout, we need to skip them
        response = None
        while True:
            response_line = process.stdout.readline()
            
            # Check for immediate crash or EOF
            if not response_line:
                stderr_out = process.stderr.read()
                if response is None:
                    pytest.fail(f"Server exited without sending JSON. Stderr:\n{stderr_out}")
                break

            print(f"Received: {response_line.strip()}")
            
            # Try to parse as JSON if it looks like it
            if response_line.strip().startswith("{"):
                try:
                    response = json.loads(response_line)
                    break
                except json.JSONDecodeError:
                    continue  # Not valid JSON, probably a log line starting with {
        
        if response is None:
             pytest.fail("Never received a valid JSON response")

        # 3. Validation
        assert response.get("jsonrpc") == "2.0"
        assert response.get("id") == 1
        assert "result" in response
        result = response["result"]
        
        # Verify server identity
        assert result["serverInfo"]["name"] == "research-kb"
        
        # Verify capabilities
        assert "tools" in result["capabilities"]

    finally:
        # Cleanup
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
