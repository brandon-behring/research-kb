"""Fixtures for end-to-end tests.

Provides:
- pdf_dispatcher: Configured PDFDispatcher instance
- simple_pdf_path: Path to small test PDF
- embedding_available: Check if embedding server is running
- grobid_available: Check if GROBID server is running
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

# Add packages to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "packages/pdf-tools/src"))
sys.path.insert(0, str(repo_root / "packages/storage/src"))
sys.path.insert(0, str(repo_root / "packages/contracts/src"))
sys.path.insert(0, str(repo_root / "packages/common/src"))


@pytest_asyncio.fixture
async def pdf_dispatcher(tmp_path):
    """Provide configured PDFDispatcher instance.

    Uses temporary DLQ path for test isolation.
    """
    from research_kb_pdf import PDFDispatcher

    dlq_path = tmp_path / "test_dlq.jsonl"

    dispatcher = PDFDispatcher(
        grobid_url="http://localhost:8070",
        dlq_path=str(dlq_path),
        embedding_socket_path="/tmp/research_kb_embed.sock",
    )

    return dispatcher


@pytest.fixture
def simple_pdf_path():
    """Return path to simple test PDF.

    Uses ai_iv_search_2024.pdf (525KB, manageable for tests).
    """
    pdf_path = Path(__file__).parent.parent.parent / "fixtures/papers/ai_iv_search_2024.pdf"

    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")

    return pdf_path


@pytest.fixture
def small_pdf_path():
    """Return path to smallest available test PDF.

    Uses athey_imbens_hte_2016.pdf (216KB, fastest to process).
    """
    pdf_path = Path(__file__).parent.parent.parent / "fixtures/papers/athey_imbens_hte_2016.pdf"

    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")

    return pdf_path


@pytest_asyncio.fixture
async def embedding_available():
    """Check if embedding server is running.

    Returns True if available, False otherwise.
    Used to skip tests that require real embeddings.
    """
    import socket

    sock_path = "/tmp/research_kb_embed.sock"

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(sock_path)
        sock.close()
        return True
    except (FileNotFoundError, ConnectionRefusedError):
        return False


@pytest.fixture
def grobid_available():
    """Check if GROBID server is running.

    Returns True if available, False otherwise.
    Used to skip tests that require GROBID.
    """
    import socket

    try:
        # Try to connect to GROBID port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(("localhost", 8070))
        sock.close()
        return result == 0
    except Exception:
        return False


@pytest.fixture
def mock_embedding_client():
    """Provide mock embedding client for tests without real embedding server.

    Returns embeddings as zero vectors (valid but not meaningful).
    """
    client = AsyncMock()

    # Mock embed() to return 1024-dim zero vector
    client.embed = AsyncMock(return_value=[0.0] * 1024)

    # Mock batch_embed() to return list of zero vectors
    client.batch_embed = AsyncMock(
        side_effect=lambda texts: [[0.0] * 1024 for _ in texts]
    )

    return client


@pytest.fixture
def corrupted_pdf_path(tmp_path):
    """Create corrupted PDF file for error handling tests."""
    corrupted = tmp_path / "corrupted.pdf"
    corrupted.write_bytes(b"Not a valid PDF file")
    return corrupted
