"""Tests for Pydantic models in contracts package.

Coverage targets: 90%+ (core package)
Focus: Validators, edge cases, JSONB flexibility
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from research_kb_contracts import (
    Chunk,
    IngestionStage,
    IngestionStatus,
    SearchResult,
    Source,
    SourceType,
)


class TestSourceType:
    """Test SourceType enum."""

    def test_all_source_types_available(self):
        """Verify all expected source types are defined."""
        assert SourceType.TEXTBOOK == "textbook"
        assert SourceType.PAPER == "paper"
        assert SourceType.CODE_REPO == "code_repo"

    def test_source_type_string_conversion(self):
        """Verify SourceType can be created from strings."""
        assert SourceType("textbook") == SourceType.TEXTBOOK
        assert SourceType("paper") == SourceType.PAPER
        assert SourceType("code_repo") == SourceType.CODE_REPO


class TestSource:
    """Test Source model matching PostgreSQL sources table."""

    def test_minimal_source_creation(self):
        """Test creating source with minimal required fields."""
        now = datetime.now(timezone.utc)
        source = Source(
            id=uuid4(),
            source_type=SourceType.TEXTBOOK,
            title="Test Book",
            file_hash="sha256:abc123",
            created_at=now,
            updated_at=now,
        )

        assert source.source_type == SourceType.TEXTBOOK
        assert source.title == "Test Book"
        assert source.file_hash == "sha256:abc123"
        assert source.authors == []
        assert source.metadata == {}

    def test_source_with_full_metadata(self):
        """Test source with complete metadata (textbook example)."""
        now = datetime.now(timezone.utc)
        source = Source(
            id=uuid4(),
            source_type=SourceType.TEXTBOOK,
            title="Causality: Models, Reasoning, and Inference",
            authors=["Judea Pearl"],
            year=2009,
            file_path="/test/pearl_causality.pdf",
            file_hash="sha256:def456",
            metadata={
                "isbn": "978-0521895606",
                "publisher": "Cambridge University Press",
                "total_pages": 464,
            },
            created_at=now,
            updated_at=now,
        )

        assert source.authors == ["Judea Pearl"]
        assert source.year == 2009
        assert source.metadata["isbn"] == "978-0521895606"
        assert source.metadata["total_pages"] == 464

    def test_source_paper_metadata_extensibility(self):
        """Test paper source with journal/DOI metadata."""
        now = datetime.now(timezone.utc)
        source = Source(
            id=uuid4(),
            source_type=SourceType.PAPER,
            title="Double/debiased machine learning",
            authors=["Victor Chernozhukov", "Denis Chetverikov"],
            year=2018,
            file_hash="sha256:ghi789",
            metadata={
                "doi": "10.1111/ectj.12097",
                "journal": "Econometrics Journal",
                "authority_tier": "canonical",
            },
            created_at=now,
            updated_at=now,
        )

        assert source.metadata["doi"] == "10.1111/ectj.12097"
        assert source.metadata["authority_tier"] == "canonical"

    def test_source_code_repo_metadata(self):
        """Test code repository source with git metadata."""
        now = datetime.now(timezone.utc)
        source = Source(
            id=uuid4(),
            source_type=SourceType.CODE_REPO,
            title="scikit-learn/linear_model",
            authors=["scikit-learn developers"],
            year=2023,
            file_hash="sha256:jkl012",
            metadata={
                "git_url": "https://github.com/scikit-learn/scikit-learn",
                "language": "python",
                "license": "BSD-3-Clause",
            },
            created_at=now,
            updated_at=now,
        )

        assert (
            source.metadata["git_url"] == "https://github.com/scikit-learn/scikit-learn"
        )
        assert source.metadata["language"] == "python"

    def test_source_file_hash_validation(self):
        """Test file_hash validator rejects empty strings."""
        now = datetime.now(timezone.utc)

        with pytest.raises(ValidationError) as exc_info:
            Source(
                id=uuid4(),
                source_type=SourceType.TEXTBOOK,
                title="Test",
                file_hash="",  # Empty hash should fail
                created_at=now,
                updated_at=now,
            )

        assert "file_hash must be non-empty" in str(exc_info.value)

    def test_source_file_hash_strips_whitespace(self):
        """Test file_hash validator strips whitespace."""
        now = datetime.now(timezone.utc)
        source = Source(
            id=uuid4(),
            source_type=SourceType.TEXTBOOK,
            title="Test",
            file_hash="  sha256:abc  ",
            created_at=now,
            updated_at=now,
        )

        assert source.file_hash == "sha256:abc"


class TestChunk:
    """Test Chunk model matching PostgreSQL chunks table."""

    def test_minimal_chunk_creation(self):
        """Test creating chunk with minimal required fields."""
        now = datetime.now(timezone.utc)
        chunk = Chunk(
            id=uuid4(),
            source_id=uuid4(),
            content="Test content",
            content_hash="sha256:chunk123",
            created_at=now,
        )

        assert chunk.content == "Test content"
        assert chunk.content_hash == "sha256:chunk123"
        assert chunk.metadata == {}
        assert chunk.embedding is None

    def test_chunk_with_full_metadata(self):
        """Test chunk with complete metadata (theorem example)."""
        now = datetime.now(timezone.utc)
        chunk = Chunk(
            id=uuid4(),
            source_id=uuid4(),
            content="The backdoor criterion states...",
            content_hash="sha256:chunk456",
            location="Chapter 3, Section 3.3, Theorem 3.3.1, p. 73",
            page_start=73,
            page_end=74,
            embedding=[0.1] * 1024,  # 1024-dim vector (BGE-large-en-v1.5)
            metadata={
                "chunk_type": "theorem",
                "chapter_num": 3,
                "section_num": "3.3",
                "theorem_name": "Backdoor Criterion",
                "has_proof": True,
            },
            created_at=now,
        )

        assert chunk.location == "Chapter 3, Section 3.3, Theorem 3.3.1, p. 73"
        assert chunk.page_start == 73
        assert chunk.metadata["theorem_name"] == "Backdoor Criterion"
        assert len(chunk.embedding) == 1024

    def test_chunk_code_metadata(self):
        """Test chunk with code-specific metadata."""
        now = datetime.now(timezone.utc)
        chunk = Chunk(
            id=uuid4(),
            source_id=uuid4(),
            content="class LogisticRegression(BaseEstimator): ...",
            content_hash="sha256:code789",
            location="sklearn/linear_model/_logistic.py:lines 1200-1450",
            metadata={
                "chunk_type": "class_definition",
                "file_path": "sklearn/linear_model/_logistic.py",
                "start_line": 1200,
                "end_line": 1450,
                "class_name": "LogisticRegression",
                "language": "python",
            },
            created_at=now,
        )

        assert chunk.metadata["class_name"] == "LogisticRegression"
        assert chunk.metadata["language"] == "python"
        assert chunk.page_start is None  # Code has no pages

    def test_chunk_content_validation(self):
        """Test content validator rejects empty strings."""
        now = datetime.now(timezone.utc)

        with pytest.raises(ValidationError) as exc_info:
            Chunk(
                id=uuid4(),
                source_id=uuid4(),
                content="   ",  # Whitespace-only should fail
                content_hash="sha256:test",
                created_at=now,
            )

        assert "content must be non-empty" in str(exc_info.value)

    def test_chunk_embedding_dimension_validation(self):
        """Test embedding validator enforces 1024 dimensions (BGE-large-en-v1.5)."""
        now = datetime.now(timezone.utc)

        # Correct dimension should pass
        chunk = Chunk(
            id=uuid4(),
            source_id=uuid4(),
            content="Test",
            content_hash="sha256:test",
            embedding=[0.1] * 1024,
            created_at=now,
        )
        assert len(chunk.embedding) == 1024

        # Wrong dimension should fail
        with pytest.raises(ValidationError) as exc_info:
            Chunk(
                id=uuid4(),
                source_id=uuid4(),
                content="Test",
                content_hash="sha256:test",
                embedding=[0.1] * 128,  # Wrong dimension
                created_at=now,
            )

        assert "embedding must be 1024 dimensions" in str(exc_info.value)

    def test_chunk_embedding_optional(self):
        """Test embedding is optional (None allowed)."""
        now = datetime.now(timezone.utc)
        chunk = Chunk(
            id=uuid4(),
            source_id=uuid4(),
            content="Test",
            content_hash="sha256:test",
            embedding=None,
            created_at=now,
        )

        assert chunk.embedding is None


class TestIngestionStatus:
    """Test IngestionStatus model for pipeline tracking."""

    def test_ingestion_status_creation(self):
        """Test creating ingestion status."""
        now = datetime.now(timezone.utc)
        status = IngestionStatus(
            source_id=uuid4(),
            stage=IngestionStage.CHUNKING,
            progress=0.5,
            chunks_created=42,
            updated_at=now,
        )

        assert status.stage == IngestionStage.CHUNKING
        assert status.progress == 0.5
        assert status.chunks_created == 42
        assert status.error_message is None

    def test_ingestion_status_failed_state(self):
        """Test ingestion status in failed state with error message."""
        now = datetime.now(timezone.utc)
        status = IngestionStatus(
            source_id=uuid4(),
            stage=IngestionStage.FAILED,
            progress=0.3,
            error_message="GROBID connection timeout",
            chunks_created=15,
            updated_at=now,
        )

        assert status.stage == IngestionStage.FAILED
        assert status.error_message == "GROBID connection timeout"

    def test_ingestion_progress_validation(self):
        """Test progress field validates 0.0-1.0 range."""
        now = datetime.now(timezone.utc)

        # Valid progress
        status = IngestionStatus(
            source_id=uuid4(),
            stage=IngestionStage.EMBEDDING,
            progress=0.0,
            updated_at=now,
        )
        assert status.progress == 0.0

        status = IngestionStatus(
            source_id=uuid4(),
            stage=IngestionStage.COMPLETED,
            progress=1.0,
            updated_at=now,
        )
        assert status.progress == 1.0

        # Invalid progress (out of range)
        with pytest.raises(ValidationError):
            IngestionStatus(
                source_id=uuid4(),
                stage=IngestionStage.PENDING,
                progress=1.5,
                updated_at=now,
            )


class TestSearchResult:
    """Test SearchResult model for hybrid search."""

    def test_search_result_creation(self):
        """Test creating search result with FTS and vector scores."""
        now = datetime.now(timezone.utc)
        source = Source(
            id=uuid4(),
            source_type=SourceType.TEXTBOOK,
            title="Test Book",
            file_hash="sha256:abc",
            created_at=now,
            updated_at=now,
        )
        chunk = Chunk(
            id=uuid4(),
            source_id=source.id,
            content="Test content",
            content_hash="sha256:chunk",
            created_at=now,
        )

        result = SearchResult(
            chunk=chunk,
            source=source,
            fts_score=0.72,
            vector_score=0.15,
            combined_score=0.65,
            rank=1,
        )

        assert result.fts_score == 0.72
        assert result.vector_score == 0.15
        assert result.combined_score == 0.65
        assert result.rank == 1

    def test_search_result_vector_only(self):
        """Test search result with only vector score (no FTS)."""
        now = datetime.now(timezone.utc)
        source = Source(
            id=uuid4(),
            source_type=SourceType.PAPER,
            title="Test Paper",
            file_hash="sha256:def",
            created_at=now,
            updated_at=now,
        )
        chunk = Chunk(
            id=uuid4(),
            source_id=source.id,
            content="Abstract text",
            content_hash="sha256:abstract",
            created_at=now,
        )

        result = SearchResult(
            chunk=chunk,
            source=source,
            fts_score=None,  # No FTS score
            vector_score=0.08,
            combined_score=0.92,  # 1 - (vector_score / 2) normalization
            rank=3,
        )

        assert result.fts_score is None
        assert result.vector_score == 0.08
        assert result.combined_score == 0.92

    def test_search_result_combined_score_validation(self):
        """Test combined_score validator rejects negative values."""
        now = datetime.now(timezone.utc)
        source = Source(
            id=uuid4(),
            source_type=SourceType.TEXTBOOK,
            title="Test",
            file_hash="sha256:test",
            created_at=now,
            updated_at=now,
        )
        chunk = Chunk(
            id=uuid4(),
            source_id=source.id,
            content="Test",
            content_hash="sha256:test",
            created_at=now,
        )

        with pytest.raises(ValidationError) as exc_info:
            SearchResult(
                chunk=chunk,
                source=source,
                combined_score=-0.5,  # Negative should fail
                rank=1,
            )

        assert "combined_score must be non-negative" in str(exc_info.value)

    def test_search_result_rank_validation(self):
        """Test rank field must be >= 1."""
        now = datetime.now(timezone.utc)
        source = Source(
            id=uuid4(),
            source_type=SourceType.TEXTBOOK,
            title="Test",
            file_hash="sha256:test",
            created_at=now,
            updated_at=now,
        )
        chunk = Chunk(
            id=uuid4(),
            source_id=source.id,
            content="Test",
            content_hash="sha256:test",
            created_at=now,
        )

        # Valid rank
        result = SearchResult(chunk=chunk, source=source, combined_score=0.5, rank=1)
        assert result.rank == 1

        # Invalid rank (0 or negative)
        with pytest.raises(ValidationError):
            SearchResult(chunk=chunk, source=source, combined_score=0.5, rank=0)
