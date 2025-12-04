"""Tests for output formatters."""

import json


class TestMarkdownFormatter:
    """Tests for markdown output formatting."""

    def test_markdown_single_result(self, mock_search_results):
        """Test markdown formatting with one result."""
        from research_kb_cli.formatters import format_results_markdown

        output = format_results_markdown(
            results=mock_search_results[:1],
            query="test query",
            show_content=True,
        )

        assert "## Result 1" in output
        assert "Test Paper on Causal Inference" in output
        assert "score:" in output
        assert "instrumental variables" in output

    def test_markdown_multiple_results(self, mock_search_results):
        """Test markdown with multiple results."""
        from research_kb_cli.formatters import format_results_markdown
        from copy import deepcopy

        # Create 3 unique results with different ranks
        results = []
        for i in range(3):
            result = deepcopy(mock_search_results[0])
            result.rank = i + 1
            results.append(result)

        output = format_results_markdown(results, query="test", show_content=True)

        assert "## Result 1" in output
        assert "## Result 2" in output
        assert "## Result 3" in output

    def test_markdown_no_results(self):
        """Test markdown formatting with empty results."""
        from research_kb_cli.formatters import format_results_markdown

        output = format_results_markdown([], query="no match query")

        assert "No results found" in output
        assert "no match query" in output

    def test_markdown_without_content(self, mock_search_results):
        """Test markdown with content snippets hidden."""
        from research_kb_cli.formatters import format_results_markdown

        output = format_results_markdown(
            results=mock_search_results[:1],
            query="test",
            show_content=False,
        )

        # Should have source info but not the content snippet
        assert "Test Paper on Causal Inference" in output
        # Content should not be included
        assert len(output) < 300  # Shorter without content


class TestJsonFormatter:
    """Tests for JSON output formatting."""

    def test_json_structure(self, mock_search_results):
        """Test JSON output structure."""
        from research_kb_cli.formatters import format_results_json

        output = format_results_json(mock_search_results, query="test")

        # Should be valid JSON
        data = json.loads(output)

        assert data["query"] == "test"
        assert data["result_count"] == 1
        assert len(data["results"]) == 1

    def test_json_result_fields(self, mock_search_results):
        """Test JSON result contains required fields."""
        from research_kb_cli.formatters import format_results_json

        output = format_results_json(mock_search_results, query="test")
        data = json.loads(output)

        result = data["results"][0]

        # Check result fields
        assert "rank" in result
        assert "score" in result
        assert "fts_score" in result
        assert "vector_score" in result

        # Check source fields
        assert "source" in result
        assert "id" in result["source"]
        assert "title" in result["source"]
        assert "authors" in result["source"]
        assert "year" in result["source"]

        # Check chunk fields
        assert "chunk" in result
        assert "id" in result["chunk"]
        assert "content" in result["chunk"]

    def test_json_empty_results(self):
        """Test JSON formatting with empty results."""
        from research_kb_cli.formatters import format_results_json

        output = format_results_json([], query="no match")
        data = json.loads(output)

        assert data["query"] == "no match"
        assert data["result_count"] == 0
        assert data["results"] == []


class TestAgentFormatter:
    """Tests for agent-optimized output formatting."""

    def test_agent_format_structure(self, mock_search_results):
        """Test agent-optimized format structure."""
        from research_kb_cli.formatters import format_results_agent

        output = format_results_agent(
            mock_search_results, query="test", context_type="balanced"
        )

        assert output.startswith("RESEARCH_KB_RESULTS")
        assert "QUERY: test" in output
        assert "CONTEXT: balanced" in output
        assert "COUNT: 1" in output
        assert "---USAGE:" in output.replace("\n", " ") or "USAGE:" in output

    def test_agent_format_provenance(self, mock_search_results):
        """Test agent format includes provenance information."""
        from research_kb_cli.formatters import format_results_agent

        output = format_results_agent(mock_search_results, query="test")

        # Should include citation format
        assert "CITE:" in output
        # Should include author
        assert "Author" in output

    def test_agent_format_no_results(self):
        """Test agent format with no results."""
        from research_kb_cli.formatters import format_results_agent

        output = format_results_agent([], query="no match")

        assert "[NO RESULTS]" in output
        assert "no match" in output

    def test_agent_format_context_type(self, mock_search_results):
        """Test agent format includes context type."""
        from research_kb_cli.formatters import format_results_agent

        output = format_results_agent(
            mock_search_results, query="test", context_type="building"
        )

        assert "CONTEXT: building" in output


class TestFormatterEdgeCases:
    """Tests for edge cases and error handling in formatters."""

    def test_markdown_long_content_truncation(self):
        """Test markdown truncates long content."""
        from research_kb_cli.formatters import format_result_markdown
        from research_kb_contracts import Chunk, Source, SearchResult, SourceType
        from datetime import datetime
        from uuid import uuid4

        # Create result with very long content
        chunk = Chunk(
            id=uuid4(),
            source_id=uuid4(),
            content="A" * 1000,  # 1000 characters
            content_hash="hash",
            page_start=1,
            page_end=1,
            metadata={},
            created_at=datetime.now(),
        )

        source = Source(
            id=uuid4(),
            title="Test Source",
            authors=["Author"],
            year=2024,
            source_type=SourceType.PAPER,
            file_hash="hash",
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        result = SearchResult(
            chunk=chunk,
            source=source,
            fts_score=0.8,
            vector_score=0.7,
            graph_score=None,
            combined_score=0.75,
            rank=1,
        )

        output = format_result_markdown(result, show_content=True)

        # Should truncate to 500 chars + "..."
        assert "..." in output
        # Content section should be shorter than full content
        assert len(output) < 1500

    def test_json_special_characters(self):
        """Test JSON formatter handles special characters."""
        from research_kb_cli.formatters import format_results_json
        from research_kb_contracts import Chunk, Source, SearchResult, SourceType
        from datetime import datetime
        from uuid import uuid4

        # Create result with special characters
        chunk = Chunk(
            id=uuid4(),
            source_id=uuid4(),
            content='Content with "quotes" and \n newlines',
            content_hash="hash",
            page_start=1,
            page_end=1,
            metadata={},
            created_at=datetime.now(),
        )

        source = Source(
            id=uuid4(),
            title="Test: Special & Characters",
            authors=["O'Brien"],
            year=2024,
            source_type=SourceType.PAPER,
            file_hash="hash",
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        result = SearchResult(
            chunk=chunk,
            source=source,
            fts_score=0.8,
            vector_score=0.7,
            graph_score=None,
            combined_score=0.75,
            rank=1,
        )

        output = format_results_json([result], query="test")

        # Should be valid JSON
        data = json.loads(output)
        assert data["results"][0]["chunk"]["content"] == chunk.content
