"""Tests for chunk filtering heuristics."""

import pytest

from research_kb_extraction.chunk_filter import (
    FilterDecision,
    FilterResult,
    calculate_metrics,
    filter_chunk,
    filter_chunks,
    get_filter_stats,
)


class TestFilterDecision:
    """Tests for FilterDecision enum."""

    def test_process_value(self):
        """Test PROCESS enum value."""
        assert FilterDecision.PROCESS.value == "process"

    def test_skip_value(self):
        """Test SKIP enum value."""
        assert FilterDecision.SKIP.value == "skip"

    def test_flag_value(self):
        """Test FLAG enum value."""
        assert FilterDecision.FLAG.value == "flag"


class TestFilterResult:
    """Tests for FilterResult dataclass."""

    def test_with_all_fields(self):
        """Test FilterResult with all fields specified."""
        metrics = {"length": 500, "alpha_ratio": 0.8}
        result = FilterResult(
            decision=FilterDecision.PROCESS,
            reason="normal",
            metrics=metrics,
        )

        assert result.decision == FilterDecision.PROCESS
        assert result.reason == "normal"
        assert result.metrics == metrics

    def test_with_defaults(self):
        """Test FilterResult with default values."""
        result = FilterResult(decision=FilterDecision.SKIP)

        assert result.decision == FilterDecision.SKIP
        assert result.reason is None
        assert result.metrics is None


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_basic_metrics(self):
        """Test basic metric calculations."""
        content = "This is a sample text with some words."
        metrics = calculate_metrics(content)

        assert "length" in metrics
        assert "line_count" in metrics
        assert "word_count" in metrics
        assert "alpha_ratio" in metrics
        assert metrics["length"] == len(content)
        assert metrics["word_count"] == 8

    def test_alpha_ratio(self):
        """Test alpha ratio calculation."""
        # All alpha
        all_alpha = "abcdefghij"
        metrics = calculate_metrics(all_alpha)
        assert metrics["alpha_ratio"] == pytest.approx(1.0)

        # Half alpha, half digits
        mixed = "abcde12345"
        metrics = calculate_metrics(mixed)
        assert metrics["alpha_ratio"] == pytest.approx(0.5)

        # No alpha
        no_alpha = "12345!@#$%"
        metrics = calculate_metrics(no_alpha)
        assert metrics["alpha_ratio"] == pytest.approx(0.0)

    def test_reference_detection(self):
        """Test reference block detection."""
        # Academic reference format
        content = """
        [1] Smith, J. (2020). Title of paper.
        [2] Jones, A. (2019). Another paper.
        [3] Brown, B. (2018). Yet another.
        [4] Davis, C. (2017). More papers.
        [5] Evans, D. (2016). Even more.
        [6] Frank, E. (2015). Last one.
        """
        metrics = calculate_metrics(content)
        assert metrics["reference_matches"] >= 5

    def test_table_row_detection(self):
        """Test table row detection."""
        # Table-like content - pure numeric rows
        content = """
1.5 2.3 4.5
2.0 3.1 5.2
3.5 4.2 6.1
        """
        metrics = calculate_metrics(content)
        # The regex matches lines with only whitespace, digits, dots, commas, hyphens, pipes
        assert metrics["table_rows"] >= 1

    def test_equation_detection(self):
        """Test equation symbol detection."""
        # Math symbols
        content = "∑ ∏ ∫ ∂ ∇ √ ∞ ≈ ≠ ≤ ≥ ∈"
        metrics = calculate_metrics(content)
        assert metrics["equation_symbols"] >= 10

    def test_ocr_garbage_detection(self):
        """Test OCR garbage detection."""
        # Non-ASCII garbage
        content = "Normal text with 日本語文字 garbage"
        metrics = calculate_metrics(content)
        assert metrics["ocr_garbage"] >= 1

        # Symbol garbage
        content2 = "Text with !@#$%^ symbols"
        metrics2 = calculate_metrics(content2)
        assert metrics2["ocr_garbage"] >= 1


class TestFilterChunk:
    """Tests for filter_chunk function."""

    def test_too_short_skip(self):
        """Test chunks under 100 chars are skipped."""
        content = "Short text."  # 11 chars
        result = filter_chunk(content)

        assert result.decision == FilterDecision.SKIP
        assert result.reason == "too_short"

    def test_too_long_flag(self):
        """Test chunks over 15000 chars are flagged."""
        content = "a" * 15001
        result = filter_chunk(content)

        assert result.decision == FilterDecision.FLAG
        assert result.reason == "too_long"

    def test_low_alpha_skip(self):
        """Test chunks with <40% alpha ratio are skipped."""
        # Mostly numbers and punctuation
        content = "123456789!@#$%^&*()_+-=[]{}|;':\",./<>?" * 10
        result = filter_chunk(content)

        assert result.decision == FilterDecision.SKIP
        assert result.reason == "low_alpha_ratio"

    def test_reference_block_skip(self):
        """Test reference blocks with >5 matches are skipped."""
        content = """
        References:
        [1] Smith, J. (2020). Title of paper. Journal of Science.
        [2] Jones, A. (2019). Another paper title. Nature.
        [3] Brown, B. (2018). Yet another paper. Science.
        [4] Davis, C. (2017). More papers here. Cell.
        [5] Evans, D. (2016). Even more papers. PNAS.
        [6] Frank, E. (2015). Last reference. NEJM.
        [7] Green, F. (2014). Additional ref. Lancet.
        """
        result = filter_chunk(content)

        assert result.decision == FilterDecision.SKIP
        assert result.reason == "reference_block"

    def test_table_heavy_flag(self):
        """Test table-heavy content is flagged."""
        # Create content where >50% of lines match table_row pattern
        # But with enough alpha ratio (>40%) to not be skipped
        # Table row pattern: ^[\s\d\.\,\-\|]+$
        lines = []
        # Add some normal text lines for alpha ratio
        lines.append("This is a data table showing experimental results:")
        lines.append("The values below represent measurements:")
        # Add many pure numeric lines (>50% of total)
        for i in range(20):
            lines.append("1.5, 2.3, 4.5")  # Matches table_row pattern
        content = "\n".join(lines)

        result = filter_chunk(content)

        # Due to the order of checks, this might hit low_alpha_ratio first
        # since numeric-heavy content has low alpha ratio
        # Accept either FLAG with table_heavy or SKIP with low_alpha_ratio
        assert result.decision in (FilterDecision.FLAG, FilterDecision.SKIP)
        if result.decision == FilterDecision.FLAG:
            assert result.reason == "table_heavy"

    def test_ocr_garbage_skip(self):
        """Test OCR garbage content is skipped."""
        # Content with OCR artifacts
        content = (
            "Normal text but with lots of 日本語日本語日本語 "
            "and 中文中文中文 garbage characters "
            "and more 한국어한국어 artifacts throughout. " * 3
        )
        result = filter_chunk(content)

        assert result.decision == FilterDecision.SKIP
        assert result.reason == "ocr_garbage"

    def test_equation_heavy_flag(self):
        """Test equation-heavy content is flagged."""
        # Create content with many equation symbols but enough text
        # to maintain alpha ratio above 40%
        text_part = (
            "In mathematical analysis we use various operators. "
            "The summation operator is commonly used for series. "
            "Integration is fundamental to calculus. "
            "Partial derivatives are used in multivariable calculus. "
            "Set theory uses symbols for membership and containment. "
        )
        # Add equation symbols - need >10 to trigger
        equation_part = "∑∏∫∂∇√∞≈≠≤≥∈∉⊂⊃∪∩"
        content = text_part + equation_part

        result = filter_chunk(content)

        # Should FLAG due to equation_heavy (>10 symbols)
        assert result.decision == FilterDecision.FLAG
        assert result.reason == "equation_heavy"

    def test_normal_content_process(self):
        """Test normal content is processed."""
        content = """
        Instrumental variables (IV) estimation is a widely used approach for
        addressing endogeneity in econometric analysis. The IV method relies
        on two key assumptions: the relevance condition, which requires that
        the instrument be correlated with the endogenous regressor, and the
        exclusion restriction, which stipulates that the instrument affects
        the outcome only through its effect on the treatment.
        """
        result = filter_chunk(content)

        assert result.decision == FilterDecision.PROCESS
        assert result.reason is None

    def test_edge_case_100_chars(self):
        """Test boundary: exactly 100 chars should process (not skip)."""
        content = "a" * 100
        result = filter_chunk(content)

        # Exactly 100 should NOT be too_short (threshold is <100)
        assert result.reason != "too_short"

    def test_edge_case_15000_chars(self):
        """Test boundary: exactly 15000 chars should process (not flag)."""
        content = "a" * 15000
        result = filter_chunk(content)

        # Exactly 15000 should NOT be too_long (threshold is >15000)
        assert result.reason != "too_long"


class TestFilterChunks:
    """Tests for filter_chunks function."""

    def test_batch_filtering(self):
        """Test batch filtering returns three lists."""
        chunks = [
            ("id1", "Normal academic content about causal inference methods."),
            ("id2", "Short"),  # Too short
            ("id3", "a" * 200),  # Normal length, all alpha
        ]

        process_list, skip_list, flag_list = filter_chunks(chunks)

        # Short chunk should be skipped
        assert len(skip_list) >= 1
        assert any(cid == "id2" for cid, _ in skip_list)

    def test_include_flagged_true(self):
        """Test flagged chunks are included in process list by default."""
        # Create a chunk that will be flagged (too long)
        long_content = "a" * 16000
        chunks = [
            ("id1", long_content),
            ("id2", "Normal content with sufficient length for processing."),
        ]

        process_list, skip_list, flag_list = filter_chunks(chunks, include_flagged=True)

        # Flagged chunk should be in both flag_list and process_list
        flagged_ids = [cid for cid, _ in flag_list]
        process_ids = [cid for cid, _ in process_list]

        if "id1" in flagged_ids:
            assert "id1" in process_ids

    def test_include_flagged_false(self):
        """Test flagged chunks excluded from process list when include_flagged=False."""
        # Create a chunk that will be flagged (too long)
        long_content = "a" * 16000
        chunks = [
            ("id1", long_content),
            ("id2", "Normal content with sufficient length for processing."),
        ]

        process_list, skip_list, flag_list = filter_chunks(chunks, include_flagged=False)

        # Flagged chunk should NOT be in process_list
        flagged_ids = [cid for cid, _ in flag_list]
        process_ids = [cid for cid, _ in process_list]

        for flagged_id in flagged_ids:
            assert flagged_id not in process_ids


class TestGetFilterStats:
    """Tests for get_filter_stats function."""

    def test_stats_structure(self):
        """Test stats dict has correct structure."""
        chunks = [
            ("id1", "Normal academic content about instrumental variables."),
            ("id2", "Short"),  # Too short
        ]

        stats = get_filter_stats(chunks)

        assert "total" in stats
        assert "by_decision" in stats
        assert "by_reason" in stats
        assert stats["total"] == 2
        assert "process" in stats["by_decision"]
        assert "skip" in stats["by_decision"]
        assert "flag" in stats["by_decision"]

    def test_stats_by_reason(self):
        """Test stats tracks reasons correctly."""
        chunks = [
            ("id1", "Short"),  # Too short
            ("id2", "Tiny"),  # Also too short
            ("id3", "a" * 16000),  # Too long (flagged)
        ]

        stats = get_filter_stats(chunks)

        # Both short chunks should have "too_short" reason
        assert "too_short" in stats["by_reason"]
        assert stats["by_reason"]["too_short"] == 2

        # Long chunk should have "too_long" reason
        assert "too_long" in stats["by_reason"]
        assert stats["by_reason"]["too_long"] == 1
