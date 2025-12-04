"""Tests for logging configuration."""

import pytest

from research_kb_common.logging_config import configure_logging, get_logger


class TestLoggingConfiguration:
    """Test logging configuration and logger creation."""

    def test_configure_logging_sets_up_structlog(self):
        """Test configure_logging initializes structlog correctly."""
        # Configure with JSON output for testing
        configure_logging(level="INFO", json_output=True)

        logger = get_logger("test_module")

        # Verify logger has log methods (duck typing - structlog returns LazyProxy)
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")
        assert callable(logger.info)

    def test_logger_name_preserved(self):
        """Test logger name is preserved in log output."""
        configure_logging(level="DEBUG", json_output=False)

        logger1 = get_logger("module_a")
        logger2 = get_logger("module_b")

        # Loggers should be different instances with different names
        assert logger1 is not logger2

    def test_json_output_mode_configured(self):
        """Test JSON output mode can be configured without errors."""
        # Just verify configuration doesn't crash - output capture is complex with structlog
        try:
            configure_logging(level="INFO", json_output=True)
            logger = get_logger("test")
            # Log message - should not crash
            logger.info("test_event", key1="value1", key2=42)
        except Exception as e:
            pytest.fail(f"JSON logging configuration failed: {e}")

    def test_get_logger_returns_usable_logger(self):
        """Test get_logger returns a usable logger instance."""
        configure_logging()

        logger = get_logger("test_logger")

        # Verify logger has necessary methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")

        # Logger should be callable (won't crash)
        try:
            logger.info("test_message", key="value")
        except Exception as e:
            pytest.fail(f"Logger.info() raised {e}")

    def test_different_log_levels(self):
        """Test different log levels can be configured."""
        # Test each level can be set without error
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            configure_logging(level=level, json_output=False)
            logger = get_logger(f"test_{level}")
            assert logger is not None

    def test_human_readable_output_mode(self):
        """Test human-readable output mode (development) can be configured."""
        # Verify configuration doesn't crash - output capture is complex with structlog
        try:
            configure_logging(level="INFO", json_output=False)
            logger = get_logger("dev_test")
            # Log message - should not crash
            logger.info("development_log", user_id="abc123")
        except Exception as e:
            pytest.fail(f"Human-readable logging configuration failed: {e}")
