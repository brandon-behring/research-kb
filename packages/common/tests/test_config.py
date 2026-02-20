"""Tests for configuration management module.

Tests cover:
- Settings defaults
- Environment variable overrides
- Field validators (log_level, log_format)
- Settings caching (lru_cache)
- .env file loading
"""

from __future__ import annotations

import os
import pytest

from pydantic import ValidationError

from research_kb_common.config import Settings, get_settings

pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def clean_env():
    """Provide a clean environment without config-related vars."""
    # Store original values
    env_vars = [
        "DATABASE_URL",
        "GROBID_URL",
        "EMBEDDING_MODEL",
        "EMBEDDING_CACHE_DIR",
        "LOG_LEVEL",
        "LOG_FORMAT",
        "API_HOST",
        "API_PORT",
        "S2_API_KEY",
        "DAEMON_SOCKET_PATH",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    ]
    original_values = {var: os.environ.get(var) for var in env_vars}

    # Remove all config vars
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore original values
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


@pytest.fixture
def clear_settings_cache():
    """Clear the settings cache before and after each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# =============================================================================
# Test Default Values
# =============================================================================


class TestSettingsDefaults:
    """Test Settings has correct default values."""

    def test_database_url_default(self, clean_env):
        """Test database_url has development default."""
        settings = Settings()

        assert settings.database_url == "postgresql://postgres:postgres@localhost:5432/research_kb"

    def test_grobid_url_default(self, clean_env):
        """Test grobid_url has development default."""
        settings = Settings()

        assert settings.grobid_url == "http://localhost:8070"

    def test_embedding_model_default(self, clean_env):
        """Test embedding_model has correct default."""
        settings = Settings()

        assert settings.embedding_model == "BAAI/bge-large-en-v1.5"

    def test_embedding_cache_dir_default(self, clean_env):
        """Test embedding_cache_dir has correct default."""
        settings = Settings()

        assert settings.embedding_cache_dir == "~/.cache/sentence_transformers"

    def test_log_level_default(self, clean_env):
        """Test log_level defaults to INFO."""
        settings = Settings()

        assert settings.log_level == "INFO"

    def test_log_format_default(self, clean_env):
        """Test log_format defaults to console."""
        settings = Settings()

        assert settings.log_format == "console"

    def test_api_host_default(self, clean_env):
        """Test api_host defaults to 0.0.0.0."""
        settings = Settings()

        assert settings.api_host == "0.0.0.0"

    def test_api_port_default(self, clean_env):
        """Test api_port defaults to 8000."""
        settings = Settings()

        assert settings.api_port == 8000

    def test_s2_api_key_default(self, clean_env):
        """Test s2_api_key defaults to None."""
        settings = Settings()

        assert settings.s2_api_key is None

    def test_daemon_socket_path_default(self, clean_env):
        """Test daemon_socket_path has correct default."""
        settings = Settings()

        assert settings.daemon_socket_path == "/tmp/research_kb_daemon.sock"

    def test_otel_endpoint_default(self, clean_env):
        """Test otel_exporter_otlp_endpoint defaults to None."""
        settings = Settings()

        assert settings.otel_exporter_otlp_endpoint is None


# =============================================================================
# Test Environment Variable Overrides
# =============================================================================


class TestEnvironmentOverrides:
    """Test Settings can be overridden via environment variables."""

    def test_database_url_override(self, clean_env):
        """Test DATABASE_URL environment variable override."""
        os.environ["DATABASE_URL"] = "postgresql://user:pass@prod:5432/db"

        settings = Settings()

        assert settings.database_url == "postgresql://user:pass@prod:5432/db"

    def test_grobid_url_override(self, clean_env):
        """Test GROBID_URL environment variable override."""
        os.environ["GROBID_URL"] = "http://grobid:8070"

        settings = Settings()

        assert settings.grobid_url == "http://grobid:8070"

    def test_embedding_model_override(self, clean_env):
        """Test EMBEDDING_MODEL environment variable override."""
        os.environ["EMBEDDING_MODEL"] = "custom/model"

        settings = Settings()

        assert settings.embedding_model == "custom/model"

    def test_embedding_cache_dir_override(self, clean_env):
        """Test EMBEDDING_CACHE_DIR environment variable override."""
        os.environ["EMBEDDING_CACHE_DIR"] = "/custom/cache"

        settings = Settings()

        assert settings.embedding_cache_dir == "/custom/cache"

    def test_log_level_override(self, clean_env):
        """Test LOG_LEVEL environment variable override."""
        os.environ["LOG_LEVEL"] = "DEBUG"

        settings = Settings()

        assert settings.log_level == "DEBUG"

    def test_log_format_override(self, clean_env):
        """Test LOG_FORMAT environment variable override."""
        os.environ["LOG_FORMAT"] = "json"

        settings = Settings()

        assert settings.log_format == "json"

    def test_api_host_override(self, clean_env):
        """Test API_HOST environment variable override."""
        os.environ["API_HOST"] = "127.0.0.1"

        settings = Settings()

        assert settings.api_host == "127.0.0.1"

    def test_api_port_override(self, clean_env):
        """Test API_PORT environment variable override."""
        os.environ["API_PORT"] = "9000"

        settings = Settings()

        assert settings.api_port == 9000

    def test_s2_api_key_override(self, clean_env):
        """Test S2_API_KEY environment variable override."""
        os.environ["S2_API_KEY"] = "secret-key-123"

        settings = Settings()

        assert settings.s2_api_key == "secret-key-123"

    def test_daemon_socket_path_override(self, clean_env):
        """Test DAEMON_SOCKET_PATH environment variable override."""
        os.environ["DAEMON_SOCKET_PATH"] = "/var/run/research_kb.sock"

        settings = Settings()

        assert settings.daemon_socket_path == "/var/run/research_kb.sock"

    def test_otel_endpoint_override(self, clean_env):
        """Test OTEL_EXPORTER_OTLP_ENDPOINT environment variable override."""
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://otel:4317"

        settings = Settings()

        assert settings.otel_exporter_otlp_endpoint == "http://otel:4317"

    def test_case_insensitive_env_vars(self, clean_env):
        """Test environment variables are case insensitive."""
        os.environ["database_url"] = "postgresql://lower:case@localhost/db"

        settings = Settings()

        assert settings.database_url == "postgresql://lower:case@localhost/db"


# =============================================================================
# Test Field Validators
# =============================================================================


class TestLogLevelValidator:
    """Test log_level validator."""

    def test_log_level_debug(self, clean_env):
        """Test DEBUG log level is valid."""
        settings = Settings(log_level="DEBUG")
        assert settings.log_level == "DEBUG"

    def test_log_level_info(self, clean_env):
        """Test INFO log level is valid."""
        settings = Settings(log_level="INFO")
        assert settings.log_level == "INFO"

    def test_log_level_warning(self, clean_env):
        """Test WARNING log level is valid."""
        settings = Settings(log_level="WARNING")
        assert settings.log_level == "WARNING"

    def test_log_level_error(self, clean_env):
        """Test ERROR log level is valid."""
        settings = Settings(log_level="ERROR")
        assert settings.log_level == "ERROR"

    def test_log_level_critical(self, clean_env):
        """Test CRITICAL log level is valid."""
        settings = Settings(log_level="CRITICAL")
        assert settings.log_level == "CRITICAL"

    def test_log_level_lowercase_converted(self, clean_env):
        """Test lowercase log level is converted to uppercase."""
        settings = Settings(log_level="debug")
        assert settings.log_level == "DEBUG"

    def test_log_level_mixed_case_converted(self, clean_env):
        """Test mixed case log level is converted to uppercase."""
        settings = Settings(log_level="Warning")
        assert settings.log_level == "WARNING"

    def test_log_level_invalid_raises(self, clean_env):
        """Test invalid log level raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(log_level="INVALID")

        errors = exc_info.value.errors()
        assert any("log_level" in str(e) for e in errors)

    def test_log_level_invalid_via_env(self, clean_env):
        """Test invalid log level via environment variable raises error."""
        os.environ["LOG_LEVEL"] = "TRACE"

        with pytest.raises(ValidationError):
            Settings()


class TestLogFormatValidator:
    """Test log_format validator."""

    def test_log_format_json(self, clean_env):
        """Test json log format is valid."""
        settings = Settings(log_format="json")
        assert settings.log_format == "json"

    def test_log_format_console(self, clean_env):
        """Test console log format is valid."""
        settings = Settings(log_format="console")
        assert settings.log_format == "console"

    def test_log_format_uppercase_converted(self, clean_env):
        """Test uppercase log format is converted to lowercase."""
        settings = Settings(log_format="JSON")
        assert settings.log_format == "json"

    def test_log_format_mixed_case_converted(self, clean_env):
        """Test mixed case log format is converted to lowercase."""
        settings = Settings(log_format="Console")
        assert settings.log_format == "console"

    def test_log_format_invalid_raises(self, clean_env):
        """Test invalid log format raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(log_format="xml")

        errors = exc_info.value.errors()
        assert any("log_format" in str(e) for e in errors)

    def test_log_format_invalid_via_env(self, clean_env):
        """Test invalid log format via environment variable raises error."""
        os.environ["LOG_FORMAT"] = "yaml"

        with pytest.raises(ValidationError):
            Settings()


# =============================================================================
# Test Settings Caching
# =============================================================================


class TestGetSettings:
    """Test get_settings function and caching."""

    def test_get_settings_returns_settings(self, clean_env, clear_settings_cache):
        """Test get_settings returns a Settings instance."""
        settings = get_settings()

        assert isinstance(settings, Settings)

    def test_get_settings_cached(self, clean_env, clear_settings_cache):
        """Test get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_cache_clear_reloads_settings(self, clean_env, clear_settings_cache):
        """Test clearing cache causes reload."""
        settings1 = get_settings()

        # Modify environment
        os.environ["LOG_LEVEL"] = "DEBUG"

        # Should still return cached instance
        settings2 = get_settings()
        assert settings2 is settings1

        # Clear cache
        get_settings.cache_clear()

        # Should return new instance with updated value
        settings3 = get_settings()
        assert settings3 is not settings1
        assert settings3.log_level == "DEBUG"

    def test_get_settings_cache_info(self, clean_env, clear_settings_cache):
        """Test cache_info is available."""
        get_settings()
        get_settings()

        cache_info = get_settings.cache_info()

        assert cache_info.hits >= 1
        assert cache_info.misses >= 1


# =============================================================================
# Test Model Configuration
# =============================================================================


class TestModelConfig:
    """Test Settings model configuration."""

    def test_extra_fields_ignored(self, clean_env):
        """Test extra fields are ignored (not raising errors)."""
        os.environ["UNKNOWN_SETTING"] = "value"

        # Should not raise
        settings = Settings()

        assert not hasattr(settings, "UNKNOWN_SETTING")
        assert not hasattr(settings, "unknown_setting")

    def test_settings_immutable_fields(self, clean_env):
        """Test settings fields can be accessed."""
        settings = Settings()

        # All fields should be accessible
        assert hasattr(settings, "database_url")
        assert hasattr(settings, "grobid_url")
        assert hasattr(settings, "embedding_model")
        assert hasattr(settings, "log_level")
        assert hasattr(settings, "api_host")
        assert hasattr(settings, "api_port")


# =============================================================================
# Test Type Coercion
# =============================================================================


class TestTypeCoercion:
    """Test type coercion for settings."""

    def test_api_port_string_to_int(self, clean_env):
        """Test API_PORT string is coerced to int."""
        os.environ["API_PORT"] = "3000"

        settings = Settings()

        assert settings.api_port == 3000
        assert isinstance(settings.api_port, int)

    def test_api_port_invalid_raises(self, clean_env):
        """Test invalid API_PORT raises validation error."""
        os.environ["API_PORT"] = "not_a_number"

        with pytest.raises(ValidationError):
            Settings()


# =============================================================================
# Test .env File Loading
# =============================================================================


class TestEnvFileLoading:
    """Test .env file loading behavior."""

    def test_env_file_config(self):
        """Test env_file is configured."""
        # Check model config
        assert Settings.model_config.get("env_file") == ".env"

    def test_env_file_encoding_config(self):
        """Test env_file encoding is configured."""
        assert Settings.model_config.get("env_file_encoding") == "utf-8"


# =============================================================================
# Test Optional Fields
# =============================================================================


class TestOptionalFields:
    """Test optional field handling."""

    def test_s2_api_key_optional(self, clean_env):
        """Test s2_api_key can be None."""
        settings = Settings()

        assert settings.s2_api_key is None

    def test_s2_api_key_set_empty_string(self, clean_env):
        """Test s2_api_key empty string becomes empty string."""
        os.environ["S2_API_KEY"] = ""

        settings = Settings()

        # Empty string should be kept as empty string
        assert settings.s2_api_key == ""

    def test_otel_endpoint_optional(self, clean_env):
        """Test otel_exporter_otlp_endpoint can be None."""
        settings = Settings()

        assert settings.otel_exporter_otlp_endpoint is None


# =============================================================================
# Test Field Descriptions
# =============================================================================


class TestFieldDescriptions:
    """Test field descriptions for documentation."""

    def test_fields_have_descriptions(self):
        """Test all fields have descriptions."""
        schema = Settings.model_json_schema()
        properties = schema.get("properties", {})

        # Key fields should have descriptions
        assert "database_url" in properties
        assert "description" in properties["database_url"]

        assert "grobid_url" in properties
        assert "description" in properties["grobid_url"]

        assert "embedding_model" in properties
        assert "description" in properties["embedding_model"]

        assert "log_level" in properties
        assert "description" in properties["log_level"]


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases in configuration."""

    def test_database_url_with_special_chars(self, clean_env):
        """Test database URL with special characters in password."""
        os.environ["DATABASE_URL"] = "postgresql://user:p@ss%word@host/db"

        settings = Settings()

        assert settings.database_url == "postgresql://user:p@ss%word@host/db"

    def test_socket_path_with_spaces(self, clean_env):
        """Test socket path with spaces."""
        os.environ["DAEMON_SOCKET_PATH"] = "/path with spaces/socket.sock"

        settings = Settings()

        assert settings.daemon_socket_path == "/path with spaces/socket.sock"

    def test_empty_embedding_cache_dir(self, clean_env):
        """Test empty embedding cache dir."""
        os.environ["EMBEDDING_CACHE_DIR"] = ""

        settings = Settings()

        assert settings.embedding_cache_dir == ""

    def test_api_port_zero(self, clean_env):
        """Test API port can be zero (random port)."""
        os.environ["API_PORT"] = "0"

        settings = Settings()

        assert settings.api_port == 0

    def test_api_port_high_number(self, clean_env):
        """Test API port can be high port number."""
        os.environ["API_PORT"] = "65535"

        settings = Settings()

        assert settings.api_port == 65535

    def test_grobid_url_with_path(self, clean_env):
        """Test GROBID URL with path."""
        os.environ["GROBID_URL"] = "http://localhost:8070/api/processFulltextDocument"

        settings = Settings()

        assert settings.grobid_url == "http://localhost:8070/api/processFulltextDocument"


# =============================================================================
# Test Serialization
# =============================================================================


class TestSerialization:
    """Test Settings serialization."""

    def test_model_dump(self, clean_env):
        """Test Settings can be serialized to dict."""
        settings = Settings()

        data = settings.model_dump()

        assert isinstance(data, dict)
        assert "database_url" in data
        assert "grobid_url" in data
        assert "log_level" in data

    def test_model_dump_json(self, clean_env):
        """Test Settings can be serialized to JSON."""
        settings = Settings()

        json_str = settings.model_dump_json()

        assert isinstance(json_str, str)
        assert "database_url" in json_str
        assert "log_level" in json_str
