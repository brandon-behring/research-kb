"""Tests for S2 Client custom exceptions.

Tests exception hierarchy, error messages, attributes, and exception handling patterns.
"""

import pytest

from s2_client.errors import (
    S2APIError,
    S2CacheError,
    S2ConfigError,
    S2Error,
    S2NotFoundError,
    S2RateLimitError,
)


# -----------------------------------------------------------------------------
# Base Exception Tests
# -----------------------------------------------------------------------------


class TestS2Error:
    """Tests for base S2Error exception."""

    def test_is_exception(self):
        """S2Error should be an Exception."""
        assert issubclass(S2Error, Exception)

    def test_can_raise(self):
        """S2Error should be raisable."""
        with pytest.raises(S2Error):
            raise S2Error("Test error")

    def test_message_preserved(self):
        """Error message should be preserved."""
        error = S2Error("Custom error message")
        assert str(error) == "Custom error message"

    def test_can_catch_as_exception(self):
        """S2Error should be catchable as Exception."""
        try:
            raise S2Error("Test")
        except Exception as e:
            assert isinstance(e, S2Error)


# -----------------------------------------------------------------------------
# S2APIError Tests
# -----------------------------------------------------------------------------


class TestS2APIError:
    """Tests for S2APIError exception."""

    def test_inherits_from_s2error(self):
        """S2APIError should inherit from S2Error."""
        assert issubclass(S2APIError, S2Error)

    def test_constructor_with_all_args(self):
        """Test constructor with all arguments."""
        error = S2APIError(
            status_code=500,
            message="Internal server error",
            endpoint="/paper/123",
        )

        assert error.status_code == 500
        assert error.message == "Internal server error"
        assert error.endpoint == "/paper/123"

    def test_constructor_default_endpoint(self):
        """Test constructor with default endpoint."""
        error = S2APIError(status_code=400, message="Bad request")

        assert error.status_code == 400
        assert error.message == "Bad request"
        assert error.endpoint == ""

    def test_string_representation(self):
        """Test error string representation."""
        error = S2APIError(
            status_code=500,
            message="Server error",
            endpoint="/search",
        )

        error_str = str(error)
        assert "500" in error_str
        assert "/search" in error_str
        assert "Server error" in error_str

    def test_can_catch_as_s2error(self):
        """S2APIError should be catchable as S2Error."""
        try:
            raise S2APIError(500, "Test")
        except S2Error as e:
            assert isinstance(e, S2APIError)

    def test_common_status_codes(self):
        """Test common HTTP status codes."""
        codes = [400, 401, 403, 404, 429, 500, 502, 503]

        for code in codes:
            error = S2APIError(code, f"Error {code}")
            assert error.status_code == code


# -----------------------------------------------------------------------------
# S2RateLimitError Tests
# -----------------------------------------------------------------------------


class TestS2RateLimitError:
    """Tests for S2RateLimitError exception."""

    def test_inherits_from_s2apierror(self):
        """S2RateLimitError should inherit from S2APIError."""
        assert issubclass(S2RateLimitError, S2APIError)

    def test_inherits_from_s2error(self):
        """S2RateLimitError should also be catchable as S2Error."""
        assert issubclass(S2RateLimitError, S2Error)

    def test_status_code_is_429(self):
        """Status code should always be 429."""
        error = S2RateLimitError()
        assert error.status_code == 429

    def test_with_retry_after(self):
        """Test with retry_after value."""
        error = S2RateLimitError(retry_after=60.0)

        assert error.retry_after == 60.0
        assert "60" in str(error)

    def test_without_retry_after(self):
        """Test without retry_after value."""
        error = S2RateLimitError()

        assert error.retry_after is None
        assert "Rate limit exceeded" in str(error)

    def test_with_endpoint(self):
        """Test with endpoint specified."""
        error = S2RateLimitError(retry_after=30.0, endpoint="/search")

        assert error.endpoint == "/search"

    def test_message_includes_retry_hint(self):
        """Message should include retry hint when available."""
        error = S2RateLimitError(retry_after=120.0)

        message = str(error)
        assert "retry after" in message.lower()
        assert "120" in message

    def test_float_retry_after(self):
        """Retry after can be a float."""
        error = S2RateLimitError(retry_after=30.5)

        assert error.retry_after == 30.5

    def test_can_catch_as_api_error(self):
        """Should be catchable as S2APIError."""
        try:
            raise S2RateLimitError(retry_after=60)
        except S2APIError as e:
            assert e.status_code == 429


# -----------------------------------------------------------------------------
# S2NotFoundError Tests
# -----------------------------------------------------------------------------


class TestS2NotFoundError:
    """Tests for S2NotFoundError exception."""

    def test_inherits_from_s2apierror(self):
        """S2NotFoundError should inherit from S2APIError."""
        assert issubclass(S2NotFoundError, S2APIError)

    def test_status_code_is_404(self):
        """Status code should always be 404."""
        error = S2NotFoundError(resource_id="paper123")
        assert error.status_code == 404

    def test_with_resource_id(self):
        """Test with resource ID."""
        error = S2NotFoundError(resource_id="649def34f8be52c8b66281af98ae884c09aef38b")

        assert error.resource_id == "649def34f8be52c8b66281af98ae884c09aef38b"
        assert "649def34" in str(error)

    def test_default_resource_type(self):
        """Default resource type should be 'paper'."""
        error = S2NotFoundError(resource_id="123")

        assert error.resource_type == "paper"
        assert "Paper" in str(error) or "paper" in str(error).lower()

    def test_custom_resource_type(self):
        """Test with custom resource type."""
        error = S2NotFoundError(resource_id="author123", resource_type="author")

        assert error.resource_type == "author"
        assert "Author" in str(error) or "author" in str(error).lower()

    def test_message_includes_resource_info(self):
        """Message should include resource ID and type."""
        error = S2NotFoundError(resource_id="DOI:10.1234/test", resource_type="paper")

        message = str(error)
        assert "DOI:10.1234/test" in message or "10.1234" in message

    def test_can_catch_as_api_error(self):
        """Should be catchable as S2APIError."""
        try:
            raise S2NotFoundError("test123")
        except S2APIError as e:
            assert e.status_code == 404


# -----------------------------------------------------------------------------
# S2CacheError Tests
# -----------------------------------------------------------------------------


class TestS2CacheError:
    """Tests for S2CacheError exception."""

    def test_inherits_from_s2error(self):
        """S2CacheError should inherit from S2Error."""
        assert issubclass(S2CacheError, S2Error)

    def test_not_api_error(self):
        """S2CacheError should NOT inherit from S2APIError."""
        assert not issubclass(S2CacheError, S2APIError)

    def test_can_raise_with_message(self):
        """Should be raisable with custom message."""
        with pytest.raises(S2CacheError, match="Cache not initialized"):
            raise S2CacheError("Cache not initialized")

    def test_message_preserved(self):
        """Error message should be preserved."""
        error = S2CacheError("Database connection failed")
        assert str(error) == "Database connection failed"

    def test_can_catch_as_s2error(self):
        """Should be catchable as S2Error."""
        try:
            raise S2CacheError("Test")
        except S2Error as e:
            assert isinstance(e, S2CacheError)

    def test_not_caught_as_api_error(self):
        """Should NOT be caught as S2APIError."""
        with pytest.raises(S2CacheError):
            try:
                raise S2CacheError("Test")
            except S2APIError:
                pass  # Should not catch


# -----------------------------------------------------------------------------
# S2ConfigError Tests
# -----------------------------------------------------------------------------


class TestS2ConfigError:
    """Tests for S2ConfigError exception."""

    def test_inherits_from_s2error(self):
        """S2ConfigError should inherit from S2Error."""
        assert issubclass(S2ConfigError, S2Error)

    def test_not_api_error(self):
        """S2ConfigError should NOT inherit from S2APIError."""
        assert not issubclass(S2ConfigError, S2APIError)

    def test_can_raise_with_message(self):
        """Should be raisable with custom message."""
        with pytest.raises(S2ConfigError, match="Invalid API key"):
            raise S2ConfigError("Invalid API key")

    def test_message_preserved(self):
        """Error message should be preserved."""
        error = S2ConfigError("Missing required setting: api_key")
        assert "Missing required setting" in str(error)

    def test_can_catch_as_s2error(self):
        """Should be catchable as S2Error."""
        try:
            raise S2ConfigError("Test")
        except S2Error as e:
            assert isinstance(e, S2ConfigError)


# -----------------------------------------------------------------------------
# Exception Hierarchy Tests
# -----------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Tests for exception hierarchy and catch patterns."""

    def test_catch_all_s2_errors(self):
        """All S2 exceptions should be catchable as S2Error."""
        exceptions = [
            S2Error("base"),
            S2APIError(500, "api"),
            S2RateLimitError(),
            S2NotFoundError("id"),
            S2CacheError("cache"),
            S2ConfigError("config"),
        ]

        for exc in exceptions:
            try:
                raise exc
            except S2Error:
                pass  # All should be caught

    def test_catch_api_errors(self):
        """API-related errors should be catchable as S2APIError."""
        api_errors = [
            S2APIError(500, "server error"),
            S2RateLimitError(retry_after=60),
            S2NotFoundError("paper123"),
        ]

        for exc in api_errors:
            try:
                raise exc
            except S2APIError:
                pass  # All should be caught

    def test_specific_catch_before_general(self):
        """Specific exceptions should be caught before general ones."""
        # Rate limit should be caught specifically
        try:
            raise S2RateLimitError(retry_after=60)
        except S2RateLimitError as e:
            assert e.retry_after == 60
        except S2APIError:
            pytest.fail("Should have caught S2RateLimitError")

    def test_not_found_specific_catch(self):
        """NotFoundError should be caught specifically."""
        try:
            raise S2NotFoundError("test123", "paper")
        except S2NotFoundError as e:
            assert e.resource_id == "test123"
        except S2APIError:
            pytest.fail("Should have caught S2NotFoundError")


# -----------------------------------------------------------------------------
# Error Handling Pattern Tests
# -----------------------------------------------------------------------------


class TestErrorHandlingPatterns:
    """Tests demonstrating recommended error handling patterns."""

    def test_handle_rate_limit_with_retry(self):
        """Pattern: Handle rate limit with retry logic."""

        def make_api_call():
            raise S2RateLimitError(retry_after=60)

        retry_after = None
        try:
            make_api_call()
        except S2RateLimitError as e:
            retry_after = e.retry_after

        assert retry_after == 60

    def test_handle_not_found_gracefully(self):
        """Pattern: Handle not found with fallback."""

        def get_paper(paper_id: str):
            raise S2NotFoundError(paper_id)

        result = None
        try:
            result = get_paper("invalid123")
        except S2NotFoundError as e:
            # Log and return None
            assert e.resource_id == "invalid123"
            result = None

        assert result is None

    def test_handle_api_errors_generically(self):
        """Pattern: Handle all API errors generically."""

        def api_call():
            raise S2APIError(503, "Service unavailable", "/search")

        error_code = None
        try:
            api_call()
        except S2APIError as e:
            error_code = e.status_code

        assert error_code == 503

    def test_handle_cache_errors_separately(self):
        """Pattern: Handle cache errors separately from API errors."""

        def cache_operation():
            raise S2CacheError("Database locked")

        cache_failed = False
        api_failed = False

        try:
            cache_operation()
        except S2CacheError:
            cache_failed = True
        except S2APIError:
            api_failed = True

        assert cache_failed is True
        assert api_failed is False


# -----------------------------------------------------------------------------
# Edge Cases Tests
# -----------------------------------------------------------------------------


class TestErrorEdgeCases:
    """Tests for edge cases in error handling."""

    def test_empty_message(self):
        """Errors should handle empty messages."""
        error = S2APIError(500, "")
        assert error.status_code == 500

    def test_none_retry_after(self):
        """RateLimitError should handle None retry_after."""
        error = S2RateLimitError(retry_after=None)
        assert error.retry_after is None

    def test_zero_retry_after(self):
        """RateLimitError should handle zero retry_after."""
        error = S2RateLimitError(retry_after=0)
        assert error.retry_after == 0

    def test_empty_resource_id(self):
        """NotFoundError should handle empty resource ID."""
        error = S2NotFoundError(resource_id="")
        assert error.resource_id == ""

    def test_special_characters_in_message(self):
        """Errors should handle special characters."""
        error = S2APIError(
            500,
            "Error: <script>alert('xss')</script>",
            "/paper/test",
        )
        assert "<script>" in error.message

    def test_unicode_in_message(self):
        """Errors should handle Unicode in messages."""
        error = S2APIError(400, "Invalid query: 机器学习")
        assert "机器学习" in error.message

    def test_very_long_message(self):
        """Errors should handle very long messages."""
        long_message = "Error: " + "x" * 10000
        error = S2APIError(500, long_message)
        assert len(error.message) == len(long_message)

    def test_error_repr(self):
        """Errors should have reasonable repr."""
        error = S2NotFoundError("test123", "paper")
        repr_str = repr(error)
        # Should contain class name and some info
        assert "S2NotFoundError" in repr_str or "404" in repr_str or "test123" in repr_str
