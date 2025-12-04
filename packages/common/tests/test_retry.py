"""Tests for retry and backoff patterns."""

import pytest

from research_kb_common.retry import retry_on_exception, with_exponential_backoff


class FlakyCounter:
    """Test helper that fails N times before succeeding."""

    def __init__(self, failures_before_success: int):
        self.attempts = 0
        self.failures_before_success = failures_before_success

    def call(self) -> str:
        """Call that fails N times then succeeds."""
        self.attempts += 1
        if self.attempts <= self.failures_before_success:
            raise ConnectionError(f"Attempt {self.attempts} failed")
        return "success"


class TestRetryOnException:
    """Test retry_on_exception decorator."""

    def test_succeeds_on_first_attempt(self):
        """Test function that succeeds immediately doesn't retry."""
        counter = FlakyCounter(failures_before_success=0)

        @retry_on_exception((ConnectionError,), max_attempts=3)
        def stable_function() -> str:
            return counter.call()

        result = stable_function()

        assert result == "success"
        assert counter.attempts == 1

    def test_succeeds_after_retries(self):
        """Test function that fails twice then succeeds."""
        counter = FlakyCounter(failures_before_success=2)

        @retry_on_exception((ConnectionError,), max_attempts=5, min_wait_seconds=0.01)
        def flaky_function() -> str:
            return counter.call()

        result = flaky_function()

        assert result == "success"
        assert counter.attempts == 3  # Failed 2 times, succeeded on 3rd

    def test_exhausts_retries_and_raises(self):
        """Test function that always fails exhausts retries and raises."""
        counter = FlakyCounter(failures_before_success=10)

        @retry_on_exception((ConnectionError,), max_attempts=3, min_wait_seconds=0.01)
        def always_fails() -> str:
            return counter.call()

        with pytest.raises(ConnectionError) as exc_info:
            always_fails()

        assert counter.attempts == 3  # Tried 3 times
        assert "Attempt 3 failed" in str(exc_info.value)

    def test_retries_only_on_specified_exceptions(self):
        """Test decorator only retries on specified exception types."""
        attempts = 0

        @retry_on_exception((ConnectionError,), max_attempts=3)
        def fails_with_different_error() -> str:
            nonlocal attempts
            attempts += 1
            raise ValueError("Wrong error type")

        # ValueError should not trigger retry (only ConnectionError retries)
        with pytest.raises(ValueError):
            fails_with_different_error()

        assert attempts == 1  # No retries for ValueError

    def test_retries_on_multiple_exception_types(self):
        """Test decorator retries on multiple exception types."""
        # Alternate between ConnectionError and TimeoutError
        attempt = 0

        @retry_on_exception(
            (ConnectionError, TimeoutError), max_attempts=5, min_wait_seconds=0.01
        )
        def multi_exception_flaky() -> str:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise ConnectionError("Connection failed")
            elif attempt == 2:
                raise TimeoutError("Timeout occurred")
            else:
                return "success"

        result = multi_exception_flaky()

        assert result == "success"
        assert attempt == 3


class TestWithExponentialBackoff:
    """Test with_exponential_backoff decorator."""

    def test_succeeds_without_retries(self):
        """Test function that succeeds immediately."""
        counter = FlakyCounter(failures_before_success=0)

        @with_exponential_backoff(max_attempts=3)
        def stable_function() -> str:
            return counter.call()

        result = stable_function()

        assert result == "success"
        assert counter.attempts == 1

    def test_retries_on_any_exception(self):
        """Test decorator retries on ANY exception (not just specific types)."""
        attempts = 0

        @with_exponential_backoff(max_attempts=3, min_wait_seconds=0.01)
        def fails_with_value_error() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("Some error")
            return "success"

        result = fails_with_value_error()

        assert result == "success"
        assert attempts == 3  # Failed 2 times with ValueError, then succeeded

    def test_exhausts_retries(self):
        """Test function that always fails exhausts retries."""
        attempts = 0

        @with_exponential_backoff(max_attempts=3, min_wait_seconds=0.01)
        def always_fails() -> str:
            nonlocal attempts
            attempts += 1
            raise RuntimeError(f"Fail {attempts}")

        with pytest.raises(RuntimeError) as exc_info:
            always_fails()

        assert attempts == 3
        assert "Fail 3" in str(exc_info.value)


class TestRetryConfigurationEdgeCases:
    """Test edge cases in retry configuration."""

    def test_max_attempts_one_means_no_retries(self):
        """Test max_attempts=1 means no retries (single attempt)."""
        attempts = 0

        @retry_on_exception((ConnectionError,), max_attempts=1)
        def single_attempt() -> str:
            nonlocal attempts
            attempts += 1
            raise ConnectionError("Failed")

        with pytest.raises(ConnectionError):
            single_attempt()

        assert attempts == 1  # No retries

    def test_async_function_not_tested_here(self):
        """Note: Async function retry testing requires pytest-asyncio fixture.

        The retry decorators support async functions but testing them requires
        pytest-asyncio marks. See integration tests for async retry validation.
        """
        # Placeholder test to document async support
        pass
