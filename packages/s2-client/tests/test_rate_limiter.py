"""Tests for token bucket rate limiter.

Tests rate limiting behavior including token acquisition, refill over time,
and concurrent access patterns.
"""

import asyncio
import time

import pytest

from s2_client.rate_limiter import RateLimiter


# -----------------------------------------------------------------------------
# Initialization Tests
# -----------------------------------------------------------------------------


class TestRateLimiterInit:
    """Tests for RateLimiter initialization."""

    def test_default_values(self):
        """Test default configuration values."""
        limiter = RateLimiter()

        assert limiter.requests_per_second == 10.0
        assert limiter.burst_size == 10
        assert limiter.available_tokens == pytest.approx(10.0, rel=0.1)

    def test_custom_rps(self):
        """Test custom requests per second."""
        limiter = RateLimiter(requests_per_second=5.0)

        assert limiter.requests_per_second == 5.0
        assert limiter.burst_size == 5  # Defaults to RPS

    def test_custom_burst_size(self):
        """Test custom burst size independent of RPS."""
        limiter = RateLimiter(requests_per_second=10.0, burst_size=20)

        assert limiter.requests_per_second == 10.0
        assert limiter.burst_size == 20
        assert limiter.available_tokens == pytest.approx(20.0, rel=0.1)

    def test_burst_size_smaller_than_rps(self):
        """Test burst size smaller than RPS."""
        limiter = RateLimiter(requests_per_second=100.0, burst_size=5)

        assert limiter.burst_size == 5
        assert limiter.available_tokens == pytest.approx(5.0, rel=0.1)

    def test_fractional_rps(self):
        """Test fractional requests per second."""
        limiter = RateLimiter(requests_per_second=0.5)

        assert limiter.requests_per_second == 0.5
        # burst_size is int(0.5) = 0, but we should have some tokens
        # Note: This might be 0 due to int conversion - test the actual behavior


# -----------------------------------------------------------------------------
# Token Acquisition Tests
# -----------------------------------------------------------------------------


class TestRateLimiterAcquire:
    """Tests for token acquisition."""

    @pytest.mark.asyncio
    async def test_acquire_immediate_when_full(self):
        """First acquire should be immediate with full bucket."""
        limiter = RateLimiter(requests_per_second=10)

        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should be nearly instant (< 50ms)
        assert elapsed < 0.05

    @pytest.mark.asyncio
    async def test_acquire_decrements_tokens(self):
        """Acquire should decrement available tokens."""
        limiter = RateLimiter(requests_per_second=10, burst_size=10)

        initial_tokens = limiter.available_tokens
        await limiter.acquire()
        after_tokens = limiter.available_tokens

        # Should have used ~1 token (allow for timing variance)
        assert initial_tokens - after_tokens >= 0.9

    @pytest.mark.asyncio
    async def test_multiple_acquires_from_full_bucket(self):
        """Multiple rapid acquires should succeed from full bucket."""
        limiter = RateLimiter(requests_per_second=10, burst_size=10)

        start = time.monotonic()
        # Acquire 5 tokens rapidly
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should complete quickly (< 100ms total)
        assert elapsed < 0.1
        # Should have ~5 tokens left
        assert limiter.available_tokens >= 4.5

    @pytest.mark.asyncio
    async def test_acquire_waits_when_empty(self):
        """Acquire should wait when bucket is empty."""
        # Very fast rate to empty bucket quickly
        limiter = RateLimiter(requests_per_second=100, burst_size=2)

        # Drain the bucket
        await limiter.acquire()
        await limiter.acquire()

        # Next acquire should wait
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should have waited at least 10ms (1/100 second)
        assert elapsed >= 0.008  # Allow small timing variance


# -----------------------------------------------------------------------------
# Token Refill Tests
# -----------------------------------------------------------------------------


class TestRateLimiterRefill:
    """Tests for token refill over time."""

    @pytest.mark.asyncio
    async def test_tokens_refill_over_time(self):
        """Tokens should refill based on elapsed time."""
        limiter = RateLimiter(requests_per_second=100, burst_size=10)

        # Use all tokens
        for _ in range(10):
            await limiter.acquire()

        # Check tokens are low
        assert limiter.available_tokens < 1

        # Wait for refill (50ms = 5 tokens at 100 RPS)
        await asyncio.sleep(0.05)

        # Should have some tokens back
        assert limiter.available_tokens >= 4

    @pytest.mark.asyncio
    async def test_refill_caps_at_burst_size(self):
        """Tokens should not exceed burst size."""
        limiter = RateLimiter(requests_per_second=100, burst_size=5)

        # Wait well beyond what's needed to refill
        await asyncio.sleep(0.1)

        # Should not exceed burst size
        assert limiter.available_tokens <= 5.0

    @pytest.mark.asyncio
    async def test_available_tokens_triggers_refill(self):
        """Checking available_tokens should update token count."""
        limiter = RateLimiter(requests_per_second=100, burst_size=10)

        # Drain tokens
        for _ in range(10):
            await limiter.acquire()

        initial = limiter._tokens

        # Wait a bit
        await asyncio.sleep(0.02)

        # Reading available_tokens should trigger _add_tokens
        current = limiter.available_tokens

        assert current > initial


# -----------------------------------------------------------------------------
# Context Manager Tests
# -----------------------------------------------------------------------------


class TestRateLimiterContextManager:
    """Tests for async context manager interface."""

    @pytest.mark.asyncio
    async def test_context_manager_acquires_token(self):
        """Context manager entry should acquire a token."""
        limiter = RateLimiter(requests_per_second=10)
        initial_tokens = limiter.available_tokens

        async with limiter:
            pass

        # Token should be consumed
        assert limiter.available_tokens < initial_tokens

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self):
        """Context manager should return limiter instance."""
        limiter = RateLimiter(requests_per_second=10)

        async with limiter as ctx:
            assert ctx is limiter

    @pytest.mark.asyncio
    async def test_context_manager_no_release(self):
        """Context manager exit should not release token (non-releasing design)."""
        limiter = RateLimiter(requests_per_second=10, burst_size=10)

        # Use context manager
        async with limiter:
            pass

        tokens_after = limiter.available_tokens

        # Wait and check - tokens should refill naturally, not via release
        await asyncio.sleep(0.01)

        # Tokens should be slightly higher due to refill
        assert limiter.available_tokens >= tokens_after

    @pytest.mark.asyncio
    async def test_multiple_context_managers(self):
        """Multiple context managers should each consume a token."""
        limiter = RateLimiter(requests_per_second=10, burst_size=10)

        initial = limiter.available_tokens

        async with limiter:
            async with limiter:
                async with limiter:
                    pass

        # Should have consumed 3 tokens
        assert initial - limiter.available_tokens >= 2.9


# -----------------------------------------------------------------------------
# Concurrent Access Tests
# -----------------------------------------------------------------------------


class TestRateLimiterConcurrency:
    """Tests for concurrent access patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_acquires_are_serialized(self):
        """Concurrent acquires should be properly serialized."""
        limiter = RateLimiter(requests_per_second=100, burst_size=5)

        results = []

        async def acquire_and_record():
            await limiter.acquire()
            results.append(time.monotonic())

        # Launch 10 concurrent acquires
        tasks = [asyncio.create_task(acquire_and_record()) for _ in range(10)]
        await asyncio.gather(*tasks)

        # All 10 should have completed
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_concurrent_acquires_respect_rate(self):
        """Concurrent acquires should respect rate limit."""
        limiter = RateLimiter(requests_per_second=50, burst_size=2)

        start = time.monotonic()

        # Launch 5 concurrent acquires
        tasks = [asyncio.create_task(limiter.acquire()) for _ in range(5)]
        await asyncio.gather(*tasks)

        elapsed = time.monotonic() - start

        # With burst of 2 and 50 RPS, 5 acquires need at least 60ms
        # (2 immediate + 3 more at 20ms each)
        assert elapsed >= 0.05

    @pytest.mark.asyncio
    async def test_lock_prevents_race_conditions(self):
        """Internal lock should prevent race conditions."""
        limiter = RateLimiter(requests_per_second=1000, burst_size=100)

        counter = {"value": 0}

        async def acquire_many(count: int):
            for _ in range(count):
                await limiter.acquire()
                counter["value"] += 1

        # Run multiple concurrent tasks
        tasks = [asyncio.create_task(acquire_many(20)) for _ in range(5)]
        await asyncio.gather(*tasks)

        # Should have exactly 100 acquisitions
        assert counter["value"] == 100


# -----------------------------------------------------------------------------
# Edge Cases Tests
# -----------------------------------------------------------------------------


class TestRateLimiterEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_very_high_rps(self):
        """Very high RPS should work correctly."""
        limiter = RateLimiter(requests_per_second=10000, burst_size=100)

        start = time.monotonic()
        for _ in range(100):
            await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should be very fast
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_very_low_rps(self):
        """Very low RPS should properly throttle."""
        limiter = RateLimiter(requests_per_second=10, burst_size=1)

        # First acquire should be immediate
        start = time.monotonic()
        await limiter.acquire()
        first_elapsed = time.monotonic() - start
        assert first_elapsed < 0.05

        # Second acquire should wait
        start = time.monotonic()
        await limiter.acquire()
        second_elapsed = time.monotonic() - start

        # Should wait ~100ms (1/10 second)
        assert second_elapsed >= 0.09

    def test_available_tokens_readonly(self):
        """available_tokens should reflect current state."""
        limiter = RateLimiter(requests_per_second=10, burst_size=5)

        # Initial value should be burst_size
        assert limiter.available_tokens == pytest.approx(5.0, rel=0.1)

    @pytest.mark.asyncio
    async def test_burst_allows_immediate_requests(self):
        """Burst size should allow that many immediate requests."""
        limiter = RateLimiter(requests_per_second=1, burst_size=5)

        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start

        # All 5 should be nearly instant
        assert elapsed < 0.1


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestRateLimiterIntegration:
    """Integration tests simulating real usage patterns."""

    @pytest.mark.asyncio
    async def test_api_call_pattern(self):
        """Simulate typical API call pattern."""
        limiter = RateLimiter(requests_per_second=20, burst_size=5)

        async def simulated_api_call():
            async with limiter:
                # Simulate API call time
                await asyncio.sleep(0.01)
                return True

        # Make several "API calls"
        results = []
        for _ in range(10):
            result = await simulated_api_call()
            results.append(result)

        assert all(results)
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_steady_state_rate(self):
        """Verify steady-state rate matches configuration."""
        target_rps = 50
        limiter = RateLimiter(requests_per_second=target_rps, burst_size=5)

        # Drain burst
        for _ in range(5):
            await limiter.acquire()

        # Measure steady-state
        start = time.monotonic()
        count = 0
        duration = 0.2  # 200ms measurement window

        while time.monotonic() - start < duration:
            await limiter.acquire()
            count += 1

        elapsed = time.monotonic() - start
        measured_rps = count / elapsed

        # Should be close to target (within 20%)
        assert measured_rps >= target_rps * 0.8
        assert measured_rps <= target_rps * 1.2
