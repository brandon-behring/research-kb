"""Token bucket rate limiter for S2 API.

Semantic Scholar API limits:
- Without API key: 1000 RPS shared across all unauthenticated users
- With API key: 1 RPS guaranteed (but can burst higher)

We default to conservative 10 RPS to be a good citizen.
"""

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class RateLimiter:
    """Token bucket rate limiter for async operations.

    Implements a simple token bucket algorithm:
    - Bucket fills at `requests_per_second` rate
    - Maximum bucket size is `burst_size`
    - Each request consumes one token
    - If no tokens available, wait until one is available

    Example:
        >>> limiter = RateLimiter(requests_per_second=10)
        >>> async with limiter:
        ...     await make_api_call()

    Attributes:
        requests_per_second: Rate at which tokens are added (default: 10)
        burst_size: Maximum tokens in bucket (default: same as RPS)
    """

    requests_per_second: float = 10.0
    burst_size: int | None = None
    _tokens: float = field(init=False, default=0.0)
    _last_update: float = field(init=False, default_factory=time.monotonic)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        """Initialize token bucket."""
        if self.burst_size is None:
            self.burst_size = int(self.requests_per_second)
        self._tokens = float(self.burst_size)

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary.

        This method is thread-safe and will block until a token is available.
        """
        async with self._lock:
            await self._wait_for_token()
            self._tokens -= 1

    async def _wait_for_token(self) -> None:
        """Wait until at least one token is available."""
        while True:
            self._add_tokens()
            if self._tokens >= 1:
                return

            # Calculate wait time for next token
            tokens_needed = 1 - self._tokens
            wait_time = tokens_needed / self.requests_per_second
            await asyncio.sleep(wait_time)

    def _add_tokens(self) -> None:
        """Add tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now

        # Add tokens proportional to elapsed time
        new_tokens = elapsed * self.requests_per_second
        self._tokens = min(self._tokens + new_tokens, float(self.burst_size or 10))

    async def __aenter__(self) -> "RateLimiter":
        """Async context manager entry - acquires token."""
        await self.acquire()
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        pass

    @property
    def available_tokens(self) -> float:
        """Current number of available tokens (for monitoring)."""
        self._add_tokens()
        return self._tokens
