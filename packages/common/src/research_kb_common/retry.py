"""Retry and backoff patterns using tenacity.

Provides resilient retry strategies for:
- Network calls (GROBID API, LLM APIs)
- Database operations with transient failures
- File I/O operations
"""

from typing import Callable, Type

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def retry_on_exception(
    exception_types: tuple[Type[Exception], ...],
    max_attempts: int = 3,
    min_wait_seconds: float = 1.0,
    max_wait_seconds: float = 10.0,
) -> Callable:
    """Decorator for retrying functions that may raise specific exceptions.

    Uses exponential backoff: wait = min(max_wait, min_wait * 2^(attempt-1))

    Args:
        exception_types: Tuple of exception types to retry on
        max_attempts: Maximum number of attempts (default: 3)
        min_wait_seconds: Minimum wait time between retries (default: 1.0s)
        max_wait_seconds: Maximum wait time between retries (default: 10.0s)

    Returns:
        Decorator function

    Example:
        >>> @retry_on_exception((ConnectionError, TimeoutError), max_attempts=5)
        ... async def call_grobid_api(pdf_bytes: bytes) -> dict:
        ...     response = await client.post("/api/processFulltextDocument", data=pdf_bytes)
        ...     return response.json()
    """
    return retry(
        retry=retry_if_exception_type(exception_types),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=min_wait_seconds,
            min=min_wait_seconds,
            max=max_wait_seconds,
        ),
        reraise=True,  # Re-raise exception after exhausting retries
    )


def with_exponential_backoff(
    max_attempts: int = 3,
    min_wait_seconds: float = 1.0,
    max_wait_seconds: float = 10.0,
) -> Callable:
    """Decorator for retrying any function with exponential backoff.

    Retries on ANY exception (use with caution - prefer retry_on_exception for specific errors).

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        min_wait_seconds: Minimum wait time between retries (default: 1.0s)
        max_wait_seconds: Maximum wait time between retries (default: 10.0s)

    Returns:
        Decorator function

    Example:
        >>> @with_exponential_backoff(max_attempts=5, min_wait_seconds=2.0)
        ... def flaky_network_call() -> dict:
        ...     return requests.get("https://api.example.com/data").json()
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=min_wait_seconds,
            min=min_wait_seconds,
            max=max_wait_seconds,
        ),
        reraise=True,
    )
