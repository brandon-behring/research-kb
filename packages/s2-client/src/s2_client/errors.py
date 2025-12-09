"""S2 Client error types.

All errors inherit from S2Error for easy catching.
Follow "fail fast" principle with explicit, actionable messages.
"""


class S2Error(Exception):
    """Base exception for all S2 client errors."""

    pass


class S2APIError(S2Error):
    """Error from Semantic Scholar API response.

    Attributes:
        status_code: HTTP status code from the API
        message: Error message from the API or generated
        endpoint: The API endpoint that was called
    """

    def __init__(self, status_code: int, message: str, endpoint: str = ""):
        self.status_code = status_code
        self.message = message
        self.endpoint = endpoint
        super().__init__(f"S2 API Error {status_code} at {endpoint}: {message}")


class S2RateLimitError(S2APIError):
    """Rate limit exceeded (HTTP 429).

    Includes retry_after hint if provided by API.
    """

    def __init__(self, retry_after: float | None = None, endpoint: str = ""):
        self.retry_after = retry_after
        message = "Rate limit exceeded"
        if retry_after:
            message += f" (retry after {retry_after}s)"
        super().__init__(429, message, endpoint)


class S2NotFoundError(S2APIError):
    """Resource not found (HTTP 404).

    Common when paper ID is invalid or paper has been removed.
    """

    def __init__(self, resource_id: str, resource_type: str = "paper"):
        self.resource_id = resource_id
        self.resource_type = resource_type
        super().__init__(404, f"{resource_type.capitalize()} not found: {resource_id}")


class S2CacheError(S2Error):
    """Error accessing the response cache."""

    pass


class S2ConfigError(S2Error):
    """Configuration error (invalid settings)."""

    pass
