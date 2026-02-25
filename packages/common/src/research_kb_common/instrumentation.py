"""OpenTelemetry instrumentation helpers.

Provides:
- Tracer access for spans
- Function decorators for automatic span creation
- Golden signals tracking (latency, errors, requests)
"""

from functools import wraps
from typing import Any, Callable

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# Global tracer provider (initialized once)
_tracer_provider: TracerProvider | None = None


def init_telemetry(service_name: str = "research-kb") -> None:
    """Initialize OpenTelemetry tracing.

    Call this once at application startup.

    Args:
        service_name: Name of the service for traces (default: "research-kb")

    Example:
        >>> from research_kb_common import init_telemetry
        >>> init_telemetry(service_name="research-kb-ingestion")
    """
    global _tracer_provider

    if _tracer_provider is not None:
        return  # Already initialized

    # Create tracer provider
    _tracer_provider = TracerProvider()

    # Add console exporter for development (replace with OTLP exporter for production)
    console_exporter = ConsoleSpanExporter()
    span_processor = BatchSpanProcessor(console_exporter)
    _tracer_provider.add_span_processor(span_processor)

    # Set as global tracer provider
    trace.set_tracer_provider(_tracer_provider)


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer for creating spans.

    Args:
        name: Tracer name (typically module name like "research_kb.storage")

    Returns:
        Tracer instance

    Example:
        >>> tracer = get_tracer("research_kb.storage")
        >>> with tracer.start_as_current_span("insert_chunk"):
        ...     # ... database operation
        ...     pass
    """
    if _tracer_provider is None:
        init_telemetry()  # Auto-initialize if not done

    return trace.get_tracer(name)


def instrument_function(span_name: str | None = None) -> Callable:
    """Decorator to automatically create a span for a function.

    Args:
        span_name: Name for the span (default: function name)

    Returns:
        Decorator function

    Example:
        >>> from research_kb_common import instrument_function
        >>>
        >>> @instrument_function("ingest_pdf")
        ... async def ingest_source(file_path: str) -> Source:
        ...     # Function automatically wrapped in a span
        ...     source = await process_pdf(file_path)
        ...     return source
    """

    def decorator(func: Callable) -> Callable:
        actual_span_name = span_name or func.__name__
        tracer = get_tracer(func.__module__)

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(actual_span_name):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(actual_span_name):
                return func(*args, **kwargs)

        # Return appropriate wrapper based on whether function is async
        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
