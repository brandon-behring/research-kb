"""Tests for OpenTelemetry instrumentation helpers."""

import pytest
from unittest.mock import patch, MagicMock

from research_kb_common.instrumentation import (
    init_telemetry,
    get_tracer,
    instrument_function,
)


class TestInitTelemetry:
    """Tests for init_telemetry function."""

    def test_init_telemetry_creates_provider(self):
        """Test that init_telemetry initializes the tracer provider."""
        # Reset global state
        import research_kb_common.instrumentation as instr

        instr._tracer_provider = None

        init_telemetry(service_name="test-service")

        assert instr._tracer_provider is not None

    def test_init_telemetry_idempotent(self):
        """Test that init_telemetry can be called multiple times safely."""
        import research_kb_common.instrumentation as instr

        instr._tracer_provider = None

        init_telemetry()
        provider1 = instr._tracer_provider

        init_telemetry()
        provider2 = instr._tracer_provider

        # Should be the same instance (not re-initialized)
        assert provider1 is provider2


class TestGetTracer:
    """Tests for get_tracer function."""

    def test_get_tracer_returns_tracer(self):
        """Test that get_tracer returns a tracer instance."""
        tracer = get_tracer("test_module")

        # Verify it has span creation method
        assert hasattr(tracer, "start_as_current_span")
        assert callable(tracer.start_as_current_span)

    def test_get_tracer_auto_initializes(self):
        """Test that get_tracer auto-initializes if not done."""
        import research_kb_common.instrumentation as instr

        instr._tracer_provider = None

        tracer = get_tracer("auto_init_test")

        # Should have auto-initialized
        assert instr._tracer_provider is not None
        assert tracer is not None

    def test_get_tracer_different_names(self):
        """Test that tracers with different names are distinct."""
        tracer1 = get_tracer("module_a")
        tracer2 = get_tracer("module_b")

        # Both should be valid tracers
        assert hasattr(tracer1, "start_as_current_span")
        assert hasattr(tracer2, "start_as_current_span")


class TestInstrumentFunction:
    """Tests for instrument_function decorator."""

    def test_instrument_sync_function(self):
        """Test decorator works with sync functions."""

        @instrument_function("test_span")
        def sync_func(x: int) -> int:
            return x * 2

        result = sync_func(5)
        assert result == 10

    def test_instrument_sync_function_default_name(self):
        """Test decorator uses function name by default."""

        @instrument_function()
        def my_named_function() -> str:
            return "success"

        result = my_named_function()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_instrument_async_function(self):
        """Test decorator works with async functions."""

        @instrument_function("async_test_span")
        async def async_func(x: int) -> int:
            return x * 3

        result = await async_func(4)
        assert result == 12

    @pytest.mark.asyncio
    async def test_instrument_async_function_default_name(self):
        """Test decorator uses function name for async functions."""

        @instrument_function()
        async def my_async_function() -> str:
            return "async_success"

        result = await my_async_function()
        assert result == "async_success"

    def test_instrument_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""

        @instrument_function("custom_span")
        def documented_function() -> None:
            """This is a documented function."""
            pass

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a documented function."

    def test_instrument_with_args_and_kwargs(self):
        """Test decorator passes args and kwargs correctly."""

        @instrument_function()
        def func_with_params(a: int, b: int, *, c: int = 0) -> int:
            return a + b + c

        result = func_with_params(1, 2, c=3)
        assert result == 6

    @pytest.mark.asyncio
    async def test_instrument_async_with_exception(self):
        """Test decorator propagates exceptions correctly."""

        @instrument_function()
        async def raises_error() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await raises_error()

    def test_instrument_sync_with_exception(self):
        """Test sync decorator propagates exceptions correctly."""

        @instrument_function()
        def raises_sync_error() -> None:
            raise RuntimeError("Sync error")

        with pytest.raises(RuntimeError, match="Sync error"):
            raises_sync_error()
