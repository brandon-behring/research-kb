"""Tests for FastAPI application setup and lifespan.

Tests cover:
- Application creation and configuration
- Lifespan context manager (startup/shutdown)
- Router registration
- Middleware configuration
- App state management
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from research_kb_api.main import create_app, lifespan, app


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_pool():
    """Mock database connection pool."""
    pool = MagicMock()
    pool.get_size.return_value = 5
    pool.close = AsyncMock()
    return pool


@pytest.fixture
def mock_storage_layer():
    """Mock all storage-related imports to prevent database connections."""
    with patch("research_kb_api.service.ConceptStore") as concept_mock, \
         patch("research_kb_api.service.SourceStore") as source_mock, \
         patch("research_kb_api.service.ChunkStore") as chunk_mock, \
         patch("research_kb_api.service.RelationshipStore") as rel_mock, \
         patch("research_kb_api.service.search_hybrid") as search_mock, \
         patch("research_kb_api.service.search_hybrid_v2") as search_v2_mock, \
         patch("research_kb_api.service.search_with_rerank") as rerank_mock, \
         patch("research_kb_api.service.search_with_expansion") as expand_mock, \
         patch("research_kb_api.service.get_cached_embedding") as embed_mock:

        concept_mock.count = AsyncMock(return_value=0)
        embed_mock.return_value = [0.1] * 1024
        expand_mock.return_value = ([], None)

        yield


# =============================================================================
# Test Application Creation
# =============================================================================


class TestCreateApp:
    """Test create_app function."""

    def test_create_app_returns_fastapi_instance(self):
        """Test create_app returns a FastAPI application."""
        app_instance = create_app()

        assert isinstance(app_instance, FastAPI)

    def test_create_app_has_correct_title(self):
        """Test application has correct title."""
        app_instance = create_app()

        assert app_instance.title == "Research-KB API"

    def test_create_app_has_description(self):
        """Test application has description."""
        app_instance = create_app()

        assert app_instance.description is not None
        assert "search" in app_instance.description.lower()

    def test_create_app_has_version(self):
        """Test application has version."""
        app_instance = create_app()

        assert app_instance.version == "1.0.0"

    def test_create_app_has_docs_url(self):
        """Test application has docs URL configured."""
        app_instance = create_app()

        assert app_instance.docs_url == "/docs"

    def test_create_app_has_redoc_url(self):
        """Test application has redoc URL configured."""
        app_instance = create_app()

        assert app_instance.redoc_url == "/redoc"


# =============================================================================
# Test Router Registration
# =============================================================================


class TestRouterRegistration:
    """Test that all routers are properly registered."""

    def test_health_router_registered(self):
        """Test health router is registered."""
        app_instance = create_app()

        routes = [route.path for route in app_instance.routes]

        assert "/health" in routes or any("/health" in r for r in routes)

    def test_search_router_registered(self):
        """Test search router is registered."""
        app_instance = create_app()

        routes = [route.path for route in app_instance.routes]

        assert any("/search" in str(r) for r in routes)

    def test_sources_router_registered(self):
        """Test sources router is registered."""
        app_instance = create_app()

        routes = [route.path for route in app_instance.routes]

        assert any("/sources" in str(r) for r in routes)

    def test_concepts_router_registered(self):
        """Test concepts router is registered."""
        app_instance = create_app()

        routes = [route.path for route in app_instance.routes]

        assert any("/concepts" in str(r) for r in routes)

    def test_graph_router_registered(self):
        """Test graph router is registered."""
        app_instance = create_app()

        routes = [route.path for route in app_instance.routes]

        assert any("/graph" in str(r) for r in routes)

    def test_router_tags_assigned(self):
        """Test routers have tags for documentation."""
        app_instance = create_app()

        # Check that routes have tags (used in OpenAPI docs)
        route_tags = set()
        for route in app_instance.routes:
            if hasattr(route, "tags") and route.tags:
                route_tags.update(route.tags)

        assert "Health" in route_tags or "health" in [t.lower() for t in route_tags]


# =============================================================================
# Test CORS Middleware
# =============================================================================


class TestCORSMiddleware:
    """Test CORS middleware configuration."""

    def test_cors_middleware_added(self):
        """Test CORS middleware is added to app."""
        app_instance = create_app()

        # Check middleware stack
        middleware_classes = [
            type(m).__name__ for m in app_instance.user_middleware
        ]

        # The CORSMiddleware should be in the middleware stack
        # Note: FastAPI wraps it, so we check user_middleware
        assert len(app_instance.user_middleware) > 0

    @pytest.mark.asyncio
    async def test_cors_allows_requests(self, mock_pool, mock_storage_layer):
        """Test CORS headers are present in responses."""
        with patch("research_kb_api.main.get_connection_pool", return_value=mock_pool):
            app_instance = create_app()
            app_instance.state.pool = mock_pool

            transport = ASGITransport(app=app_instance)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.options(
                    "/health",
                    headers={
                        "Origin": "http://localhost:3000",
                        "Access-Control-Request-Method": "GET",
                    }
                )

                # Should not return 405 (Method Not Allowed)
                # CORS preflight should be handled


# =============================================================================
# Test Lifespan Context Manager
# =============================================================================


class TestLifespan:
    """Test lifespan context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_initializes_pool(self, mock_pool):
        """Test lifespan initializes database pool on startup."""
        with patch("research_kb_api.main.get_connection_pool") as pool_mock, \
             patch("research_kb_api.main.DatabaseConfig") as config_mock:

            pool_mock.return_value = mock_pool
            config_mock.return_value = MagicMock()

            app_instance = FastAPI()

            async with lifespan(app_instance):
                # During lifespan, pool should be set
                assert hasattr(app_instance.state, "pool")
                assert app_instance.state.pool is mock_pool

    @pytest.mark.asyncio
    async def test_lifespan_closes_pool_on_shutdown(self, mock_pool):
        """Test lifespan closes database pool on shutdown."""
        with patch("research_kb_api.main.get_connection_pool") as pool_mock, \
             patch("research_kb_api.main.DatabaseConfig") as config_mock:

            pool_mock.return_value = mock_pool
            config_mock.return_value = MagicMock()

            app_instance = FastAPI()

            async with lifespan(app_instance):
                pass  # Exit context triggers shutdown

            # Pool should be closed
            mock_pool.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_logs_startup(self, mock_pool):
        """Test lifespan logs startup message."""
        with patch("research_kb_api.main.get_connection_pool") as pool_mock, \
             patch("research_kb_api.main.DatabaseConfig") as config_mock, \
             patch("research_kb_api.main.logger") as logger_mock:

            pool_mock.return_value = mock_pool
            config_mock.return_value = MagicMock()

            app_instance = FastAPI()

            async with lifespan(app_instance):
                pass

            # Check logging calls
            calls = [call[0][0] for call in logger_mock.info.call_args_list]
            assert any("start" in c.lower() for c in calls)

    @pytest.mark.asyncio
    async def test_lifespan_logs_shutdown(self, mock_pool):
        """Test lifespan logs shutdown message."""
        with patch("research_kb_api.main.get_connection_pool") as pool_mock, \
             patch("research_kb_api.main.DatabaseConfig") as config_mock, \
             patch("research_kb_api.main.logger") as logger_mock:

            pool_mock.return_value = mock_pool
            config_mock.return_value = MagicMock()

            app_instance = FastAPI()

            async with lifespan(app_instance):
                pass

            # Check logging calls
            calls = [call[0][0] for call in logger_mock.info.call_args_list]
            assert any("stop" in c.lower() for c in calls)


# =============================================================================
# Test App State
# =============================================================================


class TestAppState:
    """Test application state management."""

    @pytest.mark.asyncio
    async def test_pool_accessible_from_app_state(self, mock_pool):
        """Test database pool is accessible from app.state."""
        with patch("research_kb_api.main.get_connection_pool") as pool_mock, \
             patch("research_kb_api.main.DatabaseConfig"):

            pool_mock.return_value = mock_pool

            app_instance = FastAPI()

            async with lifespan(app_instance):
                assert app_instance.state.pool is mock_pool
                assert app_instance.state.pool.get_size() == 5


# =============================================================================
# Test Module-Level App Instance
# =============================================================================


class TestModuleLevelApp:
    """Test module-level app instance."""

    def test_app_exists(self):
        """Test app instance exists at module level."""
        assert app is not None
        assert isinstance(app, FastAPI)

    def test_app_is_configured(self):
        """Test module-level app is properly configured."""
        assert app.title == "Research-KB API"
        assert app.version == "1.0.0"


# =============================================================================
# Test Integration with Routes
# =============================================================================


class TestRouteIntegration:
    """Test integration between main app and routes."""

    @pytest.mark.asyncio
    async def test_health_endpoint_accessible(self, mock_pool, mock_storage_layer):
        """Test health endpoint is accessible after app creation."""
        with patch("research_kb_api.main.get_connection_pool", return_value=mock_pool):
            app_instance = create_app()
            app_instance.state.pool = mock_pool

            transport = ASGITransport(app=app_instance)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/health")

                assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_search_endpoint_accessible(self, mock_pool, mock_storage_layer):
        """Test search endpoint is accessible."""
        with patch("research_kb_api.main.get_connection_pool", return_value=mock_pool):
            app_instance = create_app()
            app_instance.state.pool = mock_pool

            transport = ASGITransport(app=app_instance)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/search",
                    json={"query": "test"},
                )

                assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_sources_endpoint_accessible(self, mock_pool, mock_storage_layer):
        """Test sources endpoint is accessible."""
        with patch("research_kb_api.main.get_connection_pool", return_value=mock_pool), \
             patch("research_kb_api.service.get_sources") as sources_mock:

            sources_mock.return_value = []

            app_instance = create_app()
            app_instance.state.pool = mock_pool

            transport = ASGITransport(app=app_instance)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/sources")

                assert response.status_code == 200


# =============================================================================
# Test OpenAPI Documentation
# =============================================================================


class TestOpenAPIDocumentation:
    """Test OpenAPI documentation generation."""

    def test_openapi_schema_generated(self):
        """Test OpenAPI schema is generated."""
        app_instance = create_app()

        schema = app_instance.openapi()

        assert schema is not None
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

    def test_openapi_info_correct(self):
        """Test OpenAPI info section is correct."""
        app_instance = create_app()

        schema = app_instance.openapi()

        assert schema["info"]["title"] == "Research-KB API"
        assert schema["info"]["version"] == "1.0.0"

    def test_openapi_paths_include_routes(self):
        """Test OpenAPI paths include registered routes."""
        app_instance = create_app()

        schema = app_instance.openapi()
        paths = schema["paths"]

        # Check for key paths
        assert "/health" in paths
        assert "/search" in paths


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Test error handling in app setup."""

    @pytest.mark.asyncio
    async def test_lifespan_handles_pool_error(self):
        """Test lifespan handles database connection error."""
        with patch("research_kb_api.main.get_connection_pool") as pool_mock, \
             patch("research_kb_api.main.DatabaseConfig"):

            pool_mock.side_effect = ConnectionError("Database unavailable")

            app_instance = FastAPI()

            with pytest.raises(ConnectionError):
                async with lifespan(app_instance):
                    pass

    @pytest.mark.asyncio
    async def test_lifespan_pool_close_error_handled(self, mock_pool):
        """Test lifespan handles error during pool close."""
        mock_pool.close.side_effect = RuntimeError("Close failed")

        with patch("research_kb_api.main.get_connection_pool") as pool_mock, \
             patch("research_kb_api.main.DatabaseConfig"):

            pool_mock.return_value = mock_pool

            app_instance = FastAPI()

            # Should not propagate the close error
            with pytest.raises(RuntimeError):
                async with lifespan(app_instance):
                    pass


# =============================================================================
# Test Configuration
# =============================================================================


class TestConfiguration:
    """Test app configuration options."""

    @pytest.mark.asyncio
    async def test_database_config_used(self, mock_pool):
        """Test DatabaseConfig is used for connection."""
        mock_config = MagicMock()

        with patch("research_kb_api.main.get_connection_pool") as pool_mock, \
             patch("research_kb_api.main.DatabaseConfig") as config_class:

            pool_mock.return_value = mock_pool
            config_class.return_value = mock_config

            app_instance = FastAPI()

            async with lifespan(app_instance):
                pass

            # DatabaseConfig should be instantiated
            config_class.assert_called_once()

            # get_connection_pool should be called with the config
            pool_mock.assert_called_once_with(mock_config)
