"""Tests for concept MCP tools."""

from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock
from uuid import uuid4
from datetime import datetime

from research_kb_contracts import Concept, ConceptType, Chunk, ChunkConcept
from research_kb_mcp.tools.concepts import register_concept_tools


class MockFastMCP:
    """Mock FastMCP server for testing tool registration."""

    def __init__(self):
        self.tools = {}

    def tool(self, **kwargs):
        """Decorator that captures tool functions."""
        def decorator(func):
            self.tools[func.__name__] = {
                "func": func,
                "kwargs": kwargs,
            }
            return func
        return decorator


class TestConceptToolRegistration:
    """Tests for concept tool registration."""

    def test_list_concepts_registered(self):
        """List concepts tool is registered correctly."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        assert "research_kb_list_concepts" in mcp.tools
        doc = mcp.tools["research_kb_list_concepts"]["func"].__doc__
        assert doc is not None
        assert "concepts" in doc.lower()

    def test_get_concept_registered(self):
        """Get concept tool is registered correctly."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        assert "research_kb_get_concept" in mcp.tools
        doc = mcp.tools["research_kb_get_concept"]["func"].__doc__
        assert doc is not None
        assert "relationship" in doc.lower()

    def test_chunk_concepts_registered(self):
        """Chunk concepts tool is registered correctly."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        assert "research_kb_chunk_concepts" in mcp.tools


class TestListConcepts:
    """Tests for list concepts tool functionality."""

    @pytest.fixture
    def sample_concepts(self):
        """Create sample concepts for testing."""
        return [
            Concept(
                id=uuid4(),
                name="Double Machine Learning",
                canonical_name="double_machine_learning",
                concept_type=ConceptType.METHOD,
                created_at=datetime.now(),
            ),
            Concept(
                id=uuid4(),
                name="Unconfoundedness",
                canonical_name="unconfoundedness",
                concept_type=ConceptType.ASSUMPTION,
                created_at=datetime.now(),
            ),
        ]

    @pytest.mark.asyncio
    async def test_list_concepts_empty_query(self, sample_concepts):
        """List concepts returns all when no query provided."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        with patch("research_kb_mcp.tools.concepts.get_concepts") as get_mock:
            get_mock.return_value = sample_concepts

            result = await mcp.tools["research_kb_list_concepts"]["func"]()

            get_mock.assert_called_once_with(query=None, limit=50, concept_type=None)
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_list_concepts_with_search(self, sample_concepts):
        """List concepts filters by search query."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        with patch("research_kb_mcp.tools.concepts.get_concepts") as get_mock:
            get_mock.return_value = [sample_concepts[0]]

            await mcp.tools["research_kb_list_concepts"]["func"](
                query="machine learning",
            )

            get_mock.assert_called_once_with(
                query="machine learning",
                limit=50,
                concept_type=None,
            )

    @pytest.mark.asyncio
    async def test_list_concepts_with_type_filter(self, sample_concepts):
        """List concepts filters by type."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        with patch("research_kb_mcp.tools.concepts.get_concepts") as get_mock:
            get_mock.return_value = [sample_concepts[0]]

            await mcp.tools["research_kb_list_concepts"]["func"](
                concept_type="METHOD",
            )

            get_mock.assert_called_once_with(
                query=None,
                limit=50,
                concept_type="METHOD",
            )

    @pytest.mark.asyncio
    async def test_list_concepts_limit_clamping(self, sample_concepts):
        """List concepts clamps limit to valid range."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        with patch("research_kb_mcp.tools.concepts.get_concepts") as get_mock:
            get_mock.return_value = sample_concepts

            # Test upper bound
            await mcp.tools["research_kb_list_concepts"]["func"](limit=200)
            assert get_mock.call_args[1]["limit"] == 100

            # Test lower bound
            await mcp.tools["research_kb_list_concepts"]["func"](limit=0)
            assert get_mock.call_args[1]["limit"] == 1


class TestGetConcept:
    """Tests for get concept tool functionality."""

    @pytest.fixture
    def sample_concept(self):
        """Create a sample concept for testing."""
        return Concept(
            id=uuid4(),
            name="Instrumental Variables",
            canonical_name="instrumental_variables",
            concept_type=ConceptType.METHOD,
            definition="A method for causal inference when there is unmeasured confounding.",
            created_at=datetime.now(),
        )

    @pytest.fixture
    def sample_relationships(self):
        """Create sample relationships for testing."""
        # Create mock relationship objects with required attributes
        class MockRelationship:
            def __init__(self, rel_type, target_id):
                self.relationship_type = rel_type
                self.target_id = target_id

        return [
            MockRelationship("USES", uuid4()),
            MockRelationship("ADDRESSES", uuid4()),
        ]

    @pytest.mark.asyncio
    async def test_get_concept_success(self, sample_concept, sample_relationships):
        """Get concept returns formatted details."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        with patch("research_kb_mcp.tools.concepts.get_concept_by_id") as get_mock, \
             patch("research_kb_mcp.tools.concepts.get_concept_relationships") as rel_mock:
            get_mock.return_value = sample_concept
            rel_mock.return_value = sample_relationships

            result = await mcp.tools["research_kb_get_concept"]["func"](
                concept_id=str(sample_concept.id),
            )

            get_mock.assert_called_once()
            rel_mock.assert_called_once()
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_get_concept_without_relationships(self, sample_concept):
        """Get concept skips relationships when not requested."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        with patch("research_kb_mcp.tools.concepts.get_concept_by_id") as get_mock, \
             patch("research_kb_mcp.tools.concepts.get_concept_relationships") as rel_mock:
            get_mock.return_value = sample_concept

            await mcp.tools["research_kb_get_concept"]["func"](
                concept_id=str(sample_concept.id),
                include_relationships=False,
            )

            rel_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_concept_not_found(self):
        """Get concept returns error for missing concept."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        with patch("research_kb_mcp.tools.concepts.get_concept_by_id") as get_mock:
            get_mock.return_value = None

            result = await mcp.tools["research_kb_get_concept"]["func"](
                concept_id="nonexistent-id",
            )

            assert "Error" in result
            assert "not found" in result
