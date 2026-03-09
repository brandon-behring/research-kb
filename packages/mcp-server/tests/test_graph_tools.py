"""Tests for graph MCP tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from research_kb_mcp.tools.graph import register_graph_tools

pytestmark = pytest.mark.unit


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


class TestGraphToolRegistration:
    """Tests for graph tool registration."""

    def test_neighborhood_registered(self):
        """Graph neighborhood tool is registered correctly."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        assert "research_kb_graph_neighborhood" in mcp.tools
        doc = mcp.tools["research_kb_graph_neighborhood"]["func"].__doc__
        assert doc is not None
        assert "neighborhood" in doc.lower()
        assert "hops" in doc.lower()

    def test_path_registered(self):
        """Graph path tool is registered correctly."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        assert "research_kb_graph_path" in mcp.tools
        doc = mcp.tools["research_kb_graph_path"]["func"].__doc__
        assert doc is not None
        assert "path" in doc.lower()


class TestGraphNeighborhood:
    """Tests for graph neighborhood tool functionality."""

    @pytest.fixture
    def sample_neighborhood(self):
        """Create sample neighborhood data for testing."""
        return {
            "center": {
                "id": str(uuid4()),
                "name": "Double Machine Learning",
                "type": "METHOD",
            },
            "concepts": [
                {
                    "id": str(uuid4()),
                    "name": "Neyman Orthogonality",
                    "type": "THEOREM",
                    "distance": 1,
                },
                {
                    "id": str(uuid4()),
                    "name": "Cross-Fitting",
                    "type": "TECHNIQUE",
                    "distance": 1,
                },
            ],
            "relationships": [
                {
                    "source": "Double Machine Learning",
                    "target": "Neyman Orthogonality",
                    "type": "USES",
                },
                {
                    "source": "Double Machine Learning",
                    "target": "Cross-Fitting",
                    "type": "USES",
                },
            ],
        }

    async def test_neighborhood_default_hops(self, sample_neighborhood):
        """Neighborhood uses default hop count."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_neighborhood") as get_mock:
            get_mock.return_value = sample_neighborhood

            result = await mcp.tools["research_kb_graph_neighborhood"]["func"](
                concept_name="Double Machine Learning",
            )

            get_mock.assert_called_once_with(
                concept_name="Double Machine Learning",
                hops=2,  # default
                limit=50,  # default
            )
            assert isinstance(result, str)

    async def test_neighborhood_custom_hops(self, sample_neighborhood):
        """Neighborhood respects custom hop count."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_neighborhood") as get_mock:
            get_mock.return_value = sample_neighborhood

            await mcp.tools["research_kb_graph_neighborhood"]["func"](
                concept_name="DML",
                hops=1,
                limit=20,
            )

            get_mock.assert_called_once_with(
                concept_name="DML",
                hops=1,
                limit=20,
            )

    async def test_neighborhood_hops_clamping(self, sample_neighborhood):
        """Neighborhood clamps hops to valid range."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_neighborhood") as get_mock:
            get_mock.return_value = sample_neighborhood

            # Test upper bound
            await mcp.tools["research_kb_graph_neighborhood"]["func"](
                concept_name="test",
                hops=10,
            )
            assert get_mock.call_args[1]["hops"] == 3  # clamped to max

            # Test lower bound
            await mcp.tools["research_kb_graph_neighborhood"]["func"](
                concept_name="test",
                hops=0,
            )
            assert get_mock.call_args[1]["hops"] == 1  # clamped to min

    async def test_neighborhood_limit_clamping(self, sample_neighborhood):
        """Neighborhood clamps limit to valid range."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_neighborhood") as get_mock:
            get_mock.return_value = sample_neighborhood

            await mcp.tools["research_kb_graph_neighborhood"]["func"](
                concept_name="test",
                limit=200,
            )
            assert get_mock.call_args[1]["limit"] == 100

    async def test_neighborhood_not_found(self):
        """Neighborhood handles concept not found."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_neighborhood") as get_mock:
            # Use correct keys: nodes and edges (not concepts and relationships)
            get_mock.return_value = {"center": None, "nodes": [], "edges": []}

            result = await mcp.tools["research_kb_graph_neighborhood"]["func"](
                concept_name="nonexistent concept xyz",
            )

            assert isinstance(result, str)


class TestGraphPath:
    """Tests for graph path tool functionality."""

    @pytest.fixture
    def sample_path(self):
        """Create sample path data for testing."""
        return {
            "found": True,
            "path": [
                {
                    "id": str(uuid4()),
                    "name": "Regression Discontinuity",
                    "type": "METHOD",
                },
                {
                    "id": str(uuid4()),
                    "name": "Local Average Treatment Effect",
                    "type": "DEFINITION",
                },
                {
                    "id": str(uuid4()),
                    "name": "Instrumental Variables",
                    "type": "METHOD",
                },
            ],
            "length": 2,
        }

    async def test_path_found(self, sample_path):
        """Path returns formatted result when path exists."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_path") as get_mock:
            get_mock.return_value = sample_path

            result = await mcp.tools["research_kb_graph_path"]["func"](
                concept_a="Regression Discontinuity",
                concept_b="Instrumental Variables",
            )

            get_mock.assert_called_once()
            call_kwargs = get_mock.call_args.kwargs
            assert call_kwargs["concept_a"] == "Regression Discontinuity"
            assert call_kwargs["concept_b"] == "Instrumental Variables"
            assert isinstance(result, str)

    async def test_path_no_connection(self):
        """Path handles no connection between concepts."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_path") as get_mock:
            get_mock.return_value = {"found": False, "path": [], "length": 0}

            result = await mcp.tools["research_kb_graph_path"]["func"](
                concept_a="Concept A",
                concept_b="Concept B",
            )

            assert isinstance(result, str)

    async def test_path_same_concept(self):
        """Path handles same concept for both endpoints."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_path") as get_mock:
            get_mock.return_value = {
                "found": True,
                "path": [{"id": str(uuid4()), "name": "DML", "type": "METHOD"}],
                "length": 0,
            }

            result = await mcp.tools["research_kb_graph_path"]["func"](
                concept_a="DML",
                concept_b="DML",
            )

            assert isinstance(result, str)


class TestCrossDomainRegistration:
    """Tests for cross-domain tool registration."""

    def test_cross_domain_registered(self):
        """Cross-domain concepts tool is registered."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        assert "research_kb_cross_domain_concepts" in mcp.tools
        doc = mcp.tools["research_kb_cross_domain_concepts"]["func"].__doc__
        assert doc is not None
        assert "cross-domain" in doc.lower() or "domain" in doc.lower()


@dataclass
class MockConcept:
    """Lightweight mock concept for cross-domain tests."""

    id: str
    name: str
    concept_type: str = "METHOD"
    domain_id: Optional[str] = None
    embedding: Optional[list] = None
    definition: Optional[str] = None


class TestCrossDomainJsonFormat:
    """Tests for cross_domain_concepts JSON output_format."""

    async def test_concept_id_json(self):
        """JSON format works with concept_id path."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        concept_id = str(uuid4())
        mock_links = [
            {
                "linked_concept_name": "Granger Causality",
                "linked_concept_id": str(uuid4()),
                "linked_domain": "time_series",
                "link_type": "ANALOGOUS",
                "confidence_score": 0.92,
            }
        ]

        with patch("research_kb_mcp.tools.graph.CrossDomainStore") as store_mock:
            store_mock.get_cross_domain_concepts = AsyncMock(return_value=mock_links)

            result = await mcp.tools["research_kb_cross_domain_concepts"]["func"](
                source_domain="causal_inference",
                target_domain="time_series",
                concept_id=concept_id,
                output_format="json",
            )

            parsed = json.loads(result)
            assert parsed["source_domain"] == "causal_inference"
            assert parsed["target_domain"] == "time_series"
            assert parsed["match_count"] == 1
            assert parsed["matches"][0]["name"] == "Granger Causality"
            assert parsed["matches"][0]["link_type"] == "ANALOGOUS"
            assert parsed["matches"][0]["confidence"] == 0.92

    async def test_concept_id_markdown_default(self):
        """Default markdown format unchanged for concept_id path."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        concept_id = str(uuid4())
        mock_links = [
            {
                "linked_concept_name": "Granger Causality",
                "linked_concept_id": str(uuid4()),
                "linked_domain": "time_series",
                "link_type": "ANALOGOUS",
                "confidence_score": 0.92,
            }
        ]

        with patch("research_kb_mcp.tools.graph.CrossDomainStore") as store_mock:
            store_mock.get_cross_domain_concepts = AsyncMock(return_value=mock_links)

            result = await mcp.tools["research_kb_cross_domain_concepts"]["func"](
                source_domain="causal_inference",
                target_domain="time_series",
                concept_id=concept_id,
            )

            assert "## Cross-Domain Concept Links" in result
            assert "Granger Causality" in result

    async def test_concept_name_existing_links_json(self):
        """JSON format works with concept_name + existing links path."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        source_id = uuid4()
        mock_concept = MockConcept(id=str(source_id), name="DAG")

        mock_links = [
            {
                "linked_concept_name": "Directed Acyclic Graph",
                "linked_concept_id": str(uuid4()),
                "linked_domain": "time_series",
                "link_type": "EQUIVALENT",
                "confidence_score": 0.97,
            }
        ]

        with (
            patch("research_kb_mcp.tools.graph.ConceptStore") as concept_mock,
            patch("research_kb_mcp.tools.graph.CrossDomainStore") as store_mock,
        ):
            concept_mock.search = AsyncMock(return_value=[mock_concept])
            store_mock.get_cross_domain_concepts = AsyncMock(return_value=mock_links)

            result = await mcp.tools["research_kb_cross_domain_concepts"]["func"](
                source_domain="causal_inference",
                target_domain="time_series",
                concept_name="DAG",
                output_format="json",
            )

            parsed = json.loads(result)
            assert parsed["source_concept"]["name"] == "DAG"
            assert parsed["matches"][0]["name"] == "Directed Acyclic Graph"
            assert parsed["matches"][0]["link_type"] == "EQUIVALENT"

    async def test_discovery_json(self):
        """JSON format works with general discovery path."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        mock_links = [
            {
                "source_name": "selection bias",
                "target_name": "sample selection",
                "similarity": 0.93,
            },
            {
                "source_name": "IV",
                "target_name": "instrument",
                "similarity": 0.88,
            },
        ]

        with patch("research_kb_mcp.tools.graph.CrossDomainStore") as store_mock:
            store_mock.discover_links = AsyncMock(return_value=mock_links)

            result = await mcp.tools["research_kb_cross_domain_concepts"]["func"](
                source_domain="causal_inference",
                target_domain="econometrics",
                output_format="json",
            )

            parsed = json.loads(result)
            assert parsed["source_domain"] == "causal_inference"
            assert parsed["target_domain"] == "econometrics"
            assert parsed["match_count"] == 2
            # Sorted by similarity desc
            assert parsed["matches"][0]["source_name"] == "selection bias"
            assert parsed["matches"][0]["link_type"] == "ANALOGOUS"
            assert parsed["matches"][1]["link_type"] == "RELATED"

    async def test_no_links_returns_markdown_message(self):
        """Empty results return markdown error message even with json format."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        concept_id = str(uuid4())

        with patch("research_kb_mcp.tools.graph.CrossDomainStore") as store_mock:
            store_mock.get_cross_domain_concepts = AsyncMock(return_value=[])

            result = await mcp.tools["research_kb_cross_domain_concepts"]["func"](
                source_domain="causal_inference",
                target_domain="time_series",
                concept_id=concept_id,
                output_format="json",
            )

            # Error messages are returned before format dispatch
            assert "No cross-domain links found" in result
