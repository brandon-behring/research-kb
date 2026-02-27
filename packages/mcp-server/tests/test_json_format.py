"""Tests for JSON output_format across all 7 MCP tools.

Phase Z: Validates that output_format="json" returns valid, structured JSON
for programmatic consumers (research-agent), while output_format="markdown"
(default) produces unchanged backward-compatible output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from research_kb_contracts import (
    Concept,
    ConceptRelationship,
    ConceptType,
    RelationshipType,
    Source,
    SourceType,
)
from research_kb_mcp.tools.search import register_search_tools
from research_kb_mcp.tools.concepts import register_concept_tools
from research_kb_mcp.tools.citations import register_citation_tools
from research_kb_mcp.tools.graph import register_graph_tools
from research_kb_mcp.tools.assumptions import register_assumption_tools

pytestmark = pytest.mark.unit


# ─── Shared Test Infrastructure ──────────────────────────────────────────────


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


@dataclass
class MockScoreBreakdown:
    """Mock score breakdown matching service.ScoreBreakdown."""

    fts: float = 0.756
    vector: float = 0.891
    graph: float = 0.823
    citation: float = 0.0
    combined: float = 0.892


@dataclass
class MockSourceSummary:
    """Mock source summary matching service.SourceSummary."""

    id: str = ""
    title: str = "Instrumental Variables Methods"
    authors: list = field(default_factory=lambda: ["Angrist, J.", "Imbens, G."])
    year: Optional[int] = 1995
    source_type: Optional[str] = "paper"


@dataclass
class MockChunkSummary:
    """Mock chunk summary matching service.ChunkSummary."""

    id: str = ""
    content: str = "Instrumental variables provide a way to estimate causal effects."
    page_start: Optional[int] = 10
    page_end: Optional[int] = 10
    section: Optional[str] = "Introduction"


@dataclass
class MockSearchResultDetail:
    """Mock search result detail matching service.SearchResultDetail."""

    source: MockSourceSummary = field(default_factory=MockSourceSummary)
    chunk: MockChunkSummary = field(default_factory=MockChunkSummary)
    concepts: list = field(default_factory=list)
    scores: MockScoreBreakdown = field(default_factory=MockScoreBreakdown)
    combined_score: float = 0.892


@dataclass
class MockSearchResponse:
    """Mock search response matching service.SearchResponse."""

    query: str = "instrumental variables"
    expanded_query: Optional[str] = "instrumental variables OR IV OR 2SLS"
    results: list = field(default_factory=list)
    execution_time_ms: float = 450.2
    embedding_time_ms: float = 50.0
    search_time_ms: float = 400.2


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def source_id():
    return uuid4()


@pytest.fixture
def chunk_id():
    return uuid4()


@pytest.fixture
def search_response(source_id, chunk_id):
    """Search response with one result."""
    result = MockSearchResultDetail(
        source=MockSourceSummary(id=str(source_id)),
        chunk=MockChunkSummary(id=str(chunk_id)),
    )
    return MockSearchResponse(results=[result])


@pytest.fixture
def empty_search_response():
    """Search response with no results."""
    return MockSearchResponse(query="nonexistent topic", results=[], expanded_query=None)


@pytest.fixture
def sample_source(source_id):
    """Source object for citation tests."""
    return Source(
        id=source_id,
        title="Double Machine Learning for Treatment Effects",
        source_type=SourceType.PAPER,
        authors=["Chernozhukov, V.", "Chetverikov, D."],
        year=2018,
        domain_id="causal_inference",
        file_hash="abc123",
        metadata={},
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def citing_sources():
    """Sources that cite the sample source."""
    return [
        Source(
            id=uuid4(),
            title="Causal Forest Applications",
            source_type=SourceType.PAPER,
            authors=["Wager, S."],
            year=2019,
            domain_id="causal_inference",
            file_hash="def456",
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    ]


@pytest.fixture
def cited_sources():
    """Sources cited by the sample source."""
    return [
        Source(
            id=uuid4(),
            title="Rubin Causal Model",
            source_type=SourceType.PAPER,
            authors=["Rubin, D."],
            year=1974,
            domain_id="causal_inference",
            file_hash="ghi789",
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    ]


@pytest.fixture
def similar_biblio():
    """Bibliographic coupling results."""
    return [
        {
            "source_id": uuid4(),
            "title": "Similar Paper 1",
            "authors": ["Other, B."],
            "year": 2021,
            "shared_references": 5,
            "coupling_strength": 0.45,
        },
    ]


@pytest.fixture
def sample_concept():
    """Concept for detail tests."""
    return Concept(
        id=uuid4(),
        name="Double Machine Learning",
        canonical_name="double_machine_learning",
        concept_type=ConceptType.METHOD,
        definition="A framework for valid inference on low-dimensional parameters.",
        domain_id="causal_inference",
        created_at=datetime.now(),
    )


@pytest.fixture
def sample_relationships(sample_concept):
    """Relationships for concept detail tests."""
    target_id = uuid4()
    return [
        ConceptRelationship(
            id=uuid4(),
            source_concept_id=sample_concept.id,
            target_concept_id=target_id,
            relationship_type=RelationshipType.REQUIRES,
            created_at=datetime.now(),
        ),
    ]


@pytest.fixture
def neighborhood_data():
    """Graph neighborhood dict."""
    center_id = str(uuid4())
    node_id = str(uuid4())
    return {
        "center": {"id": center_id, "name": "double machine learning", "type": "METHOD"},
        "nodes": [{"id": node_id, "name": "Neyman orthogonality", "type": "THEOREM"}],
        "edges": [
            {"type": "REQUIRES", "source": center_id, "target": node_id},
        ],
    }


# ─── Search Tools JSON Tests ────────────────────────────────────────────────


class TestSearchJsonFormat:
    """Tests for search tools with output_format='json'."""

    async def test_search_json_returns_valid_json(self, search_response):
        """research_kb_search with output_format='json' returns parseable JSON."""
        mcp = MockFastMCP()
        register_search_tools(mcp)

        with patch("research_kb_mcp.tools.search.search") as search_mock:
            search_mock.return_value = search_response

            result = await mcp.tools["research_kb_search"]["func"](
                query="instrumental variables",
                output_format="json",
            )

            data = json.loads(result)
            assert data["query"] == "instrumental variables"
            assert data["expanded_query"] == "instrumental variables OR IV OR 2SLS"
            assert isinstance(data["execution_time_ms"], float)
            assert data["result_count"] == 1
            assert isinstance(data["results"], list)

    async def test_search_json_result_structure(self, search_response, source_id, chunk_id):
        """Search JSON result has correct fields and types."""
        mcp = MockFastMCP()
        register_search_tools(mcp)

        with patch("research_kb_mcp.tools.search.search") as search_mock:
            search_mock.return_value = search_response

            result = await mcp.tools["research_kb_search"]["func"](
                query="IV",
                output_format="json",
            )

            data = json.loads(result)
            r = data["results"][0]

            assert r["rank"] == 1
            assert r["title"] == "Instrumental Variables Methods"
            assert isinstance(r["authors"], list)
            assert r["year"] == 1995
            assert r["source_type"] == "paper"
            assert r["source_id"] == str(source_id)
            assert r["chunk_id"] == str(chunk_id)
            assert r["page_start"] == 10
            assert r["section"] == "Introduction"
            assert isinstance(r["content"], str)

            # Score breakdown
            scores = r["scores"]
            assert isinstance(scores["combined"], float)
            assert isinstance(scores["fts"], float)
            assert isinstance(scores["vector"], float)
            assert isinstance(scores["graph"], float)
            assert isinstance(scores["citation"], float)

    async def test_search_json_empty_results(self, empty_search_response):
        """Search JSON with empty results returns valid JSON with empty array."""
        mcp = MockFastMCP()
        register_search_tools(mcp)

        with patch("research_kb_mcp.tools.search.search") as search_mock:
            search_mock.return_value = empty_search_response

            result = await mcp.tools["research_kb_search"]["func"](
                query="nonexistent",
                output_format="json",
            )

            data = json.loads(result)
            assert data["result_count"] == 0
            assert data["results"] == []
            assert data["expanded_query"] is None

    async def test_search_markdown_default_unchanged(self, search_response):
        """Default output_format='markdown' still calls markdown formatter."""
        mcp = MockFastMCP()
        register_search_tools(mcp)

        with (
            patch("research_kb_mcp.tools.search.search") as search_mock,
            patch("research_kb_mcp.tools.search.format_search_results") as fmt_mock,
        ):
            search_mock.return_value = search_response
            fmt_mock.return_value = "## Search Results\n\nFormatted"

            result = await mcp.tools["research_kb_search"]["func"](
                query="IV",
            )

            fmt_mock.assert_called_once()
            assert "Search Results" in result

    async def test_fast_search_json_returns_valid_json(self, search_response):
        """research_kb_fast_search with output_format='json' returns parseable JSON."""
        mcp = MockFastMCP()
        register_search_tools(mcp)

        with patch("research_kb_mcp.tools.search.search") as search_mock:
            search_mock.return_value = search_response

            result = await mcp.tools["research_kb_fast_search"]["func"](
                query="IV",
                output_format="json",
            )

            data = json.loads(result)
            assert data["query"] == "instrumental variables"
            assert isinstance(data["results"], list)
            assert data["result_count"] == 1

    async def test_fast_search_markdown_default(self, search_response):
        """fast_search defaults to markdown format."""
        mcp = MockFastMCP()
        register_search_tools(mcp)

        with (
            patch("research_kb_mcp.tools.search.search") as search_mock,
            patch("research_kb_mcp.tools.search.format_search_results") as fmt_mock,
        ):
            search_mock.return_value = search_response
            fmt_mock.return_value = "## Results"

            result = await mcp.tools["research_kb_fast_search"]["func"](
                query="IV",
            )

            fmt_mock.assert_called_once()


# ─── Concept Detail JSON Tests ───────────────────────────────────────────────


class TestConceptJsonFormat:
    """Tests for concept detail with output_format='json'."""

    async def test_concept_json_returns_valid_json(self, sample_concept, sample_relationships):
        """research_kb_get_concept with output_format='json' returns parseable JSON."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        with (
            patch("research_kb_mcp.tools.concepts.get_concept_by_id") as get_mock,
            patch("research_kb_mcp.tools.concepts.get_concept_relationships") as rel_mock,
        ):
            get_mock.return_value = sample_concept
            rel_mock.return_value = sample_relationships

            result = await mcp.tools["research_kb_get_concept"]["func"](
                concept_id=str(sample_concept.id),
                output_format="json",
            )

            data = json.loads(result)
            assert data["concept_id"] == str(sample_concept.id)
            assert data["name"] == "Double Machine Learning"
            assert data["concept_type"] == ConceptType.METHOD.value
            assert data["definition"] is not None
            assert isinstance(data["relationships"], list)
            assert len(data["relationships"]) == 1

    async def test_concept_json_relationship_structure(self, sample_concept, sample_relationships):
        """Concept JSON relationships have correct fields."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        with (
            patch("research_kb_mcp.tools.concepts.get_concept_by_id") as get_mock,
            patch("research_kb_mcp.tools.concepts.get_concept_relationships") as rel_mock,
        ):
            get_mock.return_value = sample_concept
            rel_mock.return_value = sample_relationships

            result = await mcp.tools["research_kb_get_concept"]["func"](
                concept_id=str(sample_concept.id),
                output_format="json",
            )

            data = json.loads(result)
            rel = data["relationships"][0]
            assert rel["type"] == "REQUIRES"
            assert isinstance(rel["target_id"], str)

    async def test_concept_json_no_relationships(self, sample_concept):
        """Concept JSON with no relationships returns empty array."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        with (patch("research_kb_mcp.tools.concepts.get_concept_by_id") as get_mock,):
            get_mock.return_value = sample_concept

            result = await mcp.tools["research_kb_get_concept"]["func"](
                concept_id=str(sample_concept.id),
                include_relationships=False,
                output_format="json",
            )

            data = json.loads(result)
            assert data["relationships"] == []

    async def test_concept_markdown_default(self, sample_concept):
        """Concept defaults to markdown format."""
        mcp = MockFastMCP()
        register_concept_tools(mcp)

        with (
            patch("research_kb_mcp.tools.concepts.get_concept_by_id") as get_mock,
            patch("research_kb_mcp.tools.concepts.get_concept_relationships") as rel_mock,
        ):
            get_mock.return_value = sample_concept
            rel_mock.return_value = []

            result = await mcp.tools["research_kb_get_concept"]["func"](
                concept_id=str(sample_concept.id),
            )

            # Markdown should contain ##, not be valid JSON
            assert "##" in result
            with pytest.raises(json.JSONDecodeError):
                json.loads(result)


# ─── Citation Network JSON Tests ─────────────────────────────────────────────


class TestCitationNetworkJsonFormat:
    """Tests for citation network with output_format='json'."""

    async def test_citation_network_json_valid(self, sample_source, citing_sources, cited_sources):
        """citation_network with output_format='json' returns parseable JSON."""
        mcp = MockFastMCP()
        register_citation_tools(mcp)

        with (
            patch("research_kb_mcp.tools.citations.get_source_by_id") as src_mock,
            patch("research_kb_mcp.tools.citations.get_citing_sources") as citing_mock,
            patch("research_kb_mcp.tools.citations.get_cited_sources") as cited_mock,
        ):
            src_mock.return_value = sample_source
            citing_mock.return_value = citing_sources
            cited_mock.return_value = cited_sources

            result = await mcp.tools["research_kb_citation_network"]["func"](
                source_id=str(sample_source.id),
                output_format="json",
            )

            data = json.loads(result)
            assert data["source_id"] == str(sample_source.id)
            assert data["source_title"] == sample_source.title
            assert isinstance(data["citing"], list)
            assert isinstance(data["cited_by"], list)
            assert len(data["citing"]) == 1
            assert len(data["cited_by"]) == 1

    async def test_citation_network_json_source_structure(
        self, sample_source, citing_sources, cited_sources
    ):
        """Citation JSON entries have correct fields."""
        mcp = MockFastMCP()
        register_citation_tools(mcp)

        with (
            patch("research_kb_mcp.tools.citations.get_source_by_id") as src_mock,
            patch("research_kb_mcp.tools.citations.get_citing_sources") as citing_mock,
            patch("research_kb_mcp.tools.citations.get_cited_sources") as cited_mock,
        ):
            src_mock.return_value = sample_source
            citing_mock.return_value = citing_sources
            cited_mock.return_value = cited_sources

            result = await mcp.tools["research_kb_citation_network"]["func"](
                source_id=str(sample_source.id),
                output_format="json",
            )

            data = json.loads(result)
            citing = data["citing"][0]
            assert "source_id" in citing
            assert "title" in citing
            assert "year" in citing
            assert isinstance(citing["authors"], list)

            cited = data["cited_by"][0]
            assert cited["year"] == 1974

    async def test_citation_network_json_empty(self, sample_source):
        """Citation network JSON with no citations returns empty arrays."""
        mcp = MockFastMCP()
        register_citation_tools(mcp)

        with (
            patch("research_kb_mcp.tools.citations.get_source_by_id") as src_mock,
            patch("research_kb_mcp.tools.citations.get_citing_sources") as citing_mock,
            patch("research_kb_mcp.tools.citations.get_cited_sources") as cited_mock,
        ):
            src_mock.return_value = sample_source
            citing_mock.return_value = []
            cited_mock.return_value = []

            result = await mcp.tools["research_kb_citation_network"]["func"](
                source_id=str(sample_source.id),
                output_format="json",
            )

            data = json.loads(result)
            assert data["citing"] == []
            assert data["cited_by"] == []

    async def test_citation_network_markdown_default(
        self, sample_source, citing_sources, cited_sources
    ):
        """Citation network defaults to markdown."""
        mcp = MockFastMCP()
        register_citation_tools(mcp)

        with (
            patch("research_kb_mcp.tools.citations.get_source_by_id") as src_mock,
            patch("research_kb_mcp.tools.citations.get_citing_sources") as citing_mock,
            patch("research_kb_mcp.tools.citations.get_cited_sources") as cited_mock,
        ):
            src_mock.return_value = sample_source
            citing_mock.return_value = citing_sources
            cited_mock.return_value = cited_sources

            result = await mcp.tools["research_kb_citation_network"]["func"](
                source_id=str(sample_source.id),
            )

            assert "## Citation Network" in result


# ─── Biblio Coupling JSON Tests ──────────────────────────────────────────────


class TestBiblioCouplingJsonFormat:
    """Tests for bibliographic coupling with output_format='json'."""

    async def test_biblio_json_valid(self, sample_source, similar_biblio):
        """biblio_coupling with output_format='json' returns parseable JSON."""
        mcp = MockFastMCP()
        register_citation_tools(mcp)

        with (
            patch("research_kb_mcp.tools.citations.get_source_by_id") as src_mock,
            patch("research_kb_mcp.tools.citations.BiblioStore") as biblio_mock,
        ):
            src_mock.return_value = sample_source
            biblio_mock.get_similar_sources = AsyncMock(return_value=similar_biblio)

            result = await mcp.tools["research_kb_biblio_coupling"]["func"](
                source_id=str(sample_source.id),
                output_format="json",
            )

            data = json.loads(result)
            assert data["source_id"] == str(sample_source.id)
            assert data["source_title"] == sample_source.title
            assert isinstance(data["similar"], list)
            assert len(data["similar"]) == 1

    async def test_biblio_json_result_structure(self, sample_source, similar_biblio):
        """Biblio JSON entries have correct fields and types."""
        mcp = MockFastMCP()
        register_citation_tools(mcp)

        with (
            patch("research_kb_mcp.tools.citations.get_source_by_id") as src_mock,
            patch("research_kb_mcp.tools.citations.BiblioStore") as biblio_mock,
        ):
            src_mock.return_value = sample_source
            biblio_mock.get_similar_sources = AsyncMock(return_value=similar_biblio)

            result = await mcp.tools["research_kb_biblio_coupling"]["func"](
                source_id=str(sample_source.id),
                output_format="json",
            )

            data = json.loads(result)
            s = data["similar"][0]
            assert isinstance(s["source_id"], str)
            assert s["title"] == "Similar Paper 1"
            assert isinstance(s["authors"], list)
            assert s["year"] == 2021
            assert isinstance(s["coupling_strength"], float)
            assert s["coupling_strength"] == 0.45
            assert isinstance(s["shared_references"], int)
            assert s["shared_references"] == 5

    async def test_biblio_json_empty(self, sample_source):
        """Biblio coupling JSON with no results returns empty array."""
        mcp = MockFastMCP()
        register_citation_tools(mcp)

        with (
            patch("research_kb_mcp.tools.citations.get_source_by_id") as src_mock,
            patch("research_kb_mcp.tools.citations.BiblioStore") as biblio_mock,
        ):
            src_mock.return_value = sample_source
            biblio_mock.get_similar_sources = AsyncMock(return_value=[])

            result = await mcp.tools["research_kb_biblio_coupling"]["func"](
                source_id=str(sample_source.id),
                output_format="json",
            )

            data = json.loads(result)
            assert data["similar"] == []

    async def test_biblio_markdown_default(self, sample_source, similar_biblio):
        """Biblio coupling defaults to markdown."""
        mcp = MockFastMCP()
        register_citation_tools(mcp)

        with (
            patch("research_kb_mcp.tools.citations.get_source_by_id") as src_mock,
            patch("research_kb_mcp.tools.citations.BiblioStore") as biblio_mock,
        ):
            src_mock.return_value = sample_source
            biblio_mock.get_similar_sources = AsyncMock(return_value=similar_biblio)

            result = await mcp.tools["research_kb_biblio_coupling"]["func"](
                source_id=str(sample_source.id),
            )

            assert "Bibliographically Similar" in result


# ─── Graph Neighborhood JSON Tests ───────────────────────────────────────────


class TestGraphNeighborhoodJsonFormat:
    """Tests for graph neighborhood with output_format='json'."""

    async def test_neighborhood_json_valid(self, neighborhood_data):
        """graph_neighborhood with output_format='json' returns parseable JSON."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_neighborhood") as graph_mock:
            graph_mock.return_value = neighborhood_data

            result = await mcp.tools["research_kb_graph_neighborhood"]["func"](
                concept_name="double machine learning",
                output_format="json",
            )

            data = json.loads(result)
            assert data["center"]["name"] == "double machine learning"
            assert data["center"]["type"] == "METHOD"
            assert isinstance(data["nodes"], list)
            assert isinstance(data["edges"], list)
            assert isinstance(data["relationship_type_counts"], dict)

    async def test_neighborhood_json_node_structure(self, neighborhood_data):
        """Neighborhood JSON nodes have correct fields."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_neighborhood") as graph_mock:
            graph_mock.return_value = neighborhood_data

            result = await mcp.tools["research_kb_graph_neighborhood"]["func"](
                concept_name="DML",
                output_format="json",
            )

            data = json.loads(result)
            node = data["nodes"][0]
            assert "id" in node
            assert node["name"] == "Neyman orthogonality"
            assert node["type"] == "THEOREM"

    async def test_neighborhood_json_edge_structure(self, neighborhood_data):
        """Neighborhood JSON edges have correct fields."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_neighborhood") as graph_mock:
            graph_mock.return_value = neighborhood_data

            result = await mcp.tools["research_kb_graph_neighborhood"]["func"](
                concept_name="DML",
                output_format="json",
            )

            data = json.loads(result)
            edge = data["edges"][0]
            assert edge["type"] == "REQUIRES"
            assert "source_id" in edge
            assert "target_id" in edge

    async def test_neighborhood_json_type_counts(self, neighborhood_data):
        """Neighborhood JSON relationship_type_counts is correct."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_neighborhood") as graph_mock:
            graph_mock.return_value = neighborhood_data

            result = await mcp.tools["research_kb_graph_neighborhood"]["func"](
                concept_name="DML",
                output_format="json",
            )

            data = json.loads(result)
            assert data["relationship_type_counts"]["REQUIRES"] == 1

    async def test_neighborhood_json_error(self):
        """Neighborhood JSON error returns valid JSON with error field."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_neighborhood") as graph_mock:
            graph_mock.return_value = {"error": "Concept not found"}

            result = await mcp.tools["research_kb_graph_neighborhood"]["func"](
                concept_name="nonexistent_concept",
                output_format="json",
            )

            data = json.loads(result)
            assert "error" in data

    async def test_neighborhood_json_empty_center(self):
        """Neighborhood JSON with empty center returns error."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_neighborhood") as graph_mock:
            graph_mock.return_value = {"center": None, "nodes": [], "edges": []}

            result = await mcp.tools["research_kb_graph_neighborhood"]["func"](
                concept_name="unknown",
                output_format="json",
            )

            data = json.loads(result)
            assert data["error"] == "Concept not found"

    async def test_neighborhood_markdown_default(self, neighborhood_data):
        """Neighborhood defaults to markdown."""
        mcp = MockFastMCP()
        register_graph_tools(mcp)

        with patch("research_kb_mcp.tools.graph.get_graph_neighborhood") as graph_mock:
            graph_mock.return_value = neighborhood_data

            result = await mcp.tools["research_kb_graph_neighborhood"]["func"](
                concept_name="DML",
            )

            assert "## Graph Neighborhood" in result


# ─── Assumption Audit JSON Tests ─────────────────────────────────────────────


class TestAssumptionAuditJsonFormat:
    """Tests for assumption audit with output_format='json'."""

    @pytest.fixture
    def mock_audit_result(self):
        """Create mock MethodAssumptions using actual dataclass."""
        from research_kb_storage.assumption_audit import (
            MethodAssumptions,
            AssumptionDetail,
        )

        method_id = uuid4()
        concept_id = uuid4()
        return MethodAssumptions(
            method="instrumental variables",
            method_id=method_id,
            method_aliases=["IV", "2SLS", "TSLS"],
            definition="A method for estimating causal effects when there is unmeasured confounding.",
            assumptions=[
                AssumptionDetail(
                    name="Relevance",
                    concept_id=concept_id,
                    formal_statement="Cov(Z, X) != 0",
                    plain_english="The instrument must be correlated with the endogenous variable",
                    importance="critical",
                    violation_consequence="Weak instrument bias toward OLS estimates",
                    verification_approaches=["First-stage F-statistic > 10"],
                    source_citation="Stock & Yogo (2005)",
                    relationship_type="REQUIRES",
                    confidence=0.95,
                ),
                AssumptionDetail(
                    name="Exclusion",
                    concept_id=None,
                    formal_statement="Cov(Z, u) = 0",
                    plain_english="The instrument affects outcome only through the endogenous variable",
                    importance="critical",
                    violation_consequence="Biased IV estimates",
                    verification_approaches=["Economic reasoning", "Overidentification test"],
                    source_citation="Angrist & Imbens (1994)",
                    relationship_type="REQUIRES",
                    confidence=0.95,
                ),
            ],
            source="graph+anthropic",
            code_docstring_snippet="Assumptions:\n    [CRITICAL] - relevance: Instrument correlated with endogenous var",
        )

    async def test_assumption_json_valid(self, mock_audit_result):
        """audit_assumptions with output_format='json' returns parseable JSON."""
        mcp = MockFastMCP()
        register_assumption_tools(mcp)

        with patch("research_kb_mcp.tools.assumptions.MethodAssumptionAuditor") as auditor_mock:
            auditor_mock.audit_assumptions = AsyncMock(return_value=mock_audit_result)

            result = await mcp.tools["research_kb_audit_assumptions"]["func"](
                method_name="instrumental variables",
                output_format="json",
            )

            data = json.loads(result)
            assert data["method"] == "instrumental variables"
            assert isinstance(data["method_aliases"], list)
            assert "IV" in data["method_aliases"]
            assert data["definition"] is not None
            assert data["source"] == "graph+anthropic"

    async def test_assumption_json_assumptions_structure(self, mock_audit_result):
        """Assumption audit JSON assumptions array has correct fields."""
        mcp = MockFastMCP()
        register_assumption_tools(mcp)

        with patch("research_kb_mcp.tools.assumptions.MethodAssumptionAuditor") as auditor_mock:
            auditor_mock.audit_assumptions = AsyncMock(return_value=mock_audit_result)

            result = await mcp.tools["research_kb_audit_assumptions"]["func"](
                method_name="IV",
                output_format="json",
            )

            data = json.loads(result)
            assert len(data["assumptions"]) == 2

            a = data["assumptions"][0]
            assert a["name"] == "Relevance"
            assert isinstance(a["concept_id"], str)  # UUID stringified
            assert a["formal_statement"] == "Cov(Z, X) != 0"
            assert a["plain_english"] is not None
            assert a["importance"] == "critical"
            assert a["violation_consequence"] is not None
            assert isinstance(a["verification_approaches"], list)
            assert a["source_citation"] == "Stock & Yogo (2005)"
            assert a["relationship_type"] == "REQUIRES"
            assert isinstance(a["confidence"], float)

    async def test_assumption_json_null_concept_id(self, mock_audit_result):
        """Assumption with no concept_id serializes as null."""
        mcp = MockFastMCP()
        register_assumption_tools(mcp)

        with patch("research_kb_mcp.tools.assumptions.MethodAssumptionAuditor") as auditor_mock:
            auditor_mock.audit_assumptions = AsyncMock(return_value=mock_audit_result)

            result = await mcp.tools["research_kb_audit_assumptions"]["func"](
                method_name="IV",
                output_format="json",
            )

            data = json.loads(result)
            # Second assumption has concept_id=None
            assert data["assumptions"][1]["concept_id"] is None

    async def test_assumption_json_empty_assumptions(self):
        """Assumption audit JSON with no assumptions returns empty array."""
        from research_kb_storage.assumption_audit import MethodAssumptions

        empty_result = MethodAssumptions(
            method="unknown_method",
            assumptions=[],
            source="not_found",
        )

        mcp = MockFastMCP()
        register_assumption_tools(mcp)

        with patch("research_kb_mcp.tools.assumptions.MethodAssumptionAuditor") as auditor_mock:
            auditor_mock.audit_assumptions = AsyncMock(return_value=empty_result)

            result = await mcp.tools["research_kb_audit_assumptions"]["func"](
                method_name="unknown_method",
                output_format="json",
            )

            data = json.loads(result)
            assert data["assumptions"] == []
            assert data["source"] == "not_found"

    async def test_assumption_json_has_docstring_snippet(self, mock_audit_result):
        """Assumption audit JSON includes code_docstring_snippet."""
        mcp = MockFastMCP()
        register_assumption_tools(mcp)

        with patch("research_kb_mcp.tools.assumptions.MethodAssumptionAuditor") as auditor_mock:
            auditor_mock.audit_assumptions = AsyncMock(return_value=mock_audit_result)

            result = await mcp.tools["research_kb_audit_assumptions"]["func"](
                method_name="IV",
                output_format="json",
            )

            data = json.loads(result)
            assert data["code_docstring_snippet"] is not None
            assert "Assumptions:" in data["code_docstring_snippet"]

    async def test_assumption_domain_param_passed_through(self, mock_audit_result):
        """MCP tool passes domain to auditor."""
        mcp = MockFastMCP()
        register_assumption_tools(mcp)

        with patch("research_kb_mcp.tools.assumptions.MethodAssumptionAuditor") as auditor_mock:
            auditor_mock.audit_assumptions = AsyncMock(return_value=mock_audit_result)

            await mcp.tools["research_kb_audit_assumptions"]["func"](
                method_name="RDD",
                domain="time_series",
                output_format="json",
            )

            call_kwargs = auditor_mock.audit_assumptions.call_args
            assert call_kwargs.kwargs.get("domain") == "time_series"

    async def test_assumption_scope_param_passed_through(self, mock_audit_result):
        """MCP tool passes scope to auditor."""
        mcp = MockFastMCP()
        register_assumption_tools(mcp)

        with patch("research_kb_mcp.tools.assumptions.MethodAssumptionAuditor") as auditor_mock:
            auditor_mock.audit_assumptions = AsyncMock(return_value=mock_audit_result)

            await mcp.tools["research_kb_audit_assumptions"]["func"](
                method_name="RDD",
                scope="applied",
                domain="time_series",
                output_format="json",
            )

            call_kwargs = auditor_mock.audit_assumptions.call_args
            assert call_kwargs.kwargs.get("scope") == "applied"

    async def test_assumption_domain_scope_in_json_output(self):
        """JSON output includes domain and scope fields."""
        from research_kb_storage.assumption_audit import MethodAssumptions

        scoped_result = MethodAssumptions(
            method="RDD",
            method_aliases=["regression discontinuity"],
            assumptions=[],
            source="graph",
            domain="time_series",
            scope="applied",
        )

        mcp = MockFastMCP()
        register_assumption_tools(mcp)

        with patch("research_kb_mcp.tools.assumptions.MethodAssumptionAuditor") as auditor_mock:
            auditor_mock.audit_assumptions = AsyncMock(return_value=scoped_result)

            result = await mcp.tools["research_kb_audit_assumptions"]["func"](
                method_name="RDD",
                domain="time_series",
                scope="applied",
                output_format="json",
            )

            data = json.loads(result)
            assert data["domain"] == "time_series"
            assert data["scope"] == "applied"

    async def test_assumption_markdown_default(self, mock_audit_result):
        """Assumption audit defaults to markdown."""
        mcp = MockFastMCP()
        register_assumption_tools(mcp)

        with patch("research_kb_mcp.tools.assumptions.MethodAssumptionAuditor") as auditor_mock:
            auditor_mock.audit_assumptions = AsyncMock(return_value=mock_audit_result)

            result = await mcp.tools["research_kb_audit_assumptions"]["func"](
                method_name="IV",
            )

            assert "## Assumptions for:" in result
            assert "**Aliases**" in result


# ─── Formatter Unit Tests (JSON) ─────────────────────────────────────────────


class TestFormattersJsonUnit:
    """Direct unit tests for JSON formatter functions."""

    def test_search_results_json_roundtrip(self):
        """Search JSON roundtrips through json.loads."""
        from research_kb_mcp.formatters import format_search_results_json

        response = MockSearchResponse(
            results=[
                MockSearchResultDetail(
                    source=MockSourceSummary(id=str(uuid4())),
                    chunk=MockChunkSummary(id=str(uuid4())),
                )
            ]
        )
        result = format_search_results_json(response)
        data = json.loads(result)
        assert "query" in data
        assert "results" in data

    def test_concept_detail_json_no_relationships(self):
        """Concept JSON with no relationships returns empty array."""
        from research_kb_mcp.formatters import format_concept_detail_json

        concept = Concept(
            id=uuid4(),
            name="Test",
            canonical_name="test",
            concept_type=ConceptType.METHOD,
            domain_id="causal_inference",
            created_at=datetime.now(),
        )
        result = format_concept_detail_json(concept, None)
        data = json.loads(result)
        assert data["relationships"] == []

    def test_graph_neighborhood_json_error_case(self):
        """Graph neighborhood JSON handles error dict."""
        from research_kb_mcp.formatters import format_graph_neighborhood_json

        result = format_graph_neighborhood_json({"error": "not found"})
        data = json.loads(result)
        assert data["error"] == "not found"

    def test_biblio_similar_json_empty(self):
        """Biblio JSON with empty results returns empty array."""
        from research_kb_mcp.formatters import format_biblio_similar_json

        source = Source(
            id=uuid4(),
            title="Test",
            source_type=SourceType.PAPER,
            authors=[],
            year=2020,
            domain_id="test",
            file_hash="test",
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        result = format_biblio_similar_json([], source)
        data = json.loads(result)
        assert data["similar"] == []

    def test_citation_network_json_ids_are_strings(self):
        """Citation network JSON ensures all IDs are strings."""
        from research_kb_mcp.formatters import format_citation_network_json

        src = Source(
            id=uuid4(),
            title="Center",
            source_type=SourceType.PAPER,
            authors=["A"],
            year=2020,
            domain_id="test",
            file_hash="h",
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        citing = [
            Source(
                id=uuid4(),
                title="Citer",
                source_type=SourceType.PAPER,
                authors=["B"],
                year=2021,
                domain_id="test",
                file_hash="h2",
                metadata={},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        ]
        result = format_citation_network_json(citing, [], src)
        data = json.loads(result)

        # All IDs should be strings, not UUID objects
        assert isinstance(data["source_id"], str)
        assert isinstance(data["citing"][0]["source_id"], str)

    def test_assumption_audit_json_delegates_to_dict(self):
        """Assumption audit JSON uses MethodAssumptions.to_dict()."""
        from research_kb_mcp.formatters import format_assumption_audit_json
        from research_kb_storage.assumption_audit import MethodAssumptions

        result_obj = MethodAssumptions(
            method="DML",
            method_aliases=["double machine learning"],
            source="graph",
        )
        result = format_assumption_audit_json(result_obj)
        data = json.loads(result)
        assert data["method"] == "DML"
        assert data["method_aliases"] == ["double machine learning"]
        assert data["assumptions"] == []
