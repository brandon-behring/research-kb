"""Tests for Pydantic API schemas.

Tests cover:
- Enum definitions
- Request model validation
- Response model serialization
- Field constraints and defaults
- Nested model composition
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_kb_api.schemas import (
    # Enums
    ContextType,
    ConceptType,
    RelationshipType,
    # Request Models
    SearchRequest,
    # Response Models
    ScoreBreakdown,
    SourceSummary,
    ChunkSummary,
    SearchResultItem,
    SearchResponse,
    SearchMetadata,
    SourceDetail,
    SourceListResponse,
    ChunkDetail,
    SourceWithChunks,
    ConceptDetail,
    ConceptListResponse,
    RelationshipDetail,
    ConceptWithRelationships,
    GraphNode,
    GraphEdge,
    GraphNeighborhood,
    GraphPath,
    CitationSummary,
    SourceCitations,
    DatabaseStats,
    HealthCheck,
    HealthDetail,
)


# =============================================================================
# Test Enums
# =============================================================================


class TestContextType:
    """Test ContextType enum."""

    def test_context_type_values(self):
        """Test ContextType has expected values."""
        assert ContextType.building == "building"
        assert ContextType.auditing == "auditing"
        assert ContextType.balanced == "balanced"

    def test_context_type_from_string(self):
        """Test ContextType can be created from string."""
        assert ContextType("building") == ContextType.building
        assert ContextType("auditing") == ContextType.auditing
        assert ContextType("balanced") == ContextType.balanced

    def test_context_type_invalid_raises(self):
        """Test invalid ContextType raises ValueError."""
        with pytest.raises(ValueError):
            ContextType("invalid")

    def test_context_type_is_str_enum(self):
        """Test ContextType is a string enum."""
        assert isinstance(ContextType.building, str)
        assert ContextType.building == "building"


class TestConceptType:
    """Test ConceptType enum."""

    def test_concept_type_values(self):
        """Test ConceptType has expected values."""
        assert ConceptType.method == "method"
        assert ConceptType.assumption == "assumption"
        assert ConceptType.problem == "problem"
        assert ConceptType.definition == "definition"
        assert ConceptType.theorem == "theorem"

    def test_concept_type_all_values(self):
        """Test all ConceptType values exist."""
        values = [ct.value for ct in ConceptType]
        assert "method" in values
        assert "assumption" in values
        assert "problem" in values
        assert "definition" in values
        assert "theorem" in values


class TestRelationshipType:
    """Test RelationshipType enum."""

    def test_relationship_type_values(self):
        """Test RelationshipType has expected values."""
        assert RelationshipType.requires == "REQUIRES"
        assert RelationshipType.uses == "USES"
        assert RelationshipType.addresses == "ADDRESSES"
        assert RelationshipType.generalizes == "GENERALIZES"
        assert RelationshipType.specializes == "SPECIALIZES"
        assert RelationshipType.alternative_to == "ALTERNATIVE_TO"
        assert RelationshipType.extends == "EXTENDS"

    def test_relationship_type_uppercase_values(self):
        """Test RelationshipType values are uppercase."""
        for rt in RelationshipType:
            assert rt.value == rt.value.upper()


# =============================================================================
# Test Request Models
# =============================================================================


class TestSearchRequest:
    """Test SearchRequest model."""

    def test_minimal_valid_request(self):
        """Test minimal valid SearchRequest."""
        request = SearchRequest(query="instrumental variables")

        assert request.query == "instrumental variables"
        assert request.limit == 10  # default
        assert request.context_type == ContextType.balanced  # default
        assert request.source_filter is None  # default
        assert request.use_graph is True  # default
        assert request.graph_weight == pytest.approx(0.2)  # default
        assert request.use_rerank is True  # default
        assert request.use_expand is True  # default

    def test_full_request(self):
        """Test fully specified SearchRequest."""
        request = SearchRequest(
            query="double machine learning",
            limit=20,
            context_type=ContextType.auditing,
            source_filter="PAPER",
            use_graph=False,
            graph_weight=0.3,
            use_rerank=False,
            use_expand=False,
        )

        assert request.query == "double machine learning"
        assert request.limit == 20
        assert request.context_type == ContextType.auditing
        assert request.source_filter == "PAPER"
        assert request.use_graph is False
        assert request.graph_weight == pytest.approx(0.3)
        assert request.use_rerank is False
        assert request.use_expand is False

    def test_query_min_length_validation(self):
        """Test SearchRequest query minimum length."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="")

        errors = exc_info.value.errors()
        assert any("query" in str(e) for e in errors)

    def test_limit_minimum_validation(self):
        """Test SearchRequest limit minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="test", limit=0)

        errors = exc_info.value.errors()
        assert any("limit" in str(e) for e in errors)

    def test_limit_maximum_validation(self):
        """Test SearchRequest limit maximum value."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="test", limit=101)

        errors = exc_info.value.errors()
        assert any("limit" in str(e) for e in errors)

    def test_limit_edge_values(self):
        """Test SearchRequest limit at edge values."""
        request_min = SearchRequest(query="test", limit=1)
        assert request_min.limit == 1

        request_max = SearchRequest(query="test", limit=100)
        assert request_max.limit == 100

    def test_graph_weight_minimum_validation(self):
        """Test SearchRequest graph_weight minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="test", graph_weight=-0.1)

        errors = exc_info.value.errors()
        assert any("graph_weight" in str(e) for e in errors)

    def test_graph_weight_maximum_validation(self):
        """Test SearchRequest graph_weight maximum value."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="test", graph_weight=1.1)

        errors = exc_info.value.errors()
        assert any("graph_weight" in str(e) for e in errors)

    def test_graph_weight_edge_values(self):
        """Test SearchRequest graph_weight at edge values."""
        request_zero = SearchRequest(query="test", graph_weight=0)
        assert request_zero.graph_weight == pytest.approx(0)

        request_one = SearchRequest(query="test", graph_weight=1)
        assert request_one.graph_weight == pytest.approx(1)

    def test_context_type_from_string(self):
        """Test SearchRequest accepts context_type as string."""
        request = SearchRequest(query="test", context_type="auditing")
        assert request.context_type == ContextType.auditing

    def test_serialization_to_dict(self):
        """Test SearchRequest serializes to dict."""
        request = SearchRequest(query="test", limit=5)
        data = request.model_dump()

        assert data["query"] == "test"
        assert data["limit"] == 5
        assert data["context_type"] == "balanced"


# =============================================================================
# Test Response Models - Score and Summaries
# =============================================================================


class TestScoreBreakdown:
    """Test ScoreBreakdown model."""

    def test_default_values(self):
        """Test ScoreBreakdown has zero defaults."""
        scores = ScoreBreakdown()

        assert scores.fts == pytest.approx(0.0)
        assert scores.vector == pytest.approx(0.0)
        assert scores.graph == pytest.approx(0.0)
        assert scores.citation == pytest.approx(0.0)
        assert scores.combined == pytest.approx(0.0)

    def test_custom_values(self):
        """Test ScoreBreakdown with custom values."""
        scores = ScoreBreakdown(
            fts=0.25,
            vector=0.65,
            graph=0.05,
            citation=0.05,
            combined=0.8,
        )

        assert scores.fts == pytest.approx(0.25)
        assert scores.vector == pytest.approx(0.65)
        assert scores.graph == pytest.approx(0.05)
        assert scores.citation == pytest.approx(0.05)
        assert scores.combined == pytest.approx(0.8)

    def test_serialization(self):
        """Test ScoreBreakdown serializes correctly."""
        scores = ScoreBreakdown(fts=0.3, vector=0.7)
        data = scores.model_dump()

        assert data["fts"] == pytest.approx(0.3)
        assert data["vector"] == pytest.approx(0.7)


class TestSourceSummary:
    """Test SourceSummary model."""

    def test_minimal_source_summary(self):
        """Test SourceSummary with required fields only."""
        summary = SourceSummary(id="src123", title="Test Paper")

        assert summary.id == "src123"
        assert summary.title == "Test Paper"
        assert summary.authors == []
        assert summary.year is None
        assert summary.source_type is None

    def test_full_source_summary(self):
        """Test SourceSummary with all fields."""
        summary = SourceSummary(
            id="src456",
            title="Causal Inference Textbook",
            authors=["Pearl", "Glymour"],
            year=2016,
            source_type="TEXTBOOK",
        )

        assert summary.id == "src456"
        assert summary.title == "Causal Inference Textbook"
        assert summary.authors == ["Pearl", "Glymour"]
        assert summary.year == 2016
        assert summary.source_type == "TEXTBOOK"


class TestChunkSummary:
    """Test ChunkSummary model."""

    def test_minimal_chunk_summary(self):
        """Test ChunkSummary with required fields only."""
        summary = ChunkSummary(id="chk123", content="Test content")

        assert summary.id == "chk123"
        assert summary.content == "Test content"
        assert summary.page_start is None
        assert summary.page_end is None
        assert summary.section is None

    def test_full_chunk_summary(self):
        """Test ChunkSummary with all fields."""
        summary = ChunkSummary(
            id="chk456",
            content="The backdoor criterion...",
            page_start=42,
            page_end=43,
            section="Chapter 3: Interventions",
        )

        assert summary.page_start == 42
        assert summary.page_end == 43
        assert summary.section == "Chapter 3: Interventions"


# =============================================================================
# Test Response Models - Search Results
# =============================================================================


class TestSearchResultItem:
    """Test SearchResultItem model."""

    def test_search_result_item(self):
        """Test SearchResultItem with nested models."""
        item = SearchResultItem(
            source=SourceSummary(id="src1", title="Paper"),
            chunk=ChunkSummary(id="chk1", content="Content"),
            concepts=["concept1", "concept2"],
            scores=ScoreBreakdown(fts=0.3, vector=0.7),
            combined_score=0.5,
        )

        assert item.source.id == "src1"
        assert item.chunk.id == "chk1"
        assert len(item.concepts) == 2
        assert item.combined_score == pytest.approx(0.5)

    def test_search_result_item_empty_concepts(self):
        """Test SearchResultItem with empty concepts list."""
        item = SearchResultItem(
            source=SourceSummary(id="src1", title="Paper"),
            chunk=ChunkSummary(id="chk1", content="Content"),
            scores=ScoreBreakdown(),
            combined_score=0.0,
        )

        assert item.concepts == []


class TestSearchMetadata:
    """Test SearchMetadata model."""

    def test_search_metadata(self):
        """Test SearchMetadata with timing info."""
        metadata = SearchMetadata(
            execution_time_ms=150.5,
            embedding_time_ms=50.2,
            search_time_ms=100.3,
            result_count=10,
        )

        assert metadata.execution_time_ms == pytest.approx(150.5)
        assert metadata.embedding_time_ms == pytest.approx(50.2)
        assert metadata.search_time_ms == pytest.approx(100.3)
        assert metadata.result_count == 10


class TestSearchResponse:
    """Test SearchResponse model."""

    def test_search_response(self):
        """Test SearchResponse with results."""
        result = SearchResultItem(
            source=SourceSummary(id="src1", title="Paper"),
            chunk=ChunkSummary(id="chk1", content="Content"),
            scores=ScoreBreakdown(),
            combined_score=0.5,
        )
        metadata = SearchMetadata(
            execution_time_ms=100,
            embedding_time_ms=25,
            search_time_ms=75,
            result_count=1,
        )

        response = SearchResponse(
            query="test query",
            expanded_query="expanded test query",
            results=[result],
            metadata=metadata,
        )

        assert response.query == "test query"
        assert response.expanded_query == "expanded test query"
        assert len(response.results) == 1
        assert response.metadata.result_count == 1

    def test_search_response_no_expansion(self):
        """Test SearchResponse without query expansion."""
        response = SearchResponse(
            query="test",
            results=[],
            metadata=SearchMetadata(
                execution_time_ms=50,
                embedding_time_ms=20,
                search_time_ms=30,
                result_count=0,
            ),
        )

        assert response.expanded_query is None


# =============================================================================
# Test Response Models - Source Details
# =============================================================================


class TestSourceDetail:
    """Test SourceDetail model."""

    def test_minimal_source_detail(self):
        """Test SourceDetail with required fields."""
        detail = SourceDetail(id="src1", title="Test Paper")

        assert detail.id == "src1"
        assert detail.title == "Test Paper"
        assert detail.authors == []
        assert detail.year is None
        assert detail.source_type is None
        assert detail.file_path is None
        assert detail.abstract is None
        assert detail.metadata is None
        assert detail.created_at is None

    def test_full_source_detail(self):
        """Test SourceDetail with all fields."""
        detail = SourceDetail(
            id="src456",
            title="Full Paper",
            authors=["Author One", "Author Two"],
            year=2023,
            source_type="PAPER",
            file_path="/papers/test.pdf",
            abstract="This paper presents...",
            metadata={"doi": "10.1234/test"},
            created_at="2023-01-15T10:30:00Z",
        )

        assert detail.file_path == "/papers/test.pdf"
        assert detail.abstract == "This paper presents..."
        assert detail.metadata == {"doi": "10.1234/test"}
        assert detail.created_at == "2023-01-15T10:30:00Z"


class TestSourceListResponse:
    """Test SourceListResponse model."""

    def test_source_list_response(self):
        """Test SourceListResponse with pagination."""
        sources = [
            SourceDetail(id="src1", title="Paper 1"),
            SourceDetail(id="src2", title="Paper 2"),
        ]

        response = SourceListResponse(
            sources=sources,
            total=100,
            limit=10,
            offset=20,
        )

        assert len(response.sources) == 2
        assert response.total == 100
        assert response.limit == 10
        assert response.offset == 20


class TestChunkDetail:
    """Test ChunkDetail model."""

    def test_chunk_detail(self):
        """Test ChunkDetail with all fields."""
        detail = ChunkDetail(
            id="chk1",
            content="Test content here",
            page_start=10,
            page_end=11,
            metadata={"section": "Introduction"},
        )

        assert detail.id == "chk1"
        assert detail.content == "Test content here"
        assert detail.page_start == 10
        assert detail.page_end == 11
        assert detail.metadata == {"section": "Introduction"}


class TestSourceWithChunks:
    """Test SourceWithChunks model."""

    def test_source_with_chunks(self):
        """Test SourceWithChunks composite model."""
        source = SourceDetail(id="src1", title="Paper")
        chunks = [
            ChunkDetail(id="chk1", content="Content 1"),
            ChunkDetail(id="chk2", content="Content 2"),
        ]

        response = SourceWithChunks(
            source=source,
            chunks=chunks,
            chunk_count=2,
        )

        assert response.source.id == "src1"
        assert len(response.chunks) == 2
        assert response.chunk_count == 2


# =============================================================================
# Test Response Models - Concepts
# =============================================================================


class TestConceptDetail:
    """Test ConceptDetail model."""

    def test_minimal_concept_detail(self):
        """Test ConceptDetail with required fields."""
        detail = ConceptDetail(
            id="con1",
            name="backdoor criterion",
            canonical_name="backdoor_criterion",
        )

        assert detail.id == "con1"
        assert detail.name == "backdoor criterion"
        assert detail.canonical_name == "backdoor_criterion"
        assert detail.concept_type is None
        assert detail.definition is None
        assert detail.aliases == []

    def test_full_concept_detail(self):
        """Test ConceptDetail with all fields."""
        detail = ConceptDetail(
            id="con2",
            name="instrumental variables",
            canonical_name="instrumental_variables",
            concept_type=ConceptType.method,
            definition="A method for estimating causal effects...",
            aliases=["IV", "IVs", "instruments"],
        )

        assert detail.concept_type == ConceptType.method
        assert detail.definition == "A method for estimating causal effects..."
        assert len(detail.aliases) == 3


class TestConceptListResponse:
    """Test ConceptListResponse model."""

    def test_concept_list_response(self):
        """Test ConceptListResponse."""
        concepts = [
            ConceptDetail(id="c1", name="IV", canonical_name="iv"),
            ConceptDetail(id="c2", name="DML", canonical_name="dml"),
        ]

        response = ConceptListResponse(
            concepts=concepts,
            total=50,
        )

        assert len(response.concepts) == 2
        assert response.total == 50


class TestRelationshipDetail:
    """Test RelationshipDetail model."""

    def test_relationship_detail(self):
        """Test RelationshipDetail with all fields."""
        detail = RelationshipDetail(
            id="rel1",
            source_id="con1",
            source_name="backdoor criterion",
            target_id="con2",
            target_name="adjustment set",
            relationship_type=RelationshipType.requires,
            confidence=0.95,
        )

        assert detail.id == "rel1"
        assert detail.source_id == "con1"
        assert detail.source_name == "backdoor criterion"
        assert detail.target_id == "con2"
        assert detail.target_name == "adjustment set"
        assert detail.relationship_type == RelationshipType.requires
        assert detail.confidence == pytest.approx(0.95)

    def test_relationship_detail_no_confidence(self):
        """Test RelationshipDetail without confidence."""
        detail = RelationshipDetail(
            id="rel2",
            source_id="con1",
            source_name="A",
            target_id="con2",
            target_name="B",
            relationship_type=RelationshipType.uses,
        )

        assert detail.confidence is None


class TestConceptWithRelationships:
    """Test ConceptWithRelationships model."""

    def test_concept_with_relationships(self):
        """Test ConceptWithRelationships composite model."""
        concept = ConceptDetail(id="c1", name="Test", canonical_name="test")
        relationships = [
            RelationshipDetail(
                id="r1",
                source_id="c1",
                source_name="Test",
                target_id="c2",
                target_name="Other",
                relationship_type=RelationshipType.uses,
            )
        ]

        response = ConceptWithRelationships(
            concept=concept,
            relationships=relationships,
        )

        assert response.concept.id == "c1"
        assert len(response.relationships) == 1


# =============================================================================
# Test Response Models - Graph
# =============================================================================


class TestGraphNode:
    """Test GraphNode model."""

    def test_graph_node(self):
        """Test GraphNode with all fields."""
        node = GraphNode(id="n1", name="backdoor", type="theorem")

        assert node.id == "n1"
        assert node.name == "backdoor"
        assert node.type == "theorem"

    def test_graph_node_no_type(self):
        """Test GraphNode without type."""
        node = GraphNode(id="n2", name="test")

        assert node.type is None


class TestGraphEdge:
    """Test GraphEdge model."""

    def test_graph_edge(self):
        """Test GraphEdge with all fields."""
        edge = GraphEdge(source="n1", target="n2", type="REQUIRES")

        assert edge.source == "n1"
        assert edge.target == "n2"
        assert edge.type == "REQUIRES"

    def test_graph_edge_no_type(self):
        """Test GraphEdge without type."""
        edge = GraphEdge(source="n1", target="n2")

        assert edge.type is None


class TestGraphNeighborhood:
    """Test GraphNeighborhood model."""

    def test_graph_neighborhood(self):
        """Test GraphNeighborhood composite model."""
        center = GraphNode(id="n1", name="center", type="method")
        nodes = [
            GraphNode(id="n2", name="neighbor1"),
            GraphNode(id="n3", name="neighbor2"),
        ]
        edges = [
            GraphEdge(source="n1", target="n2", type="USES"),
            GraphEdge(source="n1", target="n3", type="REQUIRES"),
        ]

        neighborhood = GraphNeighborhood(
            center=center,
            nodes=nodes,
            edges=edges,
        )

        assert neighborhood.center.id == "n1"
        assert len(neighborhood.nodes) == 2
        assert len(neighborhood.edges) == 2


class TestGraphPath:
    """Test GraphPath model."""

    def test_graph_path(self):
        """Test GraphPath with path."""
        path = GraphPath(
            from_concept="A",
            to_concept="B",
            path=[
                GraphNode(id="n1", name="A"),
                GraphNode(id="n2", name="intermediate"),
                GraphNode(id="n3", name="B"),
            ],
            path_length=3,
        )

        assert path.from_concept == "A"
        assert path.to_concept == "B"
        assert len(path.path) == 3
        assert path.path_length == 3


# =============================================================================
# Test Response Models - Citations
# =============================================================================


class TestCitationSummary:
    """Test CitationSummary model."""

    def test_citation_summary(self):
        """Test CitationSummary with all fields."""
        citation = CitationSummary(
            id="src1",
            title="Cited Paper",
            year=2020,
        )

        assert citation.id == "src1"
        assert citation.title == "Cited Paper"
        assert citation.year == 2020

    def test_citation_summary_no_year(self):
        """Test CitationSummary without year."""
        citation = CitationSummary(id="src2", title="Old Paper")

        assert citation.year is None


class TestSourceCitations:
    """Test SourceCitations model."""

    def test_source_citations(self):
        """Test SourceCitations with both directions."""
        citing = [CitationSummary(id="c1", title="Citing Paper", year=2023)]
        cited = [CitationSummary(id="c2", title="Cited Paper", year=2010)]

        citations = SourceCitations(
            source_id="src1",
            citing_sources=citing,
            cited_sources=cited,
            citation_count=1,
            reference_count=1,
        )

        assert citations.source_id == "src1"
        assert len(citations.citing_sources) == 1
        assert len(citations.cited_sources) == 1
        assert citations.citation_count == 1
        assert citations.reference_count == 1


# =============================================================================
# Test Response Models - Stats and Health
# =============================================================================


class TestDatabaseStats:
    """Test DatabaseStats model."""

    def test_database_stats(self):
        """Test DatabaseStats with all counts."""
        stats = DatabaseStats(
            sources=100,
            chunks=5000,
            concepts=200,
            relationships=500,
            citations=300,
            chunk_concepts=1000,
        )

        assert stats.sources == 100
        assert stats.chunks == 5000
        assert stats.concepts == 200
        assert stats.relationships == 500
        assert stats.citations == 300
        assert stats.chunk_concepts == 1000


class TestHealthCheck:
    """Test HealthCheck model."""

    def test_health_check_defaults(self):
        """Test HealthCheck has sensible defaults."""
        health = HealthCheck()

        assert health.status == "healthy"
        assert health.version == "1.0.0"
        assert health.database == "connected"
        assert health.embedding_model == "ready"

    def test_health_check_custom(self):
        """Test HealthCheck with custom values."""
        health = HealthCheck(
            status="degraded",
            version="2.0.0",
            database="slow",
            embedding_model="loading",
        )

        assert health.status == "degraded"
        assert health.version == "2.0.0"


class TestHealthDetail:
    """Test HealthDetail model."""

    def test_health_detail(self):
        """Test HealthDetail with components and stats."""
        stats = DatabaseStats(
            sources=50,
            chunks=2500,
            concepts=100,
            relationships=250,
            citations=150,
            chunk_concepts=500,
        )

        health = HealthDetail(
            status="healthy",
            version="1.0.0",
            components={
                "database": "connected",
                "embedding": "ready",
                "grobid": "connected",
            },
            stats=stats,
        )

        assert health.status == "healthy"
        assert len(health.components) == 3
        assert health.stats.sources == 50

    def test_health_detail_no_stats(self):
        """Test HealthDetail without stats."""
        health = HealthDetail(
            status="healthy",
            version="1.0.0",
            components={"database": "connected"},
        )

        assert health.stats is None


# =============================================================================
# Test Serialization and Deserialization
# =============================================================================


class TestJsonSerialization:
    """Test JSON serialization of models."""

    def test_search_request_json_round_trip(self):
        """Test SearchRequest serializes and deserializes correctly."""
        original = SearchRequest(
            query="test query",
            limit=20,
            context_type=ContextType.auditing,
        )

        json_data = original.model_dump()
        restored = SearchRequest(**json_data)

        assert restored.query == original.query
        assert restored.limit == original.limit
        assert restored.context_type == original.context_type

    def test_search_response_json_serialization(self):
        """Test SearchResponse serializes to valid JSON."""
        result = SearchResultItem(
            source=SourceSummary(id="src1", title="Paper"),
            chunk=ChunkSummary(id="chk1", content="Content"),
            scores=ScoreBreakdown(fts=0.3, vector=0.7),
            combined_score=0.5,
        )
        metadata = SearchMetadata(
            execution_time_ms=100,
            embedding_time_ms=25,
            search_time_ms=75,
            result_count=1,
        )

        response = SearchResponse(
            query="test",
            results=[result],
            metadata=metadata,
        )

        json_data = response.model_dump()

        assert json_data["query"] == "test"
        assert len(json_data["results"]) == 1
        assert json_data["results"][0]["source"]["id"] == "src1"
        assert json_data["metadata"]["result_count"] == 1

    def test_database_stats_json(self):
        """Test DatabaseStats serializes correctly."""
        stats = DatabaseStats(
            sources=100,
            chunks=5000,
            concepts=200,
            relationships=500,
            citations=300,
            chunk_concepts=1000,
        )

        json_data = stats.model_dump()

        assert json_data["sources"] == 100
        assert json_data["chunks"] == 5000

    def test_graph_neighborhood_json(self):
        """Test GraphNeighborhood serializes correctly."""
        neighborhood = GraphNeighborhood(
            center=GraphNode(id="n1", name="center"),
            nodes=[GraphNode(id="n2", name="neighbor")],
            edges=[GraphEdge(source="n1", target="n2")],
        )

        json_data = neighborhood.model_dump()

        assert json_data["center"]["id"] == "n1"
        assert len(json_data["nodes"]) == 1
        assert len(json_data["edges"]) == 1


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases in schema validation."""

    def test_search_request_unicode_query(self):
        """Test SearchRequest accepts unicode characters."""
        request = SearchRequest(query="causal inference with")

        assert "causal" in request.query

    def test_source_summary_empty_authors(self):
        """Test SourceSummary with explicitly empty authors."""
        summary = SourceSummary(id="src1", title="Test", authors=[])

        assert summary.authors == []

    def test_search_metadata_zero_times(self):
        """Test SearchMetadata with zero times."""
        metadata = SearchMetadata(
            execution_time_ms=0,
            embedding_time_ms=0,
            search_time_ms=0,
            result_count=0,
        )

        assert metadata.execution_time_ms == 0

    def test_graph_path_empty_path(self):
        """Test GraphPath with empty path."""
        path = GraphPath(
            from_concept="A",
            to_concept="B",
            path=[],
            path_length=0,
        )

        assert len(path.path) == 0
        assert path.path_length == 0

    def test_concept_detail_special_characters_in_name(self):
        """Test ConceptDetail with special characters."""
        detail = ConceptDetail(
            id="c1",
            name="IV (two-stage least squares)",
            canonical_name="iv_2sls",
        )

        assert "(" in detail.name
        assert ")" in detail.name

    def test_score_breakdown_negative_scores(self):
        """Test ScoreBreakdown allows negative scores."""
        # Some scoring functions might produce negative values
        scores = ScoreBreakdown(fts=-0.1, vector=0.5, combined=0.4)

        assert scores.fts == pytest.approx(-0.1)
