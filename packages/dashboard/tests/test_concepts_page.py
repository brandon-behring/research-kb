"""Tests for concepts page.

Tests the concept graph explorer page by mocking Streamlit and the API client,
then calling the page function directly to verify widget creation and
graph rendering for various states (search, neighborhood, path, errors).
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from uuid import uuid4

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_concepts_response():
    """Concepts API response with mixed types."""
    return {
        "concepts": [
            {
                "id": str(uuid4()),
                "name": "Instrumental Variables",
                "concept_type": "METHOD",
                "description": "A technique for causal identification.",
            },
            {
                "id": str(uuid4()),
                "name": "Exclusion Restriction",
                "concept_type": "ASSUMPTION",
                "description": "Instrument affects outcome only through treatment.",
            },
            {
                "id": str(uuid4()),
                "name": "Endogeneity",
                "concept_type": "PROBLEM",
                "description": "Correlation between regressor and error term.",
            },
        ],
        "total": 3,
    }


@pytest.fixture
def sample_neighborhood_response():
    """Neighborhood API response with nodes and edges."""
    node_ids = [str(uuid4()) for _ in range(4)]
    return {
        "nodes": [
            {
                "id": node_ids[0],
                "name": "double machine learning",
                "concept_type": "METHOD",
                "description": "",
            },
            {
                "id": node_ids[1],
                "name": "Neyman Orthogonality",
                "concept_type": "THEOREM",
                "description": "",
            },
            {
                "id": node_ids[2],
                "name": "Cross-Fitting",
                "concept_type": "TECHNIQUE",
                "description": "",
            },
            {
                "id": node_ids[3],
                "name": "Sample Splitting",
                "concept_type": "TECHNIQUE",
                "description": "",
            },
        ],
        "edges": [
            {"source": node_ids[0], "target": node_ids[1], "relationship_type": "USES"},
            {"source": node_ids[0], "target": node_ids[2], "relationship_type": "REQUIRES"},
            {"source": node_ids[2], "target": node_ids[3], "relationship_type": "EXTENDS"},
        ],
    }


@pytest.fixture
def sample_path_response():
    """Path finder API response."""
    return {
        "path": [
            {"name": "IV", "concept_type": "METHOD"},
            {"name": "Exclusion Restriction", "concept_type": "ASSUMPTION"},
            {"name": "Unconfoundedness", "concept_type": "ASSUMPTION"},
        ],
        "path_length": 3,
    }


def _make_mock_client(
    concepts_response=None,
    neighborhood_response=None,
    path_response=None,
    error=None,
):
    """Create a mock ResearchKBClient for concepts page testing."""
    mock_client = AsyncMock()
    mock_client.close = AsyncMock()

    if error:
        mock_client.list_concepts.side_effect = error
        mock_client.get_graph_neighborhood.side_effect = error
        mock_client.get_graph_path.side_effect = error
    else:
        mock_client.list_concepts.return_value = concepts_response or {"concepts": [], "total": 0}
        mock_client.get_graph_neighborhood.return_value = neighborhood_response or {
            "nodes": [],
            "edges": [],
        }
        mock_client.get_graph_path.return_value = path_response or {"path": [], "path_length": 0}

    return mock_client


def _make_mock_st(
    search_query="",
    center_concept="",
    path_from="",
    path_to="",
):
    """Create a mock streamlit module for concepts page testing.

    Args:
        search_query: Value for concept search text_input
        center_concept: Value for neighborhood center text_input
        path_from: Value for path finder 'from' text_input
        path_to: Value for path finder 'to' text_input
    """
    mock_st = MagicMock()

    # Track text_input calls by key/placeholder to return correct values
    text_input_values = {
        "Search concepts": search_query,
        "Center concept": center_concept,
        "From concept": path_from,
        "To concept": path_to,
    }

    def text_input_side_effect(label, **kwargs):
        return text_input_values.get(label, "")

    mock_st.text_input.side_effect = text_input_side_effect

    # selectbox returns first option
    mock_st.selectbox.side_effect = lambda label, options, **kw: options[kw.get("index", 0)]

    # number_input returns the value kwarg
    mock_st.number_input.side_effect = lambda label, **kw: kw.get("value", 50)

    # tabs return context managers
    tab1, tab2, tab3 = MagicMock(), MagicMock(), MagicMock()
    for tab in (tab1, tab2, tab3):
        tab.__enter__ = MagicMock(return_value=tab)
        tab.__exit__ = MagicMock(return_value=False)
    mock_st.tabs.return_value = [tab1, tab2, tab3]

    # columns context manager
    def make_cols(spec):
        n = spec if isinstance(spec, int) else len(spec)
        cols = []
        for _ in range(n):
            col = MagicMock()
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=False)
            col.text_input = MagicMock(side_effect=text_input_side_effect)
            col.selectbox = mock_st.selectbox
            col.number_input = mock_st.number_input
            cols.append(col)
        return cols

    mock_st.columns.side_effect = make_cols

    # spinner context manager
    mock_st.spinner.return_value.__enter__ = MagicMock()
    mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)

    # expander context manager
    exp = MagicMock()
    exp.__enter__ = MagicMock(return_value=exp)
    exp.__exit__ = MagicMock(return_value=False)
    mock_st.expander.return_value = exp

    return mock_st


# ---------------------------------------------------------------------------
# Rendering Tests
# ---------------------------------------------------------------------------


class TestConceptsPageRendering:
    """Test that the concepts page creates expected widgets."""

    def test_concepts_page_renders(self):
        """Page runs without exception when API is mocked."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client()

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        mock_st.header.assert_called_once()

    def test_concepts_page_header(self):
        """Page shows the Concept Graph Explorer header."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client()

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        header_arg = mock_st.header.call_args[0][0]
        assert "Concept Graph Explorer" in header_arg

    def test_concepts_page_has_three_tabs(self):
        """Page creates three tabs: Search, Neighborhood, Path."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client()

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        mock_st.tabs.assert_called_once()
        tab_names = mock_st.tabs.call_args[0][0]
        assert len(tab_names) == 3
        assert "Search Concepts" in tab_names
        assert "Neighborhood Explorer" in tab_names
        assert "Path Finder" in tab_names

    def test_type_filter_selectbox(self):
        """Type filter offers All + all 9 concept types."""
        mock_st = _make_mock_st()
        mock_client = _make_mock_client()

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        selectbox_calls = mock_st.selectbox.call_args_list
        type_call = None
        for c in selectbox_calls:
            if "type" in c[0][0].lower() or "filter" in c[0][0].lower():
                type_call = c
                break

        assert type_call is not None
        options = type_call[0][1]
        assert options[0] == "All"
        assert "METHOD" in options
        assert "ASSUMPTION" in options
        assert "MODEL" in options
        assert len(options) == 10  # All + 9 types


# ---------------------------------------------------------------------------
# Concept Type & Relationship Color Tests
# ---------------------------------------------------------------------------


class TestColorMaps:
    """Test CONCEPT_TYPE_COLORS and RELATIONSHIP_COLORS cover all types."""

    def test_concept_type_colors_covers_all_types(self):
        """CONCEPT_TYPE_COLORS has entries for all 9 ConceptType values."""
        from research_kb_dashboard.pages.concepts import CONCEPT_TYPE_COLORS

        expected_types = {
            "METHOD",
            "ASSUMPTION",
            "PROBLEM",
            "DEFINITION",
            "THEOREM",
            "CONCEPT",
            "PRINCIPLE",
            "TECHNIQUE",
            "MODEL",
        }
        assert set(CONCEPT_TYPE_COLORS.keys()) == expected_types

    def test_concept_type_colors_are_hex(self):
        """All concept type colors are valid hex color codes."""
        from research_kb_dashboard.pages.concepts import CONCEPT_TYPE_COLORS

        for ctype, color in CONCEPT_TYPE_COLORS.items():
            assert color.startswith("#"), f"{ctype} color {color} is not hex"
            assert len(color) == 7, f"{ctype} color {color} is not #RRGGBB"

    def test_relationship_colors_covers_all_types(self):
        """RELATIONSHIP_COLORS has entries for all 7 RelationshipType values."""
        from research_kb_dashboard.pages.concepts import RELATIONSHIP_COLORS

        expected_types = {
            "REQUIRES",
            "USES",
            "ADDRESSES",
            "GENERALIZES",
            "SPECIALIZES",
            "ALTERNATIVE_TO",
            "EXTENDS",
        }
        assert set(RELATIONSHIP_COLORS.keys()) == expected_types

    def test_relationship_colors_are_hex(self):
        """All relationship colors are valid hex color codes."""
        from research_kb_dashboard.pages.concepts import RELATIONSHIP_COLORS

        for rtype, color in RELATIONSHIP_COLORS.items():
            assert color.startswith("#"), f"{rtype} color {color} is not hex"
            assert len(color) == 7, f"{rtype} color {color} is not #RRGGBB"


# ---------------------------------------------------------------------------
# Concept Search Tests
# ---------------------------------------------------------------------------


class TestConceptSearch:
    """Test concept search tab behavior."""

    def test_concept_search_with_results(self, sample_concepts_response):
        """Searching concepts shows success message with count."""
        mock_st = _make_mock_st(search_query="instrumental variables")
        mock_client = _make_mock_client(concepts_response=sample_concepts_response)

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        mock_st.success.assert_called()
        success_msg = mock_st.success.call_args[0][0]
        assert "3" in success_msg

    def test_concept_search_empty_results(self):
        """Empty search results show info message."""
        mock_st = _make_mock_st(search_query="xyznonexistent")
        mock_client = _make_mock_client(concepts_response={"concepts": [], "total": 0})

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        mock_st.info.assert_called()
        info_calls = [c[0][0] for c in mock_st.info.call_args_list]
        assert any("no concepts" in msg.lower() for msg in info_calls)

    def test_concept_search_renders_markdown(self, sample_concepts_response):
        """Each concept is rendered via st.markdown with type and name."""
        mock_st = _make_mock_st(search_query="IV")
        mock_client = _make_mock_client(concepts_response=sample_concepts_response)

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        # Should have at least 3 markdown calls (one per concept + page description)
        markdown_calls = [c[0][0] for c in mock_st.markdown.call_args_list]
        concept_markdowns = [m for m in markdown_calls if "Instrumental Variables" in m]
        assert len(concept_markdowns) >= 1

    def test_no_query_does_not_search(self):
        """Empty search query doesn't trigger API call."""
        mock_st = _make_mock_st(search_query="")
        mock_client = _make_mock_client()

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        mock_client.list_concepts.assert_not_called()


# ---------------------------------------------------------------------------
# Neighborhood Graph Tests
# ---------------------------------------------------------------------------


class TestNeighborhoodGraph:
    """Test neighborhood visualization tab."""

    def test_neighborhood_with_data(self, sample_neighborhood_response):
        """Neighborhood with data shows node/edge counts and renders graph."""
        mock_st = _make_mock_st(center_concept="double machine learning")
        mock_client = _make_mock_client(neighborhood_response=sample_neighborhood_response)

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.concepts.create_network") as mock_create,
            patch("research_kb_dashboard.pages.concepts.render_network"),
        ):
            net = MagicMock()
            mock_create.return_value = net

            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        mock_st.success.assert_called()
        success_msg = mock_st.success.call_args[0][0]
        assert "4" in success_msg  # 4 nodes
        assert "3" in success_msg  # 3 edges

    def test_neighborhood_adds_nodes(self, sample_neighborhood_response):
        """Network receives add_node for each node in response."""
        mock_st = _make_mock_st(center_concept="double machine learning")
        mock_client = _make_mock_client(neighborhood_response=sample_neighborhood_response)

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.concepts.create_network") as mock_create,
            patch("research_kb_dashboard.pages.concepts.render_network"),
        ):
            net = MagicMock()
            mock_create.return_value = net

            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        assert net.add_node.call_count == 4

    def test_neighborhood_adds_edges(self, sample_neighborhood_response):
        """Network receives add_edge for each edge in response."""
        mock_st = _make_mock_st(center_concept="double machine learning")
        mock_client = _make_mock_client(neighborhood_response=sample_neighborhood_response)

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.concepts.create_network") as mock_create,
            patch("research_kb_dashboard.pages.concepts.render_network"),
        ):
            net = MagicMock()
            mock_create.return_value = net

            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        assert net.add_edge.call_count == 3

    def test_neighborhood_empty_shows_warning(self):
        """Empty neighborhood shows warning."""
        mock_st = _make_mock_st(center_concept="nonexistent_concept")
        mock_client = _make_mock_client(neighborhood_response={"nodes": [], "edges": []})

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        mock_st.warning.assert_called()

    def test_neighborhood_renders_legend(self, sample_neighborhood_response):
        """Neighborhood renders color legend with concept types."""
        mock_st = _make_mock_st(center_concept="double machine learning")
        mock_client = _make_mock_client(neighborhood_response=sample_neighborhood_response)

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
            patch("research_kb_dashboard.pages.concepts.create_network") as mock_create,
            patch("research_kb_dashboard.pages.concepts.render_network"),
        ):
            net = MagicMock()
            mock_create.return_value = net

            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        # Legend rendered via markdown with unsafe_allow_html
        legend_calls = [
            c for c in mock_st.markdown.call_args_list if c[1].get("unsafe_allow_html") is True
        ]
        assert len(legend_calls) >= 1
        legend_text = legend_calls[0][0][0]
        assert "Legend" in legend_text
        assert "METHOD" in legend_text

    def test_no_center_does_not_fetch_neighborhood(self):
        """Empty center concept doesn't trigger neighborhood API call."""
        mock_st = _make_mock_st(center_concept="")
        mock_client = _make_mock_client()

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        mock_client.get_graph_neighborhood.assert_not_called()


# ---------------------------------------------------------------------------
# Path Finder Tests
# ---------------------------------------------------------------------------


class TestPathFinder:
    """Test path finder tab."""

    def test_path_found(self, sample_path_response):
        """Successful path shows path length and chain."""
        mock_st = _make_mock_st(path_from="IV", path_to="unconfoundedness")
        mock_client = _make_mock_client(path_response=sample_path_response)

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        mock_st.success.assert_called()
        mock_st.code.assert_called()
        code_arg = mock_st.code.call_args[0][0]
        assert "IV" in code_arg
        assert "Unconfoundedness" in code_arg
        assert " -> " in code_arg

    def test_no_path_found(self):
        """No path shows warning."""
        mock_st = _make_mock_st(path_from="IV", path_to="unrelated_concept")
        mock_client = _make_mock_client(path_response={"path": [], "path_length": 0})

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        mock_st.warning.assert_called()
        warning_msg = mock_st.warning.call_args[0][0]
        assert "no path" in warning_msg.lower()

    def test_partial_path_input_does_not_search(self):
        """Only one concept entered doesn't trigger path search."""
        mock_st = _make_mock_st(path_from="IV", path_to="")
        mock_client = _make_mock_client()

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        mock_client.get_graph_path.assert_not_called()


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestConceptsErrors:
    """Test error handling across all tabs."""

    def test_search_connect_error(self):
        """API connection error on search shows error message."""
        import httpx

        mock_st = _make_mock_st(search_query="test")
        mock_client = _make_mock_client(error=httpx.ConnectError("Connection refused"))

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        mock_st.error.assert_called()
        error_msg = mock_st.error.call_args[0][0]
        assert "api" in error_msg.lower() or "connect" in error_msg.lower()

    def test_search_generic_error(self):
        """Non-connection error on search shows error message."""
        mock_st = _make_mock_st(search_query="test")
        mock_client = _make_mock_client(error=RuntimeError("Database error"))

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        mock_st.error.assert_called()

    def test_neighborhood_connect_error(self):
        """API connection error on neighborhood shows error."""
        import httpx

        mock_st = _make_mock_st(center_concept="DML")
        mock_client = _make_mock_client(error=httpx.ConnectError("Connection refused"))

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        mock_st.error.assert_called()

    def test_path_connect_error(self):
        """API connection error on path finder shows error."""
        import httpx

        mock_st = _make_mock_st(path_from="IV", path_to="DML")
        mock_client = _make_mock_client(error=httpx.ConnectError("Connection refused"))

        with (
            patch("research_kb_dashboard.pages.concepts.st", mock_st),
            patch(
                "research_kb_dashboard.pages.concepts.ResearchKBClient",
                return_value=mock_client,
            ),
        ):
            from research_kb_dashboard.pages.concepts import concept_graph_page

            concept_graph_page()

        mock_st.error.assert_called()
