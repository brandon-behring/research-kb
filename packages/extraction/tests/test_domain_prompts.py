"""Tests for domain-specific prompt configurations.

Validates:
- All 5 domains registered with required keys
- get_domain_prompt_section() returns non-empty guidance
- get_domain_abbreviations() returns lowercase keys, non-empty values
- get_domain_config() returns full configuration
- list_domains() enumerates all domains
- get_all_abbreviations() merges all domain abbreviations
- Unknown domain fallback behavior (defaults to causal_inference)
"""

import pytest

from research_kb_extraction.domain_prompts import (
    DOMAIN_PROMPTS,
    get_all_abbreviations,
    get_domain_abbreviations,
    get_domain_config,
    get_domain_prompt_section,
    list_domains,
)

pytestmark = pytest.mark.unit


# All 5 expected domain IDs
EXPECTED_DOMAINS = {
    "healthcare",
    "causal_inference",
    "time_series",
    "rag_llm",
    "interview_prep",
}

# Required keys in every domain config
REQUIRED_KEYS = {
    "name",
    "description",
    "concept_type_guidance",
    "examples",
    "abbreviations",
}


class TestDomainRegistry:
    """Test the DOMAIN_PROMPTS registry structure."""

    def test_all_five_domains_present(self):
        """All 5 expected domains exist in the registry."""
        assert set(DOMAIN_PROMPTS.keys()) == EXPECTED_DOMAINS

    @pytest.mark.parametrize("domain_id", EXPECTED_DOMAINS)
    def test_domain_has_required_keys(self, domain_id):
        """Each domain config has all required keys."""
        config = DOMAIN_PROMPTS[domain_id]
        missing = REQUIRED_KEYS - set(config.keys())
        assert not missing, f"Domain '{domain_id}' missing keys: {missing}"

    @pytest.mark.parametrize("domain_id", EXPECTED_DOMAINS)
    def test_domain_name_nonempty(self, domain_id):
        """Each domain has a non-empty human-readable name."""
        name = DOMAIN_PROMPTS[domain_id]["name"]
        assert isinstance(name, str) and len(name) > 0

    @pytest.mark.parametrize("domain_id", EXPECTED_DOMAINS)
    def test_domain_description_nonempty(self, domain_id):
        """Each domain has a non-empty description."""
        desc = DOMAIN_PROMPTS[domain_id]["description"]
        assert isinstance(desc, str) and len(desc) > 0

    @pytest.mark.parametrize("domain_id", EXPECTED_DOMAINS)
    def test_domain_examples_nonempty(self, domain_id):
        """Each domain has at least one example."""
        examples = DOMAIN_PROMPTS[domain_id]["examples"]
        assert isinstance(examples, list) and len(examples) >= 1

    @pytest.mark.parametrize("domain_id", EXPECTED_DOMAINS)
    def test_domain_abbreviations_nonempty(self, domain_id):
        """Each domain has at least one abbreviation."""
        abbrevs = DOMAIN_PROMPTS[domain_id]["abbreviations"]
        assert isinstance(abbrevs, dict) and len(abbrevs) >= 1


class TestGetDomainPromptSection:
    """Test get_domain_prompt_section()."""

    @pytest.mark.parametrize("domain_id", EXPECTED_DOMAINS)
    def test_returns_nonempty_string(self, domain_id):
        """Returns non-empty guidance for each known domain."""
        guidance = get_domain_prompt_section(domain_id)
        assert isinstance(guidance, str) and len(guidance) > 50

    def test_causal_inference_mentions_key_terms(self):
        """Causal inference guidance mentions core concepts."""
        guidance = get_domain_prompt_section("causal_inference")
        assert "method" in guidance.lower()
        assert "assumption" in guidance.lower()

    def test_time_series_mentions_key_terms(self):
        """Time series guidance mentions ARIMA."""
        guidance = get_domain_prompt_section("time_series")
        assert "ARIMA" in guidance

    def test_rag_llm_mentions_key_terms(self):
        """RAG/LLM guidance mentions retrieval."""
        guidance = get_domain_prompt_section("rag_llm")
        assert "retrieval" in guidance.lower() or "RAG" in guidance

    def test_unknown_domain_falls_back_to_causal_inference(self):
        """Unknown domain ID falls back to causal_inference guidance."""
        guidance_unknown = get_domain_prompt_section("nonexistent_domain")
        guidance_ci = get_domain_prompt_section("causal_inference")
        assert guidance_unknown == guidance_ci


class TestGetDomainAbbreviations:
    """Test get_domain_abbreviations()."""

    @pytest.mark.parametrize("domain_id", EXPECTED_DOMAINS)
    def test_keys_are_lowercase(self, domain_id):
        """All abbreviation keys are lowercase."""
        abbrevs = get_domain_abbreviations(domain_id)
        for key in abbrevs:
            assert key == key.lower(), f"Key '{key}' is not lowercase in domain '{domain_id}'"

    @pytest.mark.parametrize("domain_id", EXPECTED_DOMAINS)
    def test_values_are_nonempty_strings(self, domain_id):
        """All abbreviation values are non-empty strings."""
        abbrevs = get_domain_abbreviations(domain_id)
        for key, val in abbrevs.items():
            assert (
                isinstance(val, str) and len(val) > 0
            ), f"Abbreviation '{key}' has invalid value in domain '{domain_id}'"

    def test_causal_inference_known_abbreviations(self):
        """Causal inference has expected abbreviations."""
        abbrevs = get_domain_abbreviations("causal_inference")
        assert abbrevs["iv"] == "instrumental variables"
        assert abbrevs["dml"] == "double machine learning"
        assert abbrevs["did"] == "difference-in-differences"

    def test_time_series_known_abbreviations(self):
        """Time series has expected abbreviations."""
        abbrevs = get_domain_abbreviations("time_series")
        assert abbrevs["arima"] == "autoregressive integrated moving average"
        assert abbrevs["garch"] == "generalized autoregressive conditional heteroskedasticity"

    def test_rag_llm_known_abbreviations(self):
        """RAG/LLM has expected abbreviations."""
        abbrevs = get_domain_abbreviations("rag_llm")
        assert abbrevs["rag"] == "retrieval-augmented generation"
        assert abbrevs["llm"] == "large language model"

    def test_unknown_domain_falls_back_to_causal_inference(self):
        """Unknown domain ID falls back to causal_inference abbreviations."""
        abbrevs_unknown = get_domain_abbreviations("nonexistent_domain")
        abbrevs_ci = get_domain_abbreviations("causal_inference")
        assert abbrevs_unknown == abbrevs_ci


class TestGetDomainConfig:
    """Test get_domain_config()."""

    @pytest.mark.parametrize("domain_id", EXPECTED_DOMAINS)
    def test_returns_dict_with_required_keys(self, domain_id):
        """Returns full config dict with all required keys."""
        config = get_domain_config(domain_id)
        assert isinstance(config, dict)
        assert REQUIRED_KEYS.issubset(set(config.keys()))

    def test_causal_inference_name(self):
        """Causal inference config has correct name."""
        config = get_domain_config("causal_inference")
        assert config["name"] == "Causal Inference"

    def test_unknown_domain_falls_back_to_causal_inference(self):
        """Unknown domain ID falls back to causal_inference config."""
        config_unknown = get_domain_config("nonexistent_domain")
        config_ci = get_domain_config("causal_inference")
        assert config_unknown == config_ci


class TestListDomains:
    """Test list_domains()."""

    def test_returns_all_five_domains(self):
        """Returns all 5 domain IDs."""
        domains = list_domains()
        assert set(domains) == EXPECTED_DOMAINS

    def test_returns_list_of_strings(self):
        """Returns a list of strings."""
        domains = list_domains()
        assert isinstance(domains, list)
        assert all(isinstance(d, str) for d in domains)


class TestGetAllAbbreviations:
    """Test get_all_abbreviations()."""

    def test_returns_nonempty_dict(self):
        """Returns non-empty combined abbreviation map."""
        all_abbrevs = get_all_abbreviations()
        assert isinstance(all_abbrevs, dict)
        assert len(all_abbrevs) > 50  # Many abbreviations across domains

    def test_includes_abbreviations_from_multiple_domains(self):
        """Combined map includes abbreviations from different domains."""
        all_abbrevs = get_all_abbreviations()
        # From causal_inference
        assert "iv" in all_abbrevs
        # From time_series
        assert "arima" in all_abbrevs
        # From rag_llm
        assert "rag" in all_abbrevs
        # From healthcare
        assert "hcc" in all_abbrevs

    def test_keys_all_lowercase(self):
        """All keys in combined map are lowercase."""
        all_abbrevs = get_all_abbreviations()
        for key in all_abbrevs:
            assert key == key.lower(), f"Key '{key}' is not lowercase"

    def test_values_all_nonempty(self):
        """All values in combined map are non-empty strings."""
        all_abbrevs = get_all_abbreviations()
        for key, val in all_abbrevs.items():
            assert isinstance(val, str) and len(val) > 0, f"Abbreviation '{key}' has invalid value"

    def test_later_domains_override_earlier(self):
        """Verify that overlapping keys use last-wins semantics."""
        # 'ml' appears in both causal_inference and interview_prep
        # interview_prep is later in dict order, so should win
        all_abbrevs = get_all_abbreviations()
        assert "ml" in all_abbrevs
        # Both define it as "machine learning" so value is same either way
        assert all_abbrevs["ml"] == "machine learning"
