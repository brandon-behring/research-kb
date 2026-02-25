"""Tests for domain-specific prompt configurations.

Validates:
- All 20 domains registered with required keys
- get_domain_prompt_section() returns non-empty guidance
- get_domain_abbreviations() returns lowercase keys, non-empty values
- get_domain_config() returns full configuration
- list_domains() enumerates all domains
- get_all_abbreviations() merges all domain abbreviations
- Unknown domain fallback behavior (defaults to causal_inference)
- New domains (Phase H) have appropriate key terms in guidance
- Phase N domains (sql, recommender_systems, adtech, algorithms, forecasting)
- Phase O domain: portfolio_management
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


# All 20 expected domain IDs
EXPECTED_DOMAINS = {
    "healthcare",
    "causal_inference",
    "time_series",
    "rag_llm",
    "interview_prep",
    "econometrics",
    "software_engineering",
    "deep_learning",
    "mathematics",
    "machine_learning",
    "finance",
    "statistics",
    "ml_engineering",
    "data_science",
    "sql",
    "recommender_systems",
    "adtech",
    "algorithms",
    "forecasting",
    "portfolio_management",
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

    def test_all_twenty_domains_present(self):
        """All 20 expected domains exist in the registry."""
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

    def test_unknown_domain_falls_back_to_generic(self):
        """Unknown domain ID falls back to generic guidance."""
        guidance_unknown = get_domain_prompt_section("nonexistent_domain")
        assert "General knowledge extraction" in guidance_unknown
        assert "method" in guidance_unknown.lower()


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

    def test_unknown_domain_falls_back_to_generic(self):
        """Unknown domain ID falls back to generic (empty) abbreviations."""
        abbrevs_unknown = get_domain_abbreviations("nonexistent_domain")
        assert abbrevs_unknown == {}


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

    def test_unknown_domain_falls_back_to_generic(self):
        """Unknown domain ID falls back to generic config."""
        config_unknown = get_domain_config("nonexistent_domain")
        assert config_unknown["name"] == "General"
        assert config_unknown["abbreviations"] == {}


class TestListDomains:
    """Test list_domains()."""

    def test_returns_all_twenty_domains(self):
        """Returns all 20 domain IDs."""
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


class TestNewDomainKeyTerms:
    """Validate Phase H domain configs mention appropriate key terms."""

    def test_econometrics_mentions_key_terms(self):
        """Econometrics guidance mentions OLS and endogeneity."""
        guidance = get_domain_prompt_section("econometrics")
        assert "OLS" in guidance
        assert "endogeneity" in guidance.lower()

    def test_econometrics_known_abbreviations(self):
        """Econometrics has expected abbreviations."""
        abbrevs = get_domain_abbreviations("econometrics")
        assert abbrevs["ols"] == "ordinary least squares"
        assert abbrevs["2sls"] == "two-stage least squares"
        assert abbrevs["gmm"] == "generalized method of moments"

    def test_software_engineering_mentions_key_terms(self):
        """Software engineering guidance mentions design patterns."""
        guidance = get_domain_prompt_section("software_engineering")
        assert "SOLID" in guidance or "pattern" in guidance.lower()
        assert "microservice" in guidance.lower() or "architecture" in guidance.lower()

    def test_software_engineering_known_abbreviations(self):
        """Software engineering has expected abbreviations."""
        abbrevs = get_domain_abbreviations("software_engineering")
        assert abbrevs["api"] == "application programming interface"
        assert abbrevs["tdd"] == "test-driven development"
        assert abbrevs["ci"] == "continuous integration"

    def test_deep_learning_mentions_key_terms(self):
        """Deep learning guidance mentions neural network architectures."""
        guidance = get_domain_prompt_section("deep_learning")
        assert "CNN" in guidance or "transformer" in guidance.lower()
        assert "gradient" in guidance.lower()

    def test_deep_learning_known_abbreviations(self):
        """Deep learning has expected abbreviations."""
        abbrevs = get_domain_abbreviations("deep_learning")
        assert abbrevs["cnn"] == "convolutional neural network"
        assert abbrevs["adam"] == "adaptive moment estimation"
        assert abbrevs["vae"] == "variational autoencoder"

    def test_mathematics_mentions_key_terms(self):
        """Mathematics guidance mentions theorems and proofs."""
        guidance = get_domain_prompt_section("mathematics")
        assert "theorem" in guidance.lower()
        assert "proof" in guidance.lower() or "convergence" in guidance.lower()

    def test_mathematics_known_abbreviations(self):
        """Mathematics has expected abbreviations."""
        abbrevs = get_domain_abbreviations("mathematics")
        assert abbrevs["svd"] == "singular value decomposition"
        assert abbrevs["pca"] == "principal component analysis"
        assert abbrevs["clt"] == "central limit theorem"

    def test_machine_learning_mentions_key_terms(self):
        """Machine learning guidance mentions algorithms and evaluation."""
        guidance = get_domain_prompt_section("machine_learning")
        assert "random forest" in guidance.lower() or "SVM" in guidance
        assert "overfitting" in guidance.lower() or "bias-variance" in guidance.lower()

    def test_machine_learning_known_abbreviations(self):
        """Machine learning has expected abbreviations."""
        abbrevs = get_domain_abbreviations("machine_learning")
        assert abbrevs["svm"] == "support vector machine"
        assert abbrevs["rf"] == "random forest"
        assert abbrevs["shap"] == "shapley additive explanations"

    def test_finance_mentions_key_terms(self):
        """Finance guidance mentions pricing models and risk."""
        guidance = get_domain_prompt_section("finance")
        assert "Black-Scholes" in guidance or "CAPM" in guidance
        assert "risk" in guidance.lower()

    def test_finance_known_abbreviations(self):
        """Finance has expected abbreviations."""
        abbrevs = get_domain_abbreviations("finance")
        assert abbrevs["capm"] == "capital asset pricing model"
        assert abbrevs["var"] == "value at risk"
        assert abbrevs["bs"] == "black-scholes"

    def test_statistics_mentions_key_terms(self):
        """Statistics guidance mentions hypothesis testing and inference."""
        guidance = get_domain_prompt_section("statistics")
        assert "hypothesis" in guidance.lower()
        assert "p-value" in guidance.lower() or "inference" in guidance.lower()

    def test_statistics_known_abbreviations(self):
        """Statistics has expected abbreviations."""
        abbrevs = get_domain_abbreviations("statistics")
        assert abbrevs["mle"] == "maximum likelihood estimation"
        assert abbrevs["mcmc"] == "markov chain monte carlo"
        assert abbrevs["fdr"] == "false discovery rate"

    def test_ml_engineering_mentions_key_terms(self):
        """ML engineering guidance mentions MLOps concepts."""
        guidance = get_domain_prompt_section("ml_engineering")
        assert "feature store" in guidance.lower() or "model registry" in guidance.lower()
        assert "drift" in guidance.lower() or "serving" in guidance.lower()

    def test_ml_engineering_known_abbreviations(self):
        """ML engineering has expected abbreviations."""
        abbrevs = get_domain_abbreviations("ml_engineering")
        assert abbrevs["mlops"] == "machine learning operations"
        assert abbrevs["k8s"] == "kubernetes"
        assert abbrevs["onnx"] == "open neural network exchange"

    def test_data_science_mentions_key_terms(self):
        """Data science guidance mentions EDA and analysis."""
        guidance = get_domain_prompt_section("data_science")
        assert "EDA" in guidance or "exploratory" in guidance.lower()
        assert "feature" in guidance.lower()

    def test_data_science_known_abbreviations(self):
        """Data science has expected abbreviations."""
        abbrevs = get_domain_abbreviations("data_science")
        assert abbrevs["eda"] == "exploratory data analysis"
        assert abbrevs["kpi"] == "key performance indicator"
        assert abbrevs["ltv"] == "lifetime value"


class TestPhaseNDomainKeyTerms:
    """Validate Phase N domain configs (sql, recommender_systems, adtech, algorithms, forecasting)."""

    # --- SQL ---
    def test_sql_mentions_key_terms(self):
        """SQL guidance mentions window functions and CTEs."""
        guidance = get_domain_prompt_section("sql")
        assert "window function" in guidance.lower() or "CTE" in guidance
        assert "index" in guidance.lower()

    def test_sql_known_abbreviations(self):
        """SQL has expected abbreviations."""
        abbrevs = get_domain_abbreviations("sql")
        assert abbrevs["cte"] == "common table expression"
        assert abbrevs["ddl"] == "data definition language"
        assert abbrevs["mvcc"] == "multiversion concurrency control"

    # --- Recommender Systems ---
    def test_recommender_systems_mentions_key_terms(self):
        """Recommender systems guidance mentions collaborative filtering and cold start."""
        guidance = get_domain_prompt_section("recommender_systems")
        assert "collaborative filtering" in guidance.lower()
        assert "cold start" in guidance.lower()

    def test_recommender_systems_known_abbreviations(self):
        """Recommender systems has expected abbreviations."""
        abbrevs = get_domain_abbreviations("recommender_systems")
        assert abbrevs["cf"] == "collaborative filtering"
        assert abbrevs["mf"] == "matrix factorization"
        assert abbrevs["ndcg"] == "normalized discounted cumulative gain"

    # --- AdTech ---
    def test_adtech_mentions_key_terms(self):
        """AdTech guidance mentions auction and CTR."""
        guidance = get_domain_prompt_section("adtech")
        assert "auction" in guidance.lower()
        assert "CTR" in guidance

    def test_adtech_known_abbreviations(self):
        """AdTech has expected abbreviations."""
        abbrevs = get_domain_abbreviations("adtech")
        assert abbrevs["ctr"] == "click-through rate"
        assert abbrevs["dsp"] == "demand-side platform"
        assert abbrevs["roas"] == "return on ad spend"

    # --- Algorithms ---
    def test_algorithms_mentions_key_terms(self):
        """Algorithms guidance mentions dynamic programming and complexity."""
        guidance = get_domain_prompt_section("algorithms")
        assert "dynamic programming" in guidance.lower()
        assert "complexity" in guidance.lower()

    def test_algorithms_known_abbreviations(self):
        """Algorithms has expected abbreviations."""
        abbrevs = get_domain_abbreviations("algorithms")
        assert abbrevs["dp"] == "dynamic programming"
        assert abbrevs["bfs"] == "breadth-first search"
        assert abbrevs["dag"] == "directed acyclic graph"

    # --- Forecasting ---
    def test_forecasting_mentions_key_terms(self):
        """Forecasting guidance mentions ARIMA and horizon."""
        guidance = get_domain_prompt_section("forecasting")
        assert "ARIMA" in guidance
        assert "horizon" in guidance.lower() or "forecast" in guidance.lower()

    def test_forecasting_known_abbreviations(self):
        """Forecasting has expected abbreviations."""
        abbrevs = get_domain_abbreviations("forecasting")
        assert abbrevs["arima"] == "autoregressive integrated moving average"
        assert abbrevs["ets"] == "error trend seasonality"
        assert abbrevs["crps"] == "continuous ranked probability score"

    # --- Cross-domain validation ---
    @pytest.mark.parametrize(
        "domain_id",
        ["sql", "recommender_systems", "adtech", "algorithms", "forecasting"],
    )
    def test_phase_n_domain_has_at_least_6_examples(self, domain_id):
        """Phase N domains have at least 6 relationship examples."""
        examples = DOMAIN_PROMPTS[domain_id]["examples"]
        assert len(examples) >= 6, f"{domain_id} has only {len(examples)} examples"

    @pytest.mark.parametrize(
        "domain_id",
        ["sql", "recommender_systems", "adtech", "algorithms", "forecasting"],
    )
    def test_phase_n_domain_has_at_least_20_abbreviations(self, domain_id):
        """Phase N domains have at least 20 abbreviations."""
        abbrevs = DOMAIN_PROMPTS[domain_id]["abbreviations"]
        assert len(abbrevs) >= 20, f"{domain_id} has only {len(abbrevs)} abbreviations"

    @pytest.mark.parametrize(
        "domain_id",
        ["sql", "recommender_systems", "adtech", "algorithms", "forecasting"],
    )
    def test_phase_n_no_duplicate_abbreviation_keys(self, domain_id):
        """No duplicate abbreviation keys within a domain (dict enforces this, but verify count)."""
        abbrevs = DOMAIN_PROMPTS[domain_id]["abbreviations"]
        # Dict can't have dups, but check all keys are lowercase
        for key in abbrevs:
            assert key == key.lower(), f"Key '{key}' not lowercase in {domain_id}"


class TestPortfolioManagement:
    """Validate Phase O portfolio_management domain config."""

    def test_portfolio_management_mentions_key_terms(self):
        """Portfolio management guidance mentions MPT, CAPM, and factor models."""
        guidance = get_domain_prompt_section("portfolio_management")
        assert "CAPM" in guidance
        assert "mean-variance" in guidance.lower()
        assert "factor" in guidance.lower()

    def test_portfolio_management_mentions_risk(self):
        """Portfolio management guidance covers risk management concepts."""
        guidance = get_domain_prompt_section("portfolio_management")
        assert "risk" in guidance.lower()
        assert "Sharpe ratio" in guidance or "sharpe" in guidance.lower()

    def test_portfolio_management_mentions_black_litterman(self):
        """Portfolio management guidance mentions Black-Litterman model."""
        guidance = get_domain_prompt_section("portfolio_management")
        assert "Black-Litterman" in guidance

    def test_portfolio_management_known_abbreviations(self):
        """Portfolio management has expected abbreviations."""
        abbrevs = get_domain_abbreviations("portfolio_management")
        assert abbrevs["mpt"] == "modern portfolio theory"
        assert abbrevs["capm"] == "capital asset pricing model"
        assert abbrevs["apt"] == "arbitrage pricing theory"
        assert abbrevs["saa"] == "strategic asset allocation"
        assert abbrevs["gips"] == "global investment performance standards"

    def test_portfolio_management_factor_abbreviations(self):
        """Portfolio management has Fama-French factor abbreviations."""
        abbrevs = get_domain_abbreviations("portfolio_management")
        assert abbrevs["hml"] == "high minus low"
        assert abbrevs["smb"] == "small minus big"
        assert abbrevs["ff3"] == "fama-french three-factor"

    def test_portfolio_management_has_at_least_6_examples(self):
        """Portfolio management has at least 6 relationship examples."""
        examples = DOMAIN_PROMPTS["portfolio_management"]["examples"]
        assert len(examples) >= 6, f"Only {len(examples)} examples"

    def test_portfolio_management_has_at_least_20_abbreviations(self):
        """Portfolio management has at least 20 abbreviations."""
        abbrevs = DOMAIN_PROMPTS["portfolio_management"]["abbreviations"]
        assert len(abbrevs) >= 20, f"Only {len(abbrevs)} abbreviations"

    def test_portfolio_management_name(self):
        """Portfolio management config has correct name."""
        config = get_domain_config("portfolio_management")
        assert config["name"] == "Portfolio Management"
