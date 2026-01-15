"""Domain-specific prompt configurations for concept extraction.

Provides:
- DOMAIN_PROMPTS registry with per-domain concept type guidance
- Domain-specific abbreviation maps for deduplication
- Helper functions to retrieve domain configuration

Each domain specifies:
- name: Human-readable domain name
- description: What the domain covers
- concept_type_guidance: LLM prompt section for concept types
- examples: Example relationships for the domain
- abbreviations: Abbreviation -> expansion mapping for deduplication
"""

from typing import Any


# Registry of domain-specific configurations
DOMAIN_PROMPTS: dict[str, dict[str, Any]] = {
    "healthcare": {
        "name": "Healthcare Analytics",
        "description": "Healthcare ML, risk adjustment, population health, and clinical outcomes",
        "concept_type_guidance": """Healthcare analytics (uses causal inference foundations):
1. method: Statistical/ML methods applied to healthcare (e.g., risk adjustment, survival analysis, propensity matching)
2. assumption: Clinical/methodological assumptions (e.g., no unmeasured confounding, proportional hazards, missing at random)
3. problem: Healthcare challenges (e.g., selection bias, immortal time bias, indication bias, claims data limitations)
4. definition: Healthcare-specific definitions (e.g., risk score, comorbidity index, quality metric, episode of care)
5. theorem: Formal results applicable to healthcare (e.g., causal identification, bounds on effects)

Broader concepts (use when more appropriate):
6. concept: General healthcare concepts (e.g., care pathway, population health, value-based care)
7. principle: Healthcare delivery principles (e.g., prevention, early intervention, care coordination)
8. technique: Applied techniques (e.g., risk stratification, cohort matching, chart review)
9. model: Predictive/risk models (e.g., HCC model, Charlson index, Elixhauser, mortality prediction)""",
        "examples": [
            "HCC risk adjustment uses diagnosis codes to predict costs",
            "Propensity matching addresses selection bias in observational studies",
            "Immortal time bias affects survival analysis in claims data",
            "Charlson index aggregates comorbidities into a single score",
            "HEDIS measures quality of care delivery",
            "Episode grouping aggregates claims into treatment episodes",
        ],
        "abbreviations": {
            # Coding systems
            "icd": "international classification of diseases",
            "icd-10": "international classification of diseases 10th revision",
            "icd-10-cm": "international classification of diseases 10th revision clinical modification",
            "cpt": "current procedural terminology",
            "hcpcs": "healthcare common procedure coding system",
            "drg": "diagnosis-related group",
            "ms-drg": "medicare severity diagnosis-related group",
            "apr-drg": "all patient refined diagnosis-related group",
            "ndc": "national drug code",
            "rxnorm": "rx normalized",
            "snomed": "systematized nomenclature of medicine",
            "loinc": "logical observation identifiers names and codes",
            # Risk adjustment
            "hcc": "hierarchical condition categories",
            "cms-hcc": "centers for medicare and medicaid services hierarchical condition categories",
            "raf": "risk adjustment factor",
            "rxhcc": "prescription drug hierarchical condition categories",
            "cdps": "chronic illness and disability payment system",
            "dcg": "diagnostic cost groups",
            "acg": "adjusted clinical groups",
            # Organizations/programs
            "cms": "centers for medicare and medicaid services",
            "aco": "accountable care organization",
            "mco": "managed care organization",
            "pcp": "primary care provider",
            "pbm": "pharmacy benefit manager",
            "hmo": "health maintenance organization",
            "ppo": "preferred provider organization",
            "ffs": "fee for service",
            "vbp": "value-based payment",
            "vbc": "value-based care",
            "mssp": "medicare shared savings program",
            # Quality measures
            "hedis": "healthcare effectiveness data and information set",
            "cahps": "consumer assessment of healthcare providers and systems",
            "nqf": "national quality forum",
            "mips": "merit-based incentive payment system",
            "star": "star ratings",
            # Clinical/outcome
            "los": "length of stay",
            "alos": "average length of stay",
            "readmission": "hospital readmission",
            "ed": "emergency department",
            "ip": "inpatient",
            "op": "outpatient",
            "snf": "skilled nursing facility",
            "ltc": "long-term care",
            "hospice": "hospice care",
            "palliative": "palliative care",
            # Social determinants
            "sdoh": "social determinants of health",
            "adi": "area deprivation index",
            "svi": "social vulnerability index",
            # Statistical/ML
            "rvu": "relative value unit",
            "pmpm": "per member per month",
            "pppy": "per patient per year",
            "nnt": "number needed to treat",
            "nnh": "number needed to harm",
            "or": "odds ratio",
            "rr": "relative risk",
            "hr": "hazard ratio",
            "ci": "confidence interval",
            "auc": "area under the curve",
            "auroc": "area under receiver operating characteristic",
            "ppv": "positive predictive value",
            "npv": "negative predictive value",
            "sens": "sensitivity",
            "spec": "specificity",
        },
    },
    "causal_inference": {
        "name": "Causal Inference",
        "description": "Econometrics, causal ML, and treatment effect estimation",
        "concept_type_guidance": """Causal inference / econometrics:
1. method: Statistical/ML methods (e.g., IV, DiD, DML, matching, regression)
2. assumption: Required conditions for validity (e.g., parallel trends, unconfoundedness, SUTVA)
3. problem: Issues methods address (e.g., endogeneity, selection bias, confounding)
4. definition: Formal definitions of terms (e.g., ATE, CATE, LATE)
5. theorem: Formal mathematical results (e.g., Neyman orthogonality, CLT)

Broader concepts (use when more appropriate):
6. concept: General concepts not fitting other types (e.g., ergodicity, entropy, causality)
7. principle: Foundational principles (e.g., superposition, Occam's razor, conservation laws)
8. technique: Applied techniques that aren't full methods (e.g., cross-validation, bootstrap, regularization)
9. model: Formal models/architectures (e.g., DAG, SCM, transformer, LSTM)""",
        "examples": [
            "IV requires relevance, exclusion, exogeneity",
            "DiD assumes parallel trends",
            "DML addresses regularization bias",
            "Matching requires unconfoundedness",
            "RDD exploits discontinuity at threshold",
        ],
        "abbreviations": {
            "iv": "instrumental variables",
            "ivs": "instrumental variables",
            "2sls": "two-stage least squares",
            "tsls": "two-stage least squares",
            "did": "difference-in-differences",
            "dd": "difference-in-differences",
            "diff-in-diff": "difference-in-differences",
            "rdd": "regression discontinuity design",
            "rd": "regression discontinuity",
            "psm": "propensity score matching",
            "ate": "average treatment effect",
            "att": "average treatment effect on the treated",
            "atc": "average treatment effect on the controls",
            "atu": "average treatment effect on the untreated",
            "late": "local average treatment effect",
            "cate": "conditional average treatment effect",
            "itt": "intention to treat",
            "toa": "treatment on the treated",
            "ols": "ordinary least squares",
            "gls": "generalized least squares",
            "gmm": "generalized method of moments",
            "ml": "machine learning",
            "dml": "double machine learning",
            "lasso": "least absolute shrinkage and selection operator",
            "rf": "random forest",
            "gbm": "gradient boosting machine",
            "xgboost": "extreme gradient boosting",
            "dag": "directed acyclic graph",
            "scm": "structural causal model",
            "sem": "structural equation model",
            "rct": "randomized controlled trial",
            "fe": "fixed effects",
            "re": "random effects",
            "cia": "conditional independence assumption",
            "sutva": "stable unit treatment value assumption",
            "nuc": "no unmeasured confounding",
        },
    },
    "time_series": {
        "name": "Time Series",
        "description": "Forecasting, temporal modeling, and sequential data analysis",
        "concept_type_guidance": """Time series / forecasting:
1. method: Forecasting techniques (e.g., ARIMA, exponential smoothing, Prophet, VAR)
2. assumption: Model assumptions (e.g., stationarity, ergodicity, white noise residuals)
3. problem: Forecasting challenges (e.g., non-stationarity, seasonality, structural breaks)
4. definition: Key quantities (e.g., autocorrelation, partial ACF, spectral density)
5. theorem: Theoretical results (e.g., Wold decomposition, Box-Jenkins procedure)
6. model: Model architectures (e.g., SARIMA, GARCH, VAR, state-space, LSTM)
7. technique: Preprocessing steps (e.g., differencing, log transform, seasonal adjustment)

Broader concepts (use when more appropriate):
8. concept: General concepts (e.g., trend, cycle, seasonality, noise)
9. principle: Foundational principles (e.g., parsimony, stationarity principle)""",
        "examples": [
            "ARIMA requires stationarity",
            "GARCH models volatility clustering",
            "Exponential smoothing uses weighted averaging",
            "VAR captures multivariate dependencies",
            "State-space models unify ARIMA and exponential smoothing",
            "Prophet handles multiple seasonalities",
        ],
        "abbreviations": {
            "arima": "autoregressive integrated moving average",
            "sarima": "seasonal autoregressive integrated moving average",
            "arma": "autoregressive moving average",
            "ar": "autoregressive",
            "ma": "moving average",
            "garch": "generalized autoregressive conditional heteroskedasticity",
            "arch": "autoregressive conditional heteroskedasticity",
            "var": "vector autoregression",
            "vecm": "vector error correction model",
            "acf": "autocorrelation function",
            "pacf": "partial autocorrelation function",
            "adf": "augmented dickey-fuller",
            "kpss": "kwiatkowski-phillips-schmidt-shin",
            "pp": "phillips-perron",
            "aic": "akaike information criterion",
            "bic": "bayesian information criterion",
            "mse": "mean squared error",
            "rmse": "root mean squared error",
            "mae": "mean absolute error",
            "mape": "mean absolute percentage error",
            "smape": "symmetric mean absolute percentage error",
            "ets": "error trend seasonality",
            "stl": "seasonal and trend decomposition using loess",
            "lstm": "long short-term memory",
            "rnn": "recurrent neural network",
            "gru": "gated recurrent unit",
            "dft": "discrete fourier transform",
            "fft": "fast fourier transform",
            "ewma": "exponentially weighted moving average",
            "ses": "simple exponential smoothing",
            "hw": "holt-winters",
        },
    },
    "rag_llm": {
        "name": "RAG & LLM",
        "description": "Retrieval-augmented generation, language models, prompting, embeddings",
        "concept_type_guidance": """RAG and Large Language Models:
1. method: Retrieval techniques (e.g., dense retrieval, sparse retrieval, hybrid search, reranking)
2. technique: Prompting strategies (e.g., chain-of-thought, few-shot, self-consistency, tree-of-thought)
3. model: Architectures (e.g., transformer, attention, decoder-only, encoder-decoder)
4. concept: Theoretical ideas (e.g., in-context learning, emergent abilities, scaling laws)
5. problem: Challenges (e.g., hallucination, context window limits, latency, catastrophic forgetting)
6. definition: Key terms (e.g., embedding, tokenization, fine-tuning, alignment)

Broader concepts (use when more appropriate):
7. assumption: Implicit assumptions (e.g., IID data, task transfer)
8. theorem: Theoretical results (e.g., universal approximation, attention complexity)
9. principle: Design principles (e.g., RLHF, constitutional AI, instruction tuning)""",
        "examples": [
            "dense retrieval USES embedding similarity",
            "chain-of-thought EXTENDS few-shot prompting",
            "RAG ADDRESSES hallucination",
            "RLHF USES reward modeling",
            "transformer REQUIRES attention mechanism",
            "in-context learning EXHIBITS emergent abilities",
        ],
        "abbreviations": {
            "rag": "retrieval-augmented generation",
            "llm": "large language model",
            "llms": "large language models",
            "slm": "small language model",
            "vlm": "vision language model",
            "cot": "chain-of-thought",
            "icl": "in-context learning",
            "rlhf": "reinforcement learning from human feedback",
            "dpo": "direct preference optimization",
            "sft": "supervised fine-tuning",
            "ppo": "proximal policy optimization",
            "kto": "kahneman-tversky optimization",
            "peft": "parameter-efficient fine-tuning",
            "lora": "low-rank adaptation",
            "qlora": "quantized low-rank adaptation",
            "moe": "mixture of experts",
            "kv": "key-value",
            "mlp": "multi-layer perceptron",
            "ffn": "feed-forward network",
            "gpt": "generative pre-trained transformer",
            "bert": "bidirectional encoder representations from transformers",
            "bpe": "byte pair encoding",
            "qkv": "query key value",
            "mha": "multi-head attention",
            "gqa": "grouped query attention",
            "mqa": "multi-query attention",
            "rope": "rotary position embedding",
            "rmsnorm": "root mean square normalization",
            "ce": "cross-entropy",
            "nll": "negative log-likelihood",
            "ppl": "perplexity",
            "flops": "floating point operations per second",
            "tflops": "tera floating point operations per second",
            "ctx": "context",
            "tok": "token",
            "emb": "embedding",
            "sim": "similarity",
        },
    },
}


def get_domain_prompt_section(domain_id: str) -> str:
    """Get domain-specific concept type guidance for injection into extraction prompt.

    Args:
        domain_id: Domain identifier (e.g., 'causal_inference', 'time_series')

    Returns:
        Concept type guidance string for the domain

    Example:
        >>> guidance = get_domain_prompt_section("time_series")
        >>> "ARIMA" in guidance
        True
    """
    config = DOMAIN_PROMPTS.get(domain_id, DOMAIN_PROMPTS["causal_inference"])
    return config["concept_type_guidance"]


def get_domain_abbreviations(domain_id: str) -> dict[str, str]:
    """Get domain-specific abbreviation map for deduplication.

    Args:
        domain_id: Domain identifier

    Returns:
        Dict mapping lowercase abbreviations to full expansions

    Example:
        >>> abbrevs = get_domain_abbreviations("time_series")
        >>> abbrevs["arima"]
        'autoregressive integrated moving average'
    """
    config = DOMAIN_PROMPTS.get(domain_id, DOMAIN_PROMPTS["causal_inference"])
    return config.get("abbreviations", {})


def get_domain_config(domain_id: str) -> dict[str, Any]:
    """Get full domain configuration.

    Args:
        domain_id: Domain identifier

    Returns:
        Complete domain configuration dict

    Example:
        >>> config = get_domain_config("causal_inference")
        >>> config["name"]
        'Causal Inference'
    """
    return DOMAIN_PROMPTS.get(domain_id, DOMAIN_PROMPTS["causal_inference"])


def list_domains() -> list[str]:
    """List all available domain IDs.

    Returns:
        List of domain identifiers

    Example:
        >>> domains = list_domains()
        >>> "causal_inference" in domains
        True
    """
    return list(DOMAIN_PROMPTS.keys())


def get_all_abbreviations() -> dict[str, str]:
    """Get combined abbreviation map from all domains.

    Useful for cross-domain deduplication. In case of conflicts,
    later domains override earlier ones.

    Returns:
        Combined abbreviation dict from all domains
    """
    combined = {}
    for domain_config in DOMAIN_PROMPTS.values():
        combined.update(domain_config.get("abbreviations", {}))
    return combined
