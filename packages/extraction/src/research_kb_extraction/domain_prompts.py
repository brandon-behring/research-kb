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
    "econometrics": {
        "name": "Econometrics",
        "description": "Econometric theory, estimation, inference, and applied microeconometrics",
        "concept_type_guidance": """Econometrics:
1. method: Estimation methods (e.g., OLS, GLS, 2SLS, GMM, MLE, quantile regression, panel estimators)
2. assumption: Conditions for consistency/efficiency (e.g., exogeneity, homoskedasticity, rank condition, correct specification)
3. problem: Estimation challenges (e.g., endogeneity, multicollinearity, heteroskedasticity, serial correlation, weak instruments)
4. theorem: Asymptotic results (e.g., Gauss-Markov, CLT, LLN, delta method, Cramér-Rao bound)
5. definition: Formal quantities (e.g., estimator, standard error, t-statistic, F-statistic, R-squared)
6. model: Econometric models (e.g., linear regression, probit, logit, tobit, Heckman selection, simultaneous equations)

Broader concepts (use when more appropriate):
7. technique: Applied procedures (e.g., HAC standard errors, cluster-robust SE, bootstrap, jackknife)
8. concept: General ideas (e.g., identification, consistency, efficiency, asymptotic normality)
9. principle: Foundational principles (e.g., parsimony, Frisch-Waugh-Lovell, law of iterated expectations)""",
        "examples": [
            "OLS requires exogeneity and homoskedasticity for BLUE",
            "2SLS addresses endogeneity using instrumental variables",
            "Heteroskedasticity-robust SE relaxes constant variance assumption",
            "Hausman test compares FE vs RE estimators",
            "GMM generalizes IV estimation to overidentified models",
            "Heckman selection corrects for sample selection bias",
        ],
        "abbreviations": {
            "ols": "ordinary least squares",
            "gls": "generalized least squares",
            "wls": "weighted least squares",
            "fgls": "feasible generalized least squares",
            "2sls": "two-stage least squares",
            "3sls": "three-stage least squares",
            "gmm": "generalized method of moments",
            "mle": "maximum likelihood estimation",
            "qr": "quantile regression",
            "iv": "instrumental variables",
            "fe": "fixed effects",
            "re": "random effects",
            "hac": "heteroskedasticity and autocorrelation consistent",
            "se": "standard error",
            "aic": "akaike information criterion",
            "bic": "bayesian information criterion",
            "dw": "durbin-watson",
            "lm": "lagrange multiplier",
            "lr": "likelihood ratio",
            "wald": "wald test",
            "blue": "best linear unbiased estimator",
            "clt": "central limit theorem",
            "lln": "law of large numbers",
            "sur": "seemingly unrelated regressions",
            "var": "vector autoregression",
            "vecm": "vector error correction model",
            "arma": "autoregressive moving average",
            "garch": "generalized autoregressive conditional heteroskedasticity",
            "probit": "probit model",
            "logit": "logit model",
            "tobit": "tobit model",
            "fwl": "frisch-waugh-lovell",
            "iid": "independent and identically distributed",
            "dgp": "data generating process",
            "pdf": "probability density function",
            "cdf": "cumulative distribution function",
            "mse": "mean squared error",
            "rmse": "root mean squared error",
            "r2": "r-squared",
            "vif": "variance inflation factor",
        },
    },
    "software_engineering": {
        "name": "Software Engineering",
        "description": "Software design, architecture, patterns, testing, and development practices",
        "concept_type_guidance": """Software engineering:
1. concept: Core concepts (e.g., abstraction, encapsulation, cohesion, coupling, modularity, separation of concerns)
2. technique: Development techniques (e.g., TDD, code review, continuous integration, refactoring, pair programming)
3. model: Design patterns and architectures (e.g., MVC, microservices, event-driven, hexagonal, CQRS)
4. principle: Design principles (e.g., SOLID, DRY, YAGNI, KISS, open-closed, dependency inversion)
5. problem: Engineering challenges (e.g., technical debt, race conditions, memory leaks, scaling bottlenecks)
6. definition: Formal terms (e.g., API, interface, contract, invariant, precondition, postcondition)

Broader concepts (use when more appropriate):
7. method: Formal methodologies (e.g., Agile, Scrum, Kanban, waterfall)
8. technique: Testing strategies (e.g., unit testing, integration testing, property-based testing, fuzzing)
9. model: Data structures and algorithms when architecturally relevant""",
        "examples": [
            "SOLID principles guide class design for maintainability",
            "Microservices address monolith scaling problems",
            "Dependency injection enables testability and loose coupling",
            "Event sourcing provides complete audit trail of state changes",
            "Circuit breaker pattern prevents cascading failures",
            "Technical debt accumulates from deferred refactoring",
        ],
        "abbreviations": {
            "api": "application programming interface",
            "rest": "representational state transfer",
            "grpc": "google remote procedure call",
            "graphql": "graph query language",
            "mvc": "model view controller",
            "mvvm": "model view viewmodel",
            "cqrs": "command query responsibility segregation",
            "ddd": "domain-driven design",
            "tdd": "test-driven development",
            "bdd": "behavior-driven development",
            "ci": "continuous integration",
            "cd": "continuous deployment",
            "cicd": "continuous integration continuous deployment",
            "solid": "single responsibility open-closed liskov substitution interface segregation dependency inversion",
            "dry": "don't repeat yourself",
            "yagni": "you ain't gonna need it",
            "kiss": "keep it simple stupid",
            "orm": "object-relational mapping",
            "crud": "create read update delete",
            "rpc": "remote procedure call",
            "sdk": "software development kit",
            "sla": "service level agreement",
            "slo": "service level objective",
            "sre": "site reliability engineering",
            "k8s": "kubernetes",
            "cli": "command-line interface",
            "gui": "graphical user interface",
            "ide": "integrated development environment",
            "vcs": "version control system",
            "pr": "pull request",
            "cr": "code review",
            "lsp": "language server protocol",
            "ast": "abstract syntax tree",
            "gc": "garbage collection",
            "jit": "just-in-time compilation",
            "aot": "ahead-of-time compilation",
            "ioc": "inversion of control",
            "di": "dependency injection",
        },
    },
    "deep_learning": {
        "name": "Deep Learning",
        "description": "Neural networks, architectures, training techniques, and representation learning",
        "concept_type_guidance": """Deep learning:
1. model: Network architectures (e.g., CNN, RNN, transformer, GAN, VAE, diffusion model, U-Net)
2. technique: Training techniques (e.g., dropout, batch normalization, learning rate scheduling, gradient clipping, mixed precision)
3. concept: Theoretical ideas (e.g., backpropagation, gradient flow, representation learning, transfer learning, attention)
4. problem: Training challenges (e.g., vanishing gradients, overfitting, mode collapse, catastrophic forgetting, distribution shift)
5. definition: Key terms (e.g., loss function, activation function, weight initialization, learning rate, batch size)
6. method: Optimization algorithms (e.g., SGD, Adam, AdamW, LAMB, SAM)

Broader concepts (use when more appropriate):
7. principle: Design heuristics (e.g., residual connections, skip connections, normalization before activation)
8. theorem: Theoretical results (e.g., universal approximation, lottery ticket hypothesis, neural tangent kernel)
9. assumption: Implicit assumptions (e.g., IID training data, stationarity, smoothness of loss landscape)""",
        "examples": [
            "Residual connections address vanishing gradient in deep networks",
            "Batch normalization stabilizes training by normalizing activations",
            "Attention mechanism enables transformer to model long-range dependencies",
            "Dropout acts as implicit ensemble regularization",
            "Transfer learning leverages pretrained representations for new tasks",
            "GAN training suffers from mode collapse",
        ],
        "abbreviations": {
            "cnn": "convolutional neural network",
            "rnn": "recurrent neural network",
            "lstm": "long short-term memory",
            "gru": "gated recurrent unit",
            "gan": "generative adversarial network",
            "vae": "variational autoencoder",
            "ae": "autoencoder",
            "mlp": "multi-layer perceptron",
            "ffn": "feed-forward network",
            "bn": "batch normalization",
            "ln": "layer normalization",
            "gn": "group normalization",
            "relu": "rectified linear unit",
            "gelu": "gaussian error linear unit",
            "silu": "sigmoid linear unit",
            "swish": "swish activation",
            "sgd": "stochastic gradient descent",
            "adam": "adaptive moment estimation",
            "adamw": "adam with weight decay",
            "lr": "learning rate",
            "wd": "weight decay",
            "nll": "negative log-likelihood",
            "ce": "cross-entropy",
            "bce": "binary cross-entropy",
            "mse": "mean squared error",
            "mae": "mean absolute error",
            "kl": "kullback-leibler divergence",
            "gpu": "graphics processing unit",
            "tpu": "tensor processing unit",
            "fp16": "half-precision floating point",
            "bf16": "brain floating point 16",
            "flops": "floating point operations per second",
            "resnet": "residual network",
            "vgg": "visual geometry group network",
            "unet": "u-shaped network",
            "vit": "vision transformer",
            "clip": "contrastive language-image pretraining",
            "ddpm": "denoising diffusion probabilistic model",
            "ema": "exponential moving average",
            "ntk": "neural tangent kernel",
            "nas": "neural architecture search",
        },
    },
    "mathematics": {
        "name": "Mathematics",
        "description": "Pure and applied mathematics: linear algebra, probability, optimization, analysis",
        "concept_type_guidance": """Mathematics:
1. theorem: Named results and propositions (e.g., Bayes' theorem, spectral theorem, Bolzano-Weierstrass, dominated convergence)
2. definition: Formal definitions (e.g., metric space, eigenvalue, sigma-algebra, Hilbert space, convexity)
3. concept: Abstract ideas (e.g., convergence, continuity, orthogonality, duality, compactness)
4. method: Computational methods (e.g., Gaussian elimination, Newton's method, simplex, gradient descent, SVD)
5. principle: Mathematical principles (e.g., pigeonhole, mathematical induction, contraction mapping, minimax)
6. technique: Proof and computation techniques (e.g., epsilon-delta, diagonalization, Lagrange multipliers, change of variables)

Broader concepts (use when more appropriate):
7. model: Mathematical models/frameworks (e.g., Markov chain, random walk, linear program, stochastic process)
8. problem: Open problems or problem classes (e.g., NP-completeness, halting problem, traveling salesman)
9. assumption: Regularity conditions (e.g., differentiability, integrability, measurability, boundedness)""",
        "examples": [
            "Spectral theorem decomposes symmetric matrices into eigenvalue-eigenvector pairs",
            "Bayes' theorem relates posterior to prior and likelihood",
            "SVD provides optimal low-rank approximation",
            "Gradient descent minimizes differentiable functions iteratively",
            "Central limit theorem guarantees asymptotic normality of sample means",
            "Contraction mapping principle ensures fixed-point existence and uniqueness",
        ],
        "abbreviations": {
            "svd": "singular value decomposition",
            "pca": "principal component analysis",
            "evd": "eigenvalue decomposition",
            "qr": "qr decomposition",
            "lu": "lower-upper decomposition",
            "clt": "central limit theorem",
            "lln": "law of large numbers",
            "pdf": "probability density function",
            "cdf": "cumulative distribution function",
            "pmf": "probability mass function",
            "mgf": "moment generating function",
            "iid": "independent and identically distributed",
            "rv": "random variable",
            "ev": "expected value",
            "var": "variance",
            "cov": "covariance",
            "mle": "maximum likelihood estimation",
            "map": "maximum a posteriori",
            "kl": "kullback-leibler",
            "lp": "linear program",
            "qp": "quadratic program",
            "sdp": "semidefinite program",
            "np": "nondeterministic polynomial",
            "iff": "if and only if",
            "wlog": "without loss of generality",
            "rhs": "right-hand side",
            "lhs": "left-hand side",
            "ode": "ordinary differential equation",
            "pde": "partial differential equation",
            "sde": "stochastic differential equation",
            "bvp": "boundary value problem",
            "ivp": "initial value problem",
            "fps": "formal power series",
            "gf": "generating function",
        },
    },
    "machine_learning": {
        "name": "Machine Learning",
        "description": "Classical ML algorithms, model selection, feature engineering, and evaluation",
        "concept_type_guidance": """Machine learning:
1. method: Learning algorithms (e.g., random forest, SVM, k-NN, gradient boosting, logistic regression, naive Bayes)
2. technique: ML workflow techniques (e.g., cross-validation, hyperparameter tuning, feature selection, dimensionality reduction)
3. concept: Core ideas (e.g., bias-variance tradeoff, generalization, regularization, ensemble learning, kernel trick)
4. problem: ML challenges (e.g., overfitting, underfitting, class imbalance, curse of dimensionality, data leakage)
5. definition: Key metrics and terms (e.g., accuracy, precision, recall, F1, AUC-ROC, confusion matrix)
6. model: Model families (e.g., decision tree, linear model, kernel machine, ensemble, Bayesian model)

Broader concepts (use when more appropriate):
7. theorem: Theoretical results (e.g., no free lunch, PAC learning, VC dimension bounds)
8. principle: ML best practices (e.g., Occam's razor, train/test split, stratified sampling)
9. assumption: Model assumptions (e.g., linearity, independence, distributional assumptions)""",
        "examples": [
            "Random forest reduces variance through bagging and feature subsampling",
            "SVM maximizes margin between classes using kernel trick",
            "Cross-validation estimates generalization error without held-out test set",
            "L1 regularization induces sparsity in feature weights",
            "Gradient boosting sequentially fits residuals to reduce bias",
            "Class imbalance causes classifiers to favor majority class",
        ],
        "abbreviations": {
            "ml": "machine learning",
            "svm": "support vector machine",
            "svr": "support vector regression",
            "knn": "k-nearest neighbors",
            "rf": "random forest",
            "gbm": "gradient boosting machine",
            "xgb": "extreme gradient boosting",
            "lgbm": "light gradient boosting machine",
            "dt": "decision tree",
            "nb": "naive bayes",
            "lr": "logistic regression",
            "lda": "linear discriminant analysis",
            "pca": "principal component analysis",
            "tsne": "t-distributed stochastic neighbor embedding",
            "umap": "uniform manifold approximation and projection",
            "cv": "cross-validation",
            "auc": "area under the curve",
            "roc": "receiver operating characteristic",
            "pr": "precision-recall",
            "f1": "f1 score",
            "tp": "true positive",
            "fp": "false positive",
            "tn": "true negative",
            "fn": "false negative",
            "fpr": "false positive rate",
            "tpr": "true positive rate",
            "vc": "vapnik-chervonenkis",
            "pac": "probably approximately correct",
            "erm": "empirical risk minimization",
            "srm": "structural risk minimization",
            "rbf": "radial basis function",
            "smote": "synthetic minority oversampling technique",
            "shap": "shapley additive explanations",
            "lime": "local interpretable model-agnostic explanations",
        },
    },
    "finance": {
        "name": "Finance",
        "description": "Quantitative finance, asset pricing, risk management, and financial modeling",
        "concept_type_guidance": """Quantitative finance:
1. model: Pricing and risk models (e.g., Black-Scholes, CAPM, Fama-French, GARCH, Hull-White, copula)
2. concept: Core ideas (e.g., arbitrage, risk-neutral pricing, no-arbitrage, efficient market hypothesis, alpha, beta)
3. method: Quantitative methods (e.g., Monte Carlo simulation, bootstrap, VaR calculation, delta hedging, mean-variance optimization)
4. definition: Financial terms (e.g., Sharpe ratio, volatility, Greeks, yield curve, duration, convexity)
5. problem: Financial challenges (e.g., fat tails, model risk, liquidity risk, regime changes, correlation breakdown)
6. principle: Financial principles (e.g., diversification, risk-return tradeoff, put-call parity, law of one price)

Broader concepts (use when more appropriate):
7. technique: Implementation techniques (e.g., numerical integration, finite differences, lattice methods)
8. theorem: Theoretical results (e.g., fundamental theorem of asset pricing, Girsanov, Itô's lemma)
9. assumption: Model assumptions (e.g., log-normal returns, constant volatility, complete markets)""",
        "examples": [
            "Black-Scholes assumes geometric Brownian motion and constant volatility",
            "CAPM prices assets using systematic risk (beta) only",
            "VaR estimates maximum loss at given confidence level",
            "Fama-French extends CAPM with size and value factors",
            "Monte Carlo simulation prices path-dependent derivatives",
            "Fat tails violate normal distribution assumption in risk models",
        ],
        "abbreviations": {
            "capm": "capital asset pricing model",
            "apt": "arbitrage pricing theory",
            "bs": "black-scholes",
            "var": "value at risk",
            "cvar": "conditional value at risk",
            "es": "expected shortfall",
            "emh": "efficient market hypothesis",
            "mpt": "modern portfolio theory",
            "mvo": "mean-variance optimization",
            "etf": "exchange-traded fund",
            "ipo": "initial public offering",
            "otc": "over the counter",
            "p/e": "price to earnings",
            "eps": "earnings per share",
            "roi": "return on investment",
            "irr": "internal rate of return",
            "npv": "net present value",
            "dcf": "discounted cash flow",
            "wacc": "weighted average cost of capital",
            "ebitda": "earnings before interest taxes depreciation and amortization",
            "ytm": "yield to maturity",
            "libor": "london interbank offered rate",
            "sofr": "secured overnight financing rate",
            "cds": "credit default swap",
            "mbs": "mortgage-backed security",
            "abs": "asset-backed security",
            "cdo": "collateralized debt obligation",
            "gbm": "geometric brownian motion",
            "sde": "stochastic differential equation",
            "pde": "partial differential equation",
            "mc": "monte carlo",
            "ewma": "exponentially weighted moving average",
            "garch": "generalized autoregressive conditional heteroskedasticity",
        },
    },
    "statistics": {
        "name": "Statistics",
        "description": "Statistical theory, inference, hypothesis testing, and Bayesian methods",
        "concept_type_guidance": """Statistics:
1. method: Statistical methods (e.g., hypothesis testing, confidence intervals, bootstrap, permutation test, ANOVA, chi-squared test)
2. theorem: Theoretical results (e.g., Neyman-Pearson lemma, Rao-Blackwell, sufficiency, completeness, central limit theorem)
3. definition: Formal definitions (e.g., p-value, power, type I error, type II error, sufficient statistic, estimator)
4. concept: Core ideas (e.g., likelihood, sufficiency, ancillarity, exponential family, exchangeability, conjugacy)
5. problem: Statistical challenges (e.g., multiple testing, Simpson's paradox, ecological fallacy, survivorship bias)
6. model: Statistical models (e.g., GLM, mixed effects, hierarchical Bayes, copula, nonparametric density estimation)

Broader concepts (use when more appropriate):
7. principle: Foundational principles (e.g., likelihood principle, sufficiency principle, conditionality principle)
8. technique: Computational techniques (e.g., EM algorithm, MCMC, importance sampling, variational inference)
9. assumption: Model assumptions (e.g., normality, independence, equal variance, random sampling)""",
        "examples": [
            "Neyman-Pearson lemma gives most powerful test for simple hypotheses",
            "Bootstrap estimates sampling distribution without parametric assumptions",
            "Multiple testing inflates family-wise error rate",
            "MCMC samples from posterior distribution for Bayesian inference",
            "Sufficient statistic captures all information about parameter",
            "Simpson's paradox reverses association direction when data aggregated",
        ],
        "abbreviations": {
            "mle": "maximum likelihood estimation",
            "map": "maximum a posteriori",
            "ols": "ordinary least squares",
            "glm": "generalized linear model",
            "gam": "generalized additive model",
            "lmm": "linear mixed model",
            "glmm": "generalized linear mixed model",
            "anova": "analysis of variance",
            "manova": "multivariate analysis of variance",
            "ancova": "analysis of covariance",
            "ci": "confidence interval",
            "pi": "prediction interval",
            "se": "standard error",
            "df": "degrees of freedom",
            "ss": "sum of squares",
            "em": "expectation maximization",
            "mcmc": "markov chain monte carlo",
            "mh": "metropolis-hastings",
            "hmc": "hamiltonian monte carlo",
            "nuts": "no-u-turn sampler",
            "vi": "variational inference",
            "bic": "bayesian information criterion",
            "aic": "akaike information criterion",
            "dic": "deviance information criterion",
            "waic": "widely applicable information criterion",
            "loo": "leave-one-out",
            "fdr": "false discovery rate",
            "fwer": "family-wise error rate",
            "bh": "benjamini-hochberg",
            "ks": "kolmogorov-smirnov",
            "qq": "quantile-quantile",
            "kde": "kernel density estimation",
            "cdf": "cumulative distribution function",
            "pdf": "probability density function",
            "iid": "independent and identically distributed",
        },
    },
    "ml_engineering": {
        "name": "ML Engineering",
        "description": "ML systems, MLOps, model deployment, monitoring, and production pipelines",
        "concept_type_guidance": """ML engineering / MLOps:
1. concept: Infrastructure concepts (e.g., feature store, model registry, data pipeline, experiment tracking, model serving)
2. technique: Operational techniques (e.g., A/B testing, canary deployment, shadow mode, model monitoring, data validation)
3. problem: Production challenges (e.g., training-serving skew, data drift, concept drift, pipeline debt, model staleness)
4. principle: Engineering principles (e.g., reproducibility, idempotency, schema evolution, graceful degradation)
5. definition: Key terms (e.g., latency, throughput, SLA, batch vs streaming, online vs offline features)
6. model: System architectures (e.g., lambda architecture, feature platform, inference server, DAG orchestrator)

Broader concepts (use when more appropriate):
7. method: Specific tools/frameworks when conceptually important (e.g., Kubernetes, Airflow, MLflow, Feast)
8. technique: Optimization for serving (e.g., model quantization, distillation, pruning, ONNX export, TensorRT)
9. assumption: Implicit assumptions (e.g., data stationarity, feature availability at serving time)""",
        "examples": [
            "Feature store ensures consistent features between training and serving",
            "Training-serving skew causes silent model degradation",
            "Canary deployment limits blast radius of model updates",
            "Data drift detection monitors input distribution changes",
            "Model quantization reduces serving latency at cost of accuracy",
            "Experiment tracking enables reproducible model comparisons",
        ],
        "abbreviations": {
            "mlops": "machine learning operations",
            "etl": "extract transform load",
            "elt": "extract load transform",
            "dag": "directed acyclic graph",
            "api": "application programming interface",
            "sdk": "software development kit",
            "sla": "service level agreement",
            "slo": "service level objective",
            "sli": "service level indicator",
            "ci": "continuous integration",
            "cd": "continuous deployment",
            "k8s": "kubernetes",
            "gpu": "graphics processing unit",
            "tpu": "tensor processing unit",
            "onnx": "open neural network exchange",
            "grpc": "google remote procedure call",
            "rest": "representational state transfer",
            "p50": "50th percentile latency",
            "p99": "99th percentile latency",
            "qps": "queries per second",
            "rps": "requests per second",
            "ttl": "time to live",
            "cron": "time-based job scheduler",
            "yaml": "yet another markup language",
            "json": "javascript object notation",
            "csv": "comma-separated values",
            "blob": "binary large object",
            "s3": "simple storage service",
            "gcs": "google cloud storage",
        },
    },
    "data_science": {
        "name": "Data Science",
        "description": "Data analysis, exploratory analysis, visualization, and applied analytics",
        "concept_type_guidance": """Data science:
1. technique: Analysis techniques (e.g., EDA, feature engineering, data wrangling, outlier detection, imputation)
2. concept: Core ideas (e.g., data quality, reproducibility, storytelling with data, data governance, data lineage)
3. method: Analytical methods (e.g., cohort analysis, funnel analysis, RFM segmentation, survival analysis, clustering)
4. problem: Data challenges (e.g., missing data, selection bias, data leakage, Simpson's paradox, confounding)
5. definition: Key terms (e.g., metric, KPI, funnel, cohort, segment, feature, target variable)
6. principle: Best practices (e.g., start simple, iterate, validate assumptions, communicate uncertainty)

Broader concepts (use when more appropriate):
7. model: Analytical models when relevant (e.g., propensity model, churn model, LTV model)
8. technique: Visualization techniques (e.g., heatmap, box plot, violin plot, pair plot)
9. assumption: Analytical assumptions (e.g., representative sample, stationarity, causal vs correlational)""",
        "examples": [
            "EDA reveals data distributions and anomalies before modeling",
            "Feature engineering transforms raw data into predictive signals",
            "Cohort analysis tracks behavior of user groups over time",
            "Data leakage inflates model performance during evaluation",
            "Missing data imputation assumes specific missingness mechanisms",
            "A/B testing requires proper randomization and sample sizing",
        ],
        "abbreviations": {
            "eda": "exploratory data analysis",
            "etl": "extract transform load",
            "kpi": "key performance indicator",
            "ltv": "lifetime value",
            "cac": "customer acquisition cost",
            "rfm": "recency frequency monetary",
            "sql": "structured query language",
            "csv": "comma-separated values",
            "json": "javascript object notation",
            "api": "application programming interface",
            "ab": "a/b testing",
            "mde": "minimum detectable effect",
            "dag": "directed acyclic graph",
            "bi": "business intelligence",
            "olap": "online analytical processing",
            "dim": "dimension",
            "fk": "foreign key",
            "pk": "primary key",
            "null": "null value",
            "nan": "not a number",
            "iqr": "interquartile range",
            "pca": "principal component analysis",
            "tsne": "t-distributed stochastic neighbor embedding",
        },
    },
    "interview_prep": {
        "name": "Interview Preparation",
        "description": "ML/DS interview questions, solutions, and study materials",
        "concept_type_guidance": """Interview Prep (uses ML and data science foundations):
1. problem: Interview questions or design challenges (e.g., "Design a recommendation system")
2. solution: Approaches or solution frameworks (e.g., "Start with user-item matrix")
3. technique: Specific methods or approaches (e.g., A/B testing, feature engineering)
4. concept: General concepts relevant to interviews (e.g., trade-offs, scalability)
5. definition: Definitions of key terms (e.g., precision, recall, F1 score)
6. redflag: Common mistakes to avoid (e.g., "Don't forget to handle class imbalance")
7. principle: Best practices (e.g., "Always clarify requirements first")

Broader concepts (use when more appropriate):
8. method: Statistical/ML methods when they're the focus
9. model: Model architectures when discussing specifics""",
        "examples": [
            "system design REQUIRES scalability considerations",
            "A/B testing USES randomization",
            "feature engineering ADDRESSES data quality",
            "precision-recall tradeoff REQUIRES understanding class imbalance",
            "clarifying requirements PREVENTS scope creep",
            "edge case handling DEMONSTRATES thoroughness",
        ],
        "abbreviations": {
            # ML basics
            "ml": "machine learning",
            "dl": "deep learning",
            "nn": "neural network",
            "dnn": "deep neural network",
            "cnn": "convolutional neural network",
            "rnn": "recurrent neural network",
            "lstm": "long short-term memory",
            "gru": "gated recurrent unit",
            # Metrics
            "auc": "area under the curve",
            "roc": "receiver operating characteristic",
            "pr": "precision-recall",
            "f1": "f1 score",
            "mae": "mean absolute error",
            "mse": "mean squared error",
            "rmse": "root mean squared error",
            "mape": "mean absolute percentage error",
            "ndcg": "normalized discounted cumulative gain",
            "mrr": "mean reciprocal rank",
            # Systems
            "api": "application programming interface",
            "etl": "extract transform load",
            "dag": "directed acyclic graph",
            "sql": "structured query language",
            "nosql": "not only sql",
            "olap": "online analytical processing",
            "oltp": "online transaction processing",
            # A/B testing
            "ab": "a/b testing",
            "mde": "minimum detectable effect",
            "aa": "a/a test",
            "cuped": "controlled-experiment using pre-experiment data",
            # Recommendation
            "cf": "collaborative filtering",
            "mf": "matrix factorization",
            "als": "alternating least squares",
            "bpr": "bayesian personalized ranking",
            # Interview specific
            "ds": "data science",
            "mle": "machine learning engineer",
            "swe": "software engineer",
            "pm": "product manager",
            "tpm": "technical program manager",
        },
    },
    "sql": {
        "name": "SQL & Databases",
        "description": "SQL querying, database internals, query optimization, and relational data modeling",
        "concept_type_guidance": """SQL & databases:
1. method: Query techniques (e.g., window functions, CTEs, recursive queries, pivot, lateral join, merge/upsert)
2. assumption: Correctness conditions (e.g., referential integrity, ACID compliance, isolation level guarantees, normalization)
3. problem: Database challenges (e.g., N+1 queries, deadlocks, index bloat, write amplification, cardinality estimation errors)
4. definition: Formal terms (e.g., primary key, foreign key, index, view, materialized view, transaction, WAL)
5. theorem: Formal results (e.g., CAP theorem, serializability theory, relational algebra equivalences)

Broader concepts (use when more appropriate):
6. concept: General ideas (e.g., normalization, denormalization, sharding, partitioning, replication)
7. technique: Applied procedures (e.g., query plan analysis, index tuning, vacuum, connection pooling)
8. model: Data models and architectures (e.g., star schema, snowflake schema, EAV, document model)
9. principle: Design principles (e.g., least privilege, immutable audit logs, idempotent migrations)""",
        "examples": [
            "Window functions ENABLE ranking without self-joins",
            "CTEs SIMPLIFY recursive hierarchical queries",
            "B-tree indexes ACCELERATE range scans on ordered columns",
            "MVCC ENABLES concurrent reads without blocking writes",
            "Hash joins OUTPERFORM nested loops on large unsorted tables",
            "Partitioning ADDRESSES table scan performance on large tables",
            "Write-ahead logging ENSURES crash recovery durability",
        ],
        "abbreviations": {
            "cte": "common table expression",
            "ddl": "data definition language",
            "dml": "data manipulation language",
            "dcl": "data control language",
            "tcl": "transaction control language",
            "olap": "online analytical processing",
            "oltp": "online transaction processing",
            "etl": "extract transform load",
            "elt": "extract load transform",
            "pk": "primary key",
            "fk": "foreign key",
            "uk": "unique key",
            "mvcc": "multiversion concurrency control",
            "wal": "write-ahead log",
            "acid": "atomicity consistency isolation durability",
            "base": "basically available soft state eventual consistency",
            "cap": "consistency availability partition tolerance",
            "sql": "structured query language",
            "rdbms": "relational database management system",
            "orm": "object-relational mapping",
            "dba": "database administrator",
            "iops": "input output operations per second",
            "tps": "transactions per second",
            "qps": "queries per second",
            "eav": "entity-attribute-value",
            "json": "javascript object notation",
            "jsonb": "binary json",
            "gin": "generalized inverted index",
            "gist": "generalized search tree",
            "brin": "block range index",
            "ssi": "serializable snapshot isolation",
            "2pc": "two-phase commit",
            "lsm": "log-structured merge tree",
            "scd": "slowly changing dimension",
            "udf": "user-defined function",
            "sp": "stored procedure",
            "dw": "data warehouse",
            "dwh": "data warehouse",
        },
    },
    "recommender_systems": {
        "name": "Recommender Systems",
        "description": "Collaborative filtering, content-based methods, matrix factorization, deep recsys, and evaluation",
        "concept_type_guidance": """Recommender systems:
1. method: Recommendation algorithms (e.g., collaborative filtering, content-based filtering, matrix factorization, SVD, ALS, two-tower, NCF)
2. assumption: Model assumptions (e.g., low-rank user-item matrix, item feature availability, implicit feedback reliability, missing-at-random)
3. problem: RecSys challenges (e.g., cold start, popularity bias, filter bubbles, position bias, sparsity, scalability)
4. definition: Formal terms (e.g., user-item matrix, implicit feedback, explicit rating, interaction, embedding, candidate generation)
5. theorem: Formal results (e.g., matrix completion bounds, regret bounds for bandits, convergence of ALS)

Broader concepts (use when more appropriate):
6. concept: General ideas (e.g., serendipity, diversity, fairness, exploration-exploitation, session-based recommendation)
7. technique: Applied procedures (e.g., negative sampling, approximate nearest neighbors, feature hashing, A/B testing for recsys)
8. model: Architectures (e.g., two-tower model, wide-and-deep, DeepFM, sequence models, graph neural networks for recsys)
9. principle: Design principles (e.g., recency weighting, user-centricity, multi-objective optimization)""",
        "examples": [
            "Collaborative filtering REQUIRES sufficient user-item interactions to overcome sparsity",
            "Matrix factorization DECOMPOSES user-item matrix into latent factor embeddings",
            "Two-tower models ENABLE large-scale candidate generation via ANN retrieval",
            "BPR OPTIMIZES pairwise ranking loss for implicit feedback",
            "Cold start LIMITS collaborative filtering for new users and items",
            "Position bias CONFOUNDS click-through rate interpretation in ranking",
            "Multi-armed bandits BALANCE exploration and exploitation in online recommendation",
        ],
        "abbreviations": {
            "cf": "collaborative filtering",
            "cbf": "content-based filtering",
            "mf": "matrix factorization",
            "svd": "singular value decomposition",
            "als": "alternating least squares",
            "nmf": "non-negative matrix factorization",
            "ctr": "click-through rate",
            "ncf": "neural collaborative filtering",
            "bpr": "bayesian personalized ranking",
            "ndcg": "normalized discounted cumulative gain",
            "map": "mean average precision",
            "mrr": "mean reciprocal rank",
            "hr": "hit rate",
            "ann": "approximate nearest neighbor",
            "knn": "k-nearest neighbors",
            "dnn": "deep neural network",
            "gnn": "graph neural network",
            "vae": "variational autoencoder",
            "rl": "reinforcement learning",
            "mab": "multi-armed bandit",
            "ucb": "upper confidence bound",
            "ts": "thompson sampling",
            "ips": "inverse propensity scoring",
            "snips": "self-normalized inverse propensity scoring",
            "dae": "denoising autoencoder",
            "rnn": "recurrent neural network",
            "gru": "gated recurrent unit",
            "fm": "factorization machine",
            "ffm": "field-aware factorization machine",
            "deepfm": "deep factorization machine",
            "din": "deep interest network",
            "dien": "deep interest evolution network",
            "sasrec": "self-attentive sequential recommendation",
        },
    },
    "adtech": {
        "name": "Ads & AdTech",
        "description": "Auction mechanics, CTR prediction, bid optimization, attribution, and incrementality",
        "concept_type_guidance": """Ads & AdTech:
1. method: Optimization methods (e.g., CTR prediction, bid shading, budget pacing, uplift modeling, attribution modeling)
2. assumption: Model assumptions (e.g., independent clicks, truthful bidding, last-touch validity, SUTVA for incrementality)
3. problem: AdTech challenges (e.g., click fraud, ad fatigue, frequency capping, budget waste, attribution misallocation)
4. definition: Formal terms (e.g., impression, click, conversion, second-price auction, reserve price, quality score)
5. theorem: Formal results (e.g., revenue equivalence theorem, Myerson optimal auction, VCG mechanism properties)

Broader concepts (use when more appropriate):
6. concept: General ideas (e.g., demand-side platform, supply-side platform, real-time bidding, programmatic advertising)
7. technique: Applied procedures (e.g., A/B testing for ads, geo experiments, ghost ads, PSA holdout)
8. model: Prediction models (e.g., logistic regression for CTR, calibration models, deep CTR models, conversion delay models)
9. principle: Design principles (e.g., incentive compatibility, individual rationality, advertiser ROI optimization)""",
        "examples": [
            "Second-price auctions INCENTIVIZE truthful bidding by advertisers",
            "CTR prediction REQUIRES calibrated probabilities for bid optimization",
            "Incrementality testing MEASURES causal lift of ad exposure via holdout groups",
            "Budget pacing DISTRIBUTES daily spend evenly to avoid early exhaustion",
            "Attribution models ALLOCATE conversion credit across touchpoints",
            "Bid shading REDUCES overpayment in first-price auction environments",
            "Frequency capping PREVENTS ad fatigue while maintaining reach",
        ],
        "abbreviations": {
            "ctr": "click-through rate",
            "cvr": "conversion rate",
            "cpc": "cost per click",
            "cpm": "cost per mille",
            "cpa": "cost per acquisition",
            "cpi": "cost per install",
            "cpv": "cost per view",
            "roas": "return on ad spend",
            "roi": "return on investment",
            "ltv": "lifetime value",
            "arpu": "average revenue per user",
            "rtb": "real-time bidding",
            "dsp": "demand-side platform",
            "ssp": "supply-side platform",
            "dmp": "data management platform",
            "cdp": "customer data platform",
            "gdn": "google display network",
            "ecpm": "effective cost per mille",
            "vcg": "vickrey-clarke-groves",
            "psa": "public service announcement",
            "mta": "multi-touch attribution",
            "mmm": "marketing mix modeling",
            "ita": "intent-to-treat analysis",
            "itt": "intention to treat",
            "sutva": "stable unit treatment value assumption",
            "ate": "average treatment effect",
            "att": "average treatment effect on the treated",
            "ghb": "ghost bid",
            "qps": "queries per second",
            "bid": "bid",
            "gsp": "generalized second-price",
            "fp": "first-price",
        },
    },
    "algorithms": {
        "name": "Algorithms & Data Structures",
        "description": "Algorithm design, complexity analysis, data structures, and optimization techniques",
        "concept_type_guidance": """Algorithms & data structures:
1. method: Algorithmic techniques (e.g., dynamic programming, greedy, divide-and-conquer, backtracking, branch-and-bound, A*)
2. assumption: Correctness conditions (e.g., optimal substructure, greedy choice property, DAG for topological sort, comparison model)
3. problem: Computational problems (e.g., sorting, shortest path, minimum spanning tree, knapsack, traveling salesman, maximum flow)
4. theorem: Complexity results (e.g., master theorem, amortized bounds, NP-completeness reductions, lower bounds)
5. definition: Formal terms (e.g., time complexity, space complexity, Big-O, amortized cost, recurrence relation)

Broader concepts (use when more appropriate):
6. concept: General ideas (e.g., memoization, locality of reference, cache efficiency, online vs offline algorithms)
7. technique: Implementation techniques (e.g., two pointers, sliding window, monotonic stack, union-find with path compression)
8. model: Data structures (e.g., binary search tree, B-tree, red-black tree, skip list, trie, segment tree, Bloom filter)
9. principle: Design principles (e.g., invariant maintenance, loop invariants, problem reduction, amortized analysis)""",
        "examples": [
            "Dynamic programming REQUIRES optimal substructure and overlapping subproblems",
            "Dijkstra's algorithm ASSUMES non-negative edge weights for correctness",
            "Binary search ACHIEVES O(log n) on sorted arrays via divide-and-conquer",
            "Red-black trees GUARANTEE O(log n) insertion with balanced height invariant",
            "Union-find with path compression ACHIEVES near-constant amortized operations",
            "Topological sort REQUIRES directed acyclic graph structure",
            "NP-completeness REDUCES known hard problems to prove intractability",
        ],
        "abbreviations": {
            "dp": "dynamic programming",
            "bfs": "breadth-first search",
            "dfs": "depth-first search",
            "mst": "minimum spanning tree",
            "dag": "directed acyclic graph",
            "bst": "binary search tree",
            "avl": "adelson-velsky and landis tree",
            "rbt": "red-black tree",
            "np": "nondeterministic polynomial time",
            "p": "polynomial time",
            "lp": "linear programming",
            "ilp": "integer linear programming",
            "bt": "backtracking",
            "bnb": "branch and bound",
            "uf": "union-find",
            "dsu": "disjoint set union",
            "lru": "least recently used",
            "lfu": "least frequently used",
            "fifo": "first in first out",
            "lifo": "last in first out",
            "bigo": "big-o notation",
            "scc": "strongly connected components",
            "apsp": "all-pairs shortest path",
            "sssp": "single-source shortest path",
            "lis": "longest increasing subsequence",
            "lcs": "longest common subsequence",
            "fft": "fast fourier transform",
            "gcd": "greatest common divisor",
            "lcm": "least common multiple",
            "mfmc": "max-flow min-cut",
            "tsp": "traveling salesman problem",
            "sat": "boolean satisfiability",
            "knapsack": "knapsack problem",
            "rng": "random number generator",
        },
    },
    "portfolio_management": {
        "name": "Portfolio Management",
        "description": "Portfolio construction, asset allocation, factor models, risk management, and investment performance evaluation",
        "concept_type_guidance": """Portfolio management:
1. method: Portfolio methods (e.g., mean-variance optimization, Black-Litterman, risk parity, factor investing, liability-driven investing, Monte Carlo simulation, rebalancing)
2. assumption: Model assumptions (e.g., normally distributed returns, efficient markets, rational investors, constant correlations, stationarity of factor premia)
3. problem: Portfolio challenges (e.g., estimation error, curse of dimensionality, regime changes, transaction costs, illiquidity, benchmark misfit)
4. definition: Formal terms (e.g., efficient frontier, Sharpe ratio, information ratio, tracking error, alpha, beta, drawdown, IPS)
5. theorem: Formal results (e.g., CAPM, APT, two-fund separation, mutual fund theorem, mean-variance spanning)

Broader concepts (use when more appropriate):
6. concept: General ideas (e.g., diversification, risk budgeting, strategic vs tactical allocation, factor exposure, style drift)
7. technique: Applied procedures (e.g., corner portfolio method, shrinkage estimation, robust optimization, scenario analysis, stress testing)
8. model: Model architectures (e.g., CAPM, Fama-French three-factor, Carhart four-factor, Barra risk model, Black-Litterman)
9. principle: Investment principles (e.g., diversification, risk-return tradeoff, time diversification, rebalancing discipline, GIPS compliance)""",
        "examples": [
            "CAPM REQUIRES efficient market hypothesis and mean-variance preferences",
            "Risk parity ALTERNATIVE_TO mean-variance optimization for balanced risk allocation",
            "Fama-French three-factor model EXTENDS CAPM with size and value factors",
            "Black-Litterman COMBINES market equilibrium with investor views",
            "APT GENERALIZES CAPM to multiple systematic risk factors",
            "Liability-driven investing ADDRESSES pension fund matching obligations",
            "Tracking error MEASURES deviation of portfolio returns from benchmark",
        ],
        "abbreviations": {
            "mpt": "modern portfolio theory",
            "capm": "capital asset pricing model",
            "apt": "arbitrage pricing theory",
            "var": "value at risk",
            "cvar": "conditional value at risk",
            "es": "expected shortfall",
            "mvo": "mean-variance optimization",
            "ldi": "liability-driven investing",
            "ips": "investment policy statement",
            "saa": "strategic asset allocation",
            "taa": "tactical asset allocation",
            "gips": "global investment performance standards",
            "sml": "security market line",
            "cml": "capital market line",
            "hml": "high minus low",
            "smb": "small minus big",
            "umd": "up minus down",
            "etf": "exchange-traded fund",
            "nav": "net asset value",
            "aum": "assets under management",
            "ir": "information ratio",
            "te": "tracking error",
            "mdd": "maximum drawdown",
            "ewma": "exponentially weighted moving average",
            "bl": "black-litterman",
            "dcc": "dynamic conditional correlation",
            "pca": "principal component analysis",
            "ff3": "fama-french three-factor",
            "ff5": "fama-french five-factor",
            "rmw": "robust minus weak",
            "cma": "conservative minus aggressive",
            "wml": "winners minus losers",
            "roe": "return on equity",
            "pe": "price to earnings",
            "pb": "price to book",
        },
    },
    "forecasting": {
        "name": "Forecasting",
        "description": "Time-series forecasting methods, evaluation, uncertainty quantification, and production systems",
        "concept_type_guidance": """Forecasting:
1. method: Forecasting methods (e.g., ARIMA, SARIMA, ETS, Prophet, DeepAR, TFT, N-BEATS, theta method, ensemble forecasting)
2. assumption: Model assumptions (e.g., stationarity, seasonality patterns, trend decomposability, Gaussian errors, ergodicity)
3. problem: Forecasting challenges (e.g., regime changes, cold start for new series, concept drift, intermittent demand, hierarchical reconciliation)
4. definition: Formal terms (e.g., forecast horizon, prediction interval, point forecast, quantile forecast, lead time)
5. theorem: Formal results (e.g., Wold decomposition, forecast combination optimality, bias-variance decomposition for forecasts)

Broader concepts (use when more appropriate):
6. concept: General ideas (e.g., backtesting, cross-validation for time series, feature engineering for forecasting, exogenous variables)
7. technique: Applied procedures (e.g., differencing, seasonal decomposition, Box-Cox transformation, rolling-origin evaluation)
8. model: Model architectures (e.g., global models, local models, foundation models for time series, hierarchical forecasting)
9. principle: Design principles (e.g., parsimony, forecast combination, probabilistic calibration, temporal aggregation)""",
        "examples": [
            "ARIMA REQUIRES stationarity achieved via differencing",
            "ETS DECOMPOSES series into error, trend, and seasonality components",
            "Prophet HANDLES multiple seasonalities and holiday effects",
            "DeepAR LEARNS cross-series patterns via global autoregressive model",
            "N-BEATS ACHIEVES interpretable decomposition without time-series-specific components",
            "Hierarchical reconciliation ENSURES forecast coherence across aggregation levels",
            "CRPS EVALUATES full predictive distribution quality beyond point accuracy",
        ],
        "abbreviations": {
            "arima": "autoregressive integrated moving average",
            "sarima": "seasonal autoregressive integrated moving average",
            "ets": "error trend seasonality",
            "var": "vector autoregression",
            "vecm": "vector error correction model",
            "mae": "mean absolute error",
            "mse": "mean squared error",
            "rmse": "root mean squared error",
            "mape": "mean absolute percentage error",
            "smape": "symmetric mean absolute percentage error",
            "mase": "mean absolute scaled error",
            "crps": "continuous ranked probability score",
            "wql": "weighted quantile loss",
            "stl": "seasonal and trend decomposition using loess",
            "acf": "autocorrelation function",
            "pacf": "partial autocorrelation function",
            "adf": "augmented dickey-fuller",
            "kpss": "kwiatkowski-phillips-schmidt-shin",
            "bic": "bayesian information criterion",
            "aic": "akaike information criterion",
            "tft": "temporal fusion transformer",
            "nbeats": "neural basis expansion analysis for time series",
            "nhits": "neural hierarchical interpolation for time series",
            "deepar": "deep autoregressive",
            "prophet": "facebook prophet",
            "tbats": "trigonometric box-cox arma trend seasonal",
            "es": "exponential smoothing",
            "hw": "holt-winters",
            "ar": "autoregressive",
            "ma": "moving average",
            "garch": "generalized autoregressive conditional heteroskedasticity",
            "dcc": "dynamic conditional correlation",
            "pi": "prediction interval",
        },
    },
}

# Generic fallback for unknown domains — domain-agnostic extraction guidance
_GENERIC_DOMAIN_CONFIG: dict[str, Any] = {
    "name": "General",
    "description": "Generic knowledge domain (no domain-specific guidance)",
    "concept_type_guidance": """General knowledge extraction:
1. method: Procedures, algorithms, or systematic approaches
2. assumption: Conditions or requirements for validity
3. problem: Issues, challenges, or questions being addressed
4. definition: Formal definitions of terms or concepts
5. theorem: Formal mathematical or logical results
6. concept: General concepts not fitting other types
7. principle: Foundational principles or laws
8. technique: Applied techniques or practices
9. model: Formal models, frameworks, or architectures""",
    "examples": [],
    "abbreviations": {},
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
    config = DOMAIN_PROMPTS.get(domain_id, _GENERIC_DOMAIN_CONFIG)
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
    config = DOMAIN_PROMPTS.get(domain_id, _GENERIC_DOMAIN_CONFIG)
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
    return DOMAIN_PROMPTS.get(domain_id, _GENERIC_DOMAIN_CONFIG)


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
