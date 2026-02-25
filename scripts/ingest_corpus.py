#!/usr/bin/env python3
"""Ingest textbooks and papers for Phase 1 corpus.

This script:
1. Ingests 2 textbooks (Pearl, Angrist/Pischke)
2. Ingests 12 arXiv papers (causal inference focused)
3. Reports total chunk count and validates ~500 target

Usage:
    python scripts/ingest_corpus.py          # Normal output
    python scripts/ingest_corpus.py --quiet  # Errors + summary only
"""

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "pdf-tools" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "storage" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "contracts" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "common" / "src"))

from research_kb_common import get_logger
from research_kb_contracts import SourceType
from research_kb_pdf import (
    EmbeddingClient,
    chunk_with_sections,
    extract_with_headings,
)
from research_kb_storage import (
    ChunkStore,
    DatabaseConfig,
    SourceStore,
    get_connection_pool,
)

logger = get_logger(__name__)


# Textbooks - canonical causal inference references
TEXTBOOKS = [
    {
        "file": "fixtures/textbooks/pearl_causality_2009.pdf",
        "title": "Causality: Models, Reasoning and Inference",
        "authors": ["Pearl, Judea"],
        "year": 2009,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Cambridge University Press",
            "edition": "2nd",
            "domain": "causal inference",
            "authority": "canonical",
        },
    },
    {
        "file": "fixtures/textbooks/angrist_pischke_mostly_harmless_2009.pdf",
        "title": "Mostly Harmless Econometrics: An Empiricist's Companion",
        "authors": ["Angrist, Joshua D.", "Pischke, Jörn-Steffen"],
        "year": 2009,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Princeton University Press",
            "domain": "econometrics",
            "authority": "canonical",
        },
    },
    # ===== NEW TEXTBOOKS =====
    {
        "file": "fixtures/textbooks/halpern_actual_causality_2016.pdf",
        "title": "Actual Causality",
        "authors": ["Halpern, Joseph Y."],
        "year": 2016,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "MIT Press",
            "domain": "causal inference",
            "authority": "canonical",
        },
    },
    {
        "file": "fixtures/textbooks/hernan_robins_whatif_2024.pdf",
        "title": "Causal Inference: What If",
        "authors": ["Hernán, Miguel A.", "Robins, James M."],
        "year": 2024,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Chapman & Hall/CRC",
            "domain": "causal inference",
            "authority": "canonical",
        },
    },
    {
        "file": "fixtures/textbooks/kleinberg_causality_probability_time_2012.pdf",
        "title": "Causality, Probability, and Time",
        "authors": ["Kleinberg, Samantha"],
        "year": 2012,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Cambridge University Press",
            "domain": "causal inference",
            "key_concept": "temporal causality",
        },
    },
    {
        "file": "fixtures/textbooks/kleinberg_why_2015.pdf",
        "title": "Why: A Guide to Finding and Using Causes",
        "authors": ["Kleinberg, Samantha"],
        "year": 2015,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "O'Reilly Media",
            "domain": "causal inference",
        },
    },
    {
        "file": "fixtures/textbooks/kohavi_ab_testing_2020.pdf",
        "title": "Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing",
        "authors": ["Kohavi, Ron", "Tang, Diane", "Xu, Ya"],
        "year": 2020,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Cambridge University Press",
            "domain": "experimentation",
            "key_concept": "A/B testing",
        },
    },
    {
        "file": "fixtures/textbooks/spirtes_causation_prediction_search_2001.pdf",
        "title": "Causation, Prediction, and Search",
        "authors": ["Spirtes, Peter", "Glymour, Clark", "Scheines, Richard"],
        "year": 2001,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "MIT Press",
            "edition": "2nd",
            "domain": "causal discovery",
            "authority": "canonical",
        },
    },
    {
        "file": "fixtures/textbooks/vanderweele_explanation_causal_inference_2015.pdf",
        "title": "Explanation in Causal Inference: Methods for Mediation and Interaction",
        "authors": ["VanderWeele, Tyler J."],
        "year": 2015,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Oxford University Press",
            "domain": "causal inference",
            "key_concept": "mediation",
            "authority": "canonical",
        },
    },
    {
        "file": "fixtures/textbooks/wooldridge_panel_data_2010.pdf",
        "title": "Econometric Analysis of Cross Section and Panel Data",
        "authors": ["Wooldridge, Jeffrey M."],
        "year": 2010,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "MIT Press",
            "edition": "2nd",
            "domain": "econometrics",
            "authority": "canonical",
        },
    },
    {
        "file": "fixtures/textbooks/Applied Bayesian modeling and causal inference from -- Donald B Rubin; Andrew Gelman; Xiao-Li Meng -- Wiley Series in Probability and Statistics, -- 9780470090435 -- 6b4631b748ce7a0ce1b9e18edbfee08c -- Anna's Archive (1).pdf",
        "title": "Applied Bayesian Modeling and Causal Inference from Incomplete-Data Perspectives",
        "authors": ["Gelman, Andrew", "Meng, Xiao-Li"],
        "year": 2004,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Wiley",
            "domain": "causal inference",
            "key_concept": "missing data",
        },
    },
]

# Train Discrete Choice chapters (treated as individual entries for better chunking)
TRAIN_CHAPTERS = [
    {
        "file": "fixtures/textbooks/train_discrete_choice_2009_chapters/Ch01_p1-8.pdf",
        "title": "Discrete Choice Methods with Simulation - Ch 1: Introduction",
        "authors": ["Train, Kenneth E."],
        "year": 2009,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Cambridge University Press",
            "domain": "econometrics",
            "chapter": 1,
        },
    },
    {
        "file": "fixtures/textbooks/train_discrete_choice_2009_chapters/Ch02_p9-33.pdf",
        "title": "Discrete Choice Methods with Simulation - Ch 2: Properties of Discrete Choice Models",
        "authors": ["Train, Kenneth E."],
        "year": 2009,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Cambridge University Press",
            "domain": "econometrics",
            "chapter": 2,
        },
    },
    {
        "file": "fixtures/textbooks/train_discrete_choice_2009_chapters/Ch03_p34-75.pdf",
        "title": "Discrete Choice Methods with Simulation - Ch 3: Logit",
        "authors": ["Train, Kenneth E."],
        "year": 2009,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Cambridge University Press",
            "domain": "econometrics",
            "chapter": 3,
        },
    },
    {
        "file": "fixtures/textbooks/train_discrete_choice_2009_chapters/Ch04_p76-96.pdf",
        "title": "Discrete Choice Methods with Simulation - Ch 4: GEV",
        "authors": ["Train, Kenneth E."],
        "year": 2009,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Cambridge University Press",
            "domain": "econometrics",
            "chapter": 4,
        },
    },
    {
        "file": "fixtures/textbooks/train_discrete_choice_2009_chapters/Ch05_p97-133.pdf",
        "title": "Discrete Choice Methods with Simulation - Ch 5: Probit",
        "authors": ["Train, Kenneth E."],
        "year": 2009,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Cambridge University Press",
            "domain": "econometrics",
            "chapter": 5,
        },
    },
    {
        "file": "fixtures/textbooks/train_discrete_choice_2009_chapters/Ch06_p134-150.pdf",
        "title": "Discrete Choice Methods with Simulation - Ch 6: Mixed Logit",
        "authors": ["Train, Kenneth E."],
        "year": 2009,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Cambridge University Press",
            "domain": "econometrics",
            "chapter": 6,
        },
    },
    {
        "file": "fixtures/textbooks/train_discrete_choice_2009_chapters/Ch07_p151-182.pdf",
        "title": "Discrete Choice Methods with Simulation - Ch 7: Variations on a Theme",
        "authors": ["Train, Kenneth E."],
        "year": 2009,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Cambridge University Press",
            "domain": "econometrics",
            "chapter": 7,
        },
    },
    {
        "file": "fixtures/textbooks/train_discrete_choice_2009_chapters/Ch08_p183-204.pdf",
        "title": "Discrete Choice Methods with Simulation - Ch 8: Numerical Maximization",
        "authors": ["Train, Kenneth E."],
        "year": 2009,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Cambridge University Press",
            "domain": "econometrics",
            "chapter": 8,
        },
    },
    {
        "file": "fixtures/textbooks/train_discrete_choice_2009_chapters/Ch09_p205-236.pdf",
        "title": "Discrete Choice Methods with Simulation - Ch 9: Drawing from Densities",
        "authors": ["Train, Kenneth E."],
        "year": 2009,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Cambridge University Press",
            "domain": "econometrics",
            "chapter": 9,
        },
    },
    {
        "file": "fixtures/textbooks/train_discrete_choice_2009_chapters/Ch10_p237-258.pdf",
        "title": "Discrete Choice Methods with Simulation - Ch 10: Simulation-Assisted Estimation",
        "authors": ["Train, Kenneth E."],
        "year": 2009,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Cambridge University Press",
            "domain": "econometrics",
            "chapter": 10,
        },
    },
    {
        "file": "fixtures/textbooks/train_discrete_choice_2009_chapters/Ch11_p259-281.pdf",
        "title": "Discrete Choice Methods with Simulation - Ch 11: Individual-Level Parameters",
        "authors": ["Train, Kenneth E."],
        "year": 2009,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Cambridge University Press",
            "domain": "econometrics",
            "chapter": 11,
        },
    },
    {
        "file": "fixtures/textbooks/train_discrete_choice_2009_chapters/Ch12_p282-314.pdf",
        "title": "Discrete Choice Methods with Simulation - Ch 12: Bayesian Procedures",
        "authors": ["Train, Kenneth E."],
        "year": 2009,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Cambridge University Press",
            "domain": "econometrics",
            "chapter": 12,
        },
    },
    {
        "file": "fixtures/textbooks/train_discrete_choice_2009_chapters/Ch13_p315-346.pdf",
        "title": "Discrete Choice Methods with Simulation - Ch 13: Endogeneity",
        "authors": ["Train, Kenneth E."],
        "year": 2009,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Cambridge University Press",
            "domain": "econometrics",
            "chapter": 13,
        },
    },
    {
        "file": "fixtures/textbooks/train_discrete_choice_2009_chapters/Ch14_p347-370.pdf",
        "title": "Discrete Choice Methods with Simulation - Ch 14: EM Algorithms",
        "authors": ["Train, Kenneth E."],
        "year": 2009,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Cambridge University Press",
            "domain": "econometrics",
            "chapter": 14,
        },
    },
]

# TIER 1 Books - High relevance expansion (causal AI, time series, graphs, knowledge graphs)
TIER1_BOOKS = [
    # 1A. Causal Inference & AI Integration
    {
        "file": "fixtures/textbooks/tier1_01_elements_of_causal_inference_2025.pdf",
        "title": "Elements of Causal Inference: Foundations and Learning Algorithms",
        "authors": ["Peters, Jonas", "Janzing, Dominik", "Schölkopf, Bernhard"],
        "year": 2017,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "MIT Press",
            "domain": "causal inference",
            "level": "foundational",
            "key_concepts": "causal models, graphical models, identifiability",
        },
    },
    {
        "file": "fixtures/textbooks/tier1_02_causal_ai_2025.pdf",
        "title": "Causal Artificial Intelligence: AI for Real-World Causal Understanding and Reasoning",
        "authors": ["Ay, Nihat"],
        "year": 2024,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "MIT Press",
            "domain": "causal inference",
            "level": "intermediate",
            "key_concepts": "causal discovery, machine learning, AI applications",
        },
    },
    # 1B. Time Series & Econometrics
    {
        "file": "fixtures/textbooks/tier1_03_time_series_forecasting_python_2025.pdf",
        "title": "Time Series Forecasting in Python",
        "authors": ["Bonaccorso, Giuseppe"],
        "year": 2023,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Packt Publishing",
            "domain": "forecasting",
            "level": "practical",
            "key_concepts": "ARIMA, SARIMA, Prophet, neural networks",
        },
    },
    {
        "file": "fixtures/textbooks/tier1_04_hamilton_1994_time_series_2025.pdf",
        "title": "Time Series Analysis",
        "authors": ["Hamilton, James D."],
        "year": 1994,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Princeton University Press",
            "domain": "econometrics",
            "level": "canonical reference",
            "key_concepts": "time series models, econometric methods",
            "authority": "canonical",
        },
    },
    # 1D. Knowledge Graphs & Algorithms
    {
        "file": "fixtures/textbooks/tier1_05_knowledge_graphs_llms_2025.pdf",
        "title": "Knowledge Graphs and Large Language Models in Action",
        "authors": ["Buhmann, Philipp"],
        "year": 2024,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Manning Publications",
            "domain": "knowledge representation",
            "level": "practical",
            "key_concepts": "knowledge graphs, LLM integration, semantic search",
        },
    },
    {
        "file": "fixtures/textbooks/tier1_06_graph_algorithms_data_science_2025.pdf",
        "title": "Graph Algorithms for Data Science",
        "authors": ["McCreight, Mark", "Needham, Mark"],
        "year": 2023,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "O'Reilly Media",
            "domain": "graph theory",
            "level": "practical",
            "key_concepts": "graph algorithms, PageRank, centrality, community detection",
        },
    },
    # 1C. Deep Learning Foundations
    {
        "file": "fixtures/textbooks/tier1_07_math_architectures_deep_learning_2025.pdf",
        "title": "Math and Architectures of Deep Learning",
        "authors": ["Chintala, Soumith", "Desai, Raghav"],
        "year": 2023,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Pearson",
            "domain": "deep learning",
            "level": "mathematical foundations",
            "key_concepts": "neural networks, architectures, optimization",
        },
    },
    {
        "file": "fixtures/textbooks/tier1_08_gnns_in_action_2025.pdf",
        "title": "Graph Neural Networks in Action",
        "authors": ["Hamilton, Will"],
        "year": 2022,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Manning Publications",
            "domain": "deep learning",
            "level": "specialized",
            "key_concepts": "graph neural networks, embeddings, message passing",
        },
    },
]

# TIER 2 Books - Complementary coverage (simulation, statistics, engineering)
TIER2_BOOKS = [
    # 2A. Simulation Methods
    {
        "file": "fixtures/textbooks/tier2_01_glasserman_2003_monte_carlo_2025.pdf",
        "title": "Monte Carlo Methods in Financial Engineering",
        "authors": ["Glasserman, Paul"],
        "year": 2003,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Springer",
            "domain": "statistical methods",
            "level": "advanced",
            "key_concepts": "Monte Carlo simulation, variance reduction, applications",
            "cross_domain_value": "simulation methods applicable to causal inference",
        },
    },
    # 2B. Statistics & Design
    {
        "file": "fixtures/textbooks/tier2_02_cohen_1988_statistical_power_2025.pdf",
        "title": "Statistical Power Analysis for the Behavioral Sciences",
        "authors": ["Cohen, Jacob"],
        "year": 1988,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Lawrence Erlbaum Associates",
            "domain": "statistics",
            "level": "foundational",
            "key_concepts": "power analysis, effect size, experimental design",
            "authority": "canonical",
        },
    },
    # 1E. Implementation & Best Practices
    {
        "file": "fixtures/textbooks/tier2_03_julia_data_analysis_2025.pdf",
        "title": "Julia for Data Analysis",
        "authors": ["Kwiatkowski, Bogumił"],
        "year": 2022,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Manning Publications",
            "domain": "data science",
            "level": "practical",
            "language": "Julia",
            "key_concepts": "data analysis, implementation patterns",
        },
    },
    {
        "file": "fixtures/textbooks/tier2_04_software_engineering_data_scientists_2025.pdf",
        "title": "Software Engineering for Data Scientists",
        "authors": ["Teufel, Andy", "Marian, Zack"],
        "year": 2022,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Manning Publications (MEAP)",
            "domain": "software engineering",
            "level": "practical",
            "key_concepts": "code quality, testing, best practices for ML/data science",
            "edition": "Early Access Program",
        },
    },
]


# Papers - focused on causal inference methods
PAPERS = [
    {
        "file": "fixtures/papers/chernozhukov_dml_2018.pdf",
        "title": "Double/Debiased Machine Learning for Treatment and Structural Parameters",
        "authors": [
            "Chernozhukov, Victor",
            "Chetverikov, Denis",
            "Demirer, Mert",
            "Duflo, Esther",
            "Hansen, Christian",
            "Newey, Whitney",
            "Robins, James",
        ],
        "year": 2018,
        "source_type": SourceType.PAPER,
        "metadata": {
            "arxiv_id": "1608.00060",
            "domain": "causal inference",
            "key_concept": "cross-fitting",
        },
    },
    {
        "file": "fixtures/papers/athey_imbens_hte_2016.pdf",
        "title": "Recursive Partitioning for Heterogeneous Causal Effects",
        "authors": ["Athey, Susan", "Imbens, Guido"],
        "year": 2016,
        "source_type": SourceType.PAPER,
        "metadata": {
            "arxiv_id": "1504.01132",
            "domain": "causal inference",
            "key_concept": "causal trees",
        },
    },
    {
        "file": "fixtures/papers/athey_imbens_state_2017.pdf",
        "title": "The State of Applied Econometrics: Causality and Policy Evaluation",
        "authors": ["Athey, Susan", "Imbens, Guido W."],
        "year": 2017,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "1607.00699", "domain": "econometrics"},
    },
    {
        "file": "fixtures/papers/athey_ml_economists_2019.pdf",
        "title": "Machine Learning Methods That Economists Should Know About",
        "authors": ["Athey, Susan", "Imbens, Guido W."],
        "year": 2019,
        "source_type": SourceType.PAPER,
        "metadata": {"arxiv_id": "1903.10075", "domain": "econometrics/ML"},
    },
    {
        "file": "fixtures/papers/matching_review_2011.pdf",
        "title": "Matching Methods for Causal Inference: A Review and a Look Forward",
        "authors": ["Stuart, Elizabeth A."],
        "year": 2011,
        "source_type": SourceType.PAPER,
        "metadata": {
            "arxiv_id": "1010.5586",
            "domain": "causal inference",
            "key_concept": "matching",
        },
    },
    {
        "file": "fixtures/papers/psm_revisited_2022.pdf",
        "title": "Why Propensity Scores Should Not Be Used for Matching",
        "authors": ["King, Gary", "Nielsen, Richard"],
        "year": 2022,
        "source_type": SourceType.PAPER,
        "metadata": {
            "arxiv_id": "2208.08065",
            "domain": "causal inference",
            "key_concept": "propensity scores",
        },
    },
    {
        "file": "fixtures/papers/psm_subclass_2015.pdf",
        "title": "Optimal Subclassification for Propensity Score Matching",
        "authors": ["Rosenbaum, Paul R."],
        "year": 2015,
        "source_type": SourceType.PAPER,
        "metadata": {
            "arxiv_id": "1508.06948",
            "domain": "causal inference",
            "key_concept": "propensity scores",
        },
    },
    {
        "file": "fixtures/papers/dl_causal_2018.pdf",
        "title": "Learning Representations for Counterfactual Inference",
        "authors": ["Johansson, Fredrik", "Shalit, Uri", "Sontag, David"],
        "year": 2018,
        "source_type": SourceType.PAPER,
        "metadata": {
            "arxiv_id": "1803.00149",
            "domain": "causal inference/ML",
            "key_concept": "counterfactual inference",
        },
    },
    {
        "file": "fixtures/papers/angrist_imbens_late_2024.pdf",
        "title": "On the Economics Nobel for the Local Average Treatment Effect",
        "authors": ["Angrist, Joshua D.", "Imbens, Guido W."],
        "year": 2024,
        "source_type": SourceType.PAPER,
        "metadata": {
            "arxiv_id": "2402.13023",
            "domain": "causal inference",
            "key_concept": "LATE",
        },
    },
    {
        "file": "fixtures/papers/ai_iv_search_2024.pdf",
        "title": "Finding Valid Instrumental Variables: A Machine Learning Approach",
        "authors": ["Chen, Xiaohong", "White, Halbert"],
        "year": 2024,
        "source_type": SourceType.PAPER,
        "metadata": {
            "arxiv_id": "2409.14202",
            "domain": "causal inference",
            "key_concept": "instrumental variables",
        },
    },
    {
        "file": "fixtures/papers/optimal_iv_2023.pdf",
        "title": "Optimal Instrumental Variables Estimation for Categorical Treatments",
        "authors": [
            "Blandhol, Christine",
            "Mogstad, Magne",
            "Romano, Joseph P.",
            "Shaikh, Azeem M.",
        ],
        "year": 2023,
        "source_type": SourceType.PAPER,
        "metadata": {
            "arxiv_id": "2311.17021",
            "domain": "causal inference",
            "key_concept": "instrumental variables",
        },
    },
    {
        "file": "fixtures/papers/lasso_iv_2010.pdf",
        "title": "LASSO Methods for Instrumental Variables Estimation",
        "authors": ["Belloni, Alexandre", "Chernozhukov, Victor", "Hansen, Christian"],
        "year": 2010,
        "source_type": SourceType.PAPER,
        "metadata": {
            "arxiv_id": "1012.1297",
            "domain": "econometrics",
            "key_concept": "LASSO IV",
        },
    },
    # ===== Phase 1 Cleanup: Additional Papers =====
    # NOTE: Some foundational papers (Abadie 2010, Imbens/Lemieux 2008, Cinelli 2020) are not on arXiv
    # Using arXiv alternatives where available
    {
        "file": "fixtures/papers/abadie_synthetic_control_2021.pdf",
        "title": "Synthetic Controls for Experimental Design",
        "authors": ["Abadie, Alberto", "Zhao, Jinglong"],
        "year": 2021,
        "source_type": SourceType.PAPER,
        "metadata": {
            "arxiv_id": "2108.02196",
            "domain": "causal inference",
            "key_concept": "synthetic_control",
            "authority": "canonical",
        },
    },
    {
        "file": "fixtures/papers/wager_athey_causal_forests_2015.pdf",
        "title": "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests",
        "authors": ["Wager, Stefan", "Athey, Susan"],
        "year": 2018,
        "source_type": SourceType.PAPER,
        "metadata": {
            "arxiv_id": "1510.04342",
            "domain": "causal inference",
            "key_concept": "causal_forest",
            "authority": "canonical",
        },
    },
    {
        "file": "fixtures/papers/imai_mediation_2010.pdf",
        "title": "Identification, Inference and Sensitivity Analysis for Causal Mediation Effects",
        "authors": ["Imai, Kosuke", "Keele, Luke", "Yamamoto, Teppei"],
        "year": 2010,
        "source_type": SourceType.PAPER,
        "metadata": {
            "arxiv_id": "1011.1079",
            "domain": "causal inference",
            "key_concept": "mediation",
            "authority": "canonical",
        },
    },
    {
        "file": "fixtures/papers/callaway_staggered_did_2018.pdf",
        "title": "Difference-in-Differences with Multiple Time Periods",
        "authors": ["Callaway, Brantly", "Sant'Anna, Pedro H.C."],
        "year": 2021,
        "source_type": SourceType.PAPER,
        "metadata": {
            "arxiv_id": "1803.09015",
            "domain": "econometrics",
            "key_concept": "staggered_did",
            "authority": "frontier",
        },
    },
    {
        "file": "fixtures/papers/tamer_partial_id_2010.pdf",
        "title": "Partial Identification in Econometrics",
        "authors": ["Tamer, Elie"],
        "year": 2010,
        "source_type": SourceType.PAPER,
        "metadata": {
            "arxiv_id": "1002.0729",
            "domain": "econometrics",
            "key_concept": "bounds",
            "authority": "canonical",
        },
    },
    {
        "file": "fixtures/papers/cate_estimation_2017.pdf",
        "title": "Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning",
        "authors": [
            "Künzel, Sören R.",
            "Sekhon, Jasjeet S.",
            "Bickel, Peter J.",
            "Yu, Bin",
        ],
        "year": 2019,
        "source_type": SourceType.PAPER,
        "metadata": {
            "arxiv_id": "1712.09988",
            "domain": "causal inference",
            "key_concept": "cate",
            "authority": "frontier",
        },
    },
    # ===== EXPANDED CORPUS: Additional Papers =====
    {
        "file": "fixtures/papers/abadie_synthetic_control_2010.pdf",
        "title": "Synthetic Control Methods for Comparative Case Studies",
        "authors": ["Abadie, Alberto", "Diamond, Alexis", "Hainmueller, Jens"],
        "year": 2010,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "synthetic_control"},
    },
    {
        "file": "fixtures/papers/angrist_imbens_rubin_late_1996.pdf",
        "title": "Identification of Causal Effects Using Instrumental Variables",
        "authors": ["Angrist, Joshua D.", "Imbens, Guido W.", "Rubin, Donald B."],
        "year": 1996,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "causal inference",
            "key_concept": "LATE",
            "authority": "canonical",
        },
    },
    {
        "file": "fixtures/papers/arkhangelsky_sdid_2021.pdf",
        "title": "Synthetic Difference-in-Differences",
        "authors": [
            "Arkhangelsky, Dmitry",
            "Athey, Susan",
            "Hirshberg, David A.",
            "Imbens, Guido W.",
            "Wager, Stefan",
        ],
        "year": 2021,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "synthetic_did"},
    },
    {
        "file": "fixtures/papers/athey_grf_2019.pdf",
        "title": "Generalized Random Forests",
        "authors": ["Athey, Susan", "Tibshirani, Julie", "Wager, Stefan"],
        "year": 2019,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "grf"},
    },
    {
        "file": "fixtures/papers/athey_matrix_completion_2019.pdf",
        "title": "Matrix Completion Methods for Causal Panel Data Models",
        "authors": [
            "Athey, Susan",
            "Bayati, Mohsen",
            "Doudchenko, Nikolay",
            "Imbens, Guido",
            "Khosravi, Khashayar",
        ],
        "year": 2019,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "matrix_completion"},
    },
    {
        "file": "fixtures/papers/athey_policy_2017.pdf",
        "title": "Efficient Policy Learning",
        "authors": ["Athey, Susan", "Wager, Stefan"],
        "year": 2017,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "policy_learning"},
    },
    {
        "file": "fixtures/papers/bang_robins_doubly_robust_2005.pdf",
        "title": "Doubly Robust Estimation in Missing Data and Causal Inference Models",
        "authors": ["Bang, Heejung", "Robins, James M."],
        "year": 2005,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "causal inference",
            "key_concept": "doubly_robust",
            "authority": "canonical",
        },
    },
    {
        "file": "fixtures/papers/benmichael_augsynth_2021.pdf",
        "title": "The Augmented Synthetic Control Method",
        "authors": ["Ben-Michael, Eli", "Feller, Avi", "Rothstein, Jesse"],
        "year": 2021,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "augmented_synth"},
    },
    {
        "file": "fixtures/papers/berry_levinsohn_pakes_blp_1995.pdf",
        "title": "Automobile Prices in Market Equilibrium",
        "authors": ["Berry, Steven", "Levinsohn, James", "Pakes, Ariel"],
        "year": 1995,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "econometrics",
            "key_concept": "BLP",
            "authority": "canonical",
        },
    },
    {
        "file": "fixtures/papers/bojinov_panel_experiments_2021.pdf",
        "title": "Panel Experiments and Dynamic Causal Effects",
        "authors": ["Bojinov, Iavor", "Rambachan, Ashesh", "Shephard, Neil"],
        "year": 2021,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "panel_experiments"},
    },
    {
        "file": "fixtures/papers/bojinov_timeseries_experiments_2019.pdf",
        "title": "Time Series Experiments and Causal Estimands",
        "authors": ["Bojinov, Iavor", "Shephard, Neil"],
        "year": 2019,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "causal inference",
            "key_concept": "timeseries_experiments",
        },
    },
    {
        "file": "fixtures/papers/borusyak_event_study_2021.pdf",
        "title": "Revisiting Event Study Designs: Robust and Efficient Estimation",
        "authors": ["Borusyak, Kirill", "Jaravel, Xavier", "Spiess, Jann"],
        "year": 2021,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "econometrics", "key_concept": "event_study"},
    },
    {
        "file": "fixtures/papers/brodersen_causalimpact_2015.pdf",
        "title": "Inferring Causal Impact Using Bayesian Structural Time-Series Models",
        "authors": [
            "Brodersen, Kay H.",
            "Gallusser, Fabian",
            "Koehler, Jim",
            "Remy, Nicolas",
            "Scott, Steven L.",
        ],
        "year": 2015,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "CausalImpact"},
    },
    {
        "file": "fixtures/papers/chen_causalml_2020.pdf",
        "title": "CausalML: Python Package for Causal Machine Learning",
        "authors": [
            "Chen, Huigang",
            "Harinen, Totte",
            "Lee, Jeong-Yoon",
            "Yung, Mike",
            "Zhao, Zhenyu",
        ],
        "year": 2020,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "causalml"},
    },
    {
        "file": "fixtures/papers/chernozhukov_dml_timeseries_2021.pdf",
        "title": "Automatic Debiased Machine Learning of Causal and Structural Effects",
        "authors": ["Chernozhukov, Victor", "Newey, Whitney K.", "Singh, Rahul"],
        "year": 2021,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "automatic_dml"},
    },
    {
        "file": "fixtures/papers/conlon_pyblp_2020.pdf",
        "title": "Best Practices for Differentiated Products Demand Estimation with PyBLP",
        "authors": ["Conlon, Christopher", "Gortmaker, Jeff"],
        "year": 2020,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "econometrics", "key_concept": "PyBLP"},
    },
    {
        "file": "fixtures/papers/dechaisemartin_twfe_2020.pdf",
        "title": "Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects",
        "authors": ["de Chaisemartin, Clément", "D'Haultfœuille, Xavier"],
        "year": 2020,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "econometrics", "key_concept": "TWFE"},
    },
    {
        "file": "fixtures/papers/dudik_dr_policy_2014.pdf",
        "title": "Doubly Robust Policy Evaluation and Optimization",
        "authors": [
            "Dudík, Miroslav",
            "Erhan, Dumitru",
            "Langford, John",
            "Li, Lihong",
        ],
        "year": 2014,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "dr_policy"},
    },
    {
        "file": "fixtures/papers/finkelstein_poterba_adverse_selection_2004.pdf",
        "title": "Adverse Selection in Insurance Markets: Policyholder Evidence from the U.K. Annuity Market",
        "authors": ["Finkelstein, Amy", "Poterba, James"],
        "year": 2004,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "econometrics", "key_concept": "adverse_selection"},
    },
    {
        "file": "fixtures/papers/foster_orthogonal_learning_2019.pdf",
        "title": "Orthogonal Statistical Learning",
        "authors": ["Foster, Dylan J.", "Syrgkanis, Vasilis"],
        "year": 2019,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "causal inference",
            "key_concept": "orthogonal_learning",
        },
    },
    {
        "file": "fixtures/papers/friedberg_llf_2018.pdf",
        "title": "Local Linear Forests",
        "authors": [
            "Friedberg, Rina",
            "Tibshirani, Julie",
            "Athey, Susan",
            "Wager, Stefan",
        ],
        "year": 2018,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "causal inference",
            "key_concept": "local_linear_forest",
        },
    },
    {
        "file": "fixtures/papers/hansotia_incremental_value_2002.pdf",
        "title": "Incremental Value Modeling",
        "authors": ["Hansotia, Behram", "Rukstales, Brad"],
        "year": 2002,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "uplift"},
    },
    {
        "file": "fixtures/papers/hartford_deep_iv_2017.pdf",
        "title": "Deep IV: A Flexible Approach for Counterfactual Prediction",
        "authors": [
            "Hartford, Jason",
            "Lewis, Greg",
            "Leyton-Brown, Kevin",
            "Taddy, Matt",
        ],
        "year": 2017,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "deep_iv"},
    },
    {
        "file": "fixtures/papers/hausman_new_goods_1996.pdf",
        "title": "Valuation of New Goods Under Perfect and Imperfect Competition",
        "authors": ["Hausman, Jerry A."],
        "year": 1996,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "econometrics", "key_concept": "new_goods"},
    },
    {
        "file": "fixtures/papers/kennedy_dr_hte_2020.pdf",
        "title": "Optimal Doubly Robust Estimation of Heterogeneous Causal Effects",
        "authors": ["Kennedy, Edward H."],
        "year": 2020,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "dr_hte"},
    },
    {
        "file": "fixtures/papers/koijen_yogo_life_insurers_2015.pdf",
        "title": "The Cost of Financial Frictions for Life Insurers",
        "authors": ["Koijen, Ralph S.J.", "Yogo, Motohiro"],
        "year": 2015,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "econometrics", "key_concept": "insurance"},
    },
    {
        "file": "fixtures/papers/lewis_dynamic_dml_2021.pdf",
        "title": "Double Machine Learning with Gradient Boosting and Its Application to Dynamic Causal Effects",
        "authors": ["Lewis, Greg", "Syrgkanis, Vasilis"],
        "year": 2021,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "dynamic_dml"},
    },
    {
        "file": "fixtures/papers/lim_tft_2021.pdf",
        "title": "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting",
        "authors": [
            "Lim, Bryan",
            "Arık, Sercan Ö.",
            "Loeff, Nicolas",
            "Pfister, Tomas",
        ],
        "year": 2021,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "forecasting", "key_concept": "TFT"},
    },
    {
        "file": "fixtures/papers/louizos_cevae_2017.pdf",
        "title": "Causal Effect Inference with Deep Latent-Variable Models",
        "authors": [
            "Louizos, Christos",
            "Shalit, Uri",
            "Mooij, Joris M.",
            "Sontag, David",
            "Zemel, Richard",
            "Welling, Max",
        ],
        "year": 2017,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "CEVAE"},
    },
    {
        "file": "fixtures/papers/lundberg_shap_2017.pdf",
        "title": "A Unified Approach to Interpreting Model Predictions",
        "authors": ["Lundberg, Scott M.", "Lee, Su-In"],
        "year": 2017,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "ML", "key_concept": "SHAP"},
    },
    {
        "file": "fixtures/papers/mackey_orthogonal_ml_2017.pdf",
        "title": "Orthogonal Machine Learning: Power and Limitations",
        "authors": ["Mackey, Lester", "Syrgkanis, Vasilis", "Zadik, Ilias"],
        "year": 2017,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "orthogonal_ml"},
    },
    {
        "file": "fixtures/papers/nevo_cereal_2001.pdf",
        "title": "Measuring Market Power in the Ready-to-Eat Cereal Industry",
        "authors": ["Nevo, Aviv"],
        "year": 2001,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "econometrics", "key_concept": "BLP_application"},
    },
    {
        "file": "fixtures/papers/newey_powell_iv_2003.pdf",
        "title": "Instrumental Variable Estimation of Nonparametric Models",
        "authors": ["Newey, Whitney K.", "Powell, James L."],
        "year": 2003,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "econometrics", "key_concept": "nonparametric_iv"},
    },
    {
        "file": "fixtures/papers/nie_wager_qo_2017.pdf",
        "title": "Quasi-Oracle Estimation of Heterogeneous Treatment Effects",
        "authors": ["Nie, Xinkun", "Wager, Stefan"],
        "year": 2017,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "R_learner"},
    },
    {
        "file": "fixtures/papers/oreshkin_nbeats_2020.pdf",
        "title": "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting",
        "authors": [
            "Oreshkin, Boris N.",
            "Carpov, Dmitri",
            "Chapados, Nicolas",
            "Bengio, Yoshua",
        ],
        "year": 2020,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "forecasting", "key_concept": "N-BEATS"},
    },
    {
        "file": "fixtures/papers/radcliffe_surry_uplift_trees_2011.pdf",
        "title": "Real-World Uplift Modelling with Significance-Based Uplift Trees",
        "authors": ["Radcliffe, Nicholas J.", "Surry, Patrick D."],
        "year": 2011,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "uplift_trees"},
    },
    {
        "file": "fixtures/papers/rambachan_parallel_trends_2021.pdf",
        "title": "A More Credible Approach to Parallel Trends",
        "authors": ["Rambachan, Ashesh", "Roth, Jonathan"],
        "year": 2021,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "econometrics", "key_concept": "parallel_trends"},
    },
    {
        "file": "fixtures/papers/rambachan_roth_event_study_2023.pdf",
        "title": "A More Credible Approach to Parallel Trends",
        "authors": ["Rambachan, Ashesh", "Roth, Jonathan"],
        "year": 2023,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "econometrics",
            "key_concept": "event_study_sensitivity",
        },
    },
    {
        "file": "fixtures/papers/robins_aipw_1994.pdf",
        "title": "Estimation of Regression Coefficients When Some Regressors Are Not Always Observed",
        "authors": ["Robins, James M.", "Rotnitzky, Andrea", "Zhao, Lue Ping"],
        "year": 1994,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "causal inference",
            "key_concept": "AIPW",
            "authority": "canonical",
        },
    },
    {
        "file": "fixtures/papers/rosenbaum_rubin_propensity_1983.pdf",
        "title": "The Central Role of the Propensity Score in Observational Studies for Causal Effects",
        "authors": ["Rosenbaum, Paul R.", "Rubin, Donald B."],
        "year": 1983,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "causal inference",
            "key_concept": "propensity_score",
            "authority": "canonical",
        },
    },
    {
        "file": "fixtures/papers/salinas_deepar_2019.pdf",
        "title": "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks",
        "authors": [
            "Salinas, David",
            "Flunkert, Valentin",
            "Gasthaus, Jan",
            "Januschowski, Tim",
        ],
        "year": 2019,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "forecasting", "key_concept": "DeepAR"},
    },
    {
        "file": "fixtures/papers/sharma_dowhy_2020.pdf",
        "title": "DoWhy: An End-to-End Library for Causal Inference",
        "authors": ["Sharma, Amit", "Kiciman, Emre"],
        "year": 2020,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "DoWhy"},
    },
    {
        "file": "fixtures/papers/shi_dragonnet_2019.pdf",
        "title": "Adapting Neural Networks for the Estimation of Treatment Effects",
        "authors": ["Shi, Claudia", "Blei, David M.", "Veitch, Victor"],
        "year": 2019,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "DragonNet"},
    },
    {
        "file": "fixtures/papers/sun_abraham_event_study_2021.pdf",
        "title": "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects",
        "authors": ["Sun, Liyang", "Abraham, Sarah"],
        "year": 2021,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "econometrics", "key_concept": "event_study_hte"},
    },
    {
        "file": "fixtures/papers/synbeats_panel_causal_2022.pdf",
        "title": "SyntheticBeats: Panel Causal Inference via Synthetic Time Series",
        "authors": ["Various Authors"],
        "year": 2022,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "causal inference",
            "key_concept": "synthetic_timeseries",
        },
    },
    {
        "file": "fixtures/papers/syrgkanis_ml_iv_2019.pdf",
        "title": "Machine Learning Estimation of Heterogeneous Treatment Effects with Instruments",
        "authors": [
            "Syrgkanis, Vasilis",
            "Lei, Victor",
            "Oprescu, Miruna",
            "Hei, Maggie",
            "Battocchi, Keith",
            "Lewis, Greg",
        ],
        "year": 2019,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "ml_iv"},
    },
    {
        "file": "fixtures/papers/vanderlaan_tmle_2006.pdf",
        "title": "Targeted Maximum Likelihood Learning",
        "authors": ["van der Laan, Mark J.", "Rubin, Daniel"],
        "year": 2006,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "causal inference",
            "key_concept": "TMLE",
            "authority": "canonical",
        },
    },
    {
        "file": "fixtures/papers/xu_gsynth_2017.pdf",
        "title": "Generalized Synthetic Control Method: Causal Inference with Interactive Fixed Effects Models",
        "authors": ["Xu, Yiqing"],
        "year": 2017,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "gsynth"},
    },
    {
        "file": "fixtures/papers/zhang_hyvarinen_pnl_2009.pdf",
        "title": "On the Identifiability of the Post-Nonlinear Causal Model",
        "authors": ["Zhang, Kun", "Hyvärinen, Aapo"],
        "year": 2009,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal discovery", "key_concept": "PNL"},
    },
    {
        "file": "fixtures/papers/zhao_harinen_uplift_2019.pdf",
        "title": "Uplift Modeling for Multiple Treatments with Cost Optimization",
        "authors": ["Zhao, Yan", "Fang, Xiao", "Simchi-Levi, David"],
        "year": 2019,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "uplift_multi"},
    },
    {
        "file": "fixtures/papers/zhao_uplift_features_2020.pdf",
        "title": "Feature Selection Methods for Uplift Modeling and Heterogeneous Treatment Effect Estimation",
        "authors": ["Zhao, Zhenyu", "Harinen, Totte"],
        "year": 2020,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal inference", "key_concept": "uplift_features"},
    },
    {
        "file": "fixtures/papers/zheng_causallearn_2024.pdf",
        "title": "Causal-learn: Causal Discovery in Python",
        "authors": [
            "Zheng, Yujia",
            "Huang, Biwei",
            "Chen, Wei",
            "Ramsey, Joseph",
            "Gong, Mingming",
            "Cai, Ruichu",
            "Shimizu, Shohei",
            "Spirtes, Peter",
            "Zhang, Kun",
        ],
        "year": 2024,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "causal discovery", "key_concept": "causal-learn"},
    },
    {
        "file": "fixtures/papers/zhou_informer_2021.pdf",
        "title": "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting",
        "authors": [
            "Zhou, Haoyi",
            "Zhang, Shanghang",
            "Peng, Jieqi",
            "Zhang, Shuai",
            "Li, Jianxin",
            "Xiong, Hui",
            "Zhang, Wancai",
        ],
        "year": 2021,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "forecasting", "key_concept": "Informer"},
    },
    {
        "file": "fixtures/papers/siam_2017_proceedings.pdf",
        "title": "SIAM Proceedings on Data Mining 2017",
        "authors": ["Various Authors"],
        "year": 2017,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "data_mining"},
    },
]

# ============================================================================
# TIME SERIES DOMAIN SOURCES
# Use with: python scripts/ingest_corpus.py --domain time_series
# ============================================================================
TIME_SERIES_TEXTBOOKS = [
    {
        "file": "fixtures/textbooks/hamilton_time_series_analysis_1994.pdf",
        "title": "Time Series Analysis",
        "authors": ["Hamilton, James D."],
        "year": 1994,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Princeton University Press",
            "domain": "time_series",
            "authority": "canonical",
            "key_concepts": "VAR, state-space models, Kalman filter, unit roots",
        },
    },
    {
        "file": "fixtures/textbooks/box_jenkins_time_series_2015.pdf",
        "title": "Time Series Analysis: Forecasting and Control",
        "authors": [
            "Box, George E.P.",
            "Jenkins, Gwilym M.",
            "Reinsel, Gregory C.",
            "Ljung, Greta M.",
        ],
        "year": 2015,
        "source_type": SourceType.TEXTBOOK,
        "metadata": {
            "publisher": "Wiley",
            "edition": "5th",
            "domain": "time_series",
            "authority": "canonical",
            "key_concepts": "ARIMA, Box-Jenkins methodology, seasonal adjustment",
        },
    },
]

TIME_SERIES_PAPERS = [
    {
        "file": "fixtures/papers/lim_tft_2021.pdf",
        "title": "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting",
        "authors": [
            "Lim, Bryan",
            "Arık, Sercan Ö.",
            "Loeff, Nicolas",
            "Pfister, Tomas",
        ],
        "year": 2021,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "time_series", "key_concept": "TFT"},
    },
    {
        "file": "fixtures/papers/oreshkin_nbeats_2020.pdf",
        "title": "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting",
        "authors": [
            "Oreshkin, Boris N.",
            "Carpov, Dmitri",
            "Chapados, Nicolas",
            "Bengio, Yoshua",
        ],
        "year": 2020,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "time_series", "key_concept": "N-BEATS"},
    },
    {
        "file": "fixtures/papers/salinas_deepar_2019.pdf",
        "title": "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks",
        "authors": [
            "Salinas, David",
            "Flunkert, Valentin",
            "Gasthaus, Jan",
            "Januschowski, Tim",
        ],
        "year": 2019,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "time_series", "key_concept": "DeepAR"},
    },
    {
        "file": "fixtures/papers/zhou_informer_2021.pdf",
        "title": "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting",
        "authors": [
            "Zhou, Haoyi",
            "Zhang, Shanghang",
            "Peng, Jieqi",
            "Zhang, Shuai",
            "Li, Jianxin",
            "Xiong, Hui",
            "Zhang, Wancai",
        ],
        "year": 2021,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "time_series", "key_concept": "Informer"},
    },
    {
        "file": "fixtures/papers/bojinov_timeseries_experiments_2019.pdf",
        "title": "Time Series Experiments and Causal Estimands",
        "authors": ["Bojinov, Iavor", "Shephard, Neil"],
        "year": 2019,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "time_series",
            "key_concept": "timeseries_experiments",
            "cross_domain": "causal_inference",
        },
    },
]

# ============================================================================
# RAG/LLM DOMAIN SOURCES
# Use with: python scripts/ingest_corpus.py --domain rag_llm
# ============================================================================
RAG_LLM_PAPERS = [
    # Foundational Architecture
    {
        "file": "fixtures/papers/vaswani_attention_2017.pdf",
        "title": "Attention Is All You Need",
        "authors": [
            "Vaswani, Ashish",
            "Shazeer, Noam",
            "Parmar, Niki",
            "Uszkoreit, Jakob",
            "Jones, Llion",
            "Gomez, Aidan N.",
            "Kaiser, Lukasz",
            "Polosukhin, Illia",
        ],
        "year": 2017,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "rag_llm",
            "key_concept": "transformer",
            "arxiv": "1706.03762",
        },
    },
    {
        "file": "fixtures/papers/devlin_bert_2018.pdf",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": [
            "Devlin, Jacob",
            "Chang, Ming-Wei",
            "Lee, Kenton",
            "Toutanova, Kristina",
        ],
        "year": 2018,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "rag_llm", "key_concept": "bert", "arxiv": "1810.04805"},
    },
    # In-Context Learning
    {
        "file": "fixtures/papers/brown_gpt3_2020.pdf",
        "title": "Language Models are Few-Shot Learners",
        "authors": [
            "Brown, Tom B.",
            "Mann, Benjamin",
            "Ryder, Nick",
            "Subbiah, Melanie",
        ],
        "year": 2020,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "rag_llm", "key_concept": "gpt3", "arxiv": "2005.14165"},
    },
    # RAG Foundation
    {
        "file": "fixtures/papers/lewis_rag_2020.pdf",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": [
            "Lewis, Patrick",
            "Perez, Ethan",
            "Piktus, Aleksandra",
            "Petroni, Fabio",
            "Karpukhin, Vladimir",
        ],
        "year": 2020,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "rag_llm", "key_concept": "rag", "arxiv": "2005.11401"},
    },
    # Prompting Techniques
    {
        "file": "fixtures/papers/wei_cot_2022.pdf",
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "authors": [
            "Wei, Jason",
            "Wang, Xuezhi",
            "Schuurmans, Dale",
            "Bosma, Maarten",
            "Ichter, Brian",
            "Xia, Fei",
            "Chi, Ed",
            "Le, Quoc",
            "Zhou, Denny",
        ],
        "year": 2022,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "rag_llm",
            "key_concept": "chain_of_thought",
            "arxiv": "2201.11903",
        },
    },
    {
        "file": "fixtures/papers/wang_selfconsistency_2022.pdf",
        "title": "Self-Consistency Improves Chain of Thought Reasoning in Language Models",
        "authors": [
            "Wang, Xuezhi",
            "Wei, Jason",
            "Schuurmans, Dale",
            "Le, Quoc",
            "Chi, Ed",
            "Narang, Sharan",
            "Chowdhery, Aakanksha",
            "Zhou, Denny",
        ],
        "year": 2022,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "rag_llm",
            "key_concept": "self_consistency",
            "arxiv": "2203.11171",
        },
    },
    # Alignment
    {
        "file": "fixtures/papers/ouyang_instructgpt_2022.pdf",
        "title": "Training language models to follow instructions with human feedback",
        "authors": ["Ouyang, Long", "Wu, Jeff", "Jiang, Xu", "Almeida, Diogo"],
        "year": 2022,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "rag_llm",
            "key_concept": "instructgpt",
            "arxiv": "2203.02155",
        },
    },
    {
        "file": "fixtures/papers/rafailov_dpo_2023.pdf",
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "authors": [
            "Rafailov, Rafael",
            "Sharma, Archit",
            "Mitchell, Eric",
            "Ermon, Stefano",
            "Manning, Christopher D.",
            "Finn, Chelsea",
        ],
        "year": 2023,
        "source_type": SourceType.PAPER,
        "metadata": {"domain": "rag_llm", "key_concept": "dpo", "arxiv": "2305.18290"},
    },
    # Open Models
    {
        "file": "fixtures/papers/touvron_llama_2023.pdf",
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "authors": [
            "Touvron, Hugo",
            "Lavril, Thibaut",
            "Izacard, Gautier",
            "Martinet, Xavier",
        ],
        "year": 2023,
        "source_type": SourceType.PAPER,
        "metadata": {
            "domain": "rag_llm",
            "key_concept": "llama",
            "arxiv": "2302.13971",
        },
    },
]


async def ingest_pdf(
    pdf_path: str,
    title: str,
    authors: list[str],
    year: int,
    source_type: SourceType,
    metadata: dict,
    domain_id: str = "causal_inference",
) -> tuple[str, int, int]:
    """Ingest a single PDF through full pipeline.

    Args:
        pdf_path: Path to PDF file
        title: Document title
        authors: List of authors
        year: Publication year
        source_type: Type of source
        metadata: Additional metadata
        domain_id: Knowledge domain (default: causal_inference)

    Returns:
        Tuple of (source_id, num_chunks, num_headings)
    """
    pdf_path = Path(pdf_path)

    # 1. Extract with heading detection
    logger.info("extracting_pdf", path=str(pdf_path))
    doc, headings = extract_with_headings(pdf_path)

    logger.info(
        "extraction_complete",
        path=str(pdf_path),
        pages=doc.total_pages,
        headings=len(headings),
    )

    # 2. Chunk with section tracking
    logger.info("chunking_document", path=str(pdf_path))
    chunks = chunk_with_sections(doc, headings)

    logger.info("chunking_complete", path=str(pdf_path), chunks=len(chunks))

    # 3. Calculate file hash for idempotency
    sha256_hash = hashlib.sha256()
    with pdf_path.open("rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    file_hash = sha256_hash.hexdigest()

    # 4. Create Source record
    logger.info("creating_source", title=title, domain=domain_id)
    source = await SourceStore.create(
        source_type=source_type,
        title=title,
        authors=authors,
        year=year,
        file_path=str(pdf_path),
        file_hash=file_hash,
        domain_id=domain_id,
        metadata={
            **metadata,
            "extraction_method": "pymupdf",
            "total_pages": doc.total_pages,
            "total_chars": doc.total_chars,
            "total_headings": len(headings),
            "total_chunks": len(chunks),
        },
    )

    logger.info("source_created", source_id=str(source.id))

    # 5. Generate embeddings and create Chunk records
    logger.info("generating_embeddings", chunks=len(chunks))
    embedding_client = EmbeddingClient()

    chunks_created = 0
    for chunk in chunks:
        # Sanitize content (remove null bytes and other control characters)
        sanitized_content = chunk.content.replace("\x00", "").replace("\ufffd", "")

        # Generate embedding
        embedding = embedding_client.embed(sanitized_content)

        # Calculate content hash
        content_hash = hashlib.sha256(sanitized_content.encode("utf-8")).hexdigest()

        # Create chunk record
        await ChunkStore.create(
            source_id=source.id,
            content=sanitized_content,
            content_hash=content_hash,
            page_start=chunk.start_page,
            page_end=chunk.end_page,
            embedding=embedding,
            metadata=chunk.metadata,
            domain_id=domain_id,
        )
        chunks_created += 1

        # Log progress every 50 chunks
        if chunks_created % 50 == 0:
            logger.info("chunks_progress", created=chunks_created, total=len(chunks))

    logger.info(
        "ingestion_complete",
        source_id=str(source.id),
        chunks_created=chunks_created,
        headings_detected=len(headings),
    )

    return str(source.id), chunks_created, len(headings)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest textbooks and papers into research-kb corpus."
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Minimal output (errors + final summary only)",
    )
    parser.add_argument(
        "-d",
        "--domain",
        default="causal_inference",
        help="Knowledge domain for ingested sources (default: causal_inference)",
    )
    return parser.parse_args()


async def main():
    """Ingest all textbooks and papers, report results."""
    args = parse_args()
    quiet = args.quiet
    domain_id = args.domain

    # Select sources based on domain
    if domain_id == "time_series":
        # Time series domain sources
        all_textbooks = TIME_SERIES_TEXTBOOKS
        all_papers = TIME_SERIES_PAPERS
        all_docs = all_textbooks + all_papers
    elif domain_id == "rag_llm":
        # RAG/LLM domain sources (papers only)
        all_textbooks = []
        all_papers = RAG_LLM_PAPERS
        all_docs = all_papers
    else:
        # Default: causal inference domain sources
        all_textbooks = TEXTBOOKS + TRAIN_CHAPTERS + TIER1_BOOKS + TIER2_BOOKS
        all_papers = PAPERS
        all_docs = all_textbooks + all_papers

    total_textbooks = len(all_textbooks)

    if not quiet:
        logger.info(
            "starting_corpus_ingestion",
            textbooks=total_textbooks,
            papers=len(all_papers),
            domain=domain_id,
        )

    # Initialize database connection pool
    config = DatabaseConfig()
    await get_connection_pool(config)

    results = {"textbooks": [], "papers": []}

    # Process all documents
    for doc_data in all_docs:
        pdf_path = Path(__file__).parent.parent / doc_data["file"]
        category = "textbooks" if doc_data["source_type"] == SourceType.TEXTBOOK else "papers"

        if not pdf_path.exists():
            logger.error("pdf_not_found", path=str(pdf_path))
            if not quiet:
                print(f"✗ PDF not found: {pdf_path}")
            results[category].append(
                {
                    "title": doc_data["title"],
                    "status": "not_found",
                }
            )
            continue

        try:
            source_id, num_chunks, num_headings = await ingest_pdf(
                pdf_path=str(pdf_path),
                title=doc_data["title"],
                authors=doc_data["authors"],
                year=doc_data["year"],
                source_type=doc_data["source_type"],
                metadata=doc_data["metadata"],
                domain_id=domain_id,
            )

            results[category].append(
                {
                    "title": doc_data["title"],
                    "source_id": source_id,
                    "chunks": num_chunks,
                    "headings": num_headings,
                    "status": "success",
                }
            )

            if not quiet:
                short_title = doc_data["title"][:50]
                print(f"✓ {short_title}")
                print(f"  Chunks: {num_chunks} | Headings: {num_headings}")

        except Exception as e:
            logger.error("ingestion_failed", title=doc_data["title"], error=str(e), exc_info=True)
            results[category].append(
                {
                    "title": doc_data["title"],
                    "status": "failed",
                    "error": str(e),
                }
            )
            # Always show errors even in quiet mode
            print(f"✗ {doc_data['title'][:40]}: {e}")

    # Summary
    textbook_success = [r for r in results["textbooks"] if r["status"] == "success"]
    paper_success = [r for r in results["papers"] if r["status"] == "success"]

    textbook_chunks = sum(r["chunks"] for r in textbook_success)
    paper_chunks = sum(r["chunks"] for r in paper_success)
    total_chunks = textbook_chunks + paper_chunks
    total_textbooks = len(all_textbooks)

    # Report failures (always shown)
    all_failed = [
        r
        for r in results["textbooks"] + results["papers"]
        if r["status"] in ("failed", "not_found")
    ]

    if quiet:
        # Minimal summary for quiet mode
        print(
            f"Ingested: {len(textbook_success)} textbooks, {len(paper_success)} papers | {total_chunks} chunks"
        )
        if all_failed:
            print(f"Failed: {len(all_failed)}")
            for r in all_failed:
                error = r.get("error", "not found")
                print(f"  ✗ {r['title'][:40]}: {error}")
    else:
        # Full summary
        print("\n" + "=" * 70)
        print("CORPUS INGESTION SUMMARY")
        print("=" * 70)

        print(f"\nTEXTBOOKS: {len(textbook_success)}/{total_textbooks}")
        for r in textbook_success:
            print(f"  {r['title'][:45]:45} | {r['chunks']:4} chunks")
        print(f"  Subtotal: {textbook_chunks} chunks")

        print(f"\nPAPERS: {len(paper_success)}/{len(all_papers)}")
        for r in paper_success:
            print(f"  {r['title'][:45]:45} | {r['chunks']:4} chunks")
        print(f"  Subtotal: {paper_chunks} chunks")

        print("\n" + "-" * 70)
        print(f"TOTAL CHUNKS: {total_chunks}")
        print("TARGET: ~500 chunks")

        if total_chunks >= 450:
            print("✓ Target achieved!")
        else:
            print(f"⚠ Need ~{500 - total_chunks} more chunks")

        if all_failed:
            print("\nFAILED:")
            for r in all_failed:
                error = r.get("error", "not found")
                print(f"  ✗ {r['title'][:40]}: {error}")

    if not quiet:
        logger.info(
            "corpus_ingestion_complete",
            textbooks=len(textbook_success),
            papers=len(paper_success),
            total_chunks=total_chunks,
        )


if __name__ == "__main__":
    asyncio.run(main())
