# Bayesian Inference / Statistics Repo Plan

Date: 2026-04-06

## Bottom line

You already have enough material to build a strong Bayesian inference / statistics repo without buying much first.

- `docs/owned_inventory.json` currently shows 61 indexed textbooks and 138 indexed papers that are directly relevant when filtering for Bayesian inference, statistics, time series, econometrics, forecasting, and closely related causal material.
- Your raw library also contains several canonical books that are more important than anything new you could buy right now, especially `Bayesian Data Analysis`, `Regression and Other Stories`, `Probabilistic Machine Learning`, `Bayesian Time Series Models`, and `Time Series Analysis by State Space Methods`.
- The highest-leverage move is to build the repo around a small pedagogical spine and ingest the missing owned PDFs / companion repos before spending money.

## What you already own that matters most

### Tier 1: best backbone texts

These are the books I would build the repo around first.

| Resource | Status | Why it matters | Local location |
|---|---|---|---|
| All of Statistics | Indexed | compact classical foundation for likelihood, inference, asymptotics | `fixtures/textbooks/migrated/springer_texts_in_statistics_larry_wasserman_-_all_nd.pdf` |
| Bayesian Statistics for Beginners | Indexed | best gentle on-ramp to posterior thinking and simple models | `fixtures/textbooks/migrated/therese_m_donovan__ruth_m_mickey_-_bayesian_statis_nd.pdf` |
| Bayesian Statistics: An Introduction | Indexed | cleaner conceptual Bayes treatment after the beginner text | `fixtures/textbooks/migrated/lee_peter_-_bayesian_statistics__an_introduction-j_nd.pdf` |
| OpenIntro Statistics | Indexed | quick refresh on sampling, estimators, uncertainty, tests | `fixtures/textbooks/migrated/openintro-statistics_nd.pdf` |
| Bayesian Data Analysis, 3rd ed. | Owned, raw PDF + companion repos | canonical workflow text for posterior inference, model checking, hierarchical models, computation | `fixtures/library_books/Chapman Hall_CRC Texts in Statistical Science Andrew Gelman John B Carlin Hal S Stern David B Dunson Aki Vehtari Donald B Rubin - Bayesian Data Analysis-Chapman and Hall_CRC 2014.pdf` |
| Regression and Other Stories | Owned, raw PDF | best bridge from applied statistics into Bayesian modeling and causal thinking | `fixtures/library_books/Andrew Gelman Jennifer Hill Aki Vehtari - Regression and Other Stories-Cambridge University Press 2020.pdf` |
| Probability and Statistical Inference | Owned, raw PDF | useful frequentist/probability baseline for pedagogical comparisons | `fixtures/library_books/(10) by Robert V. Hogg, Elliot Tanis, Dale Zimmerman - Probability and Statistical Inference (10th Edition) (2019).pdf` |
| Introduction to Mathematical Statistics | Owned, raw PDF | stronger mathematical backup for estimators, likelihood, asymptotics | `fixtures/library_books/Hogg RV McKean JW Craig AT - Introduction to mathematical statistics-Pearson 2019.pdf` |

### Tier 2: computation, probabilistic modeling, and workflow

| Resource | Status | Why it matters | Local location |
|---|---|---|---|
| Probabilistic Graphical Models | Indexed | latent variables, graphical factorization, message passing intuition | `fixtures/textbooks/koller_friedman_pgm_2009.pdf` |
| Machine Learning: A Probabilistic Perspective | Owned, raw PDF | broad probabilistic modeling reference; useful later for latent-variable chapters | `fixtures/library_books/(Adaptive Computation and Machine Learning series) Kevin P. Murphy - Machine Learning_ A Probabilistic Perspective-The MIT Press (2012).pdf` |
| Probabilistic Machine Learning: An Introduction | Owned, raw PDF | modern probability-first ML framing with strong Bayesian overlap | `fixtures/library_books/Kevin P. Murphy - Probabilistic Machine Learning_ An Introduction-The MIT Press (2021).pdf` |
| Computer Age Statistical Inference | Indexed | excellent link between classical stats, computation, and modern methods | `fixtures/textbooks/migrated/computer_age_statistical_inference_nd.pdf` |
| Monte Carlo Methods in Financial Engineering | Indexed | advanced Monte Carlo reference if you want a serious simulation module | `fixtures/textbooks/glasserman_monte_carlo_methods_2003.pdf` |

### Tier 3: regression, GLMs, hierarchical models

| Resource | Status | Why it matters | Local location |
|---|---|---|---|
| Multilevel Analysis | Indexed | strong support text for partial pooling and hierarchical thinking | `fixtures/textbooks/migrated/quantitative_methodology_joop_j_hox_mirjam_moerbee_nd.pdf` |
| An Introduction to Generalized Linear Models | Owned, raw PDF | practical GLM progression after linear and logistic regression | `fixtures/library_books/(Chapman & Hall Statistics Texts) Annette J. Dobson_ Adrian G Barnett - An Introduction to Generalized Linear Models-CRC Press (2018).pdf` |
| Applied Logistic Regression | Owned, raw PDF | strong reference for binary outcomes and diagnostics | `fixtures/library_books/(Wiley Series in Probability and Statistics) David W. Hosmer, Stanley Lemeshow, Rodney X. Sturdivant (auth.), Walter A. Shewhart, Samuel S. Wilks (eds.) - Applied Logistic Regression-Wiley (2013).pdf` |
| Logistic Regression: A Self-Learning Text | Owned, raw PDF | good teaching-friendly logistic regression treatment | `fixtures/library_books/(Statistics for Biology and Health) David G. Kleinbaum, Mitchel Klein (auth.) - Logistic Regression_ A Self-Learning Text-Springer-Verlag New York (2010).pdf` |
| Linear Mixed Models: A Practical Guide | Owned, raw PDF | practical extension path after multilevel chapter | `fixtures/library_books/Linear Mixed Models _ A Practical Guide Using Statistical -- Brady T_ West, Kathleen B_ Welch, Andrzej T_ Gaecki, Brenda -- 1, 3, 2022 -- Chapman and -- 9781000598261 -- 31742598b6e8e788ac6d817f5edbd62a -- Anna’s Archive.pdf` |

### Tier 4: time series and probabilistic forecasting

| Resource | Status | Why it matters | Local location |
|---|---|---|---|
| Time Series Analysis and Its Applications | Indexed | practical modern time-series core | `fixtures/textbooks/migrated/springer_texts_in_statistics_robert_h_shumway_davi_nd.pdf` |
| Time Series Analysis: Forecasting and Control | Indexed | Box-Jenkins reference | `fixtures/textbooks/box_jenkins_time_series_2015.pdf` |
| Time Series Analysis | Indexed | deeper econometric time-series treatment | `fixtures/textbooks/hamilton_time_series_analysis_1994.pdf` |
| Time Series Analysis by State Space Methods | Owned, raw PDF | state-space models, Kalman filtering, Bayesian-adjacent workflow | `fixtures/library_books/James Durbin, Siem Jan Koopman - Time Series Analysis by State Space Methods_ Second Edition (Oxford Statistical Science Series)-Oxford University Press (2012).pdf` |
| Bayesian Time Series Models | Owned, raw PDF | explicit Bayesian state-space / time-series resource | `fixtures/library_books/Barber D., Cemgil A.T., Chiappa S. (eds.) - Bayesian Time Series Models-Cambridge University Press (2011).pdf` |
| Applied Bayesian Forecasting and Time Series Analysis | Owned, raw PDF | strong pedagogy for Bayesian forecasting case studies | `fixtures/library_books/Andy Pole, Mike West, Jeff Harrison (auth.) - Applied Bayesian Forecasting and Time Series Analysis-Springer US (1994).pdf` |
| Time Series Modeling, Computation, and Inference | Owned, raw PDF | modern Bayesian time-series reference | `fixtures/library_books/(Chapman & Hall_CRC Texts in Statistical Science) West, Mike_ Prado, Raquel - Time Series_ Modeling, Computation, and Inference-Chapman and Hall_CRC (2010).pdf` |

### Tier 5: local companion repos you should actively reuse

These are especially valuable because they already contain notebooks, datasets, model files, and course structure.

| Resource | Why it matters | Local location |
|---|---|---|
| BDA Python demos | chapter-by-chapter notebooks; includes `cmdstanpy`, `ArviZ`, and classic examples | `fixtures/library_books/companion_repos/BDA_py_demos` |
| BDA R demos | posterior predictive checks, Stan demos, case studies | `fixtures/library_books/companion_repos/BDA_R_demos` |
| BDA course at Aalto | full course spine: slides, chapter notes, assignments, datasets, demos | `fixtures/library_books/companion_repos/BDA_course_Aalto` |
| BDA companion datasets | `bioassay`, `factory`, `drowning`, `algae`, `kilpisjarvi`, `diabetes`, etc. | inside the companion repos above |

## Best repo shape for your pedagogical goal

I would not organize the repo as "book notes." I would organize it as a staged learning system:

```text
bayes-stats/
  00-foundations/
  01-core-bayes/
  02-computation/
  03-regression-and-glms/
  04-hierarchical-models/
  05-model-checking-and-comparison/
  06-time-series-and-forecasting/
  07-causal-and-decision-making/
  08-probabilistic-ml/
  datasets/
  references/
  notebooks/
  exercises/
  project-templates/
```

Recommended sequence:

1. `00-foundations`
   - probability, likelihood, conditioning, simulation, estimators
   - use `OpenIntro`, `All of Statistics`, `Hogg/Tanis`
2. `01-core-bayes`
   - Bayes rule, conjugacy, prior predictive, posterior predictive, decision analysis
   - use `Bayesian Statistics for Beginners`, `Peter Lee`, `Think Bayes 2`
3. `02-computation`
   - grid approximation, Monte Carlo, importance sampling, MCMC, HMC/NUTS, VI, SBC
   - use `BDA`, `BDA_py_demos`, `Glasserman`, core papers below
4. `03-regression-and-glms`
   - linear regression, logistic regression, Poisson, interpretation, uncertainty
   - use `Regression and Other Stories`, GLM/logistic books
5. `04-hierarchical-models`
   - partial pooling, varying intercepts/slopes, shrinkage, mixed models
   - use `BDA`, `Multilevel Analysis`, `ROS`
6. `05-model-checking-and-comparison`
   - posterior predictive checks, diagnostics, PSIS-LOO, WAIC, sensitivity analysis
   - use `BDA_course_Aalto`, `ArviZ`, workflow papers
7. `06-time-series-and-forecasting`
   - ARIMA, state-space, BSTS, hierarchical forecasting, probabilistic forecasting
   - use `Shumway-Stoffer`, `Hamilton`, `Durbin-Koopman`, `West-Prado`, `Bayesian Time Series Models`
8. `07-causal-and-decision-making`
   - optional but valuable: Bayesian framing of treatment effects, MMM, decision analysis
   - use your existing causal stack plus Bayesian MMM papers already in `fixtures/papers/mmm_google`
9. `08-probabilistic-ml`
   - graphical models, latent variables, Gaussian processes, variational methods
   - use `PGM`, `Murphy`, `ProbML`

## What to ingest from your own library before buying anything

This is the real immediate acquisition queue.

| Priority | Resource | Reason |
|---|---|---|
| 1 | `Bayesian Data Analysis` | central canonical text; you already have the book and companion repos |
| 1 | `Regression and Other Stories` | best teaching text for applied statistics and regression workflow |
| 1 | `Probabilistic Machine Learning: An Introduction` | modern probabilistic reference; free official PDF also exists |
| 1 | `Bayesian Time Series Models` | directly supports a high-value advanced module |
| 1 | `Applied Bayesian Forecasting and Time Series Analysis` | pedagogically excellent for forecasting chapter design |
| 2 | `Time Series Analysis by State Space Methods` | excellent for state-space / Kalman material |
| 2 | `Time Series: Modeling, Computation, and Inference` | modern Bayesian time-series reference |
| 2 | `An Introduction to Generalized Linear Models` | clean GLM chapter support |
| 2 | `Applied Logistic Regression` / `Logistic Regression: A Self-Learning Text` | excellent for binary outcome module |
| 2 | `Linear Mixed Models` | practical mixed-model extension after hierarchical intro |
| 2 | `Probability and Statistical Inference` / `Introduction to Mathematical Statistics` | useful contrasts against Bayesian sections |
| 1 | `BDA_course_Aalto`, `BDA_py_demos`, `BDA_R_demos` | ready-made exercises, datasets, demos, and course sequencing |

## Free resources worth adding now

These are strong additions and all have official free access.

| Resource | Type | Free? | Why add it | Official source |
|---|---|---:|---|---|
| Think Bayes 2 | book + notebooks | Yes | best computational Bayes starter with runnable notebooks | <https://allendowney.github.io/ThinkBayes2/> |
| Statistical Thinking for the 21st Century | book + Python/R companions | Yes | good classical-statistics complement with modern computational framing | <https://statsthinking21.github.io/statsthinking21-core-site/> |
| Probabilistic Machine Learning: An Introduction | book + code | Yes | modern reference; stronger than many paid ML texts for this repo | <https://probml.github.io/pml-book/book1.html> |
| Probabilistic Machine Learning: Advanced Topics | book + code | Yes | good later-stage resource for VI, deep generative models, advanced topics | <https://probml.github.io/pml-book/book2.html> |
| Forecasting: Principles and Practice (3rd ed.) | book | Yes | best free applied forecasting text | <https://otexts.com/fpp3/> |
| Bayes Rules! | book | Yes | approachable applied Bayesian text; good for pedagogical tone | <https://www.bayesrulesbook.com/> |
| BDA course website | course + videos + notes | Yes | full Bayesian workflow curriculum | <https://avehtari.github.io/BDA_course_Aalto/> |
| ROS examples | code + data | Yes | examples and datasets aligned to `Regression and Other Stories` | <https://avehtari.github.io/ROS-Examples/> |
| Stan example models | repo + data | Yes | open-source models, simulators, and real data | <https://github.com/stan-dev/example-models> |
| PyMC examples | repo + notebooks | Yes | ready-made Bayesian modeling examples in Python | <https://github.com/pymc-devs/pymc-examples> |

## Paid resources still worth getting

These are worth adding if you want a tighter explanatory layer than your current library provides.

| Resource | Free? | Why get it | How to find |
|---|---:|---|---|
| Statistical Rethinking, 2nd ed. | No for book; yes for lectures/code/sample chapters | best intuition-building Bayes text; easier entry than BDA | official McElreath page: <https://xcelab.net/rm/> |
| Doing Bayesian Data Analysis | No for book; sample chapter + videos available | strong beginner-to-intermediate bridge with lots of worked examples | official site: <https://sites.google.com/site/doingbayesiandataanalysis/> |
| Bayesian Analysis with Python | No | useful if you want a Python-native book rather than R/Stan-heavy intros | official Packt listing: <https://www.packtpub.com/en-us/product/bayesian-analysis-with-python-9781836644835> |
| Print editions of BDA / ROS / Bayes Rules | No | only if you want physical annotation copies; you already have digital access paths | publisher pages linked from the official book sites above |

## Papers you should explicitly collect

These are the papers I would put into `references/papers/` for the repo.

| Topic | Paper | Free? | Why it belongs |
|---|---|---:|---|
| Workflow | Bayesian Workflow | Yes | central paper for repo philosophy, not just model fitting | <https://arxiv.org/abs/2011.01808> |
| Model comparison | Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC | Yes | PSIS-LOO / WAIC chapter backbone | <https://arxiv.org/abs/1507.04544> |
| Diagnostics | Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC | Yes | modern convergence diagnostics | <https://arxiv.org/abs/1903.08008> |
| Validation | Validating Bayesian Inference Algorithms with Simulation-Based Calibration | Yes | teaches computation validation, which most pedagogical repos skip | <https://arxiv.org/abs/1804.06788> |
| MCMC | The No-U-Turn Sampler | Yes | practical HMC/NUTS reference | <https://jmlr.org/papers/v15/hoffman14a.html> |
| Variational inference | Auto-Encoding Variational Bayes | Yes | compact VI cornerstone | <https://arxiv.org/abs/1312.6114> |

Historical papers you may also want for completeness:

- `Metropolis et al. (1953)` and `Hastings (1970)` for sampling history.
- `Gelfand and Smith (1990)` for modern Gibbs sampling context.
- These are canonical, but publisher access is often paywalled, so I would treat them as secondary history readings rather than first-priority acquisitions.

## Git repos / software stack to standardize on

You do not need every probabilistic programming system. For this repo, I would standardize on one primary stack and one comparison stack.

### Primary recommendation

- `PyMC` for approachable Python notebooks and pedagogy: <https://www.pymc.io/welcome.html>
- `ArviZ` for diagnostics, posterior predictive checks, and model comparison: <https://python.arviz.org/en/stable/>
- `PyMC examples` as runnable patterns: <https://github.com/pymc-devs/pymc-examples>

### Secondary comparison stack

- `Stan`: <https://mc-stan.org/>
- `CmdStanPy`: <https://mc-stan.org/cmdstanpy/>
- `Stan example models`: <https://github.com/stan-dev/example-models>
- `BDA_py_demos`: <https://github.com/avehtari/BDA_py_demos>

### Optional advanced stack

- `NumPyro` for fast JAX-backed inference and great tutorial coverage: <https://num.pyro.ai/en/stable/index.html>
- `Pyro` if you want deeper probabilistic programming / deep latent-variable work: <https://pyro.ai/>
- `PyMC-Marketing` if you want Bayesian MMM / CLV case studies later: <https://github.com/pymc-labs/pymc-marketing>

## Datasets to collect and keep in the repo

I would use a mix of tiny didactic datasets, medium applied datasets, and one forecasting benchmark.

| Dataset source | Free? | Best use | Notes |
|---|---:|---|---|
| `bioassay`, `factory`, `drowning`, `algae`, `kilpisjarvi`, `diabetes` from your local BDA companion repos | Yes | chapters on posterior inference, hierarchical modeling, PPCs, and prediction | already local in `fixtures/library_books/companion_repos` |
| ROS example data | Yes | regression, logistic, causal examples | official site includes code and data: <https://avehtari.github.io/ROS-Examples/> |
| Stan example-models real datasets | Yes | paired Stan/Python examples with citation structure | <https://github.com/stan-dev/example-models> |
| Think Bayes 2 data directory | Yes | simple computational Bayes exercises | <https://github.com/AllenDowney/ThinkBayes2> |
| UCI Bank Marketing | Yes | logistic regression, calibration, uncertainty, uplift-adjacent work | CC BY 4.0: <https://archive.ics.uci.edu/dataset/222/bank+marketing> |
| M4 competition dataset | Yes | forecasting, probabilistic intervals, model comparison | <https://github.com/Mcompetitions/M4-methods> |

## Practical recommendation on spend

If the goal is a repo you can start building now:

- Buy nothing first.
- Ingest the owned BDA / ROS / Bayesian time-series / GLM materials.
- Reuse the BDA companion repos as the first exercise and dataset layer.
- Add free official resources only where they clearly improve coverage.

If you do buy anything after that, I would buy in this order:

1. `Statistical Rethinking, 2nd ed.`
2. `Doing Bayesian Data Analysis`
3. nothing else until you have actually written the first 4-6 modules

## How to find more resources without polluting the repo

Use this acquisition order:

1. Search your own library first.
   - `find fixtures/library_books fixtures/textbooks -type f | rg -i 'bayes|statist|forecast|time series|glm|multilevel|probabilistic'`
2. Prefer official resource pages over mirrors.
   - official author / publisher / course page
   - arXiv or JMLR for papers
   - official GitHub org for code
   - official dataset host such as UCI or competition repo
3. Only use Semantic Scholar / Google Scholar for discovery, not as the canonical storage source.
4. Default to free and already-owned material unless a paid book fills a real pedagogical gap.

## Source notes

Local sources used:

- `docs/owned_inventory.json`
- `docs/ACQUISITION_LIST.md`
- `docs/WHAT_TO_BUY.md`
- local files under `fixtures/library_books`, `fixtures/textbooks`, and `fixtures/library_books/companion_repos`

Official external sources used:

- Think Bayes 2: <https://allendowney.github.io/ThinkBayes2/>
- Statistical Thinking for the 21st Century: <https://statsthinking21.github.io/statsthinking21-core-site/>
- ProbML books: <https://probml.github.io/pml-book/book1.html>, <https://probml.github.io/pml-book/book2.html>
- Forecasting: Principles and Practice: <https://otexts.com/fpp3/>
- Bayes Rules!: <https://www.bayesrulesbook.com/>
- Aalto BDA course: <https://avehtari.github.io/BDA_course_Aalto/>
- Regression and Other Stories examples: <https://avehtari.github.io/ROS-Examples/>
- Statistical Rethinking / McElreath page: <https://xcelab.net/rm/>
- Doing Bayesian Data Analysis: <https://sites.google.com/site/doingbayesiandataanalysis/>
- Bayesian Analysis with Python: <https://www.packtpub.com/en-us/product/bayesian-analysis-with-python-9781836644835>
- Stan / CmdStanPy / example models: <https://mc-stan.org/>, <https://mc-stan.org/cmdstanpy/>, <https://github.com/stan-dev/example-models>
- PyMC / ArviZ / PyMC examples / PyMC-Marketing: <https://www.pymc.io/welcome.html>, <https://python.arviz.org/en/stable/>, <https://github.com/pymc-devs/pymc-examples>, <https://github.com/pymc-labs/pymc-marketing>
- NumPyro / Pyro: <https://num.pyro.ai/en/stable/index.html>, <https://pyro.ai/>
- UCI Bank Marketing dataset: <https://archive.ics.uci.edu/dataset/222/bank+marketing>
- M4 competition repo: <https://github.com/Mcompetitions/M4-methods>
- Workflow / diagnostics / inference papers: <https://arxiv.org/abs/2011.01808>, <https://arxiv.org/abs/1507.04544>, <https://arxiv.org/abs/1903.08008>, <https://arxiv.org/abs/1804.06788>, <https://jmlr.org/papers/v15/hoffman14a.html>, <https://arxiv.org/abs/1312.6114>
