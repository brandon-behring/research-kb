# Bayesian Inference / Statistics Curriculum

This directory turns the Bayes/stats roadmap into a dependency-ordered curriculum rather than a flat reading list.

## Defaults

- **Audience**: technically strong self-learner building a pedagogical repo
- **Pace**: full deep-study curriculum
- **Primary stack**: `Python + PyMC + ArviZ`
- **Secondary stack**: `Stan + CmdStanPy`

## Trunk path

`01_probability_and_stat_baseline -> 02_core_bayesian_inference -> 03_computational_bayes -> 04_bayesian_workflow_and_model_checks -> 05_regression_and_glms -> 06_hierarchical_models -> 07_time_series_and_forecasting`

Advanced branches live in `08_advanced_branches` and should only be taken after the trunk is complete.

## Inventory corrections baked into this curriculum

The machine-readable registry in `material_registry.json` corrects several gaps in `docs/owned_inventory.json` for this learning track.

Promoted owned but under-indexed core materials:
- `Bayesian Data Analysis`
- `Regression and Other Stories`
- `A Student's Guide to Bayesian Statistics`
- `Probabilistic Machine Learning: An Introduction`
- `Bayesian Time Series Models`
- `Applied Bayesian Forecasting and Time Series Analysis`
- `Time Series Analysis by State Space Methods`
- `An Introduction to Generalized Linear Models`
- `Applied Logistic Regression`
- `Linear Mixed Models`

Explicit exclusions:
- Hamiltonian mechanics / physics books that appear in filename searches but are not part of the stats/Bayes curriculum.

## Files

- `material_registry.json`: corrected curriculum registry with source states and role tags
- `module_manifest.json`: module order, prerequisites, and artifact paths
- `modules/`: implementation of the curriculum modules
- `../BAYESIAN_STATISTICS_REPO_PLAN.md`: planning document that led to this implementation

## Validation

Run:

```bash
python3 scripts/validate_bayesian_curriculum.py
```

The validator checks source-state tags, role tags, module dependencies, trunk ordering, and module artifact completeness.

## Module map

| Module | Purpose |
|---|---|
| `00_inventory_and_prereqs` | normalize the learning inventory and prerequisite chain |
| `01_probability_and_stat_baseline` | probability, likelihood, uncertainty, simulation literacy |
| `02_core_bayesian_inference` | priors, posteriors, conjugacy, posterior predictive thinking |
| `03_computational_bayes` | Monte Carlo, MCMC, HMC/NUTS, probabilistic programming |
| `04_bayesian_workflow_and_model_checks` | diagnostics, PPC, sensitivity analysis, model comparison |
| `05_regression_and_glms` | linear, logistic, Poisson, GLM interpretation |
| `06_hierarchical_models` | partial pooling, varying effects, mixed-model thinking |
| `07_time_series_and_forecasting` | ARIMA, state-space, Bayesian forecasting |
| `08_advanced_branches` | probabilistic ML and causal/Bayesian decision branches |
