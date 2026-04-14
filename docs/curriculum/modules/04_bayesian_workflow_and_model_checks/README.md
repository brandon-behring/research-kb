# Bayesian Workflow and Model Checks

**Module ID**: `04_bayesian_workflow_and_model_checks`
**Prerequisites**: `03_computational_bayes`
**Goal**: Train the iterative workflow: prior design, posterior predictive checks, diagnostics, sensitivity analysis, and model comparison.

## Primary materials

- **Bayesian Data Analysis**: Canonical workflow and hierarchical-model reference. (`fixtures/library_books/Chapman Hall_CRC Texts in Statistical Science Andrew Gelman John B Carlin Hal S Stern David B Dunson Aki Vehtari Donald B Rubin - Bayesian Data Analysis-Chapman and Hall_CRC 2014.pdf`)
- **Aalto Bayesian Data Analysis course materials**: Complete course spine with schedule, assignments, demos, notes, and slides. (`fixtures/library_books/companion_repos/BDA_course_Aalto`)
- **Bayesian Workflow**: The main philosophical and practical workflow paper for the track. (https://arxiv.org/abs/2011.01808)
- **ArviZ documentation**: Diagnostics, PPC, and model comparison tooling. (https://python.arviz.org/en/stable/)

## Support materials

- **BDA Python demos**: Jupyter demos for BDA chapters 2, 3, 4, 5, 6, 10, and 11 plus CmdStanPy examples. (`fixtures/library_books/companion_repos/BDA_py_demos`)
- **BDA R demos**: Posterior predictive, model checking, and Stan/R ecosystem demos. (`fixtures/library_books/companion_repos/BDA_R_demos`)
- **PyMC documentation**: Primary implementation stack. (https://www.pymc.io/welcome.html)
- **PyMC examples**: Runnable patterns for the primary stack. (https://github.com/pymc-devs/pymc-examples)
- **Stan example models**: Secondary comparison stack and real data examples. (https://github.com/stan-dev/example-models)
- **Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC**: Model comparison backbone. (https://arxiv.org/abs/1507.04544)
- **Rank-normalization, folding, and localization: An improved R-hat**: Modern convergence diagnostics. (https://arxiv.org/abs/1903.08008)

## Artifacts

- `notebooks/`: two runnable notebook skeletons
- `exercises/worked.md`: worked prompts
- `exercises/unworked.md`: unworked prompts
- `case_study.md`: mini-project / case-study brief
- `references/`: primary, support, and optional reference lists

## Exercise sources

- BDA chapter-based demos
- Aalto assignments and PPC examples

## Checkpoint gate

- You can perform PPC, check R-hat and ESS, and compare competing models using predictive criteria.
