# Computational Bayes

**Module ID**: `03_computational_bayes`
**Prerequisites**: `02_core_bayesian_inference`
**Goal**: Move from analytic Bayes to simulation, importance sampling, MCMC, HMC/NUTS, and probabilistic programming workflows.

## Primary materials

- **A Student's Guide to Bayesian Statistics**: Best beginner-to-computation bridge in the local library. (`fixtures/textbooks/lambert_students_guide_bayesian_statistics.pdf`)
- **Lambert problem set solutions (Hotti)**: Python-heavy worked solutions, including MCMC, Gibbs, HMC, Stan, and hierarchical examples. (`fixtures/library_books/companion_repos/lambert_solutions_hotti`)
- **BDA Python demos**: Jupyter demos for BDA chapters 2, 3, 4, 5, 6, 10, and 11 plus CmdStanPy examples. (`fixtures/library_books/companion_repos/BDA_py_demos`)
- **BDA R demos**: Posterior predictive, model checking, and Stan/R ecosystem demos. (`fixtures/library_books/companion_repos/BDA_R_demos`)
- **Aalto Bayesian Data Analysis course materials**: Complete course spine with schedule, assignments, demos, notes, and slides. (`fixtures/library_books/companion_repos/BDA_course_Aalto`)
- **Bayesian Workflow**: The main philosophical and practical workflow paper for the track. (https://arxiv.org/abs/2011.01808)
- **Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC**: Model comparison backbone. (https://arxiv.org/abs/1507.04544)
- **Rank-normalization, folding, and localization: An improved R-hat**: Modern convergence diagnostics. (https://arxiv.org/abs/1903.08008)
- **Validating Bayesian Inference Algorithms with Simulation-Based Calibration**: Computational validation that most pedagogical repos skip. (https://arxiv.org/abs/1804.06788)
- **The No-U-Turn Sampler**: Primary HMC/NUTS reference. (https://jmlr.org/papers/v15/hoffman14a.html)

## Support materials

- **Lambert problem set solutions (Castillo)**: Worked exercises and data for Lambert chapters. (`fixtures/library_books/companion_repos/lambert_solutions_castillo`)
- **PyMC documentation**: Primary implementation stack. (https://www.pymc.io/welcome.html)
- **CmdStanPy documentation**: Stan comparison stack for the computational module. (https://mc-stan.org/cmdstanpy/)
- **Stan example models**: Secondary comparison stack and real data examples. (https://github.com/stan-dev/example-models)
- **Monte Carlo Methods in Financial Engineering**: Advanced simulation reference, not trunk material. (`fixtures/textbooks/glasserman_monte_carlo_methods_2003.pdf`)
- **Auto-Encoding Variational Bayes**: Variational inference anchor for the advanced branch. (https://arxiv.org/abs/1312.6114)

## Artifacts

- `notebooks/`: two runnable notebook skeletons
- `exercises/worked.md`: worked prompts
- `exercises/unworked.md`: unworked prompts
- `case_study.md`: mini-project / case-study brief
- `references/`: primary, support, and optional reference lists

## Exercise sources

- Lambert computational chapters
- BDA demos chapters 10-12
- Aalto computational demos

## Checkpoint gate

- You can run and diagnose MCMC/HMC in PyMC and compare one computational workflow to Stan/CmdStanPy.
