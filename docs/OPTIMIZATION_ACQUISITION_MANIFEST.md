# Vol23 Optimization Theory — Free Acquisitions Manifest

Staging directory for vol23 optimization theory volume. All items below are legitimately open-access from their official sources (author websites, arXiv, GitHub Pages for author-hosted texts).

**Acquired**: 2026-04-13
**Total**: 6 PDFs, 1,273 pages, ~17 MB
**Next step**: Run research-kb ingestion pipeline to add to the optimization + reinforcement_learning domains.

---

## Contents

| File | Author(s) | Pages | Size | Source |
|---|---|---|---|---|
| `agarwal_jiang_kakade_sun_rl_theory_algorithms.pdf` | Agarwal, Jiang, Kakade, Sun | 205 | 1.2 MB | https://rltheorybook.github.io/rltheorybook_AJKS_V3.pdf |
| `boyd_cvxbook_additional_exercises.pdf` | Boyd & Vandenberghe | 226 | 1.9 MB | http://seas.ucla.edu/~vandenbe/ee236b/homework/bv_cvxbook_extra_exercises.pdf |
| `bubeck_convex_optimization_algorithms_complexity.pdf` | Sébastien Bubeck | 130 | 1.2 MB | https://arxiv.org/pdf/1405.4980.pdf |
| `hartline_mechanism_design_approximation.pdf` | Jason D. Hartline | 349 | 2.5 MB | http://jasonhartline.com/MDnA/MDnA-ch1to8.pdf |
| `hazan_introduction_to_online_convex_optimization.pdf` | Elad Hazan | 260 | 5.2 MB | https://arxiv.org/pdf/1909.05207.pdf |
| `schulman_optimizing_expectations_thesis.pdf` | John Schulman (PhD thesis, UC Berkeley) | 103 | 4.8 MB | http://joschu.net/docs/thesis.pdf |

---

## Vol23 Chapter Coverage Map

| Chapter | Relevant Source(s) |
|---|---|
| Ch 00 Mathematical Prologue | Bubeck (convex analysis rigor), Boyd exercises (worked problems for KKT/duality) |
| Ch 02 Convex Analysis & Optimality | Bubeck, Boyd exercises |
| Ch 03 Lagrangian Duality & KKT | Bubeck Ch 5, Boyd exercises |
| Ch 04 First-Order Methods | Bubeck Ch 3 (GD convergence), Hazan OCO (accelerated methods), Boyd exercises |
| Ch 06 Stochastic Optimization Theory | **Hazan OCO (primary)** — FTRL, online mirror descent, bandit convex opt, regret analysis |
| Ch 10 Dynamic Programming | Agarwal RL Theory Ch 1-2 (MDPs, Bellman operators) |
| Ch 11 Stochastic DP & Optimal Control | Agarwal RL Theory Ch 3-4 |
| Ch 12 RL as Optimization | **Schulman thesis (primary)** — TRPO/PPO derivation, natural gradient, generalized advantage estimation; **Agarwal RL Theory Ch 11-13** (policy gradient theory, NPG convergence) |
| Ch 13 Marketplace & Two-Sided Matching | **Hartline Mechanism Design Ch 1-8 (primary)** — single-parameter auctions, BIC mechanisms, approximation in mechanism design, prior-free revenue maximization |

---

## Notes

- **Bubeck**: arXiv v2 (the latest available — higher version numbers 404). File tool reports "6 pages" due to non-standard linearization; `pdfinfo` correctly shows 130 pages. This IS the full Foundations & Trends monograph.
- **Agarwal RL Theory**: V3 (Feb 2022 draft). Supersedes earlier ABJKS naming — Sun and Brunskill added as authors.
- **Boyd Additional Exercises**: Vandenberghe's UCLA course page hosts the canonical version; Stanford's Boyd site has a broken link as of 2026-04-13.
- **Hazan OCO 2e**: Author-explicit "free forever as contribution to scientific community" (arXiv version).
- **Hartline MDnA**: Ch 1-8 only. Ch 9-10 marked "coming soon" on author site, not yet released.
- **Schulman thesis**: Original TRPO/PPO/GAE derivations; the F&T monograph versions are behind paywalls.
