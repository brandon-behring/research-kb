# Acquisition List (2026-03-28)

Cross-referenced against research-kb database (1,522+ sources), all 36 ~/Claude/ projects,
and post_transformers/references/wish_list.md.

Run `python scripts/generate_status.py` for current DB metrics.

## Summary

- **Corpus**: 1,500+ sources, 35 domains, 100% embedding coverage (as of 2026-03-27)
- **This list**: 42 items to buy + 5 free items just acquired + S2 auto-discovery running
- **Estimated cost**: ~$1,800 (or ~$950 with O'Reilly subscription)
- **Priority**: Critical = fills domain with <5 sources, High = interview prep or cross-project need, Medium = strengthens existing domain

---

## Just Acquired (Free, 2026-03-28)

| Title | Author(s) | Domain | Source | Status |
|-------|----------|--------|--------|--------|
| Feedback Systems: An Introduction (2nd ed) | Astrom, Murray | dynamical_systems | fbswiki.org (free PDF) | Downloaded |
| Mathematical Control Theory (2nd ed) | Sontag | dynamical_systems | sontaglab.org (free PDF) | Downloaded |
| Causal Inference: The Mixtape | Cunningham | causal_inference | mixtape.scunning.com | Already in DB |
| Geometric Numerical Integration | Hairer et al. | numerical_methods | — | Already in DB (different title) |
| Linear System Theory | Chen | dynamical_systems | — | Already in DB |

---

## Auto-Discovery (In Progress)

S2 auto-discovery running for thin domains via `scripts/s2_auto_discover.py`:
- sql, adtech, recommender_systems, economics, forecasting
- Expected yield: ~10-20 free arXiv papers per domain
- Status: check `/tmp/s2_discovery.log`

---

## Books to Buy

### Critical Priority — Fill Domain Gaps (<5 sources)

| # | Title | Author(s) | Year | Price | Domain (current count) | ISBN | Cross-project need |
|---|-------|-----------|------|-------|----------------------|------|--------------------|
| 1 | **SQL Cookbook** (2nd ed) | Molinaro, de Graaf | 2020 | $45 | sql (14) | 978-1492077442 | interview_prep vol05 |
| 2 | **Recommender Systems Handbook** (3rd ed) | Ricci, Rokach, Shapira | 2022 | $80 | recommender_systems (5) | 978-1071628966 | interview_prep vol19 |
| 3 | **Practical Recommender Systems** | Kim Falk | 2019 | $45 | recommender_systems (5) | 978-1617292705 | interview_prep vol19 |
| 4 | **Computational Advertising** | Yuan, Wang, Zhao | 2019 | $60 | adtech (3) | 978-1108481489 | interview_prep vol12 |

**Subtotal**: ~$230

### High Priority — Interview Prep + Cross-Project

| # | Title | Author(s) | Year | Price | Domain | ISBN | Cross-project need |
|---|-------|-----------|------|-------|--------|------|--------------------|
| 5 | **Elements of Programming Interviews in Python** | Aziz, Lee, Prakash | 2018 | $40 | algorithms | 978-1537713946 | interview_prep vol04 |
| 6 | **System Design Interview Vol 1** | Alex Xu | 2020 | $35 | interview_prep | 978-1736049549 | interview_prep vol15 |
| 7 | **System Design Interview Vol 2** | Alex Xu | 2022 | $40 | interview_prep | 978-1736049112 | interview_prep vol15 |
| 8 | **Designing Data-Intensive Applications** | Martin Kleppmann | 2017 | $45 | software_engineering | 978-1449373320 | interview_prep, consulting |
| 9 | **Imbens & Rubin** (Causal Inference) | Imbens, Rubin | 2015 | $80 | causal_inference | 978-0521885881 | causal_inference_mastery |
| 10 | **ML System Design Interview** | Xu, Aminian | 2022 | $40 | interview_prep | 978-1736049129 | interview_prep vol09 |
| 11 | **Generative AI System Design Interview** | Alex Xu | 2024 | $40 | interview_prep | — | interview_prep vol08 |
| 12 | **Credit Risk Scorecards** | Naeem Siddiqi | 2017 | $60 | finance | 978-1119201731 | interview_prep vol18 |

**Subtotal**: ~$380

### Medium Priority — Domain Strengthening

| # | Title | Author(s) | Year | Price | Domain | ISBN | Notes |
|---|-------|-----------|------|-------|--------|------|-------|
| 13 | **Fluent Python** (2nd ed) | Luciano Ramalho | 2022 | $50 | software_engineering | 978-1492056355 | O'Reilly eligible |
| 14 | **Effective Python** (3rd ed) | Brett Slatkin | 2024 | $45 | software_engineering | 978-0138172404 | |
| 15 | **Pattern Recognition and Machine Learning** | Christopher Bishop | 2006 | $60 | machine_learning | 978-0387310732 | O'Reilly eligible |
| 16 | **Hands-On Machine Learning** (3rd ed) | Aurelien Geron | 2022 | $55 | machine_learning | 978-1098125974 | O'Reilly eligible |
| 17 | **Deep RL Hands-On** (3rd ed) | Maxim Lapan | 2024 | $50 | reinforcement_learning | 978-1835882702 | O'Reilly eligible |
| 18 | **RL for Finance** | Yves Hilpisch | 2024 | $50 | finance | 978-1098166892 | annuity projects |
| 19 | **Practical Guide to Quant Finance Interviews** | Joshi | 2008 | $45 | finance | 978-1435712751 | interview_prep |
| 20 | **Hull's Options, Futures, Derivatives** (11th ed) | John Hull | 2021 | $100 | finance | 978-0136939979 | annuity-pricing refs |

**Subtotal**: ~$455

### Post-Transformers Research (Tier 3)

From `post_transformers/references/wish_list.md`. Only items NOT already in DB.

| # | Title | Author(s) | Year | Price | Domain | Priority | Notes |
|---|-------|-----------|------|-------|--------|----------|-------|
| 21 | **Matrix Analysis** | Horn, Johnson | 2013 | $80 | algebra | P2 | Spectral diagnostics niche |
| 22 | **Hamilton Time Series Analysis** | James Hamilton | 1994 | $80 | time_series | P2 | State-space from econometrics |
| 23 | **Solving ODEs II** (Stiff problems) | Hairer, Wanner | 1996 | $80 | numerical_methods | P2 | A-stability, exponential integrators |
| 24 | **Linear Systems** | Kailath | 1980 | $100 | dynamical_systems | P2 | Classic realization theory |
| 25 | **Introduction to Mechanics and Symmetry** | Marsden, Ratiu | 1999 | $80 | physics | P3 | PhD background extension |

**Subtotal**: ~$420

### Low Priority — Nice to Have

| # | Title | Author(s) | Year | Price | Domain | ISBN |
|---|-------|-----------|------|-------|--------|------|
| 26 | **Counterfactuals and Causal Inference** | Morgan, Winship | 2014 | $55 | causal_inference | 978-1107694163 |
| 27 | **Heard in Data Science Interviews** | Kal Mishra | 2022 | $25 | interview_prep | — |
| 28 | **Clean Code in Python** (2nd ed) | Mariano Anaya | 2021 | $35 | software_engineering | 978-1800560215 |
| 29 | **Experimentation Works** | Stefan Thomke | 2020 | $30 | statistics | 978-1633697102 |
| 30 | **Fifty Challenging Problems in Probability** | Mosteller | 1987 | $15 | statistics | 978-0486653556 |
| 31 | **Heard on the Street** | Timothy Crack | 2021 | $35 | finance | 978-0994103864 |
| 32 | **Challenging Brainteasers for Interviews** | FE Press | 2020 | $25 | finance | — |
| 33 | **Data-Driven Science and Engineering** | Brunton, Kutz | 2022 | $50 | machine_learning | 978-1009098489 |
| 34 | **Foundations of Deep RL** | Graesser, Keng | 2022 | $40 | reinforcement_learning | 978-0135172384 |
| 35 | **Multi-Agent RL** | Albrecht, Christianos, Schafer | 2024 | $60 | reinforcement_learning | 978-0262048019 |
| 36 | **Lean Analytics** | Croll, Yoskovitz | 2013 | $30 | data_science | 978-1449335670 |
| 37 | **Product Analytics** | Joanne Levin | 2022 | $35 | data_science | 978-1098131067 |
| 38 | **Fearless Salary Negotiation** | Josh Doody | 2023 | $20 | — | — |
| 39 | **Cracking the Machine Learning Interview** | Nitin Suri | 2019 | $30 | interview_prep | — |

**Subtotal**: ~$485

---

## Cost Summary

| Tier | Items | Full Price | With O'Reilly |
|------|-------|-----------|---------------|
| Critical (domain gaps) | 4 | $230 | $230 (not on O'Reilly) |
| High (interview + cross-project) | 8 | $380 | $300 (DDIA on O'Reilly) |
| Medium (strengthening) | 8 | $455 | $155 (5 on O'Reilly) |
| Post-transformers | 5 | $420 | $420 (not on O'Reilly) |
| Low (nice-to-have) | 14 | $485 | $390 (3 on O'Reilly) |
| **Total** | **39** | **$1,970** | **$1,495** |
| O'Reilly subscription | — | — | $499/yr |
| **Net with O'Reilly** | — | — | **$994 + $499/yr** |

## O'Reilly Learning Coverage

$499/yr covers these items (saves ~$475 vs individual purchase):

| Book | Individual Price |
|------|-----------------|
| SQL Cookbook | — (not on O'Reilly, must buy) |
| DDIA (Kleppmann) | $45 |
| Fluent Python | $50 |
| Pattern Recognition (Bishop) | $60 |
| Hands-On ML (Geron) | $55 |
| Deep RL Hands-On (Lapan) | $50 |
| Clean Code in Python | $35 |
| Lean Analytics | $30 |
| Recommender Systems Handbook | $80 |
| Credit Risk Scorecards | $60 |
| **Total covered** | **$465** |

---

## Free Resources — Status

| Title | Status | Notes |
|-------|--------|-------|
| RL: An Introduction (Sutton) | **Ingested** | In DB |
| Deep Learning (Goodfellow) | **Ingested** | In DB |
| Intro to Statistical Learning | **Ingested** | In DB |
| Elements of Statistical Learning | **Ingested** | In DB |
| Convex Optimization (Boyd) | **Ingested** | In DB |
| Think Stats (Downey) | **Ingested** | In DB |
| Causal Inference: The Mixtape | **Ingested** | In DB |
| Feedback Systems (Astrom & Murray) | **Downloaded** | fixtures/textbooks/, pending ingestion |
| Mathematical Control Theory (Sontag) | **Downloaded** | fixtures/textbooks/, pending ingestion |
| What If (Hernan, Robins) | **Ingested** | In DB |
| OpenIntro Statistics | **Ingested** | In DB |
| Forecasting: Principles & Practice | Unavailable | Web-only, no downloadable PDF |

---

## arXiv Papers Needing Attention

From `post_transformers/references/wish_list.md`:

| arXiv ID | Title | Issue | Action |
|----------|-------|-------|--------|
| 2212.14052 | H3 (Hungry Hungry Hippos) | Poor ingestion quality | Re-ingest with better extraction |
| 2412.06464 | Gated DeltaNet | Poor ingestion quality | Re-ingest with better extraction |

---

## Subscriptions & Platforms

| Platform | Cost/yr | Purpose | Priority |
|----------|---------|---------|----------|
| O'Reilly Learning | $499 | Covers ~10 books above | High (if buying 5+) |
| LeetCode Premium | $159 | Company-tagged problems | High (active interviewing) |
| Interview Query | $180 | SQL, product sense, ML | Medium |
| NeetCode Pro | $99 | Curated roadmaps | Medium |
| ByteByteGo | $79 | System design diagrams | Medium |

---

## Datasets

| Dataset | URL | Domain | Status |
|---------|-----|--------|--------|
| NSW Dataset | users.nber.org/~rdehejia/data/ | causal_inference | Documented |
| David Card's Data | davidcard.berkeley.edu/data_sets | econometrics | Documented |
| Dominick's Grocery | chicagobooth.edu | economics | Documented |
| FRED | fred.stlouisfed.org | time_series | Built-in (double_ml_time_series) |
| Human Mortality Database | mortality.org | actuarial_insurance | Stub only |

---

## Priority Action Plan

### Done This Session (2026-03-28)
1. Downloaded Feedback Systems + Sontag (free PDFs)
2. Started S2 auto-discovery for 5 thin domains
3. Updated this list with post_transformers + cross-project needs

### Next
4. Ingest 2 downloaded free books (three-phase workflow)
5. Complete S2 discovery, ingest free papers
6. Re-ingest 2 poor-quality post_transformers papers
7. Buy Critical tier ($230) — SQL Cookbook, Recommender handbook, Computational Advertising

### When Budget Allows
8. Buy High tier ($380) — interview prep books, Imbens & Rubin
9. Consider O'Reilly subscription ($499/yr) for Medium tier
10. Buy post_transformers books ($420) if research continues

---

*Cross-referenced against: research-kb DB, post_transformers/references/wish_list.md,
interview_prep_series volumes, annuity-pricing/references/, causal_inference_mastery,
course_learning/manning directories, and all 36 ~/Claude/ projects.*
