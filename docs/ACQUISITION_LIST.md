# Master Acquisition List (2026-03-28)

Consolidated from ALL ~/Claude/ projects:
- research-kb (this repo)
- julia_cas_exploration/docs/acquisition_wishlist.md (13 CAS textbooks)
- lever_of_archimedes/knowledge/master_bibliography/ (35 books + 59 papers)
- annuity-pricing/docs/references/ (23 papers + 4 books)
- post_transformers/references/wish_list.md (9 books + papers)
- precision_aq/reference_cache/WISHLIST.md (1 item)
- research-kb/docs/ACQUISITION_WISHLIST.md (SQL/recommender/adtech primary sources)

Run `python scripts/generate_status.py` for current DB metrics.

---

## Summary

| Category | Items | Estimated Cost | Auto-acquirable |
|----------|------:|---------------:|:---:|
| Books to buy | 78 | ~$4,200 | No |
| Books on O'Reilly | 15 of 78 | Save ~$600 with $499/yr sub | — |
| Free books (online PDFs) | 5 | $0 | Yes |
| arXiv/open-access papers | ~80 | $0 | Yes (S2 + arXiv API) |
| Journal papers (paywalled) | ~30 | Sci-Hub/library | Partially |
| **Total unique items** | **~193** | **~$4,200 books + $0 papers** | |

---

## Already Acquired (This Session)

| Title | Domain | Source | Status |
|-------|--------|--------|--------|
| Feedback Systems (Astrom & Murray, 2nd ed) | dynamical_systems | fbswiki.org | Downloaded |
| Mathematical Control Theory (Sontag, 2nd ed) | dynamical_systems | sontaglab.org | Downloaded |
| S2 auto-discovery papers | sql, adtech, recommender_systems, economics, forecasting | Semantic Scholar | Running |

---

## Books to Buy — By Domain

### Computer Algebra / Symbolic Computation
*Source: julia_cas_exploration — 0/13 owned*

| # | Title | Authors | Year | Price | Priority | Cross-project |
|---|-------|---------|------|-------|----------|---------------|
| 1 | **Algorithms for Computer Algebra** | Geddes, Czapor, Labahn | 1992 | $60 | CRITICAL | julia_cas |
| 2 | **Modern Computer Algebra** (3rd ed) | von zur Gathen, Gerhard | 2013 | $70 | CRITICAL | julia_cas |
| 3 | **Symbolic Integration I** (2nd ed) | Bronstein | 2005 | $60 | CRITICAL | julia_cas |
| 4 | **Computer Algebra and Symbolic Computation** (2 vols) | Cohen, Joel S. | 2002-03 | $80 | HIGH | julia_cas |
| 5 | **A Course in Computational Algebraic Number Theory** | Cohen, Henri | 1993 | $50 | HIGH | julia_cas |
| 6 | **Ideals, Varieties, and Algorithms** (4th ed) | Cox, Little, O'Shea | 2015 | $60 | HIGH | julia_cas |
| 7 | **Groebner Bases** | Becker, Weispfenning | 1993 | $50 | HIGH | julia_cas |
| 8 | **Term Rewriting and All That** | Baader, Nipkow | 1998 | $55 | HIGH | julia_cas |
| 9 | Computer Algebra: Systems and Algorithms | Davenport, Siret, Tournier | 1993 | $50 | MEDIUM | julia_cas |
| 10 | Polynomial Algorithms in Computer Algebra | Winkler | 1996 | $50 | MEDIUM | julia_cas |
| 11 | Mathematics for Computer Algebra | Mignotte | 1992 | $50 | MEDIUM | julia_cas |
| 12 | Computer Algebra Handbook | Grabmeier et al. | 2003 | $60 | MEDIUM | julia_cas |
| 13 | The Computer Algebra System OSCAR | Decker et al. | 2024 | $60 | MEDIUM | julia_cas |

### SQL & Databases
*Source: research-kb WISHLIST + lever — currently 14 sources in DB*

| # | Title | Authors | Year | Price | Priority | Cross-project |
|---|-------|---------|------|-------|----------|---------------|
| 14 | **SQL Cookbook** (2nd ed) | Molinaro, de Graaf | 2020 | $45 | CRITICAL | interview_prep vol05 |
| 15 | **SQL Performance Explained** | Winand | 2012 | $35 | HIGH | lever, interview_prep |
| 16 | **Database Internals** | Petrov | 2019 | $50 | HIGH | research-kb WISHLIST |
| 17 | Learning SQL (3rd ed) | Beaulieu | 2020 | $45 | MEDIUM | research-kb WISHLIST |
| 18 | SQL Antipatterns | Karwin | 2010 | $40 | LOW | research-kb WISHLIST |

### Recommender Systems & AdTech
*Source: research-kb — currently 5 recommender, 3 adtech sources*

| # | Title | Authors | Year | Price | Priority | Cross-project |
|---|-------|---------|------|-------|----------|---------------|
| 19 | **Recommender Systems Handbook** (3rd ed) | Ricci, Rokach, Shapira | 2022 | $80 | CRITICAL | interview_prep vol19 |
| 20 | **Practical Recommender Systems** | Kim Falk | 2019 | $45 | HIGH | interview_prep vol19 |
| 21 | Recommender Systems: The Textbook | Aggarwal | 2016 | $60 | MEDIUM | research-kb WISHLIST |
| 22 | **Computational Advertising** | Yuan, Wang, Zhao | 2019 | $60 | CRITICAL | interview_prep vol12 |

### Causal Inference & Econometrics
*Source: research-kb + lever — well-covered but key texts missing*

| # | Title | Authors | Year | Price | Priority | Cross-project |
|---|-------|---------|------|-------|----------|---------------|
| 23 | **Imbens & Rubin** (Causal Inference) | Imbens, Rubin | 2015 | $80 | HIGH | causal_inference_mastery, lever |
| 24 | Counterfactuals and Causal Inference | Morgan, Winship | 2014 | $55 | MEDIUM | research-kb |
| 25 | Mastering 'Metrics | Angrist, Pischke | 2014 | $35 | MEDIUM | lever |
| 26 | Observational Studies | Rosenbaum | 2002 | $80 | MEDIUM | lever |
| 27 | Trustworthy Online Controlled Experiments | Kohavi, Tang, Xu | 2020 | $50 | HIGH | lever, interview_prep |
| 28 | Experimentation Works | Thomke | 2020 | $30 | LOW | research-kb |

### Finance, Risk & Actuarial
*Source: research-kb + lever + annuity-pricing*

| # | Title | Authors | Year | Price | Priority | Cross-project |
|---|-------|---------|------|-------|----------|---------------|
| 29 | **Quantitative Risk Management** | McNeil, Frey, Embrechts | 2015 | $85 | HIGH | lever |
| 30 | **Credit Risk Scorecards** | Siddiqi | 2017 | $60 | HIGH | interview_prep vol18 |
| 31 | Value at Risk: The New Benchmark | Jorion | 2006 | $75 | MEDIUM | lever |
| 32 | Measuring Market Risk | Dowd | 2007 | $95 | MEDIUM | lever |
| 33 | Hull's Options, Futures, Derivatives (11th ed) | Hull | 2021 | $100 | MEDIUM | annuity-pricing |
| 34 | RL for Finance | Hilpisch | 2024 | $50 | MEDIUM | research-kb |
| 35 | Practical Guide to Quant Finance Interviews | Joshi | 2008 | $45 | LOW | interview_prep |
| 36 | Heard on the Street | Crack | 2021 | $35 | LOW | interview_prep |

### Machine Learning & Deep Learning
*Source: research-kb + lever — good coverage, selective adds*

| # | Title | Authors | Year | Price | Priority | Cross-project |
|---|-------|---------|------|-------|----------|---------------|
| 37 | **Pattern Recognition and Machine Learning** | Bishop | 2006 | $60 | HIGH | lever, research-kb |
| 38 | **Hands-On Machine Learning** (3rd ed) | Geron | 2022 | $55 | HIGH | research-kb |
| 39 | Learning From Data | Abu-Mostafa | 2012 | $50 | MEDIUM | lever |
| 40 | Elements of Information Theory | Cover, Thomas | 2006 | $135 | MEDIUM | lever |
| 41 | Deep RL Hands-On (3rd ed) | Lapan | 2024 | $50 | MEDIUM | research-kb |
| 42 | Foundations of Deep RL | Graesser, Keng | 2022 | $40 | LOW | research-kb |
| 43 | Multi-Agent RL | Albrecht et al. | 2024 | $60 | LOW | research-kb |

### Interview Prep & System Design
*Source: research-kb*

| # | Title | Authors | Year | Price | Priority | Cross-project |
|---|-------|---------|------|-------|----------|---------------|
| 44 | **Elements of Programming Interviews in Python** | Aziz, Lee, Prakash | 2018 | $40 | HIGH | interview_prep vol04 |
| 45 | **System Design Interview Vol 1** | Alex Xu | 2020 | $35 | HIGH | interview_prep vol15 |
| 46 | **System Design Interview Vol 2** | Alex Xu | 2022 | $40 | HIGH | interview_prep vol15 |
| 47 | **Designing Data-Intensive Applications** | Kleppmann | 2017 | $45 | HIGH | lever, interview_prep |
| 48 | ML System Design Interview | Xu, Aminian | 2022 | $40 | HIGH | interview_prep vol09 |
| 49 | Generative AI System Design Interview | Alex Xu | 2024 | $40 | MEDIUM | interview_prep vol08 |
| 50 | Heard in Data Science Interviews | Mishra | 2022 | $25 | LOW | interview_prep |
| 51 | Cracking the Machine Learning Interview | Suri | 2019 | $30 | LOW | interview_prep |

### Software Engineering & Python
*Source: research-kb + lever*

| # | Title | Authors | Year | Price | Priority | Cross-project |
|---|-------|---------|------|-------|----------|---------------|
| 52 | **Fluent Python** (2nd ed) | Ramalho | 2022 | $50 | HIGH | lever, research-kb |
| 53 | **Effective Python** (3rd ed) | Slatkin | 2024 | $45 | MEDIUM | lever, research-kb |
| 54 | Design Patterns | Gamma et al. | 1994 | $55 | MEDIUM | lever |
| 55 | Refactoring (2nd ed) | Fowler | 2018 | $50 | MEDIUM | lever |
| 56 | Clean Code in Python (2nd ed) | Anaya | 2021 | $35 | LOW | research-kb |

### Post-Transformers Research (SSM/Control Theory)
*Source: post_transformers — specialized research needs*

| # | Title | Authors | Year | Price | Priority | Cross-project |
|---|-------|---------|------|-------|----------|---------------|
| 57 | **Matrix Analysis** | Horn, Johnson | 2013 | $80 | HIGH | post_transformers |
| 58 | **Hamilton Time Series Analysis** | Hamilton | 1994 | $120 | HIGH | lever, post_transformers |
| 59 | **Solving ODEs II** (Stiff problems) | Hairer, Wanner | 1996 | $80 | MEDIUM | post_transformers |
| 60 | **Linear Systems** | Kailath | 1980 | $100 | MEDIUM | post_transformers |
| 61 | Introduction to Mechanics and Symmetry | Marsden, Ratiu | 1999 | $80 | LOW | post_transformers |

### Philosophy of Mind & Cognition
*Source: lever_of_archimedes — world models research*

| # | Title | Authors | Year | Price | Priority | Cross-project |
|---|-------|---------|------|-------|----------|---------------|
| 62 | Thinking, Fast and Slow | Kahneman | 2011 | $18 | MEDIUM | lever |
| 63 | Surfing Uncertainty | Clark | 2016 | $35 | MEDIUM | lever |
| 64 | The Embodied Mind | Varela, Thompson, Rosch | 1991 | $30 | MEDIUM | lever |
| 65 | Being and Time | Heidegger | 1927 | $22 | LOW | lever |
| 66 | Phenomenology of Perception | Merleau-Ponty | 1945 | $55 | LOW | lever |
| 67 | The Conscious Mind | Chalmers | 1996 | $25 | LOW | lever |
| 68 | Cognition in the Wild | Hutchins | 1995 | $40 | LOW | lever |
| 69 | What Computers Still Can't Do | Dreyfus | 1992 | $35 | LOW | lever |
| 70 | Supersizing the Mind | Clark | 2008 | $30 | LOW | lever |
| 71 | Design for a Brain | Ashby | 1952 | $20 | LOW | lever |
| 72 | The Nature of Explanation | Craik | 1943 | $45 | LOW | lever |
| 73 | Tractatus Logico-Philosophicus | Wittgenstein | 1921 | $8 | LOW | lever |

### Data Science & Analytics
*Source: research-kb*

| # | Title | Authors | Year | Price | Priority | Cross-project |
|---|-------|---------|------|-------|----------|---------------|
| 74 | Data-Driven Science and Engineering | Brunton, Kutz | 2022 | $50 | MEDIUM | research-kb |
| 75 | Lean Analytics | Croll, Yoskovitz | 2013 | $30 | LOW | research-kb |
| 76 | Product Analytics | Levin | 2022 | $35 | LOW | research-kb |
| 77 | Fifty Challenging Problems in Probability | Mosteller | 1987 | $15 | LOW | research-kb |
| 78 | Fearless Salary Negotiation | Doody | 2023 | $20 | LOW | — |

---

## Papers to Acquire (Paywalled)

### Actuarial / Variable Annuity Valuation
*Source: annuity-pricing — 23 papers, most with DOIs*

| Paper | Authors | Year | DOI | Priority |
|-------|---------|------|-----|----------|
| Financial valuation of GMWB | Milevsky, Salisbury | 2006 | 10.1016/j.insmatheco.2005.06.012 | HIGH |
| Titanic Option (GMDB) | Milevsky, Posner | 2001 | 10.2307/2678133 | HIGH |
| Universal Pricing Framework GMxB | Bauer, Kling, Russ | 2008 | 10.1017/S0515036100015269 | HIGH (FREE) |
| Optimal GLWB initiation | Huang, Milevsky, Salisbury | 2014 | 10.1016/j.insmatheco.2014.04.001 | HIGH |
| Effect of parameters on GMWB | Chen, Vetzal, Forsyth | 2008 | 10.1016/j.insmatheco.2008.04.003 | MEDIUM |
| GMWB in VA (PDE approach) | Dai, Kwok, Zong | 2008 | 10.1111/j.1467-9965.2008.00349.x | MEDIUM |
| VA unifying valuation | Bacinello et al. | 2011 | 10.1016/j.insmatheco.2011.05.003 | MEDIUM |
| + 16 more in annuity-pricing/docs/references/acquisition_list_extended.md | | | | |

### Causal Inference / Econometrics
*Source: lever — 15+ papers with DOIs, many in research-kb already*

Key missing papers (not yet in DB):
- Synthetic Control Methods (Abadie et al., 2010)
- Recursive Partitioning for HTE (Athey & Imbens, 2016)
- Moving Towards Best Practice IPTW (Austin & Stuart, 2015)
- Heston Closed-Form Solution (1993)

### Post-Transformers
*Source: post_transformers — 2 papers need re-ingestion*

| arXiv ID | Title | Issue |
|----------|-------|-------|
| 2212.14052 | H3 (Hungry Hungry Hippos) | Poor ingestion quality |
| 2412.06464 | Gated DeltaNet | Poor ingestion quality |

---

## Free Resources — Status

| Title | Domain | Status | Source |
|-------|--------|--------|--------|
| Feedback Systems (Astrom & Murray) | dynamical_systems | **Downloaded** | fbswiki.org |
| Mathematical Control Theory (Sontag) | dynamical_systems | **Downloaded** | sontaglab.org |
| RL: An Introduction (Sutton) | reinforcement_learning | **In DB** | incompleteideas.net |
| Deep Learning (Goodfellow) | deep_learning | **In DB** | deeplearningbook.org |
| Intro to Statistical Learning | statistics | **In DB** | statlearning.com |
| Elements of Statistical Learning | statistics | **In DB** | stanford.edu |
| Convex Optimization (Boyd) | optimization | **In DB** | stanford.edu |
| Causal Inference: The Mixtape | causal_inference | **In DB** | mixtape.scunning.com |
| Think Stats (Downey) | statistics | **In DB** | greenteapress.com |
| What If (Hernan, Robins) | causal_inference | **In DB** | hsph.harvard.edu |

---

## O'Reilly Learning Coverage

$499/yr covers ~15 items above (saves ~$600):

Books #14, 16, 17, 20, 37, 38, 41, 47, 52, 53, 56, 74, 75 and potentially others.

**Verdict**: Breaks even at 5+ books. Strongly recommended if buying Tier 1+2.

---

## Subscriptions

| Platform | Cost/yr | Priority |
|----------|---------|----------|
| O'Reilly Learning | $499 | High |
| LeetCode Premium | $159 | High (active interviewing) |
| Interview Query | $180 | Medium |
| NeetCode Pro | $99 | Medium |

---

## Auto-Acquisition Pipeline

### Already Running
- `s2_auto_discover.py` searching thin domains (sql, adtech, recommender_systems, economics, forecasting)
- Library book extraction (347 classified books, Docling processing)

### Available Tools
- `scripts/s2_auto_discover.py` — Semantic Scholar search + queue
- `scripts/classify_arxiv_papers.py` — Classify + sidecar generation
- `scripts/ingest_missing_textbooks.py` — Three-phase GPU ingestion
- `scripts/backfill_embeddings.py` — Embedding backfill
- `scripts/build_citation_graph.py` — Citation + PageRank

### Recommended Next Runs
```bash
# Thin domain paper discovery
python scripts/s2_auto_discover.py search "SQL query optimization" --domain sql
python scripts/s2_auto_discover.py search "real-time bidding CTR" --domain adtech
python scripts/s2_auto_discover.py search "collaborative filtering" --domain recommender_systems

# Lever papers (causal + finance)
python scripts/s2_auto_discover.py search "synthetic control causal" --domain causal_inference
python scripts/s2_auto_discover.py search "CVaR portfolio optimization" --domain finance
```

---

## Source Project References

| Project | Wish List Location | Items |
|---------|-------------------|-------|
| julia_cas_exploration | `docs/acquisition_wishlist.md` | 13 CAS textbooks |
| lever_of_archimedes | `knowledge/master_bibliography/ACQUISITION_LIST.md` | 35 books |
| lever_of_archimedes | `knowledge/master_bibliography/ACQUISITION_PRIORITY.md` | 59 papers |
| lever_of_archimedes | `knowledge/master_bibliography/acquisition_books.csv` | Structured CSV |
| lever_of_archimedes | `knowledge/master_bibliography/acquisition_papers.csv` | Structured CSV |
| annuity-pricing | `docs/references/acquisition_list_extended.md` | 23 papers + 4 books |
| annuity-pricing | `docs/references/acquisition_list_glwb_va.md` | GLWB-specific papers |
| post_transformers | `references/wish_list.md` | 9 books + papers |
| precision_aq | `reference_cache/WISHLIST.md` | 1 item (ASHP) |
| research-kb | `docs/ACQUISITION_WISHLIST.md` | SQL/recommender/adtech |

---

*Consolidated 2026-03-28 from 10 source files across 7 projects.*
*Previous version (33 items) superseded.*
