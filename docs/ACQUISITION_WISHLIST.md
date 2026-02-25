# Primary Textbook Acquisition Wishlist

> Canonical primary sources for empty and thin KB domains.
>
> **Principle**: Only ingest primary sources. Interview prep volumes are *synthesis* — the KB exists to *inform* those volumes, not consume them.
>
> Last updated: 2026-02-25 (Phase O)

---

## Empty Domains (Priority Acquisitions)

### SQL & Databases (`sql` — 0 sources)

| Title | Author(s) | Year | Why Canonical | Availability |
|-------|-----------|------|---------------|--------------|
| SQL Performance Explained | Markus Winand | 2012 | Definitive guide to indexing and query optimization across all major RDBMSs | Publisher direct |
| Database Internals | Alex Petrov | 2019 | O'Reilly. Covers storage engines, distributed systems, B-trees, LSM-trees | O'Reilly |
| Learning SQL | Alan Beaulieu | 2020 (3rd ed) | O'Reilly. Comprehensive SQL fundamentals with exercises | O'Reilly |
| SQL Antipatterns | Bill Karwin | 2010 | Pragmatic. Common SQL mistakes and correct patterns | Publisher direct |
| Designing Data-Intensive Applications | Martin Kleppmann | 2017 | O'Reilly. Foundational for storage, replication, partitioning concepts | O'Reilly |

**Priority**: High. SQL is a universal interview topic with zero current coverage.

### Recommender Systems (`recommender_systems` — 0 sources)

| Title | Author(s) | Year | Why Canonical | Availability |
|-------|-----------|------|---------------|--------------|
| Recommender Systems: The Textbook | Charu C. Aggarwal | 2016 | Springer. Comprehensive textbook covering CF, content-based, knowledge-based, hybrid | Springer |
| Recommender Systems Handbook | Ricci, Rokach, Shapira | 2022 (3rd ed) | Springer. Multi-author reference covering all major approaches | Springer |
| Practical Recommender Systems | Kim Falk | 2019 | Manning. Hands-on approach with production considerations | Manning |

**Priority**: Medium. Relevant for ML/product roles. Check Manning archive for Falk book availability.

### AdTech (`adtech` — 0 sources)

| Title | Author(s) | Year | Why Canonical | Availability |
|-------|-----------|------|---------------|--------------|
| Computational Advertising | Jun Wang, Weinan Zhang, Shuai Yuan | 2023 | Springer. Covers auction mechanisms, CTR prediction, attribution | Springer |
| Display Advertising with Real-Time Bidding | Jun Wang et al. | 2017 | Comprehensive RTB overview; freely available | arXiv/publisher |

**Note**: AdTech has fewer canonical textbooks than other domains. Supplement with Semantic Scholar papers on:
- Real-time bidding mechanisms (Vickrey auctions, GSP)
- CTR prediction (DeepFM, DCN, feature interactions)
- Causal attribution and incrementality testing
- Budget optimization and pacing algorithms

**Priority**: Low. Niche domain; overlap with causal_inference for incrementality.

---

## Thin Domains (Expansion Needed)

### Forecasting (`forecasting` — 1 source)

| Title | Author(s) | Year | Why Canonical | Availability |
|-------|-----------|------|---------------|--------------|
| Forecasting: Principles and Practice | Hyndman & Athanasopoulos | 2021 (3rd ed) | *The* reference for applied forecasting. Free online (OTexts) | Free (OTexts.com) |
| Time Series Forecasting in Python | Marco Peixeiro | 2022 | Manning. Already in `fixtures/textbooks/` as tier1_03 — verify ingested | Manning (owned) |

**Note**: `time_series` (47 sources) covers much foundational material. Forecasting domain focuses on applied methods, production systems, and evaluation metrics distinct from pure time series theory.

### Economics (`economics` — 1 source)

| Title | Author(s) | Year | Why Canonical | Availability |
|-------|-----------|------|---------------|--------------|
| Microeconomic Theory | Mas-Colell, Whinston, Green | 1995 | Graduate micro textbook, foundational | Publisher |
| Macroeconomics | Blanchard | 2021 (8th ed) | Standard macro textbook | Publisher |
| Quantitative Economics with Julia | Sargent & Stachurski | 2024 | Already in corpus. Verify domain tag | In corpus |

### Finance (`finance` — 1 source)

| Title | Author(s) | Year | Why Canonical | Availability |
|-------|-----------|------|---------------|--------------|
| Options, Futures, and Other Derivatives | John C. Hull | 2021 (11th ed) | *The* derivatives textbook | Publisher |
| Investment Science | David Luenberger | 2013 (2nd ed) | Comprehensive quantitative finance foundations | Publisher |

**Note**: 11 portfolio_management sources exist in DB (being added to domain_prompts in Phase O). Some finance-adjacent content may live under portfolio_management.

---

## Manning Archive Cross-Reference

Books in `fixtures/textbooks/` that may overlap with wishlist domains:

| Sidecar File | Possible Domain | Status |
|-------------|----------------|--------|
| `demand_forecasting_best_practices_nd.json` | forecasting | Already ingested |
| `data_analysis_with_python_and_pyspark_v13_meap_nd.json` | data_science | Already ingested |
| `get_programming_with_haskell_nd.json` | functional_programming | Already ingested |
| `functional_programming_in_scala_second__v2_meap_nd.json` | functional_programming | Already ingested |

No Manning titles found for: sql, recommender_systems, adtech. "Practical Recommender Systems" (Falk, Manning) may be available but is not in the local archive.

---

## Acquisition Process

1. **Check availability**: O'Reilly subscription, Manning archive, free/open access
2. **Download PDF**: Place in `fixtures/textbooks/` with sidecar JSON
3. **Ingest**: `python scripts/ingest_missing_textbooks.py --quiet`
4. **Extract concepts**: `python scripts/extract_concepts.py --backend anthropic --model haiku`
5. **Sync KuzuDB**: `python scripts/sync_kuzu.py`
6. **Activate future tests**: Remove `future` tag from retrieval test cases
7. **Update this document**: Mark as acquired

---

## Status Key

- **Needed**: Not yet acquired
- **Owned**: In local archive, not yet ingested
- **Ingested**: In database with embeddings
- **Extracted**: Concepts extracted and synced to KuzuDB
