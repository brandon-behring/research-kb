# Vol23 Optimization Theory — Acquisition List

Items that cannot be legally obtained for free and need to be purchased, borrowed, or requested through institutional access. Organized by priority for the vol23 build.

**Status**: 2026-04-13
**Already free and staged**: See `MANIFEST.md` (6 titles, 1,273 pages)
**Already in research-kb**: Boyd & Vandenberghe *Convex Optimization* (full), Bertsekas RL/DP quadrology (4 books), Sutton & Barto *RL 2e*, Szepesvári *Algorithms for RL*, MIT 6.253 lecture notes, Khamis *Optimization Algorithms*
**This list**: What's still missing

---

## P1 — Blocking for Ch 5, 7, 8 (numerical + LP rigor)

| Title | Author(s) | Publisher | Why | Cost | Notes |
|---|---|---|---|---|---|
| **Numerical Optimization (2e, 2006)** | Nocedal & Wright | Springer | Ch 5 L-BFGS, trust region, line search — no free substitute exists | ~$60 | Springer eBook sometimes free via institutional access; author-hosted errata only |
| **Introduction to Linear Optimization (1997)** | Bertsimas & Tsitsiklis | Athena Scientific | Ch 7 LP rigor (simplex proofs, interior point derivations) | ~$100 | Not available digitally from publisher; used hardcover common |
| **Nonlinear Programming (3e, 2016)** | Dimitri Bertsekas | Athena Scientific | Ch 3, 5 constrained optimization theorems beyond what Boyd covers | ~$100 | Athena Scientific never releases free drafts of this one (unlike his RL/DP books) |

---

## P2 — Fills Ch 6, 8, 9, 11 (stochastic + combinatorial + DP)

| Title | Author(s) | Publisher | Why | Cost | Notes |
|---|---|---|---|---|---|
| **Reinforcement Learning and Stochastic Optimization (2022)** | Warren Powell | Wiley | Ch 6, 11 — unified stochastic programming + ADP view | ~$140 | Some draft chapters free on Powell's Princeton page; full book paid |
| **Combinatorial Optimization: Algorithms and Complexity (1982/1998 Dover reprint)** | Papadimitriou & Steiglitz | Dover | Ch 8, 9 — complexity classes, cuts, flows | ~$20 | Dover reprint is cheap. Priority-P2 because of price. |
| **Combinatorial Optimization: Theory and Algorithms (6e, 2018)** | Korte & Vygen | Springer | Ch 9 — definitive matching, network flow, augmenting paths | ~$90 | Springer; often discounted |
| **Algorithms for Decision Making (2022)** | Kochenderfer, Wheeler, Wray | MIT Press | Ch 10-12 — unified DP/RL treatment, supplements Bertsekas | **Free (CC-BY-NC-ND)** | **Official PDF is gated behind Google Drive at `algorithmsbook.com/decisionmaking/`** — not direct-linkable but legally free. User action: click through site and drop PDF into staging dir. |

---

## P3 — RL theory, accelerated methods, online matching

| Title | Author(s) | Publisher | Why | Cost | Notes |
|---|---|---|---|---|---|
| **Lectures on Convex Optimization (2e, 2018)** | Yurii Nesterov | Springer | Ch 4 — accelerated gradient methods, complexity lower bounds | ~$70 | Nesterov's 1998 *Introductory Lectures* is cheaper and covers similar ground |
| **Online Matching and Ad Allocation (2013)** | Aranyak Mehta | NOW Publishers (F&T monograph) | Ch 9 — definitive RANKING algorithm, competitive ratios for rideshare dispatch | ~$100 | F&T monographs are expensive for their length; some chapters may appear as author preprints |
| **First-Order Methods in Optimization (2017)** | Amir Beck | SIAM | Ch 4, 6 — modern first-order treatment, proximal methods | ~$90 | SIAM member discount available |

---

## P4 — Mechanism design, auctions, game theory (Ch 13 marketplace depth)

| Title | Author(s) | Publisher | Why | Cost | Notes |
|---|---|---|---|---|---|
| **Twenty Lectures on Algorithmic Game Theory (2016)** | Tim Roughgarden | Cambridge | Ch 13 — Price of Anarchy, mechanism design lectures | ~$35 | Some lecture videos free on YouTube; book paid |
| **Algorithmic Game Theory (2007)** | Nisan, Roughgarden, Tardos, Vazirani (eds) | Cambridge | Ch 13 — canonical reference; Hartline covers mechanism design subset | ~$90 | **Cambridge allows free online hosting with permission** — several instructor pages (CMU, Penn) host PDFs; status ambiguous, treat as acquisition |
| **Putting Auction Theory to Work (2004)** | Paul Milgrom | Cambridge | Ch 13 — auction theory for surge pricing | ~$60 | Milgrom's Nobel-winning treatment |

---

## P5 — Model building and production optimization (Ch 14 depth)

| Title | Author(s) | Publisher | Why | Cost | Notes |
|---|---|---|---|---|---|
| **Model Building in Mathematical Programming (5e, 2013)** | H. Paul Williams | Wiley | Ch 7, 8 — LP/IP modeling patterns with real-world case studies | ~$70 | Closest thing to a production-optimization handbook |
| **Lectures on Stochastic Programming (3e, 2021)** | Shapiro, Dentcheva, Ruszczyński | SIAM | Ch 6 — stochastic programming rigor beyond Hazan OCO | ~$90 | Closes a real gap Hazan's OCO doesn't cover (two-stage stochastic programs) |
| **Robust Optimization (2009)** | Ben-Tal, El Ghaoui, Nemirovski | Princeton | Ch 13 — robust pricing under demand uncertainty | ~$80 | Niche but high-value for adversarial pricing topics |

---

## P6 — Optimal control (Ch 11 LQR/MPC depth)

| Title | Author(s) | Publisher | Why | Cost | Notes |
|---|---|---|---|---|---|
| **Optimal Control: Linear Quadratic Methods (1990/2007 Dover reprint)** | Anderson & Moore | Dover | Ch 11 — LQR depth if we go beyond a sketch | ~$25 | Dover cheap |
| **Optimal Control Theory: An Introduction (1970/2004 Dover reprint)** | Donald Kirk | Dover | Ch 11 — accessible optimal control intro | ~$25 | Dover cheap |
| **Predictive Control for Linear and Hybrid Systems (2017)** | Borrelli, Bemporad, Morari | Cambridge | Ch 11 — MPC depth if receding horizon gets full treatment | ~$85 | Only if Ch 11 MPC section expands beyond a sketch |

---

## Budget Summary

| Priority | Count | Min cost (used/Dover) | Max cost (new hardcover) |
|---|---|---|---|
| P1 (blocking) | 3 | ~$120 | ~$260 |
| P2 (fills critical gaps, one free) | 4 | ~$160 | ~$350 |
| P3 (nice-to-have theory depth) | 3 | ~$130 | ~$260 |
| P4 (marketplace depth) | 3 | ~$115 | ~$185 |
| P5 (production + stochastic programming) | 3 | ~$150 | ~$240 |
| P6 (optimal control) | 3 | ~$50 | ~$135 |
| **Total** | **19** | **~$725** | **~$1,430** |

**Minimum viable P1 acquisition**: ~$120-260 unlocks Ch 5, 7, 8 rigor.

**Free action items (zero cost)**:
1. Click through `algorithmsbook.com/decisionmaking/` to grab the CC-licensed Kochenderfer DM PDF and drop it into the staging dir — officially free, just Google-Drive-gated.

---

## Library / Alternative Access Suggestions

Before buying, check:
1. **Springer institutional access** via any university library (covers Nocedal & Wright, Korte & Vygen, Nesterov)
2. **Cambridge institutional access** (covers Roughgarden, Nisan et al., Milgrom)
3. **Wiley Online Library** (covers Powell, Williams)
4. **Athena Scientific** never offers institutional access — these must be bought (Bertsekas NLP, Bertsimas & Tsitsiklis LP)
5. **SIAM books** — member discounts substantial; consider SIAM student/early-career membership if >3 SIAM titles needed
6. **Dover reprints** — always cheap and legitimate (Papadimitriou & Steiglitz, Anderson & Moore, Kirk)
7. **Interlibrary loan (ILL)** — free through any university library; 2-4 week turnaround

---

## Recommendation

**Phase 1 (build vol23 Ch 00-6 with current resources)**: Proceed with what's already free and in research-kb. The 6 newly-staged titles + existing Boyd + Bertsekas quadrology + Sutton/Barto cover ~75% of the writing needs for Ch 00 through Ch 6.

**Phase 2 (acquire P1 before Ch 7)**: Budget ~$150 for Nocedal & Wright + Bertsimas & Tsitsiklis used copies before starting Ch 7 (LP) and Ch 8 (IP). These unblock rigorous treatments that the currently-free corpus can't deliver.

**Phase 3 (optional P2-P6 as content pressures appear)**: Defer all P2-P6 acquisitions until a specific chapter reveals a content gap. Many will turn out not to be needed.

**Zero-cost action that should happen regardless**: Grab Kochenderfer *Algorithms for Decision Making* from the official CC-BY-NC-ND source manually (it's legitimately free, just not directly curlable).
