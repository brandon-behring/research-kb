# Reference Cache Index

Generated: 2026-03-26
Source: `docs/mco_payer_reference/references.bib` (37 refs) + 7 audit supplements = 44 total

## Summary

| Metric | Count |
|--------|-------|
| Total references | 44 |
| Markdown snapshots cached | 32 |
| PDFs cached | 3 (4.8 MB) |
| Wishlist items | 9 (see WISHLIST.md) |
| Content coverage | 80% (35/44 fully cached) |

All 8 wishlist PMC PDFs have full-text markdown alternatives, so **effective content coverage is 98%** (only ASHP, ICER, MMIT x2, and 5 gov pages lack any cached version).

---

## Cached PDFs

| File | Bib Key | Title | Size |
|------|---------|-------|------|
| `pdfs/R09_amcp-format-5.pdf` | R09 | AMCP Format for Formulary Submissions 5.0 | 912 KB |
| `pdfs/R17_oecd-who-purchasing-quality.pdf` | R17 | OECD/WHO Purchasing for Quality Chronic Care | 3.5 MB |
| `pdfs/S03_cms-partd-manual-ch6.pdf` | S03 | CMS Part D Benefits Manual, Chapter 6 | 439 KB |

---

## Cached Markdown Snapshots

### Definitions & Regulation (R01-R08)

| File | Bib Key | Title | Method |
|------|---------|-------|--------|
| `html/R01_cms-multipayer.md` | R01 | CMS Multi-Payer Alignment | Playwright |
| `html/R04_kff-medicaid-mc-rules.md` | R04 | KFF: Medicaid MC Amid New Federal Rules | WebFetch |
| `html/R05_kff-10-things-medicaid-mc.md` | R05 | KFF: 10 Things About Medicaid Managed Care | WebFetch |
| `html/R06_kff-mcpar-reporting.md` | R06 | KFF: Medicaid MC Reporting (MCPAR) | WebFetch |
| `html/R07_kff-ma-enrollment-2025.md` | R07 | KFF: Medicare Advantage in 2025 | WebFetch |
| `html/R08_kff-pbm-regulation.md` | R08 | KFF: PBMs and Federal Regulation | WebFetch |

### Evidence & Value Communication (R09-R15)

| File | Bib Key | Title | Method |
|------|---------|-------|--------|
| `html/R10_PMC10387941.md` | R10 | Dodda et al.: PIE Best Practices (JMCP 2023) | WebFetch |
| `html/R11_brisibe-hcdm-perceptions.md` | R11 | Brisibe et al.: HCDM PIE Perceptions (JMCP 2024) | WebFetch |
| `html/R12_brixner-rwe-oncology.md` | R12 | Brixner et al.: Payer RWE in Oncology (JMCP 2021) | WebFetch |
| `html/R13_PMC12653624.md` | R13 | Chambers et al.: AMCP RWE Standards (JMCP 2025) | WebFetch |
| `html/R14_PMC12450548.md` | R14 | Abu-Shraie et al.: RWE Scoping Review (2025) | WebFetch |
| `html/R15_PMC10387901.md` | R15 | Hydery et al.: Value Assessment Tools (JMCP 2023) | WebFetch |

### Global Context (R16-R17)

| File | Bib Key | Title | Method |
|------|---------|-------|--------|
| `html/R16_who-strategic-purchasing.md` | R16 | WHO: Strategic Purchasing for UHC (2017) | WebFetch |

### Market Structure

| File | Bib Key | Title | Method |
|------|---------|-------|--------|
| `html/ncbi-mco-statpearls.md` | ncbi-mco-statpearls | StatPearls: Managed Care Organization | WebFetch |
| `html/healthinsurance-org-plan-types.md` | healthinsurance-org-plan-types | HMO vs PPO vs POS vs EPO | WebFetch |
| `html/PMC10391133_amcp-pt-principles.md` | amcp-pt-principles | AMCP P&T Committee Principles (JMCP 2020) | WebFetch |
| `html/PMC10701257_jmcp-managed-care-primer.md` | jmcp-managed-care-primer | Managed Care Pharmacy Primer (JMCP 2023) | WebFetch |
| `html/PMC10390926_jmcp-acos-medication.md` | jmcp-acos-medication | ACOs and Medication Use (JMCP 2023) | WebFetch |

### Market Access Strategy & Trends

| File | Bib Key | Title | Method |
|------|---------|-------|--------|
| `html/guidehouse-market-access-2025.md` | guidehouse-market-access-2025 | Guidehouse: Market Access as Growth Engine | WebFetch |
| `html/definitivehc-idn-guide.md` | definitivehc-idn-guide | Definitive HC: Guide to Approaching IDNs | WebFetch |
| `html/drugchannels-2025-exclusions.md` | drugchannels-2025-exclusions | Drug Channels: Big Three PBMs 2025 Exclusions | WebFetch |
| `html/simon-kucher-obc.md` | simon-kucher-obc | Simon-Kucher: Outcome-Driven Payer Contracting | WebFetch |
| `html/intuitionlabs-vbc.md` | intuitionlabs-vbc | IntuitionLabs: Value-Based Contracting | WebFetch |
| `html/amcp-emerging-trends-2025.md` | amcp-emerging-trends-2025 | AMCP: Emerging Trends 2025 | WebFetch |
| `html/psg-specialty-2025.md` | psg-specialty-2025 | PSG: Specialty Drug Cost Solutions | WebFetch |
| `html/cms-drug-negotiation.md` | cms-drug-negotiation | CMS: Medicare Drug Price Negotiation | Playwright |
| `html/cms-cgt-model.md` | cms-cgt-model | CMS: Cell and Gene Therapy Access Model | Playwright |
| `html/pharmexec-icer-trends.md` | pharmexec-icer-trends | PharmExec: Payer Reliance on ICER | WebFetch |
| `html/precisionaq-market-access.md` | precisionaq-market-access | Precision AQ: Market Access | WebFetch |

### Audit Supplements (S01-S07)

| File | Bib Key | Title | Method |
|------|---------|-------|--------|
| `html/S05_usc-21-352.md` | S05 | 21 U.S.C. § 352 — Misbranded Drugs | WebFetch |
| `html/S06_usc-42-1396r8.md` | S06 | 42 U.S.C. § 1396r-8 — Medicaid Drug Rebate | WebFetch |
| `html/S07_PMC11068649_pie-survey.md` | S07 | PIE Survey (Brisibe 2024, JMCP) | WebFetch |

---

## Re-fetch Instructions

### PDFs (curl)
```bash
cd reference_cache/pdfs
curl -L -A "Mozilla/5.0" -o "R09_amcp-format-5.pdf" "https://www.amcp.org/sites/default/files/2024-04/AMCP-Format-5.0-JMCP-web_0.pdf"
curl -L -A "Mozilla/5.0" -o "R17_oecd-who-purchasing-quality.pdf" "https://www.oecd.org/content/dam/oecd/en/publications/reports/2023/10/purchasing-for-quality-chronic-care_360ec217/66dfc7e1-en.pdf"
curl -L -A "Mozilla/5.0" -o "S03_cms-partd-manual-ch6.pdf" "https://www.cms.gov/medicare/prescription-drug-coverage/prescriptiondrugcovcontra/downloads/part-d-benefits-manual-chapter-6.pdf"
```

### Web pages (Claude Code WebFetch or Playwright)
Each markdown file contains the source URL in its YAML frontmatter. Re-fetch with:
```
WebFetch: [url from frontmatter] -> save to html/[filename].md
```

### PMC PDFs (manual browser download)
Visit each PMC article URL and click the PDF download button. URLs are in the YAML frontmatter of each `html/R*_PMC*.md` file.

---

## Research-KB Ingestion

The `healthcare` domain exists in research-kb (0 sources). To ingest:

```bash
# From ~/Claude/research-kb/
# 1. Copy PDFs to the corpus directory
cp ~/Claude/precision_aq/reference_cache/pdfs/*.pdf corpus/healthcare/

# 2. Ingest PDFs
python scripts/ingest_corpus.py --domain healthcare

# 3. For markdown files, convert to a format the ingester accepts
# (research-kb primarily ingests PDFs; markdown may need conversion)
```

After ingestion, verify with: `research_kb_search(query="formulary", domain="healthcare")`

Tags available in `index.json` for each entry: `type:{regulation|evidence|market-structure|strategy|global|supplement}` + topic-specific tags (formulary, pbm, mco, pie, rwe, vbc, icer, etc.).
