# Reference Cache Wishlist

Items that could not be automatically acquired. Each entry includes the reason and suggested manual acquisition approach.

Generated: 2026-03-26

---

## PDFs (PMC JS Download Challenge)

PMC's `/pdf/` endpoint serves a JavaScript-based download page that requires browser execution. The **markdown snapshots** for all 8 articles were successfully cached from the HTML full-text versions.

| Bib Key | Title | URL | Reason |
|---------|-------|-----|--------|
| R10 | Dodda et al., PIE Best Practices | `PMC10387941/pdf/` | JS download page; HTML cached |
| R13 | Chambers et al., AMCP RWE Standards | `PMC12653624/pdf/` | JS download page; HTML cached |
| R14 | Abu-Shraie et al., RWE Scoping Review | `PMC12450548/pdf/` | JS download page; HTML cached |
| R15 | Hydery et al., Value Assessment Tools | `PMC10387901/pdf/` | JS download page; HTML cached |
| amcp-pt-principles | AMCP P&T Committee Principles | `PMC10391133/pdf/` | JS download page; HTML cached |
| jmcp-managed-care-primer | Managed Care Pharmacy Primer | `PMC10701257/pdf/` | JS download page; HTML cached |
| jmcp-acos-medication | ACOs and Medication Use | `PMC10390926/pdf/` | JS download page; HTML cached |
| S07 | PIE Survey (Brisibe 2024) | `PMC11068649/pdf/` | JS download page; HTML cached |

**Manual acquisition**: Open each PMC article URL in a browser, click the PDF download button, save to `reference_cache/pdfs/`.

---

## Web Pages (Cloudflare / Bot Protection)

| Bib Key | Title | URL | Reason |
|---------|-------|-----|--------|
| ashp-pt-guidelines | ASHP P&T Committee Guidelines | ashp.org/.../gdl-pharmacy-therapeutics-committee-formulary-system.ashx | Cloudflare 403; PDF behind challenge |
| icer-framework | ICER Value Assessment Framework | icer.org/our-approach/methods-process/value-assessment-framework/ | Cloudflare 403 |
| mmit-market-access-101 | MMIT Market Access 101 | mmitnetwork.com/.../market-access-101-understanding-the-basics/ | nginx 403 |
| mmit-whats-ahead-2025 | MMIT What's Ahead 2025 | mmitnetwork.com/.../whats-ahead-market-access-2025/ | nginx 403 (not attempted, same domain) |

**Manual acquisition**: Open each URL in a regular browser. For ASHP, save the PDF. For others, use browser "Save as" or print-to-PDF.

---

## Web Pages (Timeout / 404)

| Bib Key | Title | URL | Reason |
|---------|-------|-----|--------|
| R02 | Medicaid Managed Care Index | medicaid.gov/medicaid/managed-care/index.html | WebFetch timeout; Playwright not attempted |
| R03 | FDA PIE Guidance | fda.gov/.../drug-and-device-manufacturer-communications... | HTTP 404 — URL may have changed |
| S01 | CMS Negotiation Fact Sheet (2026 prices) | cms.gov/newsroom/fact-sheets/...2026 | WebFetch timeout; Playwright not attempted |
| S02 | CMS 2026 Advance Notice Fact Sheet | cms.gov/newsroom/fact-sheets/...advance-notice | WebFetch timeout; Playwright not attempted |
| S04 | HRSA 340B Ceiling Price FAQ | hrsa.gov/about/faqs/how-340b-ceiling-price-calculated | HTTP 403 |

**Manual acquisition**: Visit in a browser. For CMS pages, the content loads fine in normal browsers. For R03 (FDA), search fda.gov for the current guidance document URL.

---

## Summary

| Category | Total | Cached | Wishlist |
|----------|-------|--------|----------|
| PDFs | 11 | 3 | 8 (HTML alternatives cached) |
| Markdown (WebFetch) | 29 | 29 | 0 |
| Markdown (Playwright) | 3 | 3 | 0 |
| Blocked pages | 9 | 0 | 9 |
| **Total unique content** | **44** | **35** | **9** |

Note: All 8 wishlist PMC PDFs have full-text markdown equivalents cached in `html/`. The net content gap is only the 9 blocked web pages, of which R03 (FDA) may have moved and S01/S02/R02 are likely accessible via normal browser.
