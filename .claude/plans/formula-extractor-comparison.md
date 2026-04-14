# Formula Extractor Comparison — Implementation Plan

**Created**: 2026-04-05
**Context**: Phase 4.5 of eval overhaul. 28.3% of chunks have `<!-- formula-not-decoded -->` gaps.
Eval annotation (Phase 5) is BLOCKED until this is resolved.
**Parent plan**: `.claude/plans/harmonic-brewing-lollipop.md` (Phase 4.5 section)

---

## Goal

Build `scripts/compare_formula_extractors.py` that runs 5 extractors on 5 PDFs and generates
a side-by-side comparison report. User reviews output, picks winner, then we re-extract.

## Why This Matters

- 94.9% of PDFs are born-digital (have text layers with formula content)
- Docling REPLACES extractable text with `<!-- formula-not-decoded -->` — information destruction
- Math/physics/statistics are the Tier 1 low-MRR domains in eval — formula quality directly impacts annotation

## Test PDFs (verified on disk)

```
1. fixtures/library_books/mathematics/Problems in Analytic Number Theory (M. Ram Murty).pdf
   - 44MB, 458 pages, 88.3% formula gap rate, mathematics domain

2. fixtures/library_books/First Course in Probability 7E - Ross.pdf
   - 1MB, probability domain, 91.4% gap rate

3. fixtures/textbooks/migrated/stephen_boyd_lieven_vandenberghe_-_introduction_to_nd.pdf
   - 7MB, mathematics domain, 49.8% gap rate

4. fixtures/papers/chernozhukov_dml_2018.pdf
   - Causal inference paper, <10% gap rate

5. fixtures/textbooks/manning_causal_ai_2024.pdf
   - Causal AI textbook, 1.4% gap rate (control case)
```

Pages to test: 50-54 for each (math-heavy sections). ~10-25 formulas per PDF.

## 5 Extractors to Test

### 1. `docling_default` (baseline — what we have now)
- Runs in main `.venv`
- `do_formula_enrichment = False`
- Expected: 3000+ formula gaps for Murty

### 2. `docling_patched` (raw text fallback)
- Runs in main `.venv`
- After Docling extraction, for each `<!-- formula-not-decoded -->`:
  - Use `result.document.iterate_items()` to find FormulaItem with `item.text == ""`
  - Get bounding box from `item.prov[0].bbox` (l, t, r, b coordinates)
  - Use PyMuPDF `page.get_text("text", clip=fitz.Rect(l, t, r, b))` to extract raw text
  - Replace placeholder with extracted text
- Key code reference: Docling agent found that FormulaItem has `prov` with page + bbox
- Coordinate system: may need scaling between Docling and PyMuPDF coordinate systems
- This is the MOST IMPORTANT extractor because if it works, we can patch 384K existing chunks
  without re-chunking

### 3. `pymupdf_raw` (text layer extraction)
- Runs in main `.venv`
- `pymupdf4llm.to_markdown(pdf, pages=range(50,55))`
- Known limitation: drops display formulas as `picture intentionally omitted`
- But raw `page.get_text()` preserves Unicode math
- Test BOTH: pymupdf4llm.to_markdown() AND raw page.get_text()

### 4. `mineru` (MinerU — full pipeline replacement)
- Runs in ISOLATED venv: `/tmp/venv_mineru/`
- Install: `python -m venv /tmp/venv_mineru && /tmp/venv_mineru/bin/pip install magic-pdf[full]`
- Uses UniMERNet for formula recognition (Tiny=441MB, fits RTX 2070)
- Benchmark rank 10 (9.17/10)
- CPU-capable if GPU is busy

### 5. `paddleocr_vl` (PaddleOCR-VL)
- Runs in ISOLATED venv: `/tmp/venv_paddle/`
- Install: `python -m venv /tmp/venv_paddle && /tmp/venv_paddle/bin/pip install paddleocr paddlepaddle`
- 0.9B params, CPU/GPU
- Benchmark rank 3 (9.65/10) — top open-source performer

### 6. `marker` (bonus if time permits)
- Runs in ISOLATED venv: `/tmp/venv_marker/`
- Install: `python -m venv /tmp/venv_marker && /tmp/venv_marker/bin/pip install marker-pdf`
- Uses surya for OCR, texify for equations
- Previous install in main venv caused pypdfium2 downgrade — use isolation
- Model downloads may be slow (~1GB+)

## Script Architecture

```
scripts/compare_formula_extractors.py
  --pdfs PATH          YAML file listing test PDFs (or comma-separated paths)
  --pages RANGE        Page range to extract (default: 50-54)
  --extractors LIST    Comma-separated: all, docling_default, docling_patched, pymupdf, mineru, paddle, marker
  --output DIR         Output directory (default: fixtures/eval/formula_comparison/)
  --skip-install       Skip venv creation (reuse existing)
```

### Output Structure

```
fixtures/eval/formula_comparison/
  murty/
    docling_default.md
    docling_patched.md
    pymupdf_raw.md
    mineru.md
    paddleocr.md
  ross/
    ...
  COMPARISON.md          <-- auto-generated summary
```

### COMPARISON.md Format

```markdown
# Formula Extraction Comparison

## Per-PDF Results

### Murty — Problems in Analytic Number Theory (pages 50-54)

| Extractor | Formula Gaps | LaTeX Blocks | Runtime | Sample Quality |
|-----------|-------------|-------------|---------|----------------|
| docling_default | 42 | 0 | 35s | N/A (all gaps) |
| docling_patched | 0 | 0 | 37s | Unicode: ζ(s) = Σ n^{-s} |
| pymupdf_raw | 0 | 0 | 1s | Unicode: ((s) = _s... |
| mineru | 2 | 38 | 120s | LaTeX: $\zeta(s) = \sum n^{-s}$ |
| paddleocr | 1 | 40 | 90s | LaTeX: $\zeta(s) = \sum_{n=1}^{\infty} n^{-s}$ |

#### Sample Formula Comparison (Exercise 3.2.1)

| Extractor | Output |
|-----------|--------|
| Ground truth | ζ(s) = s/(s-1) - s ∫₁^∞ {x}/x^{s+1} dx |
| docling_default | <!-- formula-not-decoded --> |
| docling_patched | ((s) = _s _ _ s (Xi {x} dx |
| pymupdf_raw | ((s) = _s _ _ s (Xi {x} dx |
| mineru | $\zeta(s) = \frac{s}{s-1} - s\int_1^\infty \frac{\{x\}}{x^{s+1}}dx$ |
| paddleocr | ... |
```

## Key Implementation Details

### Docling Patched — Bounding Box Extraction

```python
import pymupdf  # PyMuPDF

def patch_formula_gaps(docling_result, pdf_path):
    """Replace formula-not-decoded with PyMuPDF raw text from same region."""
    doc = pymupdf.open(pdf_path)

    for item, level in docling_result.document.iterate_items():
        if hasattr(item, 'label') and 'formula' in str(item.label).lower():
            if not item.text or item.text.strip() == '':
                # Formula decode failed — extract raw text from bbox
                if item.prov and len(item.prov) > 0:
                    prov = item.prov[0]
                    page_no = prov.page_no  # 0-indexed or 1-indexed? CHECK
                    bbox = prov.bbox  # BoundingBox(l, t, r, b)

                    page = doc[page_no]
                    # Docling bbox may use different coordinate system than PyMuPDF
                    # PyMuPDF uses (x0, y0, x1, y1) with origin at top-left
                    rect = pymupdf.Rect(bbox.l, bbox.t, bbox.r, bbox.b)
                    raw_text = page.get_text("text", clip=rect)

                    if raw_text.strip():
                        item.text = raw_text.strip()

    doc.close()
    return docling_result.document.export_to_markdown()
```

**WARNING**: Coordinate system alignment between Docling and PyMuPDF is unverified.
Docling may use different units (points vs pixels) or origin (top-left vs bottom-left).
Must test empirically on first PDF.

### Isolated Venv Execution

```python
import subprocess, sys, json, tempfile

def run_in_isolated_venv(venv_path, install_cmd, extract_script, pdf_path, pages):
    """Run extractor in isolated venv via subprocess."""
    # Create venv if needed
    if not os.path.exists(venv_path):
        subprocess.run([sys.executable, '-m', 'venv', venv_path], check=True)
        subprocess.run([f'{venv_path}/bin/pip', 'install'] + install_cmd.split(),
                       check=True, capture_output=True)

    # Write extraction script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(extract_script)
        script_path = f.name

    # Run in isolated venv
    result = subprocess.run(
        [f'{venv_path}/bin/python', script_path, pdf_path, str(pages[0]), str(pages[-1])],
        capture_output=True, text=True, timeout=600
    )

    os.unlink(script_path)
    return result.stdout  # markdown output
```

### Metrics Collection

For each extractor output:
```python
def compute_metrics(md_text):
    return {
        'formula_gaps': md_text.count('formula-not-decoded'),
        'picture_omitted': md_text.count('picture') + md_text.count('intentionally omitted'),
        'latex_display': md_text.count('$$') // 2,
        'latex_inline': (md_text.count('$') - md_text.count('$$') * 2) // 2,
        'total_chars': len(md_text),
        'lines': md_text.count('\n'),
    }
```

## Verification

After building the script:
1. Run on Murty (PDF 1) only with `docling_default` + `pymupdf_raw` — fast sanity check
2. Add `docling_patched` — verify bbox extraction works (coordinate system)
3. Add `mineru` and `paddleocr` — verify isolated venv setup
4. Run full comparison on all 5 PDFs
5. Generate COMPARISON.md, review

## Critical Files

- `packages/pdf-tools/src/research_kb_pdf/docling_extractor.py` — current Docling pipeline
- `scripts/test_formula_extraction.py` — existing test script (partial, can reuse patterns)
- `.venv/lib/python3.13/site-packages/docling_core/transforms/serializer/markdown.py` — where formula-not-decoded is emitted

## Hardware Constraints

- RTX 2070: 8GB VRAM total
- Stop Ollama before running GPU extractors: `sudo systemctl stop ollama`
- Run extractors sequentially (only one GPU model at a time)
- Each extractor in subprocess guarantees VRAM cleanup

## Timeline

- Step 1-2 (script + backends): ~3-4 hrs
- Step 3 (verify PDFs): ~15 min
- Step 4 (run comparison): ~2-3 hrs (mostly waiting)
- Step 5 (review): ~30 min
- Total: ~1 session (~6 hrs)

## Decisions Already Made (from parent plan)

- Decision 41: Test all 5 extractors
- Decision 42: Spectrum approach for PDFs
- Decision 43: 5 pages per PDF
- Decision 44: Isolated venvs
- Decision 45: Formula quality above all else
