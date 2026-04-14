# Formula Extraction Comparison — Final Report

Generated: 2026-04-05

## Summary Table

| PDF | Extractor | Gaps | LaTeX (display) | LaTeX (inline) | Chars | Time |
|-----|-----------|------|-----------------|----------------|-------|------|
| **murty** | docling_default | 2,990 | 0 | 1 | 427,321 | 841s |
| | docling_patched | 627 | 2,368 | 7 | 467,732 | 840s |
| | pymupdf_raw | 0 | 0 | 0 | 10,651 | 13s |
| | **mineru** | **0** | **2,734** | **6,105** | 811,035 | 553s |
| **ross** | docling_default | 949 | 0 | 1 | 112,800 | 35s |
| | docling_patched | 0 | 952 | 1 | 173,923 | 36s |
| | pymupdf_raw | 0 | 0 | 3 | 11,981 | 2s |
| | **mineru** | **0** | **—** | **2,280** | — | 216s |
| **boyd** | docling_default | 1,028 | 0 | 11 | 961,553 | 153s |
| | docling_patched | 0 | 1,028 | 11 | 984,844 | 152s |
| | pymupdf_raw | 0 | 0 | 0 | 18,102 | 2s |
| | **mineru** | **0** | **—** | **12,673** | — | 689s |
| **dml** | docling_default | 335 | 0 | 1 | 182,893 | 30s |
| | docling_patched | 0 | 335 | 1 | 198,065 | 25s |
| | pymupdf_raw | 0 | 0 | 0 | 20,890 | 2s |
| | **mineru** | **0** | **303** | **1,662** | 285,393 | 148s |
| **manning** | docling_default | 25 | 0 | 41 | 1,300,394 | 226s |
| | docling_patched | 0 | 25 | 41 | 1,300,665 | 229s |
| | pymupdf_raw | 0 | 0 | 0 | 20,304 | 2s |
| | **mineru** | **0** | **124** | **4,594** | 1,208,460 | 614s |

**PaddleOCR**: Failed on all PDFs (PaddlePaddle 3.3.x oneDNN/PIR runtime crash on CPU; not a Python version issue).

## Quality Tiers

### Tier 1: MinerU (magic-pdf 1.3.12) — Publication-quality LaTeX

```latex
\widehat{\theta}_{0} = \Big(\frac{1}{n}\sum_{i \in I} D_{i}^{2}\Big)^{-1}
\frac{1}{n}\sum_{i \in I} D_{i}\big(Y_{i} - \widehat{g}_{0}(X_{i})\big)
```

- Proper `\frac`, `\sum`, `\binom`, `\widehat`, `\operatorname`
- Handles multiline arrays, aligned equations
- Recovers image-based formulas (the 649 Murty residuals docling_patched couldn't get)
- 0 gaps across all 5 PDFs
- ~2-10 min per PDF depending on size

### Tier 2: docling_patched — Unicode text (readable but imperfect)

```
bθ0 = 1 n X i∈I D2 i −1 1 n X i∈I Di(Yi −bg0(Xi))
```

- Raw Unicode from PDF text layer via PyMuPDF bbox extraction
- Font mapping artifacts (`~(n)` instead of `φ(n)`, `II` instead of `∏`)
- 79-100% gap recovery (fails on image-based formulas)
- Zero overhead on top of Docling extraction time
- Can patch existing chunks without re-extraction

### Tier 3: docling_default — Information destruction

```
<!-- formula-not-decoded -->
```

- Current production pipeline
- 28.2% of all corpus chunks affected

### Tier 4: pymupdf_raw — Fast but no structure

- Extracts all text (no gaps) but no headings, tables, or structure
- pymupdf4llm drops display formulas as "picture intentionally omitted"
- Only useful as ground truth reference

## Decision Matrix

| Criterion | docling_patched | MinerU |
|-----------|----------------|--------|
| Formula quality | Unicode (readable) | LaTeX (publication) |
| Gap recovery | 79-100% (text-based only) | 100% (including images) |
| Retrofit existing chunks | Yes (patch in-place) | No (full re-extraction) |
| Additional dependency | PyMuPDF (already installed) | magic-pdf + PyTorch + models (~7GB) |
| GPU VRAM | 0 extra (uses Docling's model) | ~2.5 GB (doclayout-yolo + UniMERNet) |
| Time overhead | ~0s on top of Docling | 2-10 min per PDF |
| Python requirement | Any (3.13 works) | 3.12 (detectron2/transformers compat) |
| Chunking | Preserves Docling chunks | Own chunking (different structure) |

## Recommended Strategy: Two-Phase Approach

### Phase 1 (immediate): Deploy docling_patched
- Integrate PyMuPDF bbox fallback into `docling_extractor.py`
- Build `scripts/patch_formula_gaps.py` to retroactively fix ~280K existing chunks
- Unblocks eval annotation immediately
- Zero new dependencies

### Phase 2 (future): MinerU for math-heavy domains
- Re-extract the 5 worst domains (linear_algebra, topology, probability, analysis, algebra)
  with MinerU for LaTeX quality
- Requires: Python 3.12 venv, `~/magic-pdf.json` config, ~7GB model cache
- Config: `layout-config.model = "doclayout_yolo"` (avoids detectron2 dependency)
- Pin: `transformers==4.49.0` (UniMERNet compatibility)

### Phase 2 Prerequisites
- Python 3.12 installed via `uv python install 3.12`
- MinerU venv: `/tmp/venv_mineru_312/`
- Config: `~/magic-pdf.json` with `models-dir` and `layout-config`
- OCR model symlink: `ch_PP-OCRv3_det_infer.pth -> Multilingual_PP-OCRv3_det_infer.pth`

## Environment Notes

- **RTX 2070 (8GB)**: MinerU uses ~2.5 GB VRAM. Cannot coexist with Ollama or embed_server.
- **PaddleOCR**: Blocked by PaddlePaddle 3.3.x bug. May work with `paddlepaddle-gpu` or future fix.
- **Isolated venvs**: MinerU at `/tmp/venv_mineru_312/`, PaddleOCR at `/tmp/venv_paddle_312/` (will be lost on reboot)
