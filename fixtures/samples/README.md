# Sample Fixtures for CI Testing

This directory contains small sample PDFs used for CI validation. These are real open-access academic papers from arXiv on causal inference topics.

## Papers Included

| File | arXiv ID | Title | Size |
|------|----------|-------|------|
| `iv_survey_2212.05778.pdf` | [2212.05778](https://arxiv.org/abs/2212.05778) | Instrumental Variables in Causal Inference and Machine Learning: A Survey | ~1.5MB |
| `mining_causality_2409.14202.pdf` | [2409.14202](https://arxiv.org/abs/2409.14202) | Mining Causality: AI-Assisted Search for Instrumental Variables | ~0.5MB |

## Purpose

These samples enable CI workflows to test:
- PDF ingestion pipeline
- Text extraction (GROBID)
- Embedding generation
- Basic retrieval validation

## Full Corpus

For local development with the full corpus, you need additional PDFs in:
- `fixtures/textbooks/` - Causal inference textbooks
- `fixtures/papers/` - Research papers

Contact the repository maintainer for access to the full corpus, or source your own academic PDFs.

## License

All papers are open-access and distributed under arXiv's license terms.
See https://arxiv.org/help/license for details.
