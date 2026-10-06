# Repository data policy and GitHub readiness

Updated: 2026-10-06.

## What is included

The tracked tree is intended to contain application/model code, configuration, schemas, tests, documentation, and one small fictional financial-report fixture. `financial_python/sample_data/DEMO_SYNTHETIC_2025.md` and `financial_python/evaluation/demo_cases.jsonl` use invented company names, values, and citations. They exist to exercise parsing, retrieval, answer validation, and the example benchmark.

## What stays local

Real OCR reports, labeled real-report datasets, detailed evaluation outputs, review packs, and QA screenshots have been moved to `financial_python/local_data/archive/`. That directory is ignored by Git. The old local corpus path `/dataset/` also remains ignored. Do not copy items from these folders into a commit or release.

The benchmark and CI defaults use only the fictional demo. Corpus-level experiments require their local inputs and are not part of the default test suite.

## Current release gate

The sanitized tree is prepared as a parentless root commit on `codex/github-ready-clean-history`; it has no connection to the old history. The existing `main`, `origin/main`, and Codex backup refs remain untouched and still contain earlier copies of real reports, benchmark outputs, evaluation data, and screenshots. Do not publish those refs as part of this release.

The old history is preserved locally in `financial_python/local_data/archive/git-history-backup-20261006.bundle`. No push has been performed. The requested initial visibility is private; the GitHub target's actual visibility has not been checked or changed. Before publishing, confirm the target repository and push only the clean release ref.

## Verification

From `financial_python`, install the development extra and run:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy src
python -m financial_rag.evaluation.benchmark
```

These checks use the synthetic fixture and do not require the local corpus, MongoDB, Qdrant, Ollama, or a GPU. Docker integration checks are separate from the unit suite.
