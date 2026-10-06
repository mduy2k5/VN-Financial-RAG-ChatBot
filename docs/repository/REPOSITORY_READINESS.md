# Repository data policy and GitHub readiness

Updated: 2026-10-06.

## What is included

The tracked tree is intended to contain application/model code, configuration, schemas, tests, documentation, and one small fictional financial-report fixture. `financial_python/sample_data/DEMO_SYNTHETIC_2025.md` and `financial_python/evaluation/demo_cases.jsonl` use invented company names, values, and citations. They exist to exercise parsing, retrieval, answer validation, and the example benchmark.

## What stays local

Real OCR reports, labeled real-report datasets, detailed evaluation outputs, review packs, and QA screenshots have been moved to `financial_python/local_data/archive/`. That directory is ignored by Git. The old local corpus path `/dataset/` also remains ignored. Do not copy items from these folders into a commit or release.

The benchmark and CI defaults use only the fictional demo. Corpus-level experiments require their local inputs and are not part of the default test suite.

## Current release gate

Local and GitHub `main` now use a sanitized two-commit history rooted independently from the old history. The old history remains in ignored local backups and Codex backup refs; never push or publish those refs. A read-only remote check after the update found only `main` among advertised refs and no pull-request refs.

Gitleaks v8.30.0 scanned the release history and files with no findings. The push triggered GitHub Actions, but its run result has not been checked. Repository visibility was not checked or changed; verify the CI run and visibility before switching the repository to public. A history rewrite does not necessarily purge cached commit views or references held by forks, pull requests, or existing clones; review [GitHub's guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository) before changing visibility.

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
