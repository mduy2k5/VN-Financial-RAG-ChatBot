# Repository data policy and GitHub readiness

Updated: 2026-10-06.

## What is included

The tracked tree is intended to contain application/model code, configuration, schemas, tests, documentation, and one small fictional financial-report fixture. `financial_python/sample_data/DEMO_SYNTHETIC_2025.md` and `financial_python/evaluation/demo_cases.jsonl` use invented company names, values, and citations. They exist to exercise parsing, retrieval, answer validation, and the example benchmark.

## What stays local

Real OCR reports, labeled real-report datasets, detailed evaluation outputs, review packs, and QA screenshots have been moved to `financial_python/local_data/archive/`. That directory is ignored by Git. The old local corpus path `/dataset/` also remains ignored. Do not copy items from these folders into a commit or release.

The example benchmark uses only the fictional demo. Corpus-level experiments require their local inputs and are not part of the default test suite.

## Current release gate

Local and GitHub `main` use a sanitized history rooted independently from the old history. Commit author and committer addresses use the account's GitHub-provided `noreply` address. Older history remains in local Codex backups; never push those refs or use `git push --mirror`. The remote advertises only `main`.

The repository intentionally has no GitHub Actions workflow; run checks manually when needed. Gitleaks v8.30.0 scans the release history and tracked files with no findings. The local verification record is below. Keep the repository private until its owner decides to change visibility. A history rewrite does not necessarily purge cached commit views or references held by forks, pull requests, or existing clones; review [GitHub's guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository) before changing visibility.

## Verification

From `financial_python`, install the development extra and run:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy src
python -m financial_rag.evaluation.benchmark
```

These checks use the synthetic fixture and do not require the local corpus, MongoDB, Qdrant, Ollama, or a GPU. On 2026-10-06, local verification passed: 262 pytest tests, Ruff, mypy, JavaScript syntax checks, Docker Compose validation, and Gitleaks history and tracked-file scans. Docker integration checks are separate from the unit suite.
