# Repository artifact inventory

The 2026-10-06 baseline contained 408 tracked files totaling approximately 146 MiB. The largest items were real-corpus benchmark results, evaluation/review datasets, source reports, and UI QA screenshots.

The working-tree policy now keeps only a small synthetic report and two synthetic benchmark cases in Git. The previous real inputs and detailed outputs are preserved locally under the ignored `financial_python/local_data/archive/` directory (166 files, approximately 139 MiB). This archive is for local work and must not be added to GitHub.

The sanitized tree is prepared as the parentless root commit on `codex/github-ready-clean-history`. The local backup bundle is `financial_python/local_data/archive/git-history-backup-20261006.bundle`. The existing `main` and remote-tracking history remain untouched and still contain old copies of removed material; do not push those refs. Check the GitHub target and visibility before publishing the clean branch.
