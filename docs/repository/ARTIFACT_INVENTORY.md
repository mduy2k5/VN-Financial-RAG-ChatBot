# Repository artifact inventory

The tracked tree contains application code, configuration, schemas, tests, documentation, a small fictional report, and synthetic benchmark cases. Real reports, source-grounded evaluation data, run outputs, review packs, machine snapshots, and UI QA screenshots stay in ignored local storage.

The current `main` has a sanitized history rooted independently from the previous history; commit author and committer addresses use the account's GitHub-provided `noreply` address. Gitleaks v8.30.0 scanned the release history and tracked files with no findings. Older history remains only in local Codex backups; never push those refs or use `git push --mirror`. The remote advertises only `main`. The repository intentionally has no GitHub Actions workflow.

Rewriting branch history removes the old commits from advertised branch refs, but does not guarantee removal of cached commit views, pull-request references, forks, or existing clones. Before making this repository public, review [GitHub's sensitive-data removal guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository) and use its support process if eligible cached views or pull-request references need purging.
