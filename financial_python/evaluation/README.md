# Evaluation tools

The default reproducible benchmark is `demo_cases.jsonl`, which contains two fictional cases evaluated against `sample_data/DEMO_SYNTHETIC_2025.md`:

```powershell
python -m financial_rag.evaluation.benchmark
```

Corpus-scale experiments and historical labeled benchmarks are optional local workflows. Their previous OCR sources, labels, and result files are preserved under `financial_python/local_data/archive/` and are excluded from Git. Existing experiment scripts that consume those inputs resolve them there by default. Set `FINANCIAL_LOCAL_DATA_ROOT` to use another local archive. Detailed experiment output paths under `evaluation/results/` are ignored by Git.

The synthetic benchmark is a functional smoke test. It does not estimate model quality on real reports.
