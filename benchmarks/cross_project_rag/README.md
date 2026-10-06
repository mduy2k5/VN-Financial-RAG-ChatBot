# Cross-project financial RAG benchmark

This benchmark compares the end-to-end answers produced by:

1. `VN-Financial-Understanding` (`financial_rag` API).
2. Vindex `data-services` (`knowspace_search` API).

The systems keep their native ingestion and retrieval pipelines, while the input files, gold questions, answer checks, request schedule, and resource budget are shared. Supply all corpora and case sets from local paths. Do not use dev cases to tune the holdout set.

## Before running

- Keep source reports, generated corpus inventories, gold labels, run manifests, outputs, and host snapshots in local-only storage. These may contain report names, source hashes, local paths, service details, or machine metadata and are ignored by Git.
- Create and independently review source-grounded labels before reporting accuracy. Keep blind holdout labels out of development and tuning.
- Use isolated databases, collections, indexes, volumes, and ports for every system. Never point a benchmark importer at a shared or production store.
- Freeze application revisions, image digests, model IDs/revisions, prompts, limits, and environment settings before running holdout. Record hardware and provider details in a local run manifest.
- Prefer a controlled track where both systems use the same local embedding and answer models. If that is not possible without changing product code, label the run as a native-stack comparison and disclose each provider and its costs. Never send a private corpus to a hosted model unless its data path is approved.

## Benchmark design

### Corpus and isolation

- Ingest byte-identical source documents into both systems. Keep the original filename, page markers, and SHA-256. Record any normalization as a separate derived artifact; never overwrite source TXT files.
- Use benchmark-only Mongo databases/collections, Qdrant collections, Meilisearch indexes, and app ports. Do not point either importer at a developer, shared, or production store.
- Let each product use its own native parser/chunker/retriever. Comparing their native pipelines is part of the end-to-end system result. Also record parsed document count, page count, chunk count, and failed/skipped files so ingestion coverage is visible.
- Freeze application commits, image digests, collection names, model IDs/revisions, prompts/agent IDs, and all environment settings before running holdout.

### Question set

Target 180 manually grounded cases, following the categories and review requirements in `financial_python/evaluation/FINANCIAL_ANALYTICS_BENCHMARK.md`:

| Category | Target |
|---|---:|
| Metric lookup and source/page grounding | 30 |
| Period comparison and growth | 30 |
| Composition / statement-table reading | 35 |
| Ratios and multi-input calculations | 25 |
| Narrative disclosure and accounting-policy questions | 25 |
| Cross-section / multi-step questions | 15 |
| Ambiguous, out-of-scope, or evidence-absent questions | 20 |

Stratify cases across issuer, report year, separate/consolidated basis, report section, and OCR/table difficulty. Keep every case's supporting documents within one split: 120 dev and 60 blind holdout, split by source document rather than by question. Both systems index the same full corpus; the split prevents benchmark questions from leaking into tuning, it does not hide evidence the user would expect the systems to search.

Each answerable case needs source-grounded expected facts with label, value, unit, fiscal period, report basis, file, and physical extracted page. Calculation cases need the exact formula and all input facts. Unanswerable cases require a documented exhaustive scope check. Gold labels should be reviewed by a second person; holdout cases require `review_status=verified` before scoring.

### Resource and model parity

Run one system at a time on the same host, with the same container CPU and memory ceilings, GPU visibility, concurrency, warm-up count, timeout, corpus, and query order. Record actual CPU, peak RSS, GPU utilization/VRAM, disk growth, and provider token/cost usage. Do not let the two systems compete for GPU or RAM during timed runs.

Pin inference providers and exact model revisions. Prefer a controlled track where both systems use the same local embedding and answer models. If one system cannot use those models without modifying its product code, run a clearly labelled **native-stack track** with both systems under the same hardware envelope and disclose each external model/provider, token limit, and per-query cost. Do not describe that as an architecture-only comparison. Never send the private OCR corpus to a hosted LLM unless that data path is explicitly approved.

Primary result is the best quality achieved within the shared resource/cost envelope. Report a quality/latency/resource Pareto comparison instead of hiding trade-offs in one weighted score. An overall winner requires the same frozen holdout, a verified gold set, and no material differences in corpus coverage or failed cases.

### Metrics

**Answer quality** (per-case and macro-averaged by category/company):

- Exact financial fact match after deterministic number normalization; report unit and period match separately.
- Formula result match with declared rounding/tolerance, plus correctness of every input fact.
- Scope accuracy: company, fiscal year/quarter, and separate vs consolidated basis.
- Citation precision and recall: cited file/page must support the claim; also report evidence-group recall@5.
- Completeness for composition / multi-hop cases; abstention precision and recall for ambiguous or unsupported questions.
- Blind human rubric (1–5) for narrative relevance, faithfulness, and unsupported claims. Randomize answer order and mask system identity.

**Performance:**

- Ingestion wall time and throughput, full-corpus indexing completion, index size, and failed/skipped documents.
- Cold start / index readiness separately from query time.
- End-to-end query latency from submission through final answer (including asynchronous task polling), p50/p95, timeout/error rate, and warm/cold cache state.
- Peak process/container RSS, CPU-seconds, GPU utilization and peak VRAM. If remote inference is used, report provider latency and token/cost data separately.

For answer latency, use one in-flight query first and a fixed warm-up; do not count HTTP 202 as a completed answer. Keep cache hits out of the model-latency score, but report cache-hit latency separately. Use at least three independent timed runs per case if provider cost permits; set generation temperature to zero where configurable and preserve every raw response.

## Artifacts and execution gates

- `corpus_manifest.json`: generated local-only input inventory and hashes; do not commit it.
- `case.schema.json`: schema copied from the financial-analysis benchmark; extend only through a versioned change.
- `cases.dev.jsonl` / `cases.holdout.jsonl`: create after source-level grounding; holdout labels remain access-controlled during tuning.
- `run_manifest.json`: exact hardware, app revisions, models, limits, corpus hash, and benchmark settings for each run.
- `results/<run-id>/`: raw answers, task timings, resource samples, scorer output, and human review sheet.

After the gold files exist and both benchmark APIs are running on isolated stores:

```powershell
$env:BENCH_DATA_SERVICES_API_KEY = '<benchmark-local-api-key>'
python benchmarks/cross_project_rag/run_answers.py `
  --cases benchmarks/cross_project_rag/cases.dev.jsonl `
  --split dev `
  --agent-id '<configured-agent-id>' `
  --output benchmarks/cross_project_rag/results/<run-id>/answers.jsonl

python benchmarks/cross_project_rag/score_answers.py `
  --cases benchmarks/cross_project_rag/cases.dev.jsonl `
  --results benchmarks/cross_project_rag/results/<run-id>/answers.jsonl `
  --output benchmarks/cross_project_rag/results/<run-id>/scores.json

python benchmarks/cross_project_rag/sample_resources.py `
  --container <financial-api-container> `
  --container <data-services-api-container> `
  --interval 1 `
  --output benchmarks/cross_project_rag/results/<run-id>/resources.jsonl
```

The answer runner alternates system order per case, disables direct social replies, waits for the full `data-services` task, records cache hits/errors instead of hiding them, and refuses an unverified holdout. The scorer checks exact number occurrences and recognized file/page citations; use blind human review for correct metric-to-period binding and narrative quality. Timed runs must use isolated benchmark containers and storage.

Execution gates:

1. Validate file hashes and make both systems ingest the same source files into isolated stores.
2. Compare ingestion coverage and metadata extraction; fix only benchmark setup errors, not product-specific retrieval behavior.
3. Ground and independently review the blind holdout.
4. Freeze both deployments and run the same cases in randomized, alternating system order.
5. Score exact facts/citations automatically and narrative answers blind; publish per-category results and the resource/model manifest.

Commit only benchmark code, schemas, and generic documentation. Keep generated inventories, source-grounded cases, run manifests, responses, scores, and local environment snapshots ignored and outside the release history.
