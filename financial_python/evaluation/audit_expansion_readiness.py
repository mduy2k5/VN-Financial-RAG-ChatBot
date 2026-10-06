"""Build a source-bound review pack without promoting draft labels to gold."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

try:
    from .run_financial_analysis_benchmark import _validate_case, validate_holdout_isolation
except ImportError:
    from run_financial_analysis_benchmark import _validate_case, validate_holdout_isolation


def audit(cases: list[dict], manifest: dict) -> dict:
    source_index = {Path(row["source_file"]).name.casefold(): row for row in manifest["documents"]}
    reviewed = sum(case.get("review_status") == "verified" for case in cases)
    holdout = [case for case in cases if case["split"] == "holdout"]
    reasons = []
    for line, case in enumerate(cases, start=1):
        try:
            _validate_case(case, line_number=line)
        except ValueError as exc:
            reasons.append(str(exc))
        if case.get("review_status") == "verified":
            if not case.get("author_id") or not case.get("reviewer_id") or case["author_id"] == case["reviewer_id"]:
                reasons.append(f"{case['id']}: review_independence_not_established")
    if reviewed != len(cases):
        reasons.append("labels_pending_independent_review")
    if not holdout or not all(case.get("review_status") == "verified" for case in holdout):
        reasons.append("no_verified_holdout")
    try:
        validate_holdout_isolation(cases)
    except ValueError as exc:
        reasons.append(str(exc))
    bindings = []
    for case in cases:
        files = set()
        for records in (case.get("expected_facts", []), case.get("evidence_groups", [])):
            for record in records:
                for source in record.get("sources", []):
                    name = Path(str(source["file_name"]).replace("\\", "/")).name.casefold()
                    files.add(name)
                    if case.get("review_status") == "verified" and name in source_index:
                        if source.get("source_sha256") != source_index[name]["source_sha256"]:
                            reasons.append(f"{case['id']}: reviewed_source_hash_missing_or_changed")
        missing = sorted(files - source_index.keys())
        bindings.append({"case_id": case["id"], "review_status": case.get("review_status", "pending"),
                         "source_bindings": [source_index[name] for name in sorted(files & source_index.keys())],
                         "missing_runtime_sources": missing})
        if missing:
            reasons.append(f"{case['id']}: source_not_in_runtime")
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_snapshot_sha256": manifest["source_snapshot_sha256"],
            "production_accuracy_gate_passed": not reasons, "reasons": reasons,
            "case_count": len(cases), "verified_count": reviewed,
            "by_split": dict(Counter(case["split"] for case in cases)),
            "by_category": dict(Counter(case["category"] for case in cases)),
            "review_pack": [{"case": case, "binding": binding} for case, binding in zip(cases, bindings)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("financial_analysis_cases.jsonl"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    result = audit(cases, json.loads(args.manifest.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("case_count", "verified_count", "production_accuracy_gate_passed", "reasons")}))


if __name__ == "__main__":
    main()
