"""Evaluate query resolver fields against a manually reviewed annotation set."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from ..query.understanding import QueryUnderstandingEngine
from .local_data import local_data_path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = local_data_path("evaluation", "ground_truth_multi_company.jsonl")
DEFAULT_ANNOTATIONS = local_data_path("evaluation", "query_understanding_ground_truth.jsonl")
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "results" / "query_understanding_v1.json"


def run_query_understanding_benchmark(
    dataset_path: Path = DEFAULT_DATASET,
    annotations_path: Path = DEFAULT_ANNOTATIONS,
    *,
    write_result: bool = False,
) -> dict[str, Any]:
    examples = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    annotations = [
        json.loads(line)
        for line in annotations_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    annotations_by_id = {item["id"]: item for item in annotations}
    example_ids = {item["id"] for item in examples}
    annotation_ids = set(annotations_by_id)
    if len(annotations_by_id) != len(annotations):
        raise ValueError("Query annotations contain duplicate IDs")
    if example_ids != annotation_ids:
        raise ValueError(
            "Query annotations do not match the retrieval dataset IDs: "
            f"missing={sorted(example_ids - annotation_ids)}, "
            f"unexpected={sorted(annotation_ids - example_ids)}"
        )

    engine = QueryUnderstandingEngine()
    counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    metric_counts: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    errors: list[dict[str, Any]] = []

    for example in examples:
        annotation = annotations_by_id[example["id"]]
        query = example["query"]
        if annotation.get("query") != query:
            raise ValueError(f"Annotated query text changed for {example['id']}")

        parsed = engine.parse(query)
        expected_metric = annotation["expected_metric_id"]
        expected_intent = annotation["expected_intent"]
        predicted_metric = parsed.metric_id or "<none>"
        predicted_intent = parsed.intent.value
        metric_counts[expected_metric] += 1
        intent_counts[expected_intent] += 1
        confusion[(expected_intent, predicted_intent)] += 1

        row_errors: dict[str, Any] = {"id": example["id"]}
        has_error = False
        for metric_field, expected_metric_value, predicted_metric_value in (
            ("metric", expected_metric, predicted_metric),
            ("intent", expected_intent, predicted_intent),
        ):
            counts[f"{metric_field}_total"] += 1
            is_correct = expected_metric_value == predicted_metric_value
            counts[f"{metric_field}_correct"] += is_correct
            if not is_correct:
                row_errors[f"expected_{metric_field}"] = expected_metric_value
                row_errors[f"predicted_{metric_field}"] = predicted_metric_value
                has_error = True

        for metadata_field, expected_metadata_value, predicted_metadata_value in (
            ("company", example.get("company_code"), parsed.company_code),
            ("fiscal_year", example.get("fiscal_year"), parsed.fiscal_year),
        ):
            if expected_metadata_value is None:
                continue
            counts[f"{metadata_field}_total"] += 1
            is_correct = expected_metadata_value == predicted_metadata_value
            counts[f"{metadata_field}_correct"] += is_correct
            if not is_correct:
                row_errors[f"expected_{metadata_field}"] = expected_metadata_value
                row_errors[f"predicted_{metadata_field}"] = predicted_metadata_value
                has_error = True

        expected_years = example.get("fiscal_years")
        if expected_years is not None:
            counts["comparison_years_total"] += 1
            is_correct = set(expected_years) == set(parsed.fiscal_years)
            counts["comparison_years_correct"] += is_correct
            if not is_correct:
                row_errors["expected_fiscal_years"] = expected_years
                row_errors["predicted_fiscal_years"] = parsed.fiscal_years
                has_error = True

        expected_quarter = example.get("fiscal_quarter")
        if expected_quarter is not None:
            counts["fiscal_quarter_total"] += 1
            is_correct = expected_quarter == parsed.fiscal_quarter
            counts["fiscal_quarter_correct"] += is_correct
            if not is_correct:
                row_errors["expected_fiscal_quarter"] = expected_quarter
                row_errors["predicted_fiscal_quarter"] = parsed.fiscal_quarter
                has_error = True

        if has_error:
            errors.append(row_errors)

    def accuracy(field: str) -> float | None:
        total = counts[f"{field}_total"]
        if not total:
            return None
        return round(counts[f"{field}_correct"] / total, 4)

    period_fields = ("fiscal_year", "comparison_years", "fiscal_quarter")
    period_total = sum(counts[f"{field}_total"] for field in period_fields)
    period_correct = sum(counts[f"{field}_correct"] for field in period_fields)
    accuracies = {
        "metric": accuracy("metric"),
        "company": accuracy("company"),
        "period": round(period_correct / period_total, 4) if period_total else None,
        "fiscal_year": accuracy("fiscal_year"),
        "comparison_years": accuracy("comparison_years"),
        "fiscal_quarter": accuracy("fiscal_quarter"),
        "intent": accuracy("intent"),
    }
    thresholds = {"metric": 0.90, "company": 0.95, "period": 0.95, "intent": 0.90}
    thresholds_met = True
    for score_name, threshold in thresholds.items():
        score = accuracies[score_name]
        if score is None or score < threshold:
            thresholds_met = False
            break
    result: dict[str, Any] = {
        "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
        "annotations": str(annotations_path.relative_to(PROJECT_ROOT)),
        "cases": len(examples),
        "accuracies": accuracies,
        "counts": dict(counts),
        "expected_metric_distribution": dict(sorted(metric_counts.items())),
        "expected_intent_distribution": dict(sorted(intent_counts.items())),
        "intent_confusion": {
            f"{expected}->{predicted}": count
            for (expected, predicted), count in sorted(confusion.items())
        },
        "errors": errors,
        "gate_thresholds": thresholds,
        "thresholds_met_on_this_fixture": thresholds_met,
        "annotation_limitations": [
            "Nhãn v1 do một lượt rà soát, chưa có người thứ hai xác nhận độc lập.",
            "Tập 88 câu không có ví dụ intent explanation hoặc follow-up.",
            "Kết quả trên câu hỏi ngắn của benchmark không đại diện mọi cách diễn đạt thực tế.",
        ],
    }

    if write_result:
        DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_OUTPUT.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return result


if __name__ == "__main__":
    print(json.dumps(run_query_understanding_benchmark(write_result=True), ensure_ascii=False, indent=2))
