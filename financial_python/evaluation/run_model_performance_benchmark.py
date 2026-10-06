"""Run a small, repeatable generation-only benchmark against the configured Ollama model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
from typing import Any

import requests


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "model_generation_benchmark_2026-10-05.json"
NUMBER_RE = re.compile(
    r"(?<![\w])[-+]?(?:\d{1,3}(?:[.,\s]\d{3})+|\d+)(?:[.,]\d+)?%?(?!\w)"
)

CASES: list[dict[str, Any]] = [
    {
        "id": "metric_lookup",
        "category": "Tra cứu chỉ tiêu",
        "context": "BCTC hợp nhất VPB năm 2025 ghi tổng thu nhập hoạt động là 42.000 tỷ đồng.",
        "query": "Tổng thu nhập hoạt động VPB năm 2025 là bao nhiêu?",
        "expected_numbers": ["42000"],
        "required_term_groups": [["tỷ đồng"]],
    },
    {
        "id": "period_comparison",
        "category": "So sánh kỳ và tăng trưởng",
        "context": "Cho vay khách hàng: năm 2024 là 500.000 tỷ đồng; năm 2025 là 575.000 tỷ đồng.",
        "query": "So sánh hai năm, nêu mức tăng tuyệt đối và phần trăm tăng của cho vay khách hàng.",
        "expected_numbers": ["75000", "15"],
        "required_term_groups": [["tăng"], ["%"]],
    },
    {
        "id": "ratio_calculation",
        "category": "Tỷ số và phép tính",
        "context": "Lợi nhuận sau thuế năm 2025 là 12.000 tỷ đồng. Tổng tài sản bình quân năm 2025 là 600.000 tỷ đồng. Công thức ROA = lợi nhuận sau thuế / tổng tài sản bình quân × 100%.",
        "query": "Tính ROA năm 2025 theo công thức đã cho.",
        "expected_numbers": ["2"],
        "required_term_groups": [["ROA"], ["%"]],
    },
    {
        "id": "composition_reading",
        "category": "Đọc cơ cấu bảng",
        "context": "Bảng nợ phải trả năm 2025 (tỷ đồng): Tiền gửi khách hàng 70.000; Vay các tổ chức tín dụng 20.000; Các khoản phải trả khác 10.000; Tổng nợ phải trả 100.000.",
        "query": "Liệt kê ba thành phần nợ phải trả cùng số tiền; không lặp lại dòng tổng.",
        "expected_numbers": ["70000", "20000", "10000"],
        "required_term_groups": [
            ["Tiền gửi khách hàng"],
            ["Vay các tổ chức tín dụng"],
            ["Các khoản phải trả khác"],
        ],
    },
    {
        "id": "company_scope",
        "category": "Giữ đúng doanh nghiệp/phạm vi",
        "context": (
            "Mỗi dòng là một fact nguồn độc lập; các trường company, report_basis, metric và fiscal_year "
            "xác định phạm vi của value.\n"
            "company=MBB | report_basis=consolidated | metric=total_assets | fiscal_year=2024 | value=100000 | unit=tỷ đồng\n"
            "company=MBB | report_basis=consolidated | metric=total_assets | fiscal_year=2025 | value=120000 | unit=tỷ đồng\n"
            "company=VPB | report_basis=consolidated | metric=total_assets | fiscal_year=2025 | value=900000 | unit=tỷ đồng"
        ),
        "query": "Tài sản MBB hợp nhất năm 2025 là bao nhiêu? Không dùng số liệu của VPB.",
        "expected_numbers": ["120000"],
        "forbidden_numbers": ["900000"],
        "required_term_groups": [["MBB"], ["2025"]],
    },
    {
        "id": "cautious_interpretation",
        "category": "Diễn giải rủi ro có giới hạn",
        "context": "Cho vay khách hàng tăng 15% trong năm; tiền gửi khách hàng tăng 8%. Báo cáo được cung cấp không có bảng kỳ hạn tài sản và nguồn vốn.",
        "query": "Chênh lệch này có đủ để kết luận ngân hàng an toàn thanh khoản không? Giải thích ngắn gọn.",
        "expected_numbers": [],
        "required_term_groups": [[
            "chưa đủ bằng chứng", "chưa đủ cơ sở", "không thể kết luận",
            "không đủ để kết luận", "không đủ thông tin",
        ]],
    },
    {
        "id": "unsupported_metric_abstention",
        "category": "Thiếu bằng chứng/từ chối suy đoán",
        "context": "Trích đoạn BCTC được cung cấp không nêu tỷ lệ an toàn vốn (CAR) và không có đầu vào để tự tính CAR.",
        "query": "CAR của ngân hàng là bao nhiêu?",
        "expected_numbers": [],
        "required_term_groups": [[
            "chưa đủ bằng chứng", "chưa đủ dữ liệu", "dữ liệu chưa đủ", "dữ liệu không đủ",
            "không có dữ liệu", "chưa được cung cấp", "không đủ dữ liệu",
            "không đủ thông tin", "không thể xác định", "thiếu dữ liệu",
        ]],
    },
]


def normalize_number(value: str) -> str | None:
    raw = value.replace("%", "").replace(" ", "").replace("\u00a0", "")
    sign = ""
    if raw[:1] in {"-", "+"}:
        sign, raw = raw[0], raw[1:]
    if not raw:
        return None
    if "." in raw and "," in raw:
        decimal_separator = "." if raw.rfind(".") > raw.rfind(",") else ","
        grouping_separator = "," if decimal_separator == "." else "."
        raw = raw.replace(grouping_separator, "").replace(decimal_separator, ".")
    elif "." in raw or "," in raw:
        separator = "." if "." in raw else ","
        pieces = raw.split(separator)
        if len(pieces) > 2 or (len(pieces) == 2 and len(pieces[1]) == 3 and len(pieces[0]) <= 3):
            raw = "".join(pieces)
        else:
            raw = ".".join(pieces)
    try:
        number = float(sign + raw)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return str(int(number)) if number.is_integer() else format(number, ".8g")


def score_answer(case: dict[str, Any], answer: str) -> dict[str, Any]:
    folded = answer.casefold()
    answer_numbers = {value for token in NUMBER_RE.findall(answer) if (value := normalize_number(token)) is not None}
    expected_numbers = set(case["expected_numbers"])
    missing_numbers = sorted(expected_numbers - answer_numbers, key=float)
    forbidden_numbers = sorted(set(case.get("forbidden_numbers", [])) & answer_numbers, key=float)
    missing_term_groups = [
        choices for choices in case.get("required_term_groups", [])
        if not any(term.casefold() in folded for term in choices)
    ]
    passed = not missing_numbers and not forbidden_numbers and not missing_term_groups
    return {
        "passed": passed,
        "expected_numbers": sorted(expected_numbers, key=float),
        "observed_numbers": sorted(answer_numbers, key=float),
        "missing_numbers": missing_numbers,
        "forbidden_numbers_found": forbidden_numbers,
        "missing_term_groups": missing_term_groups,
    }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["model_total_ms"] for row in rows if row.get("model_total_ms") is not None]
    generation = [row["generation_ms"] for row in rows if row.get("generation_ms") is not None]
    throughputs = [row["generated_tokens_per_second"] for row in rows if row.get("generated_tokens_per_second") is not None]
    return {
        "trials": len(rows),
        "passed_trials": sum(row["score"]["passed"] for row in rows),
        "trial_accuracy": round(sum(row["score"]["passed"] for row in rows) / len(rows), 4) if rows else None,
        "mean_model_total_ms": round(statistics.fmean(latencies), 2) if latencies else None,
        "p50_model_total_ms": round(statistics.median(latencies), 2) if latencies else None,
        "p95_model_total_ms": round(percentile(latencies, .95), 2) if latencies else None,
        "mean_generation_ms": round(statistics.fmean(generation), 2) if generation else None,
        "mean_generated_tokens_per_second": round(statistics.fmean(throughputs), 2) if throughputs else None,
    }


def request_chat(base_url: str, model: str, prompt: str, timeout: float) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "think": False,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 256, "num_ctx": 4096},
    }
    started = time.perf_counter()
    response = requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    wall_ms = (time.perf_counter() - started) * 1000
    data = response.json()
    answer = data.get("message", {}).get("content", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty answer")
    return {**data, "answer": answer}, wall_ms


def build_prompt(case: dict[str, Any]) -> str:
    return (
        "Bạn là trợ lý đọc báo cáo tài chính. Chỉ dùng dữ liệu được cung cấp. "
        "Chỉ coi các trường doanh nghiệp, kỳ và cơ sở được nêu trong câu hỏi là điều kiện bắt buộc; "
        "không đòi trường không được hỏi. Bỏ qua fact xung đột với phạm vi được hỏi. Kiểm tra toàn bộ "
        "fact; nếu có fact khớp thì dùng nó, chỉ từ chối khi không có bằng chứng khớp. Giữ nguyên đơn vị, "
        "tính đúng nếu câu hỏi yêu cầu, "
        "nêu lại mã doanh nghiệp và kỳ được hỏi trong câu trả lời khi câu hỏi có nêu chúng, "
        "và trả lời ngắn gọn bằng tiếng Việt.\n\n"
        f"[NGỮ CẢNH NGUỒN]\n{case['context']}\n\n[CÂU HỎI]\n{case['query']}\n\n[CÂU TRẢ LỜI]"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3.5:4b"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    warmup, warmup_wall_ms = request_chat(
        args.base_url,
        args.model,
        "Chỉ trả lời bằng chữ số: 2 + 2 = ?",
        args.timeout,
    )
    rows: list[dict[str, Any]] = []
    for case in CASES:
        for repeat in range(1, args.repeats + 1):
            response, wall_ms = request_chat(args.base_url, args.model, build_prompt(case), args.timeout)
            duration_ns = response.get("total_duration")
            generation_ns = response.get("eval_duration")
            eval_count = response.get("eval_count")
            rows.append({
                "case_id": case["id"],
                "category": case["category"],
                "repeat": repeat,
                "wall_ms": round(wall_ms, 2),
                "model_total_ms": round(duration_ns / 1_000_000, 2) if isinstance(duration_ns, (int, float)) else None,
                "generation_ms": round(generation_ns / 1_000_000, 2) if isinstance(generation_ns, (int, float)) else None,
                "prompt_tokens": response.get("prompt_eval_count"),
                "generated_tokens": eval_count,
                "generated_tokens_per_second": round(eval_count / (generation_ns / 1_000_000_000), 2)
                if isinstance(eval_count, (int, float)) and isinstance(generation_ns, (int, float)) and generation_ns > 0
                else None,
                "score": score_answer(case, response["answer"]),
                "answer": response["answer"],
            })
            print(
                f"{'PASS' if rows[-1]['score']['passed'] else 'FAIL'} {case['id']} "
                f"repeat={repeat} model={rows[-1]['model_total_ms']}ms wall={rows[-1]['wall_ms']}ms"
            )

    by_category: dict[str, list[dict[str, Any]]] = {}
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)
        by_case.setdefault(row["case_id"], []).append(row)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_type": "generation_only_controlled_context",
        "base_url": args.base_url,
        "model": args.model,
        "settings": {"temperature": 0.1, "num_predict": 256, "num_ctx": 4096, "think": False},
        "protocol": {"warmup_requests": 1, "repeats_per_case": args.repeats, "max_inflight": 1},
        "warmup": {
            "wall_ms": round(warmup_wall_ms, 2),
            "model_total_ms": round(warmup["total_duration"] / 1_000_000, 2)
            if isinstance(warmup.get("total_duration"), (int, float)) else None,
        },
        "summary": aggregate(rows),
        "case_repeat_consistency_rate": round(
            sum(all(row["score"]["passed"] for row in case_rows) for case_rows in by_case.values())
            / len(by_case), 4,
        ),
        "by_category": {
            category: aggregate(category_rows)
            for category, category_rows in sorted(by_category.items())
        },
        "results": rows,
        "cases": [
            {key: value for key, value in case.items() if key != "context"}
            for case in CASES
        ],
        "limitations": [
            "Generation-only test with fixed synthetic context; it does not measure document retrieval or source OCR.",
            "Accuracy is mechanical matching of expected values/phrases, not a blinded human quality score.",
            "This model microbenchmark is separate from end-to-end API latency and deterministic financial calculations.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "by_category": report["by_category"]}, ensure_ascii=False, indent=2))
    print(f"Saved {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
