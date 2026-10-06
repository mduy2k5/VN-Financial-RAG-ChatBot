"""Run the same reviewed question set against both local project APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_WARMUP_QUERIES = [
    "Mô tả ngắn hoạt động kinh doanh chính của doanh nghiệp trong báo cáo thường niên.",
    "Báo cáo nêu chính sách kế toán nào cho tiền và các khoản tương đương tiền?",
    "Tóm tắt ý kiến kiểm toán và các vấn đề kiểm toán trọng yếu được nêu trong báo cáo.",
    "Những rủi ro hoạt động chính nào được doanh nghiệp mô tả trong năm tài chính?",
    "Tóm tắt biến động lưu chuyển tiền từ hoạt động kinh doanh theo báo cáo năm.",
]


class BenchmarkError(RuntimeError):
    pass


def read_cases(path: Path, split: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BenchmarkError(f"Case file not found: {path}")
    cases: list[dict[str, Any]] = []
    ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            case_id = case.get("id")
            if not isinstance(case_id, str) or not case_id or case_id in ids:
                raise BenchmarkError(f"Missing or duplicate case id at {path}:{line_number}")
            ids.add(case_id)
            if case.get("split") != split:
                continue
            if not isinstance(case.get("query"), str) or not case["query"].strip():
                raise BenchmarkError(f"{case_id}: query must be a nonempty string")
            if split == "holdout" and case.get("review_status") != "verified":
                raise BenchmarkError(
                    f"{case_id}: refusing to run unverified holdout label; "
                    "review_status must be 'verified'"
                )
            cases.append(case)
    if not cases:
        raise BenchmarkError(f"No cases found for split={split!r}")
    return cases


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float,
) -> Any:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_headers = {"Accept": "application/json", **(headers or {})}
    if data is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:4000]
        raise BenchmarkError(f"HTTP {exc.code} {method} {url}: {body}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise BenchmarkError(f"Request failed {method} {url}: {exc}") from exc
    try:
        return json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"Expected JSON from {method} {url}") from exc


def run_financial(
    base_url: str, query: str, *, timeout: float, top_k: int
) -> tuple[Any, dict[str, Any]]:
    start = time.perf_counter()
    response = request_json(
        f"{base_url.rstrip('/')}/api/v1/financial/answer",
        method="POST",
        payload={"query": query, "top_k": top_k},
        timeout=timeout,
    )
    return response, {
        "end_to_end_ms": round((time.perf_counter() - start) * 1000, 2),
        "poll_count": 0,
        "cache_hit": None,
    }


def run_data_services(
    base_url: str,
    query: str,
    *,
    timeout: float,
    poll_interval: float,
    api_key: str,
    agent_id: str,
) -> tuple[Any, dict[str, Any]]:
    root = base_url.rstrip("/")
    headers = {"key": api_key}
    start = time.perf_counter()
    task = request_json(
        f"{root}/v2.3/knowspace_search",
        method="POST",
        payload={
            "query": query,
            "search_folder": ["all"],
            "is_final_answer": True,
            "allow_direct_reply": False,
            "ai_agent_id": agent_id,
            "history_data": None,
        },
        headers=headers,
        timeout=timeout,
    )
    task_id = task.get("task_id") if isinstance(task, dict) else None
    if not task_id:
        raise BenchmarkError("data-services did not return task_id")
    poll_count = 0
    deadline = start + timeout
    while time.perf_counter() < deadline:
        poll_count += 1
        escaped_task_id = urllib.parse.quote(str(task_id), safe="")
        result = request_json(
            f"{root}/v2.3/knowspace_search/{escaped_task_id}",
            headers=headers,
            timeout=min(timeout, max(1.0, deadline - time.perf_counter())),
        )
        status = result.get("status") if isinstance(result, dict) else None
        if status == "completed":
            result_summary = result.get("result_summary") or {}
            return result, {
                "end_to_end_ms": round((time.perf_counter() - start) * 1000, 2),
                "poll_count": poll_count,
                "cache_hit": result_summary.get("from_cache"),
                "server_processing_seconds": result_summary.get("llm_generation_seconds"),
                "document_count": result_summary.get("document_count"),
                "route": result_summary.get("route"),
            }
        if status == "failed":
            detail = result.get("error") or result.get("message")
            raise BenchmarkError(f"data-services task failed: {detail}")
        time.sleep(min(poll_interval, max(0.0, deadline - time.perf_counter())))
    raise BenchmarkError(f"data-services task {task_id} exceeded {timeout}s timeout")


def run_warmup(args: argparse.Namespace, api_key: str) -> None:
    for query in DEFAULT_WARMUP_QUERIES[: args.warmup]:
        run_financial(args.financial_url, query, timeout=args.timeout, top_k=args.top_k)
        response, _ = run_data_services(
            args.data_services_url,
            query,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
            api_key=api_key,
            agent_id=args.agent_id,
        )
        if isinstance(response, dict) and response.get("status") != "completed":
            raise BenchmarkError("data-services warm-up did not finish successfully")


def verify_readiness(args: argparse.Namespace) -> dict[str, Any]:
    financial = request_json(
        f"{args.financial_url.rstrip('/')}/health/ready", timeout=min(args.timeout, 15.0)
    )
    data_services = request_json(
        f"{args.data_services_url.rstrip('/')}/v1/readyz", timeout=min(args.timeout, 15.0)
    )
    financial_components = financial.get("components", {}) if isinstance(financial, dict) else {}
    dense = financial_components.get("dense_retrieval", {})
    dense_status = dense.get("status") if isinstance(dense, dict) else dense
    data_ready = data_services.get("ready") if isinstance(data_services, dict) else False
    if financial.get("status") not in {"ready", "healthy"}:
        raise BenchmarkError(f"financial API is not ready: {financial}")
    if dense_status != "ready":
        raise BenchmarkError(
            f"financial dense retrieval must be ready before the run; status={dense_status!r}"
        )
    if data_ready is not True and data_services.get("status") not in {"ready", "healthy"}:
        raise BenchmarkError(f"data-services is not ready: {data_services}")
    return {"financial": financial, "data_services": data_services}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "holdout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--financial-url", default="http://127.0.0.1:8092")
    parser.add_argument("--data-services-url", default="http://127.0.0.1:8000")
    parser.add_argument("--data-services-key-env", default="BENCH_DATA_SERVICES_API_KEY")
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=9173)
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()

    if args.timeout <= 0 or args.poll_interval <= 0 or args.warmup < 0 or args.top_k < 1:
        parser.error("timeout/poll interval/top-k must be positive and warmup nonnegative")
    api_key = os.environ.get(args.data_services_key_env, "").strip()
    if not api_key:
        parser.error(f"Environment variable {args.data_services_key_env} is required")
    if args.output.exists():
        parser.error(f"Output already exists; choose a new path to avoid mixing runs: {args.output}")

    try:
        cases = read_cases(args.cases, args.split)
        readiness = verify_readiness(args)
        if not args.skip_warmup:
            run_warmup(args, api_key)
        rng = random.Random(args.seed)
        output_rows: list[dict[str, Any]] = []
        systems = ["financial_rag", "data_services"]
        for case in cases:
            run_order = systems.copy()
            rng.shuffle(run_order)
            for system in run_order:
                started_at = datetime.now(timezone.utc).isoformat()
                try:
                    if system == "financial_rag":
                        response, timing = run_financial(
                            args.financial_url,
                            case["query"],
                            timeout=args.timeout,
                            top_k=args.top_k,
                        )
                    else:
                        response, timing = run_data_services(
                            args.data_services_url,
                            case["query"],
                            timeout=args.timeout,
                            poll_interval=args.poll_interval,
                            api_key=api_key,
                            agent_id=args.agent_id,
                        )
                    row = {
                        "case_id": case["id"],
                        "split": args.split,
                        "category": case.get("category"),
                        "system": system,
                        "query": case["query"],
                        "started_at_utc": started_at,
                        "timing": timing,
                        "response": response,
                        "error": None,
                    }
                except Exception as exc:  # Preserve failures as benchmark outcomes.
                    row = {
                        "case_id": case["id"],
                        "split": args.split,
                        "category": case.get("category"),
                        "system": system,
                        "query": case["query"],
                        "started_at_utc": started_at,
                        "timing": None,
                        "response": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                output_rows.append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")

        digest = hashlib.sha256(args.cases.read_bytes()).hexdigest()
        print(
            json.dumps(
                {
                    "split": args.split,
                    "case_count": len(cases),
                    "result_rows": len(output_rows),
                    "case_file_sha256": digest,
                    "output": str(args.output.resolve()),
                    "readiness": readiness,
                    "errors": sum(row["error"] is not None for row in output_rows),
                    "data_services_cache_hits": sum(
                        bool((row.get("timing") or {}).get("cache_hit"))
                        for row in output_rows
                        if row["system"] == "data_services"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except BenchmarkError as exc:
        print(f"benchmark setup error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
