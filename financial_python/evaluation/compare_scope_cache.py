"""Small A/B workload: same code/corpus, one cached scope versus eight."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import requests


def compare(urls: dict[str, str], rounds: int = 8) -> dict:
    for url in urls.values():
        for attempt in range(45):
            try:
                if requests.get(url + "/health/ready", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise RuntimeError(f"API not ready: {url}")
    times = {key: [] for key in urls}
    rows = []
    for round_number in range(rounds):
        for company in ("VPB", "ACB", "MBB"):
            query = f"Đánh giá khả năng sinh lời {company} năm 2025 theo báo cáo hợp nhất, tính ROA và ROE trên số dư bình quân đầu và cuối năm."
            signatures = set()
            for label, url in urls.items():
                start = time.perf_counter()
                response = requests.post(url + "/api/v1/financial/answer", json={"query": query}, timeout=90)
                response.raise_for_status()
                elapsed = time.perf_counter() - start
                data = response.json()
                if not data["validation"]["is_valid"]:
                    raise AssertionError(f"Validation failed for {label}/{company}")
                signature = hashlib.sha256(json.dumps({key: data[key] for key in
                    ("answer", "citations", "calculations", "answer_status")}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                signatures.add(signature)
                if round_number:
                    times[label].append(elapsed)
                rows.append({"configuration": label, "round": round_number, "company": company,
                             "seconds": elapsed, "answer_sha256": signature})
            if len(signatures) != 1:
                raise AssertionError("Cache configuration changed the answer/evidence")
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(),
            "workload": "sequential, alternating VPB/ACB/MBB profitability; first round excluded",
            "dense_enabled": False, "same_code_and_corpus": True, "answers_identical": True,
            "not_a_concurrent_load_test": True,
            "summary": {label: {"samples": len(values), "p50_seconds": statistics.median(values),
                                 "p95_seconds": float(np.percentile(values, 95)),
                                 "mean_seconds": statistics.mean(values)} for label, values in times.items()},
            "requests": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-url", default="http://127.0.0.1:8094")
    parser.add_argument("--multi-url", default="http://127.0.0.1:8092")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare({"one_scope": args.single_url, "eight_scopes": args.multi_url})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
