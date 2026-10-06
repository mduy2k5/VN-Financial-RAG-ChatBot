"""Measure scoped requests at one and four clients against a fixed live corpus."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
import statistics
import time

import requests

QUESTIONS = [
    "So sánh tổng tài sản FPT năm 2023, 2024 và 2025 theo báo cáo hợp nhất",
    "Đánh giá khả năng sinh lời VPB năm 2025 theo báo cáo hợp nhất, tính ROA và ROE",
    "So sánh tổng tài sản VPB, ACB và MBB năm 2025 theo báo cáo hợp nhất",
    "Tỷ lệ nợ phải trả trên tổng tài sản BMP năm 2025 theo báo cáo hợp nhất",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8092")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    health_before = requests.get(args.base_url + "/health", timeout=20).json()

    def ask(index):
        started = time.perf_counter()
        response = requests.post(args.base_url + "/api/v1/financial/answer",
                                 json={"query": QUESTIONS[index % len(QUESTIONS)]}, timeout=90)
        response.raise_for_status()
        data = response.json()
        return {"seconds": time.perf_counter() - started, "valid": data["validation"]["is_valid"],
                "snapshot_id": data["source_snapshot"].get("id"), "question": index % len(QUESTIONS)}

    ask(0)
    runs = []
    for clients in (1, 4):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=clients) as pool:
            rows = list(pool.map(ask, range(24)))
        times = sorted(row["seconds"] for row in rows)
        runs.append({"clients": clients, "requests": len(rows), "p50_seconds": statistics.median(times),
                     "p95_seconds": times[math.ceil(len(times) * .95)-1],
                     "elapsed_seconds": time.perf_counter() - started,
                     "valid_count": sum(row["valid"] for row in rows), "samples": rows})
    output = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "dense_enabled": False,
              "independent_accuracy_benchmark": False, "health_before": health_before, "runs": runs,
              "health_after": requests.get(args.base_url + "/health", timeout=20).json()}
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{key: run[key] for key in ("clients", "requests", "p50_seconds", "p95_seconds", "valid_count")} for run in runs]))


if __name__ == "__main__":
    main()
