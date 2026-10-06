"""Sample Docker container and NVIDIA GPU usage during a benchmark run."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run_command(args: list[str], timeout: float = 10.0) -> str | None:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def sample_docker(containers: list[str]) -> list[dict[str, Any]]:
    if not containers or not shutil.which("docker"):
        return []
    template = "{{json .}}"
    raw = run_command(["docker", "stats", "--no-stream", "--format", template, *containers])
    if not raw:
        return []
    rows = []
    for line in raw.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"raw": line})
    return rows


def sample_gpu() -> list[dict[str, str]]:
    if not shutil.which("nvidia-smi"):
        return []
    raw = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not raw:
        return []
    values = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            values.append(
                {
                    "name": parts[0],
                    "utilization_percent": parts[1],
                    "memory_used_mib": parts[2],
                    "memory_total_mib": parts[3],
                }
            )
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", action="append", default=[], help="Docker container name or ID")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(f"Output already exists; choose a new path: {args.output}")
    print("Sampling resources; press Ctrl+C to stop.", flush=True)
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            while True:
                row = {
                    "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
                    "docker": sample_docker(args.container),
                    "gpu": sample_gpu(),
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"Stopped. Resource samples: {args.output.resolve()}", flush=True)
        return 0
    except OSError as exc:
        print(f"Could not write resource samples: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
