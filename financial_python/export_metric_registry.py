"""Export the runtime financial metric mapper as a portable JSON artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from financial_rag.metadata.metric_mapper import FinancialMetricMapper  # noqa: E402


def main() -> None:
    mapper = FinancialMetricMapper()
    registry = {
        "schema_version": 1,
        "metric_count": len(mapper.metrics_list),
        "metrics": [
            {
                "metric_id": metric.metric_id,
                "canonical_name": metric.canonical_name,
                "statement_type": metric.statement_type.value,
                "statement_name_vi": metric.statement_name_vi,
                "code": metric.code,
                "section": metric.section,
                "aliases": metric.aliases,
            }
            for metric in mapper.metrics_list
        ],
    }
    destination = REPOSITORY_ROOT / "financial_metric_registry.json"
    destination.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {len(mapper.metrics_list)} metrics to {destination}")


if __name__ == "__main__":
    main()
