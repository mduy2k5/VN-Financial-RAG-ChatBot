"""Score exact facts, calculations, abstention, and source-page citations."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
NUMBER_RE = re.compile(
    r"(?<![\w])[-+]?(?:\d{1,3}(?:[.,\s\u00a0\u202f]\d{3})+|\d+)"
    r"(?:[.,]\d+)?%?(?!\w)"
)
PAGE_RE = re.compile(r"\[?\s*(\d{1,4})\s*\]?")


def parse_decimal_candidates(raw: str, *, expected: bool = False) -> set[Decimal]:
    value = raw.strip().replace("\u00a0", "").replace("\u202f", "").replace(" ", "")
    value = value.rstrip("%")
    if not value:
        return set()
    sign = ""
    if value[:1] in {"-", "+"}:
        sign, value = value[0], value[1:]
    if not value or not re.fullmatch(r"\d[\d.,]*", value):
        return set()

    candidates: set[str] = set()
    if "." in value and "," in value:
        decimal_separator = "." if value.rfind(".") > value.rfind(",") else ","
        group_separator = "," if decimal_separator == "." else "."
        canonical = value.replace(group_separator, "").replace(decimal_separator, ".")
        candidates.add(sign + canonical)
    elif "." in value or "," in value:
        separator = "." if "." in value else ","
        pieces = value.split(separator)
        if len(pieces) > 2:
            if all(len(piece) == 3 for piece in pieces[1:]):
                candidates.add(sign + "".join(pieces))
            else:
                candidates.add(sign + "".join(pieces[:-1]) + "." + pieces[-1])
        elif len(pieces) == 2:
            left, right = pieces
            grouping_form = len(right) == 3 and len(left) <= 3
            if expected:
                if separator == "." and grouping_form:
                    candidates.add(sign + left + right)
                else:
                    candidates.add(sign + left + "." + right)
            else:
                candidates.add(sign + left + "." + right)
                if grouping_form:
                    candidates.add(sign + left + right)
    else:
        candidates.add(sign + value)

    parsed: set[Decimal] = set()
    for candidate in candidates:
        try:
            parsed.add(Decimal(candidate))
        except InvalidOperation:
            continue
    return parsed


def answer_text(system: str, response: Any) -> str:
    if not isinstance(response, dict):
        return ""
    if system == "financial_rag":
        value = response.get("answer")
        return value if isinstance(value, str) else ""
    data = response.get("data")
    if isinstance(data, dict):
        value = data.get("context") or data.get("raw_context")
        if isinstance(value, str):
            return value
    return ""


def citation_pairs(system: str, response: Any, text: str) -> set[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    if system == "financial_rag" and isinstance(response, dict):
        for citation in response.get("citations", []) or []:
            if not isinstance(citation, dict) or not citation.get("file_name"):
                continue
            try:
                page = int(citation["page"])
            except (KeyError, TypeError, ValueError):
                continue
            pairs.add((Path(str(citation["file_name"])).name.casefold(), page))

    for match in re.finditer(r"<<<\s*(\{.*?\})\s*>>>", text, re.DOTALL):
        try:
            source = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = source.get("file_name") if isinstance(source, dict) else None
        page_value = source.get("page_number") if isinstance(source, dict) else None
        if not name:
            continue
        page_match = PAGE_RE.search(str(page_value or ""))
        if page_match:
            pairs.add((Path(str(name)).name.casefold(), int(page_match.group(1))))
    return pairs


def sources_for_fact(fact: dict[str, Any]) -> set[tuple[str, int]]:
    output: set[tuple[str, int]] = set()
    for source in fact.get("sources", []) or []:
        name = source.get("file_name") or source.get("source")
        page = source.get("page")
        if name is not None and page is not None:
            output.add((Path(str(name)).name.casefold(), int(page)))
    return output


def check_value(expected: Any, text: str, *, tolerance: Decimal = Decimal(0)) -> bool:
    gold = parse_decimal_candidates(str(expected), expected=True)
    if not gold:
        return False
    answer_values = [parse_decimal_candidates(token) for token in NUMBER_RE.findall(text)]
    candidates = set().union(*answer_values) if answer_values else set()
    return any(abs(actual - target) <= tolerance for actual in candidates for target in gold)


def case_score(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    system = result["system"]
    text = answer_text(system, result.get("response"))
    pairs = citation_pairs(system, result.get("response"), text)
    facts = case.get("expected_facts", [])
    calculations = case.get("expected_calculations", [])
    fact_matches = [check_value(item.get("value"), text) for item in facts]
    calc_matches = [
        check_value(
            item.get("value"),
            text,
            tolerance=Decimal("0.005") if item.get("unit") == "%" else Decimal(0),
        )
        for item in calculations
    ]
    fact_cited = [bool(sources_for_fact(fact) & pairs) for fact in facts]
    expected_pairs = set().union(*(sources_for_fact(fact) for fact in facts)) if facts else set()
    if not expected_pairs:
        for group in case.get("evidence_groups", []):
            for source in group.get("sources", []):
                name = source.get("file_name") or source.get("source")
                if name and source.get("page") is not None:
                    expected_pairs.add((Path(str(name)).name.casefold(), int(source["page"])))
    citation_hits = len(pairs & expected_pairs)
    citation_precision = citation_hits / len(pairs) if pairs else (1.0 if not expected_pairs else 0.0)
    citation_recall = citation_hits / len(expected_pairs) if expected_pairs else None
    if case.get("answerable", True):
        abstention_ok = None
    else:
        folded = text.casefold()
        abstention_ok = any(
            str(marker).casefold() in folded for marker in case.get("abstention_markers", [])
        )
    return {
        "case_id": case["id"],
        "split": case.get("split"),
        "category": case.get("category"),
        "system": system,
        "request_error": result.get("error"),
        "answer_char_count": len(text),
        "latency_ms": (result.get("timing") or {}).get("end_to_end_ms"),
        "cache_hit": (result.get("timing") or {}).get("cache_hit"),
        "fact_exact_matches": sum(fact_matches),
        "fact_count": len(fact_matches),
        "all_facts_exact": all(fact_matches) if fact_matches else None,
        "calculation_matches": sum(calc_matches),
        "calculation_count": len(calc_matches),
        "all_calculations_match": all(calc_matches) if calc_matches else None,
        "facts_with_correct_source_page": sum(fact_cited),
        "citation_precision": citation_precision if expected_pairs else None,
        "citation_page_recall": citation_recall,
        "abstention_correct": abstention_ok,
        "cited_source_pages": sorted([{"file_name": name, "page": page} for name, page in pairs], key=lambda x: (x["file_name"], x["page"])),
        "answer_text": text,
    }


def mean_or_none(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_latency = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    sorted_latency = sorted(valid_latency)
    p95 = sorted_latency[max(0, int(0.95 * len(sorted_latency) + 0.999) - 1)] if sorted_latency else None
    facts = [row for row in rows if row["fact_count"]]
    calcs = [row for row in rows if row["calculation_count"]]
    citation = [row for row in rows if row["citation_precision"] is not None]
    abstention = [row for row in rows if row["abstention_correct"] is not None]
    return {
        "case_count": len(rows),
        "request_error_rate": round(sum(row["request_error"] is not None for row in rows) / len(rows), 4) if rows else None,
        "exact_fact_rate": mean_or_none([float(row["all_facts_exact"]) for row in facts]),
        "exact_calculation_rate": mean_or_none([float(row["all_calculations_match"]) for row in calcs]),
        "citation_precision": mean_or_none([row["citation_precision"] for row in citation]),
        "citation_page_recall": mean_or_none([row["citation_page_recall"] for row in citation if row["citation_page_recall"] is not None]),
        "abstention_accuracy": mean_or_none([float(row["abstention_correct"]) for row in abstention]),
        "latency_ms_p50": statistics.median(valid_latency) if valid_latency else None,
        "latency_ms_p95": p95,
        "latency_ms_mean": mean_or_none(valid_latency),
        "data_services_cache_hit_count": sum(bool(row.get("cache_hit")) for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = {
        item["id"]: item
        for item in (json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    results = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scored = []
    for result in results:
        case = cases.get(result.get("case_id"))
        if case is None:
            raise SystemExit(f"Result references unknown case: {result.get('case_id')}")
        scored.append(case_score(case, result))

    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in scored:
        by_system[row["system"]].append(row)
        by_category[row["system"]][str(row["category"] or "unknown")].append(row)
    report = {
        "scorer_version": "cross_project_exact_value_citation_v1",
        "scored_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "case_file": str(args.cases.resolve()),
        "results_file": str(args.results.resolve()),
        "summary": {
            system: {
                "overall": aggregate(rows),
                "by_category": {
                    category: aggregate(category_rows)
                    for category, category_rows in sorted(by_category[system].items())
                },
            }
            for system, rows in sorted(by_system.items())
        },
        "results": scored,
        "limitations": [
            "Exact-value matching checks whether a value occurs in the final response; it does not prove that the claim is attached to the correct period or metric.",
            "Citation scoring recognizes structured Financial RAG citations and data-services <<<JSON source>>> blocks. Other citation formats need human review.",
            "Narrative relevance, completeness, causal reasoning, and unsupported claims require blinded human review.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
