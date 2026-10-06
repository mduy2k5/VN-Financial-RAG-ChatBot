"""Run an independent financial-analysis benchmark against the live API.

The dataset is expected to contain human-reviewed gold facts, calculations,
source pages, and evidence groups. Parser-generated cases are useful for stress
testing but must not be marked ``verified`` or used as a holdout gate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_rag.validation.numeric import NumericValidator


DEFAULT_DATASET = Path(__file__).resolve().parent / "financial_analysis_cases.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "financial_analysis_latest.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8092"
FINANCIAL_AMOUNT_PATTERN = re.compile(
    r"(?<![\w])[-+]?\d[\d.,]*(?:\s*(?:%|vnd|vnđ|đồng|triệu|tỷ|nghìn|ngàn))",
    re.IGNORECASE,
)
GROUPED_NUMBER_PATTERN = re.compile(
    r"(?<![\w])[-+]?\d{1,3}(?:[.,]\d{3})+(?!\w)|(?<![\w])[-+]?\d+[.,]\d+(?!\w)"
)
PAGE_REFERENCE_PATTERN = re.compile(r"(?i)\b(?:trang|page)\s*\d+\b")
CITATION_PATTERN = re.compile(r"\[(?:c)?\d+\]", re.IGNORECASE)


class BenchmarkDataError(ValueError):
    """Raised when a benchmark case is malformed or unsafe to score as holdout."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_source(source: Any, *, case_id: str) -> None:
    if not isinstance(source, dict):
        raise BenchmarkDataError(f"{case_id}: each source must be an object")
    if not (source.get("file_name") or source.get("source")):
        raise BenchmarkDataError(f"{case_id}: each source needs file_name or source")
    try:
        page = int(source["page"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkDataError(f"{case_id}: each source needs an integer page") from exc
    if page < 1:
        raise BenchmarkDataError(f"{case_id}: source pages must be positive")


def _validate_case(case: Any, *, line_number: int) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise BenchmarkDataError(f"line {line_number}: case must be a JSON object")
    case_id = str(case.get("id", "")).strip()
    if not case_id:
        raise BenchmarkDataError(f"line {line_number}: missing id")
    for field in ("query", "category"):
        if not isinstance(case.get(field), str) or not case[field].strip():
            raise BenchmarkDataError(f"{case_id}: missing non-empty {field}")
    if case.get("split") not in {"dev", "holdout"}:
        raise BenchmarkDataError(f"{case_id}: split must be dev or holdout")
    if not isinstance(case.get("answerable"), bool):
        raise BenchmarkDataError(f"{case_id}: answerable must be boolean")
    review_status = case.get("review_status", "pending")
    if review_status not in {"draft", "pending", "verified"}:
        raise BenchmarkDataError(f"{case_id}: invalid review_status {review_status!r}")
    if review_status == "verified":
        if not isinstance(case.get("reviewer_id"), str) or not case["reviewer_id"].strip():
            raise BenchmarkDataError(f"{case_id}: verified labels need reviewer_id")
        reviewed_at = case.get("reviewed_at")
        if not isinstance(reviewed_at, str) or not reviewed_at.strip():
            raise BenchmarkDataError(f"{case_id}: verified labels need reviewed_at")
        try:
            datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise BenchmarkDataError(f"{case_id}: reviewed_at must be ISO 8601") from exc

    scope = case.get("expected_scope", {})
    if not isinstance(scope, dict):
        raise BenchmarkDataError(f"{case_id}: expected_scope must be an object")
    if not scope.get("company_codes") or not scope.get("fiscal_years"):
        raise BenchmarkDataError(f"{case_id}: expected_scope needs company_codes and fiscal_years")

    facts = case.get("expected_facts", [])
    calculations = case.get("expected_calculations", [])
    groups = case.get("evidence_groups", [])
    for field, value in (
        ("expected_facts", facts),
        ("expected_calculations", calculations),
        ("evidence_groups", groups),
    ):
        if not isinstance(value, list):
            raise BenchmarkDataError(f"{case_id}: {field} must be an array")
    if not case["answerable"] and (facts or calculations):
        raise BenchmarkDataError(f"{case_id}: unanswerable cases cannot have expected facts/calculations")
    abstention_markers = case.get("abstention_markers", [])
    if not isinstance(abstention_markers, list) or any(not isinstance(item, str) for item in abstention_markers):
        raise BenchmarkDataError(f"{case_id}: abstention_markers must be an array of strings")
    if not case["answerable"] and not abstention_markers:
        raise BenchmarkDataError(f"{case_id}: unanswerable cases need explicit abstention_markers")
    if case["answerable"] and not (facts or calculations or case.get("required_terms")):
        raise BenchmarkDataError(f"{case_id}: answerable case needs a gold fact, calculation, or term")
    if case["answerable"] and not groups:
        raise BenchmarkDataError(f"{case_id}: answerable case needs evidence_groups for recall scoring")
    for field in ("required_terms", "forbidden_terms"):
        terms = case.get(field, [])
        if not isinstance(terms, list) or any(not isinstance(item, str) for item in terms):
            raise BenchmarkDataError(f"{case_id}: {field} must be an array of strings")

    for fact in facts:
        if not isinstance(fact, dict) or fact.get("value") is None:
            raise BenchmarkDataError(f"{case_id}: each expected fact needs a value")
        sources = fact.get("sources", [])
        if not isinstance(sources, list) or not sources:
            raise BenchmarkDataError(f"{case_id}: each expected fact needs source pages")
        for source in sources:
            _require_source(source, case_id=case_id)
    for calc in calculations:
        if not isinstance(calc, dict) or not calc.get("formula_id") or calc.get("value") is None:
            raise BenchmarkDataError(f"{case_id}: each expected calculation needs formula_id and value")
        for source in calc.get("sources", []):
            _require_source(source, case_id=case_id)
    for group in groups:
        if not isinstance(group, dict) or not group.get("id"):
            raise BenchmarkDataError(f"{case_id}: each evidence group needs an id")
        sources = group.get("sources", [])
        if not isinstance(sources, list) or not sources:
            raise BenchmarkDataError(f"{case_id}: each evidence group needs source pages")
        for source in sources:
            _require_source(source, case_id=case_id)
    return case


def validate_holdout_isolation(cases: list[dict[str, Any]]) -> None:
    """Reject issuer/document leakage, even when only holdout is selected."""
    def identities(case):
        scope = case.get("expected_scope", {})
        companies = {str(code).casefold() for code in scope.get("company_codes", [])}
        sources = set()
        for records in (case.get("expected_facts", []), case.get("evidence_groups", []),
                        case.get("expected_calculations", [])):
            for record in records:
                for source in record.get("sources", []):
                    name = source.get("file_name") or source.get("source")
                    if name:
                        sources.add("file:" + Path(str(name).replace("\\", "/")).name.casefold())
                    if source.get("source_sha256"):
                        sources.add("sha256:" + source["source_sha256"])
        return companies, sources

    dev_companies, dev_sources = set(), set()
    for case in cases:
        if case["split"] == "dev":
            companies, sources = identities(case)
            dev_companies.update(companies)
            dev_sources.update(sources)
    for case in cases:
        if case["split"] != "holdout":
            continue
        companies, sources = identities(case)
        if sources & dev_sources:
            raise BenchmarkDataError(f"{case['id']}: holdout document appears in dev")
        if companies & dev_companies:
            raise BenchmarkDataError(f"{case['id']}: holdout issuer appears in dev")
        if case.get("author_id") and case.get("author_id") == case.get("reviewer_id"):
            raise BenchmarkDataError(f"{case['id']}: holdout needs an independent reviewer")


def load_cases(dataset_path: Path, split: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    all_cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            case = _validate_case(json.loads(line), line_number=line_number)
        except json.JSONDecodeError as exc:
            raise BenchmarkDataError(f"line {line_number}: invalid JSON: {exc}") from exc
        if case["id"] in seen_ids:
            raise BenchmarkDataError(f"duplicate case id: {case['id']}")
        seen_ids.add(case["id"])
        all_cases.append(case)
        if split == "all" or case["split"] == split:
            if case["split"] == "holdout" and case["review_status"] != "verified":
                raise BenchmarkDataError(
                    f"{case['id']}: holdout labels must be reviewer-verified before scoring"
                )
            cases.append(case)
    if split in {"all", "holdout"}:
        validate_holdout_isolation(all_cases)
    if not cases:
        raise BenchmarkDataError(f"No cases selected for split={split!r} in {dataset_path}")
    return cases


def _normalize_number(value: Any) -> str | None:
    normalized = NumericValidator.normalize_number(str(value))
    return normalized or None


def _answer_numbers(answer: str) -> set[str]:
    return {
        normalized
        for value in NumericValidator.extract_numbers(answer)
        if (normalized := _normalize_number(value)) is not None
    }


def _source_pair(source: dict[str, Any]) -> tuple[str, int]:
    file_name = source.get("file_name") or source.get("source") or ""
    return Path(str(file_name).replace("\\", "/")).name.casefold(), int(source["page"])


def _citation_pair(citation: dict[str, Any]) -> tuple[str, int] | None:
    file_name = citation.get("file_name")
    page = citation.get("page")
    if not file_name or page is None:
        return None
    return Path(str(file_name).replace("\\", "/")).name.casefold(), int(page)


def _citation_id(value: Any) -> str:
    return str(value or "").strip().strip("[]").casefold()


def _used_citations(answer: str, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    referenced = {_citation_id(value) for value in CITATION_PATTERN.findall(answer)}
    return [citation for citation in citations if _citation_id(citation.get("citation_id")) in referenced]


def _cited_period_claims(answer: str) -> list[tuple[str, set[int], set[str]]]:
    """Bind numeric cells/clauses to their period and adjacent citations.

    A global bag of numbers accepts swapped years or citations attached to an
    unrelated claim. Comparison tables bind each year column independently;
    ratio tables bind the whole row to its explicit year cell.
    """
    claims: list[tuple[str, set[int], set[str]]] = []
    header: list[str] = []
    context_years: set[int] = set()

    def years(text: str) -> set[int]:
        return {int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", text)}

    for line in answer.splitlines():
        if line.strip().startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r"[:\s-]+", cell) for cell in cells):
                continue
            if not header:
                header = cells
                continue
            if len(cells) != len(header):
                continue
            refs = {_citation_id(ref) for ref in CITATION_PATTERN.findall(line)}
            year_columns = {index: years(cell) for index, cell in enumerate(header) if years(cell)}
            if year_columns:
                claims.extend((cells[index], period, refs) for index, period in year_columns.items())
            else:
                explicit_year = next((
                    years(cells[index]) for index, cell in enumerate(header)
                    if cell.casefold() in {"năm", "year", "kỳ"}
                ), context_years)
                claims.append((line, explicit_year, refs))
            continue
        header = []
        for clause in re.split(r"(?<=[.!?])\s+", line):
            refs = {_citation_id(ref) for ref in CITATION_PATTERN.findall(clause)}
            period = years(clause)
            if refs:
                # A single source row can cite two explicit period/value
                # pairs in prose. Bind each pair independently; do not accept
                # an unbound bag of values under both years.
                pairs = list(re.finditer(
                    r"\b(?P<year>(?:19|20)\d{2})\b\s*:?\s*"
                    r"[-+]?\d[\d.,]*\s+(?:triệu\s+(?:đồng|vnd)|tỷ\s+(?:đồng|vnd)|vnd|vnđ|đồng)",
                    clause, flags=re.IGNORECASE,
                ))
                if len(refs) == 1 and len(pairs) >= 2:
                    claims.extend((pair.group(0), {int(pair.group("year"))}, refs) for pair in pairs)
                    continue
                claims.append((clause, period or context_years, refs))
            elif period:
                context_years = period
    return claims


def _source_supports_period_value(fact: dict[str, Any], citation: dict[str, Any]) -> bool:
    value = _normalize_number(fact["value"])
    year = fact.get("year")
    # The API emits one source-label/period-value snippet per row. Requiring
    # this evidence also catches citations to another row on the same page.
    for part in re.split(r"[|\n]", str(citation.get("snippet") or "")):
        if value not in _answer_numbers(part):
            continue
        if year is None or re.search(rf"\b{int(year)}\b", part):
            return True
    return False


def _scope_result(expected: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    expected_companies = {str(value).upper() for value in expected.get("company_codes", [])}
    actual_companies = {str(value).upper() for value in parsed.get("company_codes", [])}
    if not actual_companies and parsed.get("company_code"):
        actual_companies.add(str(parsed["company_code"]).upper())
    expected_years = {int(value) for value in expected.get("fiscal_years", [])}
    actual_years = {int(value) for value in parsed.get("fiscal_years", [])}
    if parsed.get("fiscal_year") is not None:
        actual_years.add(int(parsed["fiscal_year"]))
    company_pass = not expected_companies or expected_companies == actual_companies
    year_pass = not expected_years or expected_years == actual_years
    expected_report_type = expected.get("report_type")
    report_pass = (
        expected_report_type is None
        or str(parsed.get("report_type") or "").casefold() == str(expected_report_type).casefold()
    )
    quarter_pass = expected.get("fiscal_quarter") == parsed.get("fiscal_quarter")
    policy_pass = ("source_policy" not in expected
                   or expected["source_policy"] == parsed.get("source_policy", "source_report"))
    return {
        "passed": company_pass and year_pass and report_pass and quarter_pass and policy_pass,
        "fiscal_quarter_passed": quarter_pass,
        "source_policy_passed": policy_pass,
        "company_codes_expected": sorted(expected_companies),
        "company_codes_actual": sorted(actual_companies),
        "companies_passed": company_pass,
        "fiscal_years_expected": sorted(expected_years),
        "fiscal_years_actual": sorted(actual_years),
        "fiscal_years_passed": year_pass,
        "report_type_expected": expected_report_type,
        "report_type_actual": parsed.get("report_type"),
        "report_type_passed": report_pass,
    }


def _money_claims(answer: str) -> list[str]:
    cleaned = CITATION_PATTERN.sub("", answer)
    cleaned = PAGE_REFERENCE_PATTERN.sub("", cleaned)
    claims = set(FINANCIAL_AMOUNT_PATTERN.findall(cleaned))
    claims.update(GROUPED_NUMBER_PATTERN.findall(cleaned))
    return sorted(claim.strip() for claim in claims if claim.strip())


def _financial_amount_values(answer: str) -> set[str]:
    cleaned = CITATION_PATTERN.sub("", answer)
    cleaned = PAGE_REFERENCE_PATTERN.sub("", cleaned)
    matches = set(FINANCIAL_AMOUNT_PATTERN.findall(cleaned))
    matches.update(GROUPED_NUMBER_PATTERN.findall(cleaned))
    values: set[str] = set()
    for match in matches:
        values.update(_answer_numbers(match))
    return values


def score_case(
    case: dict[str, Any],
    answer_response: dict[str, Any],
    retrieved_response: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    answer = str(answer_response.get("answer", ""))
    citations = answer_response.get("citations", [])
    citations = citations if isinstance(citations, list) else []
    calculations = answer_response.get("calculations", [])
    calculations = calculations if isinstance(calculations, list) else []
    parsed = answer_response.get("parsed_query", {})
    parsed = parsed if isinstance(parsed, dict) else {}
    validation = answer_response.get("validation", {})
    validation = validation if isinstance(validation, dict) else {}
    used_citations = _used_citations(answer, citations)
    used_pairs = {pair for citation in used_citations if (pair := _citation_pair(citation))}
    citation_by_id = {_citation_id(citation.get("citation_id")): citation for citation in citations}
    scope = _scope_result(case.get("expected_scope", {}), parsed)

    fact_checks = []
    cited_claims = _cited_period_claims(answer)
    for fact in case.get("expected_facts", []):
        expected_value = _normalize_number(fact["value"])
        fact_year = fact.get("year")
        claim_refs: set[str] = set()
        value_match = False
        for claim_text, claim_years, refs in cited_claims:
            if expected_value in _answer_numbers(claim_text) and (
                fact_year is None or claim_years == {int(fact_year)}
            ):
                value_match = True
                claim_refs.update(refs)
        expected_sources = {_source_pair(source) for source in fact.get("sources", [])}
        supporting_citations = [
            citation for citation in used_citations
            if _citation_id(citation.get("citation_id")) in claim_refs
            and _citation_pair(citation) in expected_sources
            and _source_supports_period_value(fact, citation)
        ]
        source_match = bool(supporting_citations)
        expected_unit = fact.get("unit")
        unit_match = True
        if expected_unit:
            normalized_unit = NumericValidator._normalize_unit(str(expected_unit))
            unit_match = any(
                _citation_pair(citation) in expected_sources
                and NumericValidator._normalize_unit(str(citation.get("unit") or "")) == normalized_unit
                for citation in supporting_citations
            )
        fact_checks.append({
            "label": fact.get("label"),
            "year": fact.get("year"),
            "expected_value": str(fact["value"]),
            "value_match": value_match,
            "source_match": source_match,
            "unit_match": unit_match,
            "passed": value_match and source_match and unit_match,
        })

    calculation_checks = []
    for expected in case.get("expected_calculations", []):
        expected_value = _normalize_number(expected["value"])
        candidate_calculations = [
            calculation for calculation in calculations
            if str(calculation.get("formula_id")) == str(expected["formula_id"])
            and _normalize_number(calculation.get("value")) == expected_value
        ]
        if expected.get("unit"):
            expected_unit = NumericValidator._normalize_unit(str(expected["unit"]))
            candidate_calculations = [
                calculation for calculation in candidate_calculations
                if NumericValidator._normalize_unit(str(calculation.get("unit") or "")) == expected_unit
            ]
        if expected.get("formula_version"):
            candidate_calculations = [
                calculation for calculation in candidate_calculations
                if str(calculation.get("formula_version")) == str(expected["formula_version"])
            ]
        source_pairs = {_source_pair(source) for source in expected.get("sources", [])}
        if source_pairs:
            candidate_calculations = [
                calculation for calculation in candidate_calculations
                if any(
                    _citation_id(citation_id) in citation_by_id
                    and _citation_pair(citation_by_id[_citation_id(citation_id)]) in source_pairs
                    for citation_id in calculation.get("input_citation_ids", [])
                )
            ]
        calculation_checks.append({
            "formula_id": expected["formula_id"],
            "expected_value": str(expected["value"]),
            "matched": bool(candidate_calculations),
            "passed": bool(candidate_calculations),
        })

    all_expected_sources = {
        _source_pair(source)
        for fact in case.get("expected_facts", [])
        for source in fact.get("sources", [])
    }
    all_expected_sources.update(
        _source_pair(source)
        for group in case.get("evidence_groups", [])
        for source in group.get("sources", [])
    )
    citation_precision = (
        sum(pair in all_expected_sources for pair in used_pairs) / len(used_pairs)
        if used_pairs else (1.0 if not all_expected_sources else 0.0)
    )
    fact_citation_recall = (
        sum(check["source_match"] for check in fact_checks) / len(fact_checks)
        if fact_checks else None
    )

    evidence_group_checks = []
    retrieved_response = retrieved_response or []
    retrieved_pairs = {
        pair
        for result in retrieved_response
        if isinstance(result, dict)
        and isinstance(result.get("chunk"), dict)
        and (pair := _citation_pair({
            "file_name": result["chunk"].get("source_file"),
            "page": result["chunk"].get("page"),
        }))
    }
    for group in case.get("evidence_groups", []):
        expected_group_sources = {_source_pair(source) for source in group.get("sources", [])}
        evidence_group_checks.append({
            "id": group["id"],
            "retrieval_hit": bool(expected_group_sources & retrieved_pairs),
            "answer_citation_hit": bool(expected_group_sources & used_pairs),
        })
    retrieval_evidence_recall = (
        sum(check["retrieval_hit"] for check in evidence_group_checks) / len(evidence_group_checks)
        if evidence_group_checks else None
    )
    answer_evidence_coverage = (
        sum(check["answer_citation_hit"] for check in evidence_group_checks)
        / len(evidence_group_checks)
        if evidence_group_checks else None
    )

    lineage_checks = []
    for calculation in calculations:
        input_citations = calculation.get("input_citation_ids", [])
        input_chunks = set(calculation.get("input_chunk_ids", []))
        input_citation_objects = [
            citation_by_id.get(_citation_id(citation_id))
            for citation_id in input_citations
        ]
        mapped_input_chunks = {
            citation.get("chunk_id")
            for citation in input_citation_objects
            if citation and citation.get("chunk_id")
        }
        lineage_valid = bool(input_citations and input_chunks) and all(input_citation_objects) and (
            mapped_input_chunks == input_chunks
        )
        lineage_checks.append(lineage_valid)
    calculation_lineage_valid = all(lineage_checks)

    required_terms = [str(term).casefold() for term in case.get("required_terms", [])]
    forbidden_terms = [str(term).casefold() for term in case.get("forbidden_terms", [])]
    text_lower = answer.casefold()
    required_terms_pass = all(term in text_lower for term in required_terms)
    forbidden_terms_pass = not any(term in text_lower for term in forbidden_terms)
    abstention_markers = [
        str(term).casefold() for term in case.get("abstention_markers", [])
    ]
    abstention_marker_pass = (
        not abstention_markers or any(term in text_lower for term in abstention_markers)
    )
    response_valid = bool(
        validation.get("is_valid")
        and validation.get("unit_match", True)
        and validation.get("citation_valid", True)
    )

    answerable = bool(case["answerable"])
    financial_claims = _money_claims(answer)
    expected_numeric_values = {
        normalized_value
        for raw_value in (
            *[fact.get("value") for fact in case.get("expected_facts", [])],
            *[calculation.get("value") for calculation in case.get("expected_calculations", [])],
        )
        if (normalized_value := _normalize_number(raw_value)) is not None
    }
    observed_numeric_values = _financial_amount_values(answer)
    unexpected_numeric_values = sorted(observed_numeric_values - expected_numeric_values)
    abstention_pass = None if answerable else not (
        financial_claims or calculations or not abstention_marker_pass
    )
    facts_pass = all(check["passed"] for check in fact_checks)
    calculations_pass = all(check["passed"] for check in calculation_checks)
    if answerable:
        case_pass = bool(
            scope["passed"]
            and response_valid
            and facts_pass
            and calculations_pass
            and not unexpected_numeric_values
            and answer_evidence_coverage in {None, 1.0}
            and required_terms_pass
            and forbidden_terms_pass
            and calculation_lineage_valid
        )
    else:
        case_pass = bool(
            scope["passed"]
            and response_valid
            and abstention_pass
            and required_terms_pass
            and forbidden_terms_pass
        )

    return {
        "id": case["id"],
        "split": case["split"],
        "category": case["category"],
        "review_status": case.get("review_status", "pending"),
        "query": case["query"],
        "answerable": answerable,
        "case_pass": case_pass,
        "scope": scope,
        "response_validation": validation,
        "expected_facts": fact_checks,
        "expected_calculations": calculation_checks,
        "unexpected_numeric_values": unexpected_numeric_values,
        "financial_values_in_answer": sorted(observed_numeric_values),
        "evidence_groups": evidence_group_checks,
        "retrieval_evidence_group_recall_at_k": retrieval_evidence_recall,
        "answer_evidence_group_coverage": answer_evidence_coverage,
        "fact_citation_recall": fact_citation_recall,
        "citation_precision": round(citation_precision, 4),
        "used_citations": [
            {
                "citation_id": citation.get("citation_id"),
                "file_name": citation.get("file_name"),
                "page": citation.get("page"),
                "unit": citation.get("unit"),
            }
            for citation in used_citations
        ],
        "calculations": calculations,
        "calculation_lineage_valid": calculation_lineage_valid,
        "required_terms_pass": required_terms_pass,
        "forbidden_terms_pass": forbidden_terms_pass,
        "abstention_marker_pass": abstention_marker_pass,
        "financial_claims_in_abstention": financial_claims if not answerable else [],
        "abstention_pass": abstention_pass,
        "answer": answer,
        "parsed_query": parsed,
    }


def _mean(values: list[float | int | bool | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return round(sum(present) / len(present), 4) if present else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latency = [row["latency_ms"] for row in rows if row.get("latency_ms") is not None]
    fact_checks = [check for row in rows for check in row.get("expected_facts", [])]
    calculation_checks = [check for row in rows for check in row.get("expected_calculations", [])]
    evidence_groups = [check for row in rows for check in row.get("evidence_groups", [])]
    answerable = [row for row in rows if row.get("answerable")]
    unanswerable = [row for row in rows if not row.get("answerable")]
    p95_latency = None
    if latency:
        ordered = sorted(latency)
        p95_latency = round(ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)], 2)
    return {
        "cases": len(rows),
        "passed_cases": sum(bool(row.get("case_pass")) for row in rows),
        "case_pass_rate": _mean([row.get("case_pass") for row in rows]),
        "scope_accuracy": _mean([row.get("scope", {}).get("passed") for row in rows]),
        "api_validation_rate": _mean([
            bool(
                row.get("response_validation", {}).get("is_valid")
                and row.get("response_validation", {}).get("unit_match", True)
                and row.get("response_validation", {}).get("citation_valid", True)
            )
            if row.get("response_validation") is not None else None
            for row in rows
        ]),
        "fact_exact_match_rate": _mean([check["passed"] for check in fact_checks]),
        "calculation_exact_match_rate": _mean([check["passed"] for check in calculation_checks]),
        "fact_citation_recall": _mean([row.get("fact_citation_recall") for row in answerable]),
        "citation_precision": _mean([row.get("citation_precision") for row in answerable]),
        "retrieval_evidence_group_recall_at_k": _mean([
            check["retrieval_hit"] for check in evidence_groups
        ]),
        "answer_evidence_group_coverage": _mean([
            check["answer_citation_hit"] for check in evidence_groups
        ]),
        "abstention_accuracy": _mean([row.get("abstention_pass") for row in unanswerable]),
        "mean_latency_ms": round(sum(latency) / len(latency), 2) if latency else None,
        "p95_latency_ms": p95_latency,
    }


def category_release_gate(rows: list[dict], cases: list[dict]) -> dict:
    summary = _aggregate(rows)
    recall = summary["retrieval_evidence_group_recall_at_k"]
    mechanical = bool(rows and summary["case_pass_rate"] == 1 and summary["scope_accuracy"] == 1
                      and summary["citation_precision"] in {None, 1}
                      and ((recall is not None and recall >= .98)
                           or (recall is None and not any(row.get("answerable") for row in rows)
                               and summary["abstention_accuracy"] == 1)))
    holdout = [case for case in cases if case["split"] == "holdout"]
    reviewed = len(holdout) >= 5 and all(case.get("review_status") == "verified"
        and case.get("author_id") and case.get("reviewer_id") != case.get("author_id")
        and case.get("interpretation_score", 0) >= 4 for case in holdout)
    return {"mechanical_passed": mechanical, "independent_holdout_and_interpretation_passed": reviewed,
            "source_hash_audit_required": True, "production_release_passed": False,
            "cases": len(rows), "holdout_cases": len(holdout)}


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, timeout: float) -> Any:
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_error(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensive for broken HTTP streams
            detail = ""
        return f"HTTP {exc.code}: {detail[:1000]}"
    if isinstance(exc, URLError):
        return f"connection error: {exc.reason}"
    return f"{type(exc).__name__}: {exc}"


def run_benchmark(
    cases: list[dict[str, Any]],
    *,
    base_url: str,
    top_k: int,
    timeout: float,
    poster: Any = _post_json,
    getter: Any = _get_json,
) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    api_root = f"{base_url}/api/v1/financial"
    try:
        health = getter(f"{base_url}/health/ready", timeout)
    except Exception as exc:  # service metadata is useful but should not hide case-level errors
        health = {"health_error": _http_error(exc)}

    dense_status = (
        health.get("components", {}).get("dense_retrieval")
        if isinstance(health, dict) else None
    )
    retrieval_mode = (
        "unknown" if "health_error" in health
        else "hybrid" if dense_status == "ready"
        else "bm25_fallback"
    )
    rows = []
    holdout_cases = [case for case in cases if case["split"] == "holdout"]
    for case in cases:
        row: dict[str, Any] = {
            "id": case["id"],
            "split": case["split"],
            "category": case["category"],
            "company_codes": case.get("expected_scope", {}).get("company_codes", []),
            "answerable": case["answerable"],
            "review_status": case.get("review_status", "pending"),
            "top_k": top_k,
            "latency_ms": None,
            "case_pass": False,
        }
        started = time.perf_counter()
        try:
            retrieved = poster(
                f"{api_root}/retrieve",
                {"query": case["query"], "top_k": top_k},
                timeout,
            )
            if not isinstance(retrieved, list):
                raise ValueError("retrieve endpoint returned a non-array response")
            row["retrieval_error"] = None
        except Exception as exc:
            retrieved = []
            row["retrieval_error"] = _http_error(exc)
        try:
            response = poster(
                f"{api_root}/answer",
                {"query": case["query"], "top_k": top_k},
                timeout,
            )
            if not isinstance(response, dict):
                raise ValueError("answer endpoint returned a non-object response")
            scored = score_case(case, response, retrieved)
            row.update(scored)
            row["answer_error"] = None
        except Exception as exc:
            row["answer_error"] = _http_error(exc)
            row["case_pass"] = False
            row["answer"] = None
            row["retrieval_evidence_group_recall_at_k"] = None
            row["answer_evidence_group_coverage"] = None
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        rows.append(row)

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_company: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
        by_split[row["split"]].append(row)
        for code in row.get("company_codes", []) or ["unspecified"]:
            by_company[str(code)].append(row)

    summary = _aggregate(rows)
    summary["by_split"] = {key: _aggregate(value) for key, value in sorted(by_split.items())}
    summary["by_category"] = {key: _aggregate(value) for key, value in sorted(by_category.items())}
    summary["by_company"] = {key: _aggregate(value) for key, value in sorted(by_company.items())}
    summary["release_gates_by_category"] = {
        category: category_release_gate(category_rows,
            [case for case in cases if case["category"] == category])
        for category, category_rows in by_category.items()
    }
    evidence_recall = summary["retrieval_evidence_group_recall_at_k"]
    overall_automated = bool(
        summary["scope_accuracy"] == 1.0
        and summary["fact_exact_match_rate"] in {None, 1.0}
        and summary["calculation_exact_match_rate"] in {None, 1.0}
        and summary["fact_citation_recall"] in {None, 1.0}
        and summary["citation_precision"] in {None, 1.0}
        and (evidence_recall is None or evidence_recall >= 0.98)
        and summary["answer_evidence_group_coverage"] in {None, 1.0}
        and summary["abstention_accuracy"] in {None, 1.0}
        and summary["case_pass_rate"] == 1.0
    )
    summary["automated_gate"] = {
        "passed": overall_automated,
        "thresholds": {
            "scope_accuracy": 1.0,
            "fact_exact_match_rate": 1.0,
            "calculation_exact_match_rate": 1.0,
            "fact_citation_recall": 1.0,
            "citation_precision": 1.0,
            "retrieval_evidence_group_recall_at_k": 0.98,
            "answer_evidence_group_coverage": 1.0,
            "abstention_accuracy": 1.0,
            "case_pass_rate": 1.0,
        },
        "interpretation_rubric": "not_scored_by_automation",
        "overall_mvp_gate": "pending_human_interpretation_review",
    }
    try:
        health_after = getter(f"{base_url}/health/ready", timeout)
    except Exception as exc:
        health_after = {"health_error": _http_error(exc)}
    status_after = health_after.get("components", {}).get("dense_retrieval")
    if status_after != dense_status:
        retrieval_mode = "transitioned_during_run"
    return {
        "run_metadata": {
            "scoring_version": "period_claim_citation_v3",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_url": base_url,
            "retrieval_mode_inferred": retrieval_mode,
            "readiness": health,
            "readiness_after": health_after,
            "top_k": top_k,
            "timeout_seconds": timeout,
            "python_version": platform.python_version(),
            "case_count": len(cases),
            "verified_case_count": sum(case["review_status"] == "verified" for case in cases),
            "unanswerable_case_count": sum(not case["answerable"] for case in cases),
            "holdout_verified": bool(holdout_cases) and all(
                case["review_status"] == "verified" for case in holdout_cases
            ),
        },
        "summary": summary,
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run labeled financial-analysis cases against the live Financial RAG API"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--api-image-id")
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > 100:
        parser.error("--top-k must be between 1 and 100")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    try:
        cases = load_cases(args.dataset, args.split)
    except (OSError, BenchmarkDataError) as exc:
        parser.error(str(exc))
    result = run_benchmark(
        cases,
        base_url=args.base_url,
        top_k=args.top_k,
        timeout=args.timeout,
    )
    result["run_metadata"]["dataset_path"] = str(args.dataset.resolve())
    result["run_metadata"]["dataset_sha256"] = _sha256(args.dataset)
    if args.corpus_manifest is not None:
        manifest = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
        if manifest.get("source_snapshot_sha256"):
            from audit_expansion_readiness import audit
            source_audit = audit(cases, manifest)
            result["summary"]["source_review_audit"] = {key: source_audit[key] for key in
                ("production_accuracy_gate_passed", "verified_count", "reasons")}
            for gate in result["summary"]["release_gates_by_category"].values():
                gate["source_hash_audit_required"] = not source_audit["production_accuracy_gate_passed"]
                gate["production_release_passed"] = bool(gate["mechanical_passed"]
                    and gate["independent_holdout_and_interpretation_passed"]
                    and source_audit["production_accuracy_gate_passed"])
        result["run_metadata"].update({
            "corpus_manifest_path": str(args.corpus_manifest.resolve()),
            "corpus_manifest_sha256": _sha256(args.corpus_manifest),
            "corpus_snapshot_sha256": manifest.get("snapshot_sha256") or manifest["source_snapshot_sha256"],
            "runtime_chunk_snapshot_sha256": manifest.get("chunk_snapshot_sha256"),
            "runtime_code_snapshot_sha256": manifest.get("code_snapshot_sha256"),
            "runtime_summary": manifest.get("summary") or {
                key: manifest[key] for key in ("document_count", "declared_chunk_count", "by_company")
            },
        })
    if args.api_image_id:
        result["run_metadata"]["api_image_id"] = args.api_image_id
    output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": result["summary"]}, ensure_ascii=False, indent=2))
    failed_cases = [row for row in result["results"] if not row.get("case_pass")]
    return 1 if args.split == "holdout" and failed_cases else 0


if __name__ == "__main__":
    raise SystemExit(main())
