"""Execute bounded issuer/period matrices through existing source-checking executors."""
from decimal import Decimal

from ..schemas import ParsedQuery
from ..structure.normalized import unit_dimensions
from ..retrieval.hybrid import _scope_reports
from .overview import OverviewAnswer, build_financial_overview_answer, _select_period_fact, _format_decimal
from .ratio import build_ratio_answer, requested_ratio_inputs, _requested_formulas
from .series import build_series_answer


def combine_answers(answers: list[OverviewAnswer]) -> OverviewAnswer:
    used = {citation.citation_id: citation for answer in answers for citation in answer.citations}
    missing = tuple(item for answer in answers for item in answer.missing_components)
    return OverviewAnswer("\n\n".join(answer.answer for answer in answers), list(used.values()),
                          list(dict.fromkeys(context for answer in answers for context in answer.source_contexts)),
                          [calc for answer in answers for calc in answer.calculations],
                          answer_status="insufficient_evidence" if not used else "partial" if missing or any(answer.answer_status in
                                        {"partial", "insufficient_evidence"} for answer in answers) else "complete",
                          missing_components=missing)


def build_matrix_answer(parsed: ParsedQuery, issuer_plans: dict, metrics: dict, grouped: dict,
                        citations: list, chunks_by_id: dict) -> OverviewAnswer:
    companies = parsed.company_codes or (
        [parsed.company_code] if parsed.company_code is not None else []
    )
    years = sorted(set(parsed.fiscal_years or (
        [parsed.fiscal_year] if parsed.fiscal_year is not None else []
    )))
    answers = []
    for company in companies:
        plan = issuer_plans[company]
        issuer_query = parsed.model_copy(update={"company_codes": [company], "company_code": company})
        if parsed.structured_operation == "ranking":
            continue
        if plan.task in {"comparison", "trend"} and not requested_ratio_inputs(parsed.original_query) and not any(
                term in parsed.original_query.casefold() for term in ("roa", "roe")):
            answers.append(build_series_answer(plan, issuer_query, metrics, grouped, citations))
            continue
        for year in years:
            cell_query = issuer_query.model_copy(update={"fiscal_year": year, "fiscal_years": [year]})
            selected = {}
            for metric_id, rows in grouped.items():
                allowed = _scope_reports([row.chunk for row in rows], [year], [company],
                                         parsed.report_type, parsed.source_policy)
                ids = {chunk.chunk_id for chunk in allowed}
                selected[metric_id] = [row for row in rows if row.chunk.chunk_id in ids]
            rows = list({row.chunk.chunk_id: row for values in selected.values() for row in values}.values())
            if parsed.structured_operation == "ratio_analysis" and requested_ratio_inputs(parsed.original_query):
                ratio = build_ratio_answer(cell_query, rows, citations, [], chunks_by_id=chunks_by_id)
                answers.append(OverviewAnswer(ratio.answer, ratio.citations, ratio.source_contexts, ratio.calculations,
                                              answer_status="complete" if ratio.calculations else "insufficient_evidence",
                                              missing_components=() if ratio.calculations else (f"{company}, {year}: thiếu đầu vào tỷ số",)))
            else:
                answers.append(build_financial_overview_answer(plan, cell_query, metrics, selected, citations, [year]))
    if parsed.structured_operation == "ranking":
        return build_ranking(parsed, issuer_plans, metrics, grouped, citations)
    return combine_answers(answers)


def build_ranking(parsed, plans, metrics, grouped, citations):
    companies = parsed.company_codes or (
        [parsed.company_code] if parsed.company_code is not None else []
    )
    years = sorted(set(parsed.fiscal_years or (
        [parsed.fiscal_year] if parsed.fiscal_year is not None else []
    )))
    metric_ids = list(dict.fromkeys(metric for plan in plans.values() for metric in plan.planned_metric_ids))
    formulas = _requested_formulas(parsed.original_query)
    for term, formula in (("roa", "return_on_average_assets_after_tax_percent"),
                          ("roe", "return_on_average_equity_after_tax_percent")):
        if term in parsed.original_query.casefold():
            formulas.append(formula)
    if formulas:
        return build_ratio_ranking(parsed, plans, metrics, grouped, citations, formulas)
    if len(metric_ids) != 1:
        return OverviewAnswer("Cần chỉ rõ một chỉ tiêu để xếp hạng doanh nghiệp.", [], [], [],
                              answer_status="needs_clarification")
    metric_id = metric_ids[0]
    metric = metrics.get(metric_id)
    if metric is None:
        return OverviewAnswer("Chỉ tiêu chưa có định nghĩa thống nhất để xếp hạng.", [], [], [],
                              answer_status="insufficient_evidence")
    by_chunk = {citation.chunk_id: citation for citation in citations}
    lines, missing, used = [], [], {}
    for year in years:
        facts = []
        for company in companies:
            fact, error = _select_period_fact(metric_id, metric, year, grouped.get(metric_id, []),
                                             by_chunk, company, parsed.report_type)
            if not fact or error:
                missing.append(f"{company}, {year}: thiếu hoặc mâu thuẫn bằng chứng")
            else:
                facts.append(fact)
        conventions = {(fact.chunk.report_type, fact.chunk.fiscal_quarter,
                        unit_dimensions(fact.unit).get("currency"),
                        unit_dimensions(fact.unit).get("measurement")) for fact in facts}
        if (len(conventions) != 1 or any(not unit_dimensions(fact.unit).get("scale") for fact in facts)
                or (parsed.fiscal_quarter is None and any(fact.chunk.fiscal_quarter for fact in facts))):
            missing.append(f"{year}: không thể xếp hạng vì khác cơ sở, thời lượng hoặc đơn vị chưa xác định")
            continue
        ascending = any(term in parsed.original_query.casefold() for term in ("tăng dần", "thấp nhất", "nhỏ nhất"))
        facts.sort(key=lambda fact: fact.value * Decimal(unit_dimensions(fact.unit)["scale"]), reverse=not ascending)
        lines.extend([f"Xếp hạng {metric.canonical_name}, năm {year}, {'tăng' if ascending else 'giảm'} dần:", "",
                      "| Hạng | Doanh nghiệp | Giá trị trong nguồn | Nguồn |", "| ---: | --- | ---: | --- |"])
        previous, rank = None, 0
        for position, fact in enumerate(facts, 1):
            value = fact.value * Decimal(unit_dimensions(fact.unit)["scale"])
            if value != previous:
                rank = position
            previous = value
            lines.append(f"| {rank} | {fact.chunk.company_code} | {_format_decimal(fact.value)} {fact.unit} | [{fact.citation.citation_id}] |")
            used[fact.chunk.chunk_id] = fact
    if missing:
        lines.extend(["", "Phần chưa đủ bằng chứng:", *["- " + item for item in missing]])
    lines.extend(["", "Xếp hạng theo giá trị của cùng chỉ tiêu; quy đổi hệ số đơn vị khi rõ tiền tệ. Doanh nghiệp thiếu dữ liệu không được đặt cuối bảng. Thứ hạng này không phải xếp loại sức khỏe tài chính."])
    if not used:
        return OverviewAnswer("Không đủ bằng chứng cùng phạm vi để xếp hạng doanh nghiệp.", [], [], [],
                              answer_status="insufficient_evidence", missing_components=tuple(missing))
    return OverviewAnswer("\n".join(lines), [fact.citation for fact in used.values()],
                          [fact.chunk.content for fact in used.values()], [],
                          answer_status="partial" if missing else "complete",
                          missing_components=tuple(missing))


def build_ratio_ranking(parsed, plans, metrics, grouped, citations, formulas):
    if len(set(formulas)) != 1:
        return OverviewAnswer("Cần chọn một công thức để xếp hạng doanh nghiệp.", [], [], [],
                              answer_status="needs_clarification")
    formula = formulas[0]
    missing, lines, used, contexts, calculations = [], [], {}, [], []
    for year in sorted(set(parsed.fiscal_years or [parsed.fiscal_year])):
        candidates = []
        for company, plan in plans.items():
            query = parsed.model_copy(update={"company_code": company, "company_codes": [company],
                                               "fiscal_year": year, "fiscal_years": [year]})
            selected = {}
            for metric, rows in grouped.items():
                allowed = _scope_reports([row.chunk for row in rows], [year], [company],
                                         parsed.report_type, parsed.source_policy)
                ids = {chunk.chunk_id for chunk in allowed}
                selected[metric] = [row for row in rows if row.chunk.chunk_id in ids]
            rows = list({row.chunk.chunk_id: row for values in selected.values() for row in values}.values())
            if formula.startswith("return_on_average"):
                answer = build_financial_overview_answer(plan, query, metrics, selected, citations, [year])
            else:
                answer = build_ratio_answer(query, rows, citations, [],
                                            chunks_by_id={row.chunk.chunk_id: row.chunk for row in rows})
            matches = [calc for calc in answer.calculations if calc.formula_id == formula]
            if len(matches) != 1:
                missing.append(f"{company}, {year}: không đủ đầu vào cùng báo cáo cho công thức")
                continue
            calc = matches[0]
            by_id = {row.chunk.chunk_id: row.chunk for row in rows}
            basis = {(by_id[key].report_type, by_id[key].fiscal_quarter) for key in calc.input_chunk_ids}
            candidates.append((Decimal(calc.value), company, calc, answer, basis))
        conventions = {convention for candidate in candidates for convention in candidate[4]}
        if len(conventions) > 1:
            missing.append(f"{year}: cơ sở hoặc thời lượng báo cáo khác nhau; không xếp hạng")
            continue
        ascending = any(term in parsed.original_query.casefold() for term in ("tăng dần", "thấp nhất", "nhỏ nhất"))
        candidates.sort(key=lambda item: item[0], reverse=not ascending)
        lines.extend([f"Xếp hạng theo {formula}, năm {year}:", "", "| Hạng | Doanh nghiệp | Tỷ số (%) | Nguồn |",
                      "| ---: | --- | ---: | --- |"])
        rank, prior = 0, None
        for position, (value, company, calc, answer, _basis) in enumerate(candidates, 1):
            if value != prior:
                rank = position
            prior = value
            refs = " ".join(f"[{key}]" for key in calc.input_citation_ids)
            lines.append(f"| {rank} | {company} | {_format_decimal(value, places=2)}% | {refs} |")
            calculations.append(calc)
            used.update({citation.citation_id: citation for citation in answer.citations
                         if citation.citation_id in calc.input_citation_ids})
            contexts.extend(answer.source_contexts)
    lines.extend(["", "Cùng công thức và quy ước đầu vào; ROA/ROE dùng lợi nhuận sau thuế và bình quân số dư đầu–cuối năm. Thứ hạng tỷ số không tự trở thành xếp loại sức khỏe tài chính."])
    if missing:
        lines.extend(["", *["- " + item for item in missing]])
    if not calculations:
        return OverviewAnswer("Không đủ bằng chứng cùng phạm vi để xếp hạng theo tỷ số.", [], [], [],
                              answer_status="insufficient_evidence", missing_components=tuple(missing))
    return OverviewAnswer("\n".join(lines), list(used.values()), list(dict.fromkeys(contexts)), calculations,
                          answer_status="partial" if missing else "complete", missing_components=tuple(missing))
