"""Reconcile bank credit classifications before using them as evidence."""
from __future__ import annotations

from decimal import Decimal
from typing import cast

from ..schemas import RetrievalResult
from ..validation.numeric import NumericValidator
from .evidence import _source_amount


CREDIT_GROUP_IDS = (
    "BANK.NOTE.8.STANDARD_DEBT", "BANK.NOTE.8.SPECIAL_MENTION",
    "BANK.NOTE.8.SUBSTANDARD", "BANK.NOTE.8.DOUBTFUL", "BANK.NOTE.8.LOSS",
)
GROSS_LOANS_ID = "ASSET.CUSTOMER_LOANS.GROSS"
MARGIN_LOANS_ID = "cho_vay_giao_dich_ky_quy_va_ung_truoc_cho_khach_hang"


def reconcile_gross_loans(
    results: dict[str, list[RetrievalResult]], years: list[int],
) -> dict[str, list[RetrievalResult]]:
    """Disambiguate gross/net loan rows using the contra-asset in their table."""
    candidates = results.get(GROSS_LOANS_ID, [])
    matches = {}
    for allowance in results.get("ASSET.CUSTOMER_LOANS.ALLOWANCE", []):
        anchor = allowance.chunk
        table = anchor.metadata.get("table_id")
        if not table:
            continue
        peers = [row for row in candidates if (
            row.chunk.document_id == anchor.document_id
            and row.chunk.report_type == anchor.report_type
            and row.chunk.metadata.get("table_id") == table
            and NumericValidator._normalize_unit(row.chunk.unit or "")
            == NumericValidator._normalize_unit(anchor.unit or "")
        )]
        for gross in peers:
            for net in peers:
                amounts = tuple(_source_amount(gross.chunk, year) for year in years)
                nets = tuple(_source_amount(net.chunk, year) for year in years)
                provisions = tuple(_source_amount(anchor, year) for year in years)
                if all(
                    g is not None and n is not None and p is not None
                    and g > n > 0 and p < 0 and g + p == n
                    for g, n, p in zip(amounts, nets, provisions, strict=True)
                ):
                    matches[(anchor.document_id, anchor.report_type, amounts)] = gross
    if len(matches) == 1:
        return {**results, GROSS_LOANS_ID: [next(iter(matches.values()))]}
    return results


def reconcile_credit_groups(
    results: dict[str, list[RetrievalResult]], years: list[int],
) -> dict[str, list[RetrievalResult]]:
    """Keep a classification table only if its components match gross loans.

    Identical note labels also occur for bonds and purchased debt. Neither
    inferred schema IDs nor the note labels alone establish customer loans.
    """
    output = dict(results)
    tables: dict[tuple[str, str, str, str], dict[str, list[RetrievalResult]]] = {}
    for metric_id in (*CREDIT_GROUP_IDS, MARGIN_LOANS_ID):
        for result in results.get(metric_id, []):
            chunk = result.chunk
            table_id = str(chunk.metadata.get("table_id") or "")
            unit = NumericValidator._normalize_unit(chunk.unit or "")
            if not table_id or not unit:
                continue
            key = (chunk.document_id or chunk.source_file, chunk.report_type or "", table_id, unit)
            tables.setdefault(key, {}).setdefault(metric_id, []).append(result)

    matches: dict[tuple, tuple[dict[str, RetrievalResult], RetrievalResult]] = {}
    for (document, basis, _table, unit), groups in tables.items():
        if set(CREDIT_GROUP_IDS) - groups.keys():
            continue
        selected = {}
        for metric_id, rows in groups.items():
            signatures = {
                tuple(_source_amount(row.chunk, year) for year in years) for row in rows
            }
            if len(signatures) != 1:
                break
            amounts = next(iter(signatures))
            if any(value is None or value < 0 for value in amounts):
                break
            selected[metric_id] = max(rows, key=lambda row: row.rrf_score)
        if set(CREDIT_GROUP_IDS) - selected.keys():
            continue
        totals = tuple(
            sum((cast(Decimal, _source_amount(row.chunk, year)) for row in selected.values()), Decimal(0))
            for year in years
        )
        for gross in results.get(GROSS_LOANS_ID, []):
            chunk = gross.chunk
            if (
                (chunk.document_id or chunk.source_file) != document
                or (chunk.report_type or "") != basis
                or NumericValidator._normalize_unit(chunk.unit or "") != unit
                or tuple(_source_amount(chunk, year) for year in years) != totals
                or any(value <= 0 for value in totals)
            ):
                continue
            signature = (document, basis, totals, tuple(
                tuple(_source_amount(selected[metric_id].chunk, year) for year in years)
                for metric_id in sorted(selected)
            ))
            matches[signature] = (selected, gross)

    # Conflicting filings/tables are not silently resolved by retrieval score.
    for metric_id in (*CREDIT_GROUP_IDS, MARGIN_LOANS_ID):
        output[metric_id] = []
    if len(matches) == 1:
        selected, gross = next(iter(matches.values()))
        output.update({metric_id: [row] for metric_id, row in selected.items()})
        output[GROSS_LOANS_ID] = [gross]
    return output
