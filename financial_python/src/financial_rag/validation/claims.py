"""Expose mechanical claim lineage and enforce extractive support for free-form prose.

This is not a calibrated entailment model. Unsupported paraphrases remain
unverified rather than being promoted by lexical similarity or valid numbers.
"""
import re

from ..query.directives import normalize_directive_text
from ..schemas import FinancialChunk


def check_claims(
    answer: str,
    citations: list,
    chunks: dict[str, FinancialChunk],
    *,
    deterministic: bool,
) -> list[dict]:
    by_citation = {citation.citation_id: chunks.get(citation.chunk_id) for citation in citations}
    checks = []
    for line in answer.splitlines():
        ids = list(dict.fromkeys(re.findall(r"\[(c\d+)\]", line, flags=re.I)))
        if not ids:
            continue
        claim = re.sub(r"\[(c\d+)\]", "", line, flags=re.I).strip(" -*|\"“”:. ")
        plain = normalize_directive_text(claim)
        texts = [
            normalize_directive_text(chunk.content)
            for key in ids
            if (chunk := by_citation.get(key)) is not None
        ]
        support = bool(plain and any(plain in text for text in texts))
        status = "executor_checked" if deterministic else "source_extract" if support else "unverified"
        checks.append({"text": claim, "citation_ids": ids, "status": status,
                       "method": "deterministic_lineage" if deterministic else "exact_source_extract",
                       "human_reviewed": False})
    # Causal/normative prose without citations must never escape extraction checks.
    if not deterministic:
        for line in answer.splitlines():
            plain = normalize_directive_text(line)
            if len(plain.split()) >= 5 and not line.lstrip().startswith(("#", "| ---")):
                if not re.search(r"\[c\d+\]", line, flags=re.I) and not any(
                        term in plain for term in ("khong du", "chua du", "chua co", "khong the", "chua the", "khong tim thay")):
                    checks.append({"text": line, "citation_ids": [], "status": "unverified",
                                   "method": "uncited_prose", "human_reviewed": False})
    return checks
