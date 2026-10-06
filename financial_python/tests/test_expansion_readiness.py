from evaluation.audit_expansion_readiness import audit


def case():
    return {"id": "holdout-test", "split": "holdout", "category": "period_comparison",
            "review_status": "verified", "reviewer_id": "reviewer", "author_id": "author",
            "reviewed_at": "2026-10-02T08:00:00+07:00", "query": "Tài sản MBB 2025",
            "answerable": True, "expected_scope": {"company_codes": ["MBB"], "fiscal_years": [2025]},
            "expected_facts": [{"label": "Tài sản", "year": 2025, "value": "100", "unit": "Triệu đồng",
                                "sources": [{"file_name": "MBB.txt", "page": 1, "source_sha256": "source1"}]}],
            "evidence_groups": [{"id": "assets", "sources": [{"file_name": "MBB.txt", "page": 1,
                                                                  "source_sha256": "source1"}]}]}


def manifest():
    return {"source_snapshot_sha256": "snapshot", "documents": [{"source_file": "MBB.txt",
                                                                  "source_sha256": "source1"}]}


def test_review_audit_requires_independence_and_immutable_source_binding():
    source = case()
    assert audit([source], manifest())["production_accuracy_gate_passed"]
    source["author_id"] = "reviewer"
    assert not audit([source], manifest())["production_accuracy_gate_passed"]
    source = case()
    source["expected_facts"][0]["sources"][0].pop("source_sha256")
    assert "reviewed_source_hash" in " ".join(audit([source], manifest())["reasons"])
    source = case()
    source["reviewer_id"] = ""
    assert not audit([source], manifest())["production_accuracy_gate_passed"]


def test_pending_case_never_becomes_gold_in_the_review_pack():
    source = case()
    source["review_status"] = "pending"
    result = audit([source], manifest())
    assert not result["production_accuracy_gate_passed"] and result["verified_count"] == 0
    assert source["review_status"] == result["review_pack"][0]["case"]["review_status"] == "pending"
