import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from financial_rag.query.understanding import QueryUnderstandingEngine

def main():
    engine = QueryUnderstandingEngine()

    test_queries = [
        "Tổng nợ phải trả của ABB đầu năm 2025 là bao nhiêu?",
        "Doanh thu thuần của FPT năm 2024 tăng bao nhiêu % so với 2023?",
        "Lợi nhuận sau thuế của Vinamilk cuối năm 2024",
        "Vốn điều lệ của ngân hàng An Bình năm 2025",
        "Thuyết minh các khoản vay của ABB"
    ]

    print("=" * 80)
    print("KIỂM TRA LỚP QUERY UNDERSTANDING BÁO CÁO TÀI CHÍNH VIỆT NAM")
    print("=" * 80)

    for q in test_queries:
        res = engine.parse(q)
        print(f"\n[CÂU HỎI GỐC]: {q}")
        print(f"  ├── Mã cổ phiếu / Công ty : {res.company_code}")
        print(f"  ├── Năm tài chính         : {res.fiscal_year}")
        print(f"  ├── Kỳ báo cáo            : {res.period_type.value}")
        print(f"  ├── Báo cáo đích          : {res.statement_type.value if res.statement_type else 'N/A'}")
        print(f"  ├── Chỉ tiêu tài chính    : {res.primary_metric}")
        print(f"  ├── Ý định (Intent)       : {res.intent.value}")
        print(f"  ├── Metadata Filter       : {res.metadata_filter}")
        print(f"  ├── BM25 Query Rewrite    : {res.rewritten_query_bm25}")
        print(f"  └── Vector Query Rewrite  : {res.rewritten_query_vector}")

if __name__ == "__main__":
    main()
