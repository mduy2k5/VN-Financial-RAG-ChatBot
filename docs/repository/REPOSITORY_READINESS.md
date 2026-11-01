# Chính sách dữ liệu và phát hành

Cập nhật: 2026-10-06.

## Dữ liệu được đưa vào repository

Repository chứa mã nguồn, cấu hình, schema, kiểm thử, tài liệu và một fixture báo cáo tài chính hư cấu. `financial_python/sample_data/DEMO_SYNTHETIC_2025.md` cùng `financial_python/evaluation/demo_cases.jsonl` dùng tên doanh nghiệp, số liệu và trích dẫn được tạo để minh họa; chúng phục vụ kiểm tra parsing, retrieval, answer validation và benchmark ví dụ.

## Dữ liệu không được commit

Không commit báo cáo tài chính thật, kết quả OCR, dữ liệu khách hàng hoặc thông tin cá nhân, bản sao cơ sở dữ liệu vận hành, nhãn và kết quả benchmark phụ thuộc corpus thật, review pack, ảnh QA hay thông tin xác thực. Dùng dữ liệu hư cấu để tạo ví dụ công khai; kiểm tra nội dung staged trước mỗi commit và release.

Benchmark ví dụ chỉ dùng fixture hư cấu. Các thí nghiệm cần corpus thật nằm ngoài repository và không thuộc bộ kiểm thử mặc định.

## Kiểm tra trước khi phát hành

GitHub Actions chạy pytest, Ruff và mypy trên push/pull request. Benchmark tổng hợp, kiểm tra JavaScript/Docker Compose và secret scan vẫn cần chạy thủ công trước khi phát hành. Hãy rà soát diff, file được stage, nội dung mẫu và secret scan trước khi push.

## Lệnh kiểm tra

From `financial_python`, install the development extra and run:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy src
python -m financial_rag.evaluation.benchmark
```

Các kiểm tra mặc định dùng fixture hư cấu, không cần corpus riêng, MongoDB, Qdrant, Ollama hoặc GPU. Ngày 2026-10-06, lần xác minh gần nhất đạt 262 bài pytest, Ruff, mypy, kiểm tra cú pháp JavaScript, Docker Compose và Gitleaks trên lịch sử cùng các file được theo dõi. Kiểm tra tích hợp Docker được chạy riêng với bộ unit test.
