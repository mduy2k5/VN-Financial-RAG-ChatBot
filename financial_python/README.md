# Financial RAG Service

Dịch vụ hỏi đáp báo cáo tài chính tiếng Việt, gồm API FastAPI, giao diện web và các module parsing, retrieval, phân tích, citation validation.

## Ví dụ tổng hợp

[`sample_data/DEMO_SYNTHETIC_2025.md`](sample_data/DEMO_SYNTHETIC_2025.md) là báo cáo ngắn hoàn toàn hư cấu, dùng để thử định dạng Markdown, trích xuất kỳ/chỉ tiêu và citation. Benchmark mặc định dùng hai ca giả lập trong `evaluation/demo_cases.jsonl`.

## Cài đặt và kiểm tra

Dùng Python 3.11 trở lên. Chạy từ thư mục này:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy src
python -m financial_rag.evaluation.benchmark
```

Các lệnh mặc định không cần dữ liệu ngoài repo, cơ sở dữ liệu, Ollama hoặc GPU.

## Chạy API cục bộ

```powershell
uvicorn financial_rag.api.app:app --host 127.0.0.1 --port 8092
```

Mở `http://127.0.0.1:8092/docs` để xem API. Có thể bật nạp file tổng hợp vào bộ nhớ khi chạy demo:

```powershell
$env:FINANCIAL_LOAD_SAMPLE_DATA = "true"
uvicorn financial_rag.api.app:app --host 127.0.0.1 --port 8092
```

Mặc định service khởi động với kho rỗng. Ingest file demo qua UI/API hoặc bật biến môi trường nêu trên.

## Dữ liệu benchmark cục bộ

Báo cáo thật, nhãn benchmark, nguồn OCR, phiếu rà soát, ảnh QA và kết quả chi tiết không được theo dõi trong Git. Bản lưu local nếu có nằm trong thư mục bị ignore `local_data/archive/`. Những script corpus cũ cần đầu vào ở đường dẫn cụ thể; truyền nguồn/dataset tường minh hoặc cấu hình lại theo kho dữ liệu local trước khi chạy. CI và benchmark mặc định chỉ dùng fixture `DEMO`.

Xem [chính sách dữ liệu và trạng thái chuẩn bị GitHub](../docs/repository/REPOSITORY_READINESS.md).
