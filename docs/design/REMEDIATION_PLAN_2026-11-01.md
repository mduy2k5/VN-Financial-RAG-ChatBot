# Kế hoạch sửa chữa và hoàn thiện — 01/11/2026

## Mục tiêu nghiệm thu

Người dùng nạp báo cáo TXT/Markdown hoặc JSON được hỗ trợ, hỏi chỉ tiêu/phân tích trong phạm vi đã công bố và nhận dữ liệu đúng doanh nghiệp, kỳ, cơ sở báo cáo, chỉ tiêu, giá trị, đơn vị; mỗi kết quả truy ngược được về nguồn. Khi thiếu bằng chứng hoặc còn mơ hồ, hệ thống trả trạng thái rõ ràng thay vì chọn số gần giống.

Các ngưỡng dưới đây là đề xuất kỹ thuật ban đầu; cần cố định trước khi chạy bộ nghiệm thu cuối và không hạ ngưỡng để phù hợp kết quả.

Phạm vi bắt buộc: TXT/Markdown, hợp đồng JSON đã công bố, tra cứu có nguồn, các phép tính đang hỗ trợ, đầu ra dữ liệu có cấu trúc và vận hành cấu hình hiện tại. PDF/OCR trực tiếp và phân quyền nhiều người là các phần mở rộng theo nhu cầu triển khai.

## Cơ sở hiện tại

- Sau mốc A: lần chạy toàn bộ gần nhất đạt 265 kiểm thử; sau đó bổ sung một hồi quy đối xứng FVTPL và chạy nhóm kiểm tra tập trung: 4 đạt. Ruff và mypy đạt theo cấu hình hiện có.
- Chẩn đoán sau sửa trên 31 câu / 9 báo cáo TXT bằng BM25, không gọi LLM: 31/31 ca đạt theo nhãn draft/pending; citation precision 98,39%. Trước sửa đạt 26/31. Kết quả sau sửa là hồi quy kỹ thuật, không phải xác nhận độc lập về nghiệp vụ.
- Nhãn chẩn đoán vẫn draft/pending; không coi 26/31 là độ chính xác production. Chi tiết chỉ lưu ở `tmp/PROJECT_ASSESSMENT_2026-11-01.md` và `tmp/project_assessment_2026-11-01.json` theo chính sách dữ liệu.
- CI GitHub Actions cho pytest/Ruff/mypy đã được cấu hình; chưa có kết quả chạy trên hosted runner.
- Chưa có holdout được rà độc lập; chưa xác minh triển khai thật hoặc tải trong lần đánh giá này.

## Thứ tự thực hiện

| Phần | Ưu tiên | Kết quả cần giao | Phụ thuộc |
| --- | --- | --- | --- |
| 0. Cố định bằng chứng và hồi quy | P0 | Bộ tái hiện, fixture tổng hợp, baseline | Không |
| 1. Sửa kiểm định số và mã dòng | P0 | Không từ chối nhầm số hợp lệ; vẫn chặn số sai | 0 |
| 2. Sửa hiểu câu hỏi và alias | P0 | Phân biệt đúng tài sản, thu nhập, lãi/lỗ | 0 |
| 3. Chọn đúng dòng và kiểm định fact | P0 | Không trả dòng tổng thay dòng con | 1, 2 |
| 4. Hoàn thiện đọc và chuẩn hóa | P1 | Coverage ingestion rõ ràng, JSON đúng hợp đồng | 3 |
| 5. Phép tính và đầu ra dữ liệu | P1 | Tính toán có lineage, bảng và CSV | 3; dùng dữ liệu mới sau 4 |
| 6. Kiểm chứng độc lập và CI | P1, bắt đầu từ 0 | Gold/holdout, CI và biên bản nghiệm thu | Bắt đầu cùng 0; kết thúc sau 5 |
| 7. Kiểm chứng triển khai và tải | P1 trước sử dụng rộng | E2E trên stack thật, báo cáo hiệu năng và khôi phục | 3; nghiệm thu cuối sau 6 |

Mỗi phần là một nhóm thay đổi có thể review riêng. Mỗi PR mô tả lỗi/trigger, hành vi mới, bằng chứng kiểm thử và ảnh hưởng tới dữ liệu cũ. Khi thay đổi parser/projection phải chỉ rõ có cần lập chỉ mục lại hay không.

## Phần 0 — Cố định bằng chứng và bộ hồi quy

**Công việc**

1. Lưu commit, cấu hình BM25/dense/LLM, revision parser/schema, hash corpus và nhãn cho mỗi lần chạy.
2. Giữ 31 ca chẩn đoán thành tập dev cục bộ có thể chạy lại; không đưa chúng vào holdout.
3. Tạo fixture hư cấu tối thiểu tái hiện ba nhóm lỗi; bổ sung các biến thể gần giống để tránh sửa riêng tên doanh nghiệp hoặc một câu hỏi.
4. Lưu tách biệt kết quả hiểu câu hỏi, ứng viên retrieval, fact đã chọn, validation, câu trả lời và citation. Phân loại lỗi theo từng tầng.
5. Bắt đầu xây dựng bộ nghiệm thu độc lập và CI ở phần 6.

**Vùng sửa:** `financial_python/tests/`, `financial_python/evaluation/`, công cụ benchmark hiện có.

**Nghiệm thu:** tái hiện đủ 5 ca lỗi và giữ kết quả baseline; chạy offline được bộ fixture công khai; không commit báo cáo thật hoặc nhãn private. Các ca hồi quy chỉ được đánh dấu sửa xong sau khi đổi mã nguồn tương ứng.

## Phần 1 — Sửa validator cho mã dòng và công thức nhãn

**Công việc**

1. Phân biệt giá trị tài chính với số thứ tự, mã dòng, mã thuyết minh, ngày và biểu thức nằm trong tên chỉ tiêu.
2. Ưu tiên kiểm định fact có cấu trúc; với văn bản, chỉ loại bỏ identifier trong ngữ cảnh đã nhận diện, không bỏ rộng mọi số trong ngoặc hoặc đầu dòng.
3. Giữ nguyên quy tắc kiểm tra dấu âm, số trong ngoặc, số 0, thập phân và đơn vị.
4. Trả lý do từ chối có cấu trúc để phân biệt thiếu nguồn, sai số, sai đơn vị và nhầm identifier.

**Vùng sửa:** `validation/numeric.py`, chỗ truyền dữ liệu kiểm định trong `api/app.py`, `tests/test_validation.py`, `tests/test_api.py`.

**Nghiệm thu:** 3 ca từ chối nhầm đã xác định trả đúng số và nguồn; trường hợp số tiền sai, đơn vị sai, số âm hoặc tỷ lệ phần trăm vẫn bị phát hiện. Kiểm tra cả câu hỏi có/không có mã dòng, bảng và câu văn. Không giảm độ chặt bằng cách tắt validator.

## Phần 2 — Sửa hiểu câu hỏi và phân giải alias

**Công việc**

1. Phân biệt tài sản FVTPL với lãi/lỗ phát sinh từ FVTPL; ưu tiên cụm khái niệm đầy đủ và ngữ cảnh bảng.
2. Phân giải alias theo ngành, loại báo cáo và chỉ tiêu cụ thể; khi các tín hiệu xung đột, giữ trạng thái mơ hồ hoặc yêu cầu làm rõ.
3. Không áp dụng bộ lọc sai loại báo cáo chỉ vì gặp một từ trong tên tài sản.
4. Ghi lại alias và lý do chọn metric để chẩn đoán; chỉ thay đổi schema/alias nếu tìm thấy lỗi ở chính dữ liệu đó.

**Vùng sửa:** `query/understanding.py`, `metadata/metric_mapper.py`, `structure/report_schema_catalog.py`, alias liên quan, `tests/test_query.py`.

**Nghiệm thu:** câu hỏi tài sản FVTPL chọn đúng chỉ tiêu thuộc bảng cân đối; câu hỏi lãi/lỗ FVTPL chọn đúng chỉ tiêu thuộc kết quả kinh doanh. Ca thiếu ngữ cảnh được xử lý rõ ràng. Các query về doanh nghiệp/kỳ/cơ sở báo cáo đang đúng tiếp tục đúng.

## Phần 3 — Chọn đúng dòng và kiểm định ý nghĩa của số

**Công việc**

1. So khớp theo độ cụ thể của nhãn gốc, node/mã dòng đáng tin cậy và quan hệ cha/con. Tránh coi một nhãn ngắn nằm trong câu hỏi là đủ để thắng nhãn dài đúng yêu cầu.
2. Tách lựa chọn doanh nghiệp/kỳ/cơ sở báo cáo khỏi lựa chọn chỉ tiêu; không âm thầm nới scope khi thiếu ứng viên.
3. Tạo hoặc tái sử dụng cấu trúc fact liên kết chỉ tiêu, giá trị, đơn vị, kỳ và tọa độ nguồn. Kiểm định giá trị thuộc đúng dòng được hỏi, không chỉ có xuất hiện trong một chunk hoặc cùng trang.
4. Citation theo fact/dòng; với nguồn cũ thiếu tọa độ, giữ trạng thái unlocated và dùng chính sách hỗ trợ đã công bố, không tự tạo tọa độ.
5. Phân biệt trạng thái complete/partial/insufficient_evidence/needs_clarification nhất quán với bằng chứng đã chọn.

**Vùng sửa:** `api/app.py::_select_preferred_exact_source_rows`, `structure/source_rows.py`, `structure/source_groups.py`, `structure/normalized.py`, retrieval/generation/validation liên quan.

**Nghiệm thu:** ca dòng con đang sai được trả đúng; hỏi dòng tổng vẫn trả đúng tổng. Có kiểm tra đối kháng khi dòng tổng và dòng con cùng trang, khi cùng giá trị xuất hiện ở nhiều kỳ/chỉ tiêu, khi alias trùng và khi thiếu đúng dòng. Chạy lại 31 ca; mục tiêu đạt toàn bộ sau khi xác nhận nhãn hợp lệ, không sửa nhãn chỉ để khớp output. Mọi thay đổi nhãn phải có bằng chứng nguồn.

## Phần 4 — Hoàn thiện việc đọc và chuẩn hóa dữ liệu

**Công việc**

1. Kiểm tra toàn bộ payload JSON theo version hợp đồng trước khi xuất bản tài liệu. Báo lỗi trường/đường dẫn cụ thể; xử lý version không hỗ trợ rõ ràng.
2. Xác định ngữ nghĩa rồi bổ sung adapter cho `observations` và `notes.rows`. Với `raw_blocks`, bảo toàn bằng chứng nhưng chỉ tạo fact khi đủ nhãn/kỳ/đơn vị; tránh tạo fact trùng với statement items.
3. Theo dõi số bảng/dòng đã đọc, đã chuẩn hóa, chưa map, bị loại và lý do. Ingestion thành công phải kèm coverage và cảnh báo phù hợp.
4. Bổ sung hỗ trợ có kiểm thử cho header nhiều tầng, ô trống, số âm, OCR lỗi, bảng nhiều kỳ và nhóm cha/con. Phân biệt lỗi parser với dữ liệu nguồn thiếu.
5. Áp dụng quality gate nhất quán trên mọi đường trả lời; tài liệu không đủ điều kiện không được dùng để sinh fact hợp lệ.
6. Version hóa parser/projection và lập kế hoạch backfill/reindex có thể chạy lại; giữ lịch sử và khả năng trở về projection trước.

**Vùng sửa:** `structure/production_report_adapter.py`, `schemas/production_report.schema.json`, `parsing/`, `chunking/`, `application/ingestion.py`, lưu trữ normalized.

**Nghiệm thu:** payload sai hợp đồng bị từ chối rõ ràng; phần dữ liệu chưa hỗ trợ không bị bỏ âm thầm. Fixture statements/notes/observations truy ngược được về nguồn, không sinh fact trùng. Ingest lại idempotent; migration/backfill không làm mất bản cũ.

## Phần 5 — Phép tính và đầu ra phục vụ lấy dữ liệu

**Công việc**

1. Dùng fact đã kiểm định làm đầu vào; tái sử dụng catalog công thức và lineage đang có.
2. Bổ sung thực thi công thức từ schema bằng tập toán tử được cho phép và kiểm tra phụ thuộc; không dùng eval với nội dung schema. Từ chối công thức không hỗ trợ hoặc phụ thuộc vòng.
3. Kiểm tra tương thích đơn vị, kỳ, cơ sở; tránh cộng đồng thời subtotal và dòng con. Thiếu một toán hạng phải trả thiếu dữ liệu.
4. Chuẩn hóa đầu ra dạng hàng: chỉ tiêu, doanh nghiệp, kỳ, giá trị gốc/chuẩn hóa, đơn vị, nguồn và trạng thái. UI dựng bảng và mở bằng chứng từ cùng kết quả API.
5. Bổ sung xuất CSV từ cùng tập hàng. XLSX là phần tiếp theo nếu CSV chưa đủ nhu cầu; không tạo đường tính số riêng trong UI/export. Xử lý an toàn chuỗi có thể bị ứng dụng bảng tính hiểu thành công thức.

**Vùng sửa:** `analysis/formula_catalog.py`, `analysis/evidence.py`, engine công thức schema, `schemas.py`, endpoint answer/facts, `financial_ui/`.

**Nghiệm thu:** kiểm tra lại phép tính độc lập; mỗi toán hạng có nguồn. Bảng và CSV khớp API về giá trị, đơn vị, kỳ và nguồn; giữ phân biệt 0 với thiếu dữ liệu. Không cộng trùng thành phần hoặc tự tính từ số suy đoán.

## Phần 6 — Kiểm chứng độc lập và tự động hóa

**Công việc bắt đầu ngay từ phần 0**

1. Chuẩn bị khoảng 180 câu hỏi thật: 120 dev, 60 holdout; phân bố theo ngành, kỳ, loại báo cáo, bảng/thuyết minh và độ khó OCR. Phân tách tài liệu/doanh nghiệp theo protocol đã công bố để hạn chế rò rỉ.
2. Người rà độc lập đối chiếu nhãn với nguồn, ký nhận người rà/thời điểm. Không coi việc agent tự rà lại nhãn của mình là kiểm chứng độc lập.
3. Bao gồm tra cứu, so sánh, cơ cấu, công thức, diễn giải, thiếu dữ liệu và mơ hồ. Nhãn cơ cấu phải đủ thành phần; nhãn công thức phải đủ toán hạng.
4. Chạy CI pytest/Ruff/mypy và benchmark hư cấu; chạy benchmark private trong môi trường có quyền truy cập corpus, giữ dữ liệu thật khỏi log/artifact công khai.
5. Đóng băng commit, model/revision, schema, corpus và nhãn khi nghiệm thu. Mỗi lần dùng holdout để điều chỉnh hệ thống phải coi các ca đã lộ là dev và chuẩn bị tập holdout mới cho vòng đánh giá sau.

**Ngưỡng đề xuất để chốt trước nghiệm thu**

| Chỉ số | Ngưỡng ban đầu |
| --- | --- |
| Sai doanh nghiệp/kỳ/cơ sở hoặc sai dòng trọng yếu | 0 lỗi quan sát được trong bộ nghiệm thu |
| Ca trả đúng đầy đủ, gồm số/đơn vị/kỳ/chỉ tiêu | ≥95%, báo cáo riêng từng nhóm |
| Citation precision ở cấp fact | ≥98% |
| Độ đầy đủ thành phần ở câu hỏi cơ cấu | ≥95% |
| Phép tính trên bộ toán hạng được xác nhận | 100% đúng kết quả/lineage theo sai số làm tròn công bố |
| Từ chối đúng khi thiếu bằng chứng | Precision ≥95%, recall ≥90% |

Với tập holdout nhỏ phải báo cả số ca và bất định thống kê, không chỉ phần trăm. Diễn giải tự do cần rubric và người rà; không suy từ số khớp thành diễn giải đúng. Nếu chưa có người rà độc lập, trạng thái phần này là chưa nghiệm thu, dù kiểm tra tự động đạt.

## Phần 7 — Kiểm chứng triển khai và tải

**Công việc**

1. Chạy E2E trên kho kiểm thử riêng với cấu hình Docker/MongoDB/Qdrant hiện tại: ingest → query → citation → restart → query lại; batch/job/retry; đổi projection và xóa nguồn.
2. Kiểm tra BM25, dense ready và dense lỗi/tắt; ghi rõ chế độ retrieval thực sự được dùng, không gắn nhãn hybrid khi chỉ chạy BM25.
3. Kiểm tra nguồn mới được hỏi ngay sau publish; yêu cầu đang chạy giữ snapshot nhất quán; không dùng dữ liệu chưa publish hoặc đã bị loại.
4. Đo ingestion, cold start, p50/p95, lỗi/timeout, RAM/CPU và tải đồng thời theo các quy mô corpus. Chốt ngân sách hiệu năng sau baseline và trước vòng nghiệm thu hiệu năng.
5. Thử backup/restore trên kho kiểm thử; xác nhận tài liệu, version/projection, notebook và khả năng dựng lại index.

**Nghiệm thu:** E2E đạt trên stack thật, không mất dữ liệu sau restart/restore, không trộn scope/snapshot khi cập nhật; tải đo nằm trong ngân sách đã chốt. Không thực hiện thử nghiệm phá hủy trên kho vận hành.

## Các mốc bàn giao

- **Mốc A — sửa độ tin cậy của tra cứu:** hoàn thành 0–3 và CI nền; ba nhóm lỗi có hồi quy, tập dev được chẩn đoán lại. Chưa tuyên bố chất lượng production.
- **Mốc B — đủ luồng lấy dữ liệu:** hoàn thành 4–5; hợp đồng đầu vào, coverage, phép tính và bảng/CSV được kiểm tra từ đầu đến cuối.
- **Mốc C — nghiệm thu sử dụng:** hoàn thành 6–7; holdout được duyệt độc lập, đạt ngưỡng chất lượng và ngân sách vận hành.

Chỉ đánh dấu hoàn tất từng phần khi có kết quả kiểm tra tương ứng. Thời lượng phụ thuộc khối lượng dữ liệu cần rà, tính đa dạng của bảng và việc huy động người rà độc lập; ước lượng lại sau phần 0 và khi kết thúc từng mốc.

## Tiến độ cập nhật 01/11/2026

**Mốc A hoàn tất.** Validator lọc mã dòng/công thức trong nhãn trước khi kiểm tra số; alias TSTC/FVTPL được phân giải về tài sản tài chính; API phục hồi và ưu tiên dòng con đúng nhãn trong kho ứng viên theo cùng doanh nghiệp, kỳ, cơ sở và báo cáo. Đã thêm hồi quy cho ba nhóm lỗi và workflow CI. Kiểm chứng cục bộ: lần chạy toàn bộ gần nhất đạt 265 pytest; sau đó bổ sung kiểm tra đối xứng FVTPL lãi/lỗ và 4 hồi quy tập trung đều đạt; Ruff và mypy đạt. Tập chẩn đoán tăng từ 26/31 lên 31/31. Nhãn vẫn là draft/pending, chưa có holdout độc lập; workflow chưa được GitHub hosted runner thực thi.

**Tiếp theo:** mốc B (coverage ingestion/JSON, phép tính và đầu ra bảng/CSV), sau đó mốc C (gold/holdout độc lập và kiểm chứng triển khai/tải). Không xem kết quả mốc A là nghiệm thu production.

## Phần mở rộng theo nhu cầu

- PDF/OCR trực tiếp: đánh giá extraction riêng, có coverage trang/bảng và tập kiểm tra OCR trước khi nối vào cùng hợp đồng fact.
- Nhiều người dùng: xác thực, phân quyền tài liệu/notebook, cô lập hội thoại và kiểm thử truy cập chéo trước khi mở dịch vụ dùng chung.
- XLSX và UI nâng cao: thực hiện sau khi bảng/CSV và dữ liệu nguồn đã ổn định.
