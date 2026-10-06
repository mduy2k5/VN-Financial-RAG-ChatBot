# Giải pháp hỏi đáp theo schema báo cáo tài chính

## Mục tiêu

Cho phép hỏi một chỉ tiêu đơn lẻ, xem các dòng cấu thành một nhóm, tổng hợp các chỉ số chính, xếp hạng và xem tỷ trọng mà không phụ thuộc vào việc BM25 tình cờ lấy đủ các dòng trong bảng. Số liệu phải truy ngược được về tài liệu, trang, kỳ và đơn vị.

## Vai trò của bộ schema

- `financial_position_schema`, `income_statement_schema`, `cash_flow_schema` là ontology: mã node ổn định, tên chuẩn, quan hệ cha/con, nhóm ngành và công thức được khai báo.
- Các file `*_aliases` và overlay bổ sung cách viết theo biểu mẫu, ngành và phiên bản. Khi map dòng dữ liệu, ưu tiên nhãn chính xác; alias trùng chỉ được phân giải khi có ngữ cảnh nhóm. Không dùng số thứ tự hiển thị làm mã ngữ nghĩa.
- `notes_aliases_{bank,enterprise,insurance,securities}` cung cấp khung mục thuyết minh và alias cho các dòng con.
- `production_report.schema.json` mô tả hợp đồng dữ liệu chuẩn hóa nên được tạo sau khi đọc báo cáo: giá trị hiện tại/kỳ trước, node cha/con, dữ liệu chưa map và provenance.

## Luồng trả lời

1. **Nạp và giữ cấu trúc bảng**: tách tên dòng, các cột kỳ, đơn vị, tiêu đề nhóm, mã số, tài liệu và trang. Giữ cả dòng chưa map thay vì loại bỏ.
2. **Map ngữ nghĩa**: ghép nhãn với alias theo schema mới nhất. Lưu `schema_node_id`, node cha, node ngữ cảnh, phiên bản schema, giá trị từng kỳ và trạng thái/confidence mapping cạnh chunk.
3. **Hiểu thao tác câu hỏi**: phân biệt tra cứu, liệt kê thành phần, cơ cấu/tỷ trọng, xếp hạng, so sánh và tổng hợp. Nhận diện công ty trước, rồi chọn báo cáo đúng năm; chỉ dùng cột so sánh khi không có báo cáo đúng năm.
4. **Thực thi trên toàn bộ dòng trong báo cáo đã chọn**: câu hỏi về thành phần không dùng top-k. Đi theo node cha/con, chỉ hiện các dòng có số liệu và tránh cộng đồng thời một subtotal với các dòng con.
5. **Tính toán có kiểm soát**: tỷ trọng chỉ tính khi có tổng mẫu số và đơn vị khớp. Tổng chỉ được lấy từ dòng tổng đã báo cáo hoặc công thức được schema khai báo; không cộng tùy tiện các dòng có thể chồng lấp.
6. **Trả lời có bằng chứng**: mỗi dòng kết quả có citation riêng; số, đơn vị, kỳ và nguồn được kiểm tra. LLM chỉ diễn giải phần ngôn ngữ, không tự sinh con số.

## Phần đã nối vào Financial RAG

- Đọc schema và alias theo chuỗi overlay mới nhất cho bảng cân đối, kết quả kinh doanh, lưu chuyển tiền tệ; đọc alias note theo ngành.
- Nạp thêm schema và alias riêng cho các chỉ tiêu ngoài bảng của ngân hàng.
- Khi ingest TXT/Markdown, gắn node chuẩn và ngữ cảnh nhóm vào các financial-item chunk đã trích xuất.
- Nhận trực tiếp JSON theo hợp đồng `production_report.schema.json` v2: dùng `mapping.target_id`, giá trị hai kỳ, đơn vị và provenance để tạo cùng loại fact chunk. Báo cáo bị đánh dấu `eligible_for_use=false` không tham gia trả lời cấu trúc.
- Nhận diện câu hỏi “cơ cấu”, “thành phần”, “gồm những gì”, “liệt kê”, “xếp hạng”, “tổng hợp”. Với câu hỏi theo schema, lấy tất cả các dòng phù hợp trong tài liệu của đúng công ty/kỳ, thay vì chỉ lấy một tập top-k.
- Trả lời danh sách thành phần và tổng được báo cáo; có thể tính tỷ trọng trên tổng tương ứng khi đơn vị và số liệu mẫu số phù hợp. Dòng bị mơ hồ hoặc không map được không được đoán.
- Citation được tạo cho toàn bộ các dòng đã chọn mà không cắt theo ngân sách context của LLM. Schema được mount vào container từ thư mục gốc `schemas/`; thay đổi schema/alias được nạp lại khi khởi động service.

## Phạm vi còn lại để đạt “mọi bảng dữ liệu”

Hiện service vẫn nhận TXT/Markdown và JSON theo hợp đồng `production_report` v2. JSON adapter hiện lấy `statements.*.items`; nó chưa tiêu thụ `observations`, `notes.rows`, `raw_blocks` và chưa xác thực toàn bộ payload bằng JSON Schema. Formula engine cũng chưa thực thi các công thức khai báo trong schema. Để hỗ trợ đầy đủ bảng thuyết minh và bảng tùy biến, bước tiếp theo là:

1. Hoàn thiện việc đọc `observations`/`notes.rows` và kiểm tra đầy đủ hợp đồng JSON trước khi lập chỉ mục.
2. Bổ sung formula engine đọc công thức từ schema, ghi lại phép tính và trích dẫn cho từng đầu vào.
3. Hiển thị kết quả dạng bảng có thể mở rộng theo nhóm, kỳ, đơn vị và nguồn; cho phép mở rộng các dòng chưa map.
4. Đánh giá trên báo cáo ngân hàng, doanh nghiệp, chứng khoán và bảo hiểm, gồm alias trùng, subtotal/con, nhiều kỳ và dòng chưa ánh xạ.

Phạm vi tài liệu đầu vào của MVP vẫn là TXT/Markdown; PDF không nằm trong phạm vi.
