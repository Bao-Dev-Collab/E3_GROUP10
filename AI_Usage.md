# Tuyên bố sử dụng AI

Nhóm E3_GROUP10 có sử dụng trợ lý AI trong bài thực hành này. Nhóm ghi rõ AI đã làm gì, nhóm đã làm gì, và nhóm kiểm soát chất lượng ra sao.

## 1. Công cụ

- **Claude Code** (model Claude Opus 5.5 của Anthropic), chạy trong ứng dụng Claude trên máy của nhóm, có quyền đọc / ghi file trong thư mục project và chạy lệnh trên máy.
- Mọi lần train và đánh giá đều chạy **trên máy của nhóm** (Apple M1), không dùng dịch vụ train trên mạng.

## 2. Phân công

| Việc | AI | Nhóm |
|---|---|---|
| Hiểu đề, đối chiếu với bài giảng | Tóm tắt, chỉ ra chỗ code mẫu trong bài giảng không còn chạy được với thư viện mới | Đọc đề, xác nhận cách hiểu |
| Chọn dataset, model, cấu hình | Đề xuất IMDb và DistilBERT, đo tốc độ thật để đề xuất `max_length` | **Quyết định**: chọn IMDb, giới hạn train 2–3 tiếng, dùng toàn bộ dữ liệu, không lấy tập con |
| Cấu trúc project | Đề xuất dựa trên mẫu của giảng viên | **Quyết định** làm theo project mẫu, thêm các file md |
| Quy tắc khoa học | Áp dụng | **Đặt ra** bộ quy tắc trong `RULES.md` (test, chia dữ liệu trước, không chọn bằng test...) |
| Viết code và test | Viết toàn bộ code trong `src/` và `tests/` | Yêu cầu làm từng bước, **duyệt** sau mỗi bước |
| Chạy thí nghiệm | Chạy chuẩn bị dữ liệu, baseline, train, đánh giá | Theo dõi tiến độ, đóng ứng dụng khác khi máy thiếu RAM |
| Tìm và sửa lỗi | Tự rà sau mỗi bước, tìm nguyên nhân gốc bằng thí nghiệm | Yêu cầu "chỉ sửa đúng chỗ gây lỗi, giải thích nguyên nhân gốc, không vá tạm" |
| Tài liệu | Viết các file trong `docs/`, `REPORT.md`, notebook | Đọc, hỏi lại các chỗ chưa rõ, duyệt |

## 3. Nhóm kiểm soát chất lượng thế nào

1. **Làm từng bước, có duyệt:** 9 bước, mỗi bước xong đều dừng để nhóm kiểm tra rồi mới làm tiếp (xem [docs/plan-progress-status.md](docs/plan-progress-status.md)).
2. **Có test cho mọi phần code:** 440 test pytest, gồm trường hợp thường, biên và input sai. Toàn bộ đều pass trước khi bàn giao mỗi bước.
3. **Tự rà lỗi sau mỗi bước** và ghi lại trong [docs/AUDIT_LOG.md](docs/AUDIT_LOG.md). Các lỗi tìm ra (crash trên CPU do lệch căn hàng bộ nhớ, chia dữ liệu có thể làm mất một lớp, file CSV khác nhau giữa các hệ điều hành, cột thừa `token_type_ids`, cảnh báo early stopping sai...) đều được chứng minh bằng thí nghiệm trước khi sửa.
4. **Số liệu trong báo cáo đều lấy từ file kết quả thật** trong `outputs/reports/`, không có số tự đặt ra. Giả thuyết chưa kiểm chứng được đều ghi rõ là giả thuyết.

## 4. Lưu ý

- AI có thể sai. Nhóm chịu trách nhiệm cuối cùng về nội dung bài, và đã đọc, hiểu các phần code, kết quả chính trước khi nộp.
- Nhóm tuân theo quy định về sử dụng AI của môn học. Nếu giảng viên yêu cầu, nhóm có thể cung cấp thêm lịch sử làm việc với AI.
