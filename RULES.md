# Bộ quy tắc khoa học và chuẩn mực học thuật

Mọi phần code và thí nghiệm trong project này phải tuân theo các quy tắc dưới đây.

## 1. Quy tắc viết code

1. Có unit test (pytest) cho trường hợp bình thường, trường hợp biên và input sai (rỗng, `None`, giá trị âm, sai kiểu dữ liệu, file không tồn tại).
2. Xử lý lỗi rõ ràng: báo lỗi cụ thể kèm thông báo, không nuốt lỗi bằng `try/except` để trống.
3. Mỗi hàm có chú thích ghi rõ các giả định ở đầu hàm.
4. Code giải bài toán tổng quát: không hardcode kết quả của ví dụ, không viết `if` riêng cho từng test case, không giả định dữ liệu giống hệt ví dụ.
5. Ngoài các ví dụ có sẵn, tự nghĩ thêm ít nhất 5 test case khác (gồm trường hợp biên) và kiểm tra code qua chúng.
6. Chạy code và test trước khi bàn giao. Có lỗi thì sửa đến khi pass.

## 2. Quy tắc pipeline học máy

1. Chia train / validation / test **trước** mọi bước xử lý dữ liệu.
2. Chỉ fit scaler / encoder / imputer trên tập train, rồi mới transform validation và test (tránh rò rỉ dữ liệu). Dùng sklearn `Pipeline`.
3. Với dữ liệu theo thời gian: chia theo thời gian, không xáo trộn ngẫu nhiên.
4. Dùng cross-validation (k-fold, stratified nếu các lớp lệch nhau).
5. Không chọn siêu tham số bằng tập test.
6. Báo cáo cả điểm trên train **và** điểm trên validation/test, cảnh báo nếu chênh lệch lớn.
7. Thêm regularization / early stopping / dropout khi phù hợp.
8. Đặt random seed để tái lập kết quả.

## 3. Cách áp dụng vào bài này

| Quy tắc | Áp dụng |
|---|---|
| 2.1 Chia trước | IMDb được chia train / validation / test ngay sau khi tải, trước khi làm sạch, tokenize hay thống kê |
| 2.2 Fit trên train | TF-IDF của baseline nằm trong sklearn `Pipeline`, chỉ fit trên train (và trên từng fold). Tokenizer của DistilBERT đã được huấn luyện sẵn, không fit trên dữ liệu của nhóm. Dữ liệu là văn bản, không có giá trị thiếu nên không cần scaler hay imputer |
| 2.3 Dữ liệu thời gian | Không áp dụng: IMDb không có cột thời gian |
| 2.4 Cross-validation | Baseline dùng StratifiedKFold 5 fold. DistilBERT dùng một tập validation cố định (2.500 review), vì train 5 lần sẽ vượt xa giới hạn 2–3 tiếng |
| 2.5 Không chọn bằng test | Mọi lựa chọn (tham số baseline, `max_length`, epoch, checkpoint) dựa trên validation. Tập test chỉ chạy một lần ở cuối |
| 2.6 Train vs validation | Chấm điểm trên cả train và validation, cảnh báo khi chênh lệch vượt ngưỡng đặt trong config |
| 2.7 Regularization | `weight_decay`, dropout có sẵn trong DistilBERT, early stopping, gradient clipping. Baseline có L2 |
| 2.8 Seed | Seed cố định cho python, numpy, torch và cho bước chia dữ liệu |
