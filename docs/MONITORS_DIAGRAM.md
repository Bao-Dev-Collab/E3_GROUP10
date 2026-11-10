# Sơ đồ quy trình và giám sát

> Ghi chú: tên file lấy theo project mẫu (tên gốc bị cắt: `MONITORS_DIAGRAM_...`). Nhóm hiểu đây là sơ đồ quy trình và các điểm giám sát (monitor) của bài, nên trình bày 3 sơ đồ dưới đây. Các sơ đồ viết bằng Mermaid, VS Code (cài extension Markdown Preview Mermaid) và GitHub hiển thị được thành hình.

## 1. Toàn bộ quy trình

```mermaid
flowchart TD
    A["IMDb trên Hugging Face<br/>stanfordnlp/imdb"] -->|tải parquet, đổi sang CSV| B["data/raw/<br/>train 25.000 · test 25.000"]
    B -->|"--stage data<br/>chia TRƯỚC mọi bước xử lý"| C["data/splits/<br/>train 22.297 · validation 2.482 · test 25.000"]
    C -->|train + validation| D["--stage baseline<br/>TF-IDF + LogisticRegression<br/>5-fold CV trên train"]
    C -->|train + validation| E["--stage train<br/>fine-tune DistilBERT<br/>chọn checkpoint theo validation"]
    D --> F["outputs/baseline/"]
    E --> G["outputs/final_model/"]
    C -->|test, chỉ 1 lần| H["--stage evaluate<br/>metrics, confusion matrix,<br/>phân tích theo độ dài, McNemar"]
    F --> H
    G --> H
    G --> I["03_predict_finetuned.py<br/>dự đoán review mới"]
    J["Bài 1: 01_sentiment_analysis.py<br/>model SST-2 có sẵn"] -.độc lập.-> K["outputs/reports/exercise1_sentiment.json"]
```

## 2. Giám sát trong lúc fine-tune

```mermaid
flowchart LR
    S["Bắt đầu train<br/>lr tăng dần (warmup 10%)"] --> T["Train 697 bước<br/>(nửa epoch)<br/>log loss mỗi 50 bước"]
    T --> V["Đánh giá validation<br/>accuracy · F1 · loss"]
    V --> CK["Lưu checkpoint<br/>(giữ tốt nhất + mới nhất)"]
    CK --> Q{"Accuracy validation<br/>tăng không?"}
    Q -->|có| R["Đặt lại bộ đếm<br/>early stopping"]
    Q -->|không| P["Bộ đếm +1"]
    P --> X{"Bộ đếm = 2?"}
    X -->|có| STOP["Dừng sớm"]
    X -->|không| N
    R --> N{"Hết 2 epoch?"}
    N -->|chưa| T
    N -->|rồi| BEST
    STOP --> BEST["Nạp lại checkpoint tốt nhất<br/>theo validation"]
    BEST --> GAP["Chấm train và validation<br/>train − validation > 0,05 → cảnh báo overfitting"]
```

Trong lần train thật: đánh giá 4 lần (accuracy 0,8940 → 0,9049 → 0,9053 → 0,9073). Accuracy tăng ở mọi lần đánh giá nên early stopping không kích hoạt. Checkpoint tốt nhất là bước 2.788.

## 3. Các điểm kiểm soát chất lượng

| Điểm kiểm soát | Kiểm tra gì | Ở đâu |
|---|---|---|
| Rò rỉ dữ liệu | Review trùng giữa train / validation / test bị loại (123 review trùng với test) | `data.py` → `remove_duplicates` |
| Chia trước xử lý | Chia dữ liệu trước khi làm sạch, tokenize, thống kê | `data.py` → `prepare_data` |
| Fit chỉ trên train | TF-IDF nằm trong `Pipeline`, học lại ở từng fold | `baseline.py` |
| Tập test dùng 1 lần | Chỉ giai đoạn `evaluate` đọc `test.csv` | `02_train_finetune.py` |
| Overfitting | So accuracy train với validation, ngưỡng 0,05 | `evaluate.py` → `overfitting_gap` |
| Tái lập | Seed 42, SHA-256 của các file data | `config.py`, `data.md` |
| Thời gian / bộ nhớ | Chạy thử `--smoke` đo tốc độ trước khi train thật | `train.py` → `StepTimer` |
| Code đúng | 400+ test pytest (thường, biên, input sai) | `tests/` |
| Nhật ký | Mỗi bước ghi 5 giai đoạn Description → Gen → Smoke → Run → Check | `AUDIT_LOG.md` |
