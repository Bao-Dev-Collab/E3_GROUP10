# Hướng dẫn chạy thử

Mọi lệnh chạy từ thư mục `lab3_DL/`. Thời gian dưới đây đo trên Apple M1, 8 GB RAM.

## 0. Cài đặt (một lần)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Lần chạy đầu cần internet để tải model từ Hugging Face (khoảng 270 MB mỗi model). Các lần sau dùng bản lưu sẵn trên máy.

## 1. Toàn bộ quy trình

| # | Lệnh | Thời gian | Tạo ra |
|---|---|---|---|
| 1 | `python src/01_sentiment_analysis.py` | khoảng 10 giây | `outputs/reports/exercise1_sentiment.json` |
| 2 | `python src/02_train_finetune.py --stage data` | khoảng 15 giây | `data/splits/*.csv`, `outputs/reports/data_report.json` |
| 3 | `python src/02_train_finetune.py --stage baseline` | khoảng 3 phút | `outputs/baseline/tfidf_logreg.joblib`, `outputs/reports/baseline_report.json` |
| 4 | `python src/02_train_finetune.py --stage train --smoke` | khoảng 1 phút | `outputs/reports/smoke_report.json` (ước tính thời gian train) |
| 5 | `caffeinate -i python src/02_train_finetune.py --stage train` | **khoảng 1 giờ 11 phút** | `outputs/final_model/`, `outputs/reports/train_report.json` |
| 6 | `python src/02_train_finetune.py --stage evaluate` | khoảng 10 phút | `outputs/reports/evaluation_report.json`, 2 ảnh PNG, `test_predictions.csv` |
| 7 | `python src/03_predict_finetuned.py` | khoảng 10 giây | In kết quả ra màn hình |

Bước 1 không cần các bước khác. Bước 2 → 7 phải chạy theo đúng thứ tự (bước sau dùng kết quả của bước trước). `caffeinate -i` giữ máy Mac không ngủ khi train; trên Windows / Linux thì bỏ đi.

## 2. Dự đoán review mới (`03_predict_finetuned.py`)

Script dùng model đã fine-tune trong `outputs/final_model/`, và tự đọc `max_length` (256) từ báo cáo train để cắt review giống hệt lúc train.

```bash
python src/03_predict_finetuned.py                                    # 5 câu mẫu có sẵn
python src/03_predict_finetuned.py --text "Great movie!" --text "So boring."
python src/03_predict_finetuned.py --file reviews.txt                 # file .txt, mỗi dòng một review
python src/03_predict_finetuned.py --interactive                      # gõ từng câu, dòng trống để thoát
python src/03_predict_finetuned.py --device cpu --output ket_qua.json # chạy CPU, lưu kết quả ra JSON
```

**Kết quả thật với 5 câu mẫu** (09/11/2026, model ở `outputs/final_model/`):

| # | Review | Dự đoán | Độ tin cậy | Nhận xét |
|---|---|---|---|---|
| 1 | One of the best films I have seen in years. The acting and the soundtrack are wonderful. | POSITIVE | 99,6% | Đúng |
| 2 | A complete waste of two hours. The plot makes no sense and the dialogue is painful. | NEGATIVE | 99,6% | Đúng |
| 3 | It's not a bad movie at all, I actually enjoyed it more than I expected. | POSITIVE | 99,2% | Đúng, hiểu được phủ định "not bad" |
| 4 | Oh great, another sequel nobody asked for. Just what we needed. | POSITIVE | **58,6%** | **Sai**: câu mỉa mai. Độ tin cậy thấp cho thấy model không chắc |
| 5 | The first half is slow and boring, but the ending is absolutely brilliant. | POSITIVE | 99,2% | Hợp lý: ý chính nằm sau "but" |

Chế độ hỏi–đáp (`--interactive`):

```text
Gõ một review tiếng Anh rồi nhấn Enter (dòng trống để thoát):
I loved every minute of it.
1. POSITIVE (độ tin cậy 99.4%)
Worst film ever made.
2. NEGATIVE (độ tin cậy 99.5%)
```

Review dài hơn 256 token sẽ được ghi chú thêm `bị cắt còn 256/… token`: model chỉ đọc 256 token đầu (xem [evaluation.md](evaluation.md), mục 5).

Lưu ý: model được train trên review tiếng Anh, nên review tiếng Việt sẽ cho kết quả không đáng tin.

## 3. Chạy test

```bash
python -m pytest                  # toàn bộ 440 test (khoảng 5 phút)
python -m pytest -m "not slow"    # chỉ test nhanh, không cần tải model
```

## 4. Xử lý sự cố thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `Không tìm thấy thư mục model ... Hãy chạy ... --stage train trước` | Chạy bước 5 trước khi chạy bước 6, 7 |
| Train bị dừng giữa chừng | Chạy lại lệnh train, thêm `--resume` |
| Train báo đã có checkpoint cũ | Thêm `--resume` để chạy tiếp, hoặc `--overwrite` để train lại từ đầu |
| Báo lỗi hết bộ nhớ khi train | Thêm `--batch-size 8 --grad-accum 2`; đóng bớt ứng dụng khác |
| Cảnh báo "Some weights … newly initialized" khi train | Bình thường: lớp phân loại 2 nhãn được tạo mới để train |
