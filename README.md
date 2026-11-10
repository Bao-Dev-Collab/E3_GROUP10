# Lab 3 – Get started with Hugging Face

Nhóm E3_GROUP10 · Practice 3 · Chương 4: Fine-tune Models and Training Algorithms

## 1. Giới thiệu

- **Bài 1:** phân tích cảm xúc một câu bằng model đã train sẵn trên Hugging Face Hub: tokenize câu, tính logits, softmax, ra nhãn.
- **Bài 2:** fine-tune một model pretrained cho bài toán phân loại văn bản nhị phân (review phim tích cực / tiêu cực), làm đủ 7 bước của đề.

Chi tiết đề bài và các quyết định của nhóm: [docs/lab3-description.md](docs/lab3-description.md). **Báo cáo đầy đủ: [REPORT.md](REPORT.md).**

**Kết quả chính** (25.000 review test của IMDb):

| Model | Accuracy | F1 |
|---|---|---|
| DistilBERT fine-tune (Bài 2) | **0,9145** | **0,9153** |
| Baseline TF-IDF + LogisticRegression | 0,9032 | 0,9032 |

DistilBERT tốt hơn baseline có ý nghĩa thống kê (kiểm định McNemar, p ≈ 4,5 × 10⁻⁸).

## 2. Dữ liệu và mô hình

- **Dataset:** IMDb (`stanfordnlp/imdb`): 25.000 review train, 25.000 review test, 2 nhãn (`0` = tiêu cực, `1` = tích cực), mỗi nhãn 12.500 review. Nguồn, cách tải và cách xử lý: [docs/data.md](docs/data.md).
- **Base model cho fine-tuning (Bài 2):** `distilbert/distilbert-base-uncased`
- **Model pretrained cho Bài 1:** `distilbert/distilbert-base-uncased-finetuned-sst-2-english`

## 3. Cấu trúc thư mục

```text
lab3_DL/
├── data/
│   ├── raw/                              # IMDb gốc tải từ Hugging Face
│   │   ├── imdb_train.csv                # 25.000 dòng, cột text, label
│   │   └── imdb_test.csv                 # 25.000 dòng, cột text, label
│   └── splits/                           # sau khi chia, làm sạch, loại trùng (bước 3)
│       ├── train.csv                     # 22.297 dòng
│       ├── validation.csv                # 2.482 dòng
│       └── test.csv                      # 25.000 dòng
├── docs/                                 # tài liệu kỹ thuật từng phần
│   ├── lab3-description.md               # đề bài, tài liệu tham khảo, quyết định của nhóm
│   ├── data.md                           # dữ liệu, cách chia, làm sạch, tokenization
│   ├── model.md                          # model Bài 1, kiến trúc DistilBERT
│   ├── training.md                       # quy trình 7 bước, siêu tham số
│   ├── evaluation.md                     # kết quả, confusion matrix, phân tích lỗi
│   ├── demo.md                           # hướng dẫn chạy thử
│   ├── plan-progress-status.md           # tiến độ từng bước
│   ├── AUDIT_LOG.md                      # nhật ký 5 giai đoạn của từng bước
│   ├── plan-progress-status.html         # bản HTML của file tiến độ
│   └── MONITORS_DIAGRAM.md               # sơ đồ quy trình và các điểm giám sát
├── notebooks/
│   ├── Practice3_HuggingFace_Final.ipynb # notebook tổng hợp cả 2 bài (đã chạy sẵn kết quả)
│   └── Practice3_HuggingFace_Final.html  # bản HTML của notebook, mở bằng trình duyệt
├── src/
│   ├── 01_sentiment_analysis.py          # Bài 1
│   ├── 02_train_finetune.py              # Bài 2: fine-tune DistilBERT
│   ├── 03_predict_finetuned.py           # dự đoán câu mới bằng model đã fine-tune
│   └── lab3/                             # logic dùng chung, import được để test
│       ├── config.py                     # đường dẫn, tên model, tham số
│       ├── utils.py                      # chọn thiết bị, ghi JSON
│       ├── sentiment.py                  # Bài 1: tokenize, logits, softmax, pipeline
│       ├── data.py                       # Bài 2: đọc, chia, làm sạch, loại trùng, thống kê
│       ├── baseline.py                   # Bài 2: TF-IDF + LogisticRegression, 5-fold CV
│       ├── train.py                      # Bài 2: fine-tune DistilBERT bằng Trainer
│       ├── evaluate.py                   # metrics, confusion matrix, cảnh báo overfitting
│       ├── final_evaluation.py           # Bài 2: chấm tập test, phân tích theo độ dài, ví dụ đoán sai
│       ├── predict.py                    # dự đoán review mới bằng model đã fine-tune
│       └── plots.py                      # vẽ confusion matrix, đồ thị quá trình train
├── tests/                                # unit test (pytest)
│   ├── conftest.py                       # fixture dùng chung
│   ├── test_utils.py
│   ├── test_sentiment.py                 # test nhanh, không cần model
│   ├── test_sentiment_model.py           # test với model thật (slow)
│   ├── test_exercise1_script.py          # chạy thử script Bài 1 (slow)
│   ├── test_data.py                      # test phần dữ liệu (nhanh, tokenizer giả)
│   ├── test_data_script.py               # chạy thử giai đoạn data (slow)
│   ├── test_evaluate.py                  # test metrics, cảnh báo overfitting
│   ├── test_baseline.py                  # test baseline trên dữ liệu giả
│   ├── test_baseline_script.py           # chạy thử giai đoạn baseline
│   ├── test_train.py                     # test fine-tune (model DistilBERT tí hon, CPU)
│   ├── test_train_script.py              # chạy thử giai đoạn train (slow)
│   ├── test_final_evaluation.py          # test đánh giá cuối
│   ├── test_plots.py                     # test vẽ đồ thị
│   ├── test_evaluate_script.py           # chạy thử giai đoạn evaluate (slow)
│   ├── test_predict.py                   # test dự đoán review mới
│   ├── test_predict_script.py            # chạy thử script 03 (slow)
│   ├── test_notebook.py                  # notebook hợp lệ, chạy được từ đầu đến cuối
│   └── helpers.py                        # tạo dữ liệu giả cho test
├── outputs/
│   ├── baseline/
│   │   └── tfidf_logreg.joblib           # model baseline (4,5 MB)
│   ├── checkpoints/                      # checkpoint khi train (không nén vào zip)
│   ├── final_model/                      # model DistilBERT tốt nhất (256 MB)
│   └── reports/                          # metrics, hình vẽ
│       ├── exercise1_sentiment.json      # kết quả Bài 1
│       ├── data_report.json              # báo cáo dữ liệu Bài 2
│       ├── baseline_report.json          # kết quả cross-validation, train, validation của baseline
│       ├── smoke_report.json             # kết quả chạy thử, ước tính thời gian train
│       ├── train_report.json             # kết quả fine-tune DistilBERT
│       ├── evaluation_report.json        # kết quả trên tập test của 2 model
│       ├── confusion_matrices.png
│       ├── training_curves.png
│       ├── test_predictions.csv          # dự đoán từng review test
│       ├── demo_predictions.json         # kết quả script 03 với 5 câu mẫu
│       └── *_console.log                 # log terminal của lần train và lần đánh giá
├── REPORT.md                             # báo cáo thực hành
├── RULES.md                              # quy tắc khoa học và chuẩn mực học thuật
├── AI_Usage.md                           # tuyên bố sử dụng AI
├── requirements.txt                      # thư viện và phiên bản
└── pytest.ini                            # cấu hình pytest
```

## 4. Cài đặt

Yêu cầu Python 3.13 (đã thử trên macOS, Apple M1). Chạy từ thư mục `lab3_DL/`:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Cách chạy

Chạy từ thư mục `lab3_DL/` sau khi đã kích hoạt `.venv`. Lần chạy đầu cần internet để tải model (khoảng 270 MB), các lần sau dùng bản lưu sẵn trên máy.

**Bài 1: phân tích cảm xúc**

```bash
python src/01_sentiment_analysis.py
python src/01_sentiment_analysis.py --text "This movie is great" --text "I hated it"
python src/01_sentiment_analysis.py --device cpu
```

Kết quả in ra màn hình và lưu vào `outputs/reports/exercise1_sentiment.json`. Giải thích chi tiết: [docs/model.md](docs/model.md).

**Bài 2: chuẩn bị dữ liệu**

```bash
python src/02_train_finetune.py --stage data
```

Đọc `data/raw/`, chia train / validation / test, làm sạch, loại trùng, rồi ghi ra `data/splits/` và `outputs/reports/data_report.json`. Giải thích chi tiết: [docs/data.md](docs/data.md).

**Bài 2: baseline TF-IDF + LogisticRegression** (khoảng 3 phút)

```bash
python src/02_train_finetune.py --stage baseline
```

Chọn tham số bằng 5-fold cross-validation trên train, báo cáo điểm train và validation, không đụng tới tập test. Giải thích chi tiết: [docs/training.md](docs/training.md).

**Bài 2: fine-tune DistilBERT** (khoảng 1 giờ 20 phút trên Apple M1)

```bash
python src/02_train_finetune.py --stage train --smoke          # chạy thử 1–2 phút, in ước tính thời gian
caffeinate -i python src/02_train_finetune.py --stage train    # train thật; caffeinate giữ máy Mac không ngủ
```

- Bị dừng giữa chừng: chạy lại với `--resume` để tiếp tục từ checkpoint gần nhất.
- Muốn train lại từ đầu: thêm `--overwrite`.
- Báo lỗi hết bộ nhớ: thêm `--batch-size 8 --grad-accum 2`.
- Trên Windows / Linux bỏ `caffeinate -i`.

**Bài 2: đánh giá trên tập test** (khoảng 10 phút, chạy một lần sau khi đã chốt model)

```bash
python src/02_train_finetune.py --stage evaluate
```

Kết quả: DistilBERT accuracy **0,9145** / F1 **0,9153**, baseline 0,9032 / 0,9032. Chi tiết: [docs/evaluation.md](docs/evaluation.md).

**Dự đoán review mới bằng model đã fine-tune**

```bash
python src/03_predict_finetuned.py                       # 5 câu mẫu có sẵn
python src/03_predict_finetuned.py --text "Great movie!"
python src/03_predict_finetuned.py --interactive         # gõ từng câu
```

Toàn bộ quy trình, thời gian chạy và cách xử lý sự cố: [docs/demo.md](docs/demo.md).

**Chạy test**

```bash
python -m pytest                  # toàn bộ test (khoảng 5 phút)
python -m pytest -m "not slow"    # chỉ test nhanh, không tải model
```

**Notebook tổng hợp:** mở `notebooks/Practice3_HuggingFace_Final.ipynb` trong VS Code, chọn kernel `.venv`. Notebook đã có sẵn kết quả, chạy lại mất khoảng 15 giây (không train lại). Tiến độ từng bước: [docs/plan-progress-status.md](docs/plan-progress-status.md).

## 6. Trên GitHub

Repo không chứa `.venv/`, `outputs/checkpoints/` và `outputs/final_model/` (xem `.gitignore`). File model 255 MB vượt giới hạn 100 MB/file của GitHub. Sau khi clone về:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
caffeinate -i python src/02_train_finetune.py --stage train   # tạo lại outputs/final_model/ (khoảng 1 giờ 11 phút)
```

Dữ liệu (`data/`), các báo cáo, đồ thị trong `outputs/reports/`, model baseline và notebook đã chạy sẵn đều có trong repo. Vì vậy xem được toàn bộ kết quả mà không cần train lại.

## 7. Khi nén nộp bài

Bỏ `.venv/` (môi trường, tạo lại bằng `requirements.txt`), `outputs/checkpoints/` (1,5 GB, chỉ dùng để train tiếp) và các thư mục cache (`__pycache__`, `.pytest_cache`). Giữ `outputs/final_model/`, để chạy được bước đánh giá, script dự đoán và notebook mà không phải train lại 1 giờ 11 phút.

```bash
cd ..
zip -r -X E3_GROUP10_lab3_DL.zip lab3_DL -x "lab3_DL/.venv/*" "lab3_DL/outputs/checkpoints/*" "*/__pycache__/*" "lab3_DL/.pytest_cache/*" "*.DS_Store"
```
