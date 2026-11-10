# Huấn luyện

## Phần A – Baseline: TF-IDF + LogisticRegression (bước 4)

Đề không yêu cầu baseline. Nhóm làm thêm vì 2 lý do:
- Có mốc để so: DistilBERT phải tốt hơn một mô hình đơn giản thì việc fine-tune mới đáng công.
- Áp dụng đúng quy tắc cross-validation trong [RULES.md](../RULES.md). DistilBERT không làm k-fold được vì quá tốn thời gian.

Code: [src/lab3/baseline.py](../src/lab3/baseline.py). Chạy bằng lệnh:

```bash
python src/02_train_finetune.py --stage baseline
```

### 1. Mô hình

| Thành phần | Cấu hình | Lý do |
|---|---|---|
| TF-IDF | Đổi chữ thường, bỏ từ chỉ xuất hiện ở 1 review (`min_df=2`), lấy log tần suất (`sublinear_tf`), tối đa 100.000 từ / cụm từ | Giới hạn 100.000 để file model nhỏ (4,5 MB) khi nén nộp |
| LogisticRegression | Regularization L2, solver `liblinear`, `random_state=42` | Mô hình tuyến tính chuẩn cho văn bản |
| Pipeline | TF-IDF nằm **trong** sklearn `Pipeline` | Mỗi fold cross-validation học lại từ vựng và trọng số IDF chỉ trên phần train của fold đó, nên không rò rỉ sang phần kiểm tra |

### 2. Chọn tham số bằng 5-fold cross-validation (chỉ trên tập train)

`StratifiedKFold(5, shuffle=True, random_state=42)` trên 22.297 review train, thử 8 bộ tham số. Chọn bộ có accuracy trung bình các fold cao nhất. Tập validation và test **không** được dùng để chọn tham số.

| Hạng | n-gram | C | CV accuracy | CV F1 | Train accuracy |
|---|---|---|---|---|---|
| 1 | 1–2 | 16 | **0,9049 ± 0,0041** | 0,9060 | 0,9998 |
| 2 | 1–2 | 4 | 0,9022 ± 0,0031 | 0,9034 | 0,9912 |
| 3 | 1–2 | 1 | 0,8946 ± 0,0047 | 0,8964 | 0,9552 |
| 4 | 1–1 | 4 | 0,8946 ± 0,0040 | 0,8961 | 0,9747 |
| 5 | 1–1 | 16 | 0,8919 ± 0,0047 | 0,8932 | 0,9963 |
| 6 | 1–1 | 1 | 0,8891 ± 0,0039 | 0,8910 | 0,9380 |
| 7 | 1–2 | 0,25 | 0,8776 ± 0,0061 | 0,8805 | 0,9127 |
| 8 | 1–1 | 0,25 | 0,8751 ± 0,0052 | 0,8779 | 0,9013 |

Nhận xét:
- Dùng cả cụm 2 từ (1–2 gram) tốt hơn chỉ dùng từ đơn, vì bắt được các cụm như "not good" hay "waste of".
- `C` là mức nới regularization (C càng lớn càng ít phạt). Độ lệch chuẩn giữa các fold nhỏ (dưới 0,6%), nên thứ hạng ổn định.

**Kiểm tra C = 16 có thật là tốt nhất không.** Giá trị 16 nằm ở mép lưới, nên có thể giá trị tối ưu thật nằm ngoài lưới. Nhóm chạy thêm cross-validation, vẫn chỉ trên tập train, cho C lớn hơn:

| C | CV accuracy |
|---|---|
| 16 | **0,9049** |
| 64 | 0,9030 |
| 256 | 0,9026 |

Accuracy giảm khi C vượt quá 16, nên 16 đúng là đỉnh (nằm giữa 4 và 64), không phải do lưới quá hẹp.

### 3. Kết quả: train so với validation

Model với tham số tốt nhất được train lại trên toàn bộ tập train, rồi chấm điểm:

| Tập | Accuracy | Precision | Recall | F1 | Số review |
|---|---|---|---|---|---|
| train | 0,9997 | 0,9999 | 0,9996 | 0,9997 | 22.297 |
| validation | **0,9098** | 0,9056 | 0,9157 | **0,9106** | 2.482 |

**Cảnh báo overfitting:** accuracy train cao hơn validation 0,090, vượt ngưỡng 0,05 đặt trong `config.py`.

Giải thích: với 100.000 đặc trưng mà chỉ có 22.297 review, mô hình tuyến tính gần như luôn tách được hết tập train, nên điểm train xấp xỉ 1. Vì vậy điểm train **không** phản ánh khả năng tổng quát hoá. Con số đáng tin là điểm cross-validation (0,9049) và điểm validation (0,9098). Hai số này khớp nhau, cho thấy mô hình ổn định trên dữ liệu chưa thấy.

Tăng regularization (C nhỏ hơn) thì khoảng chênh thu hẹp, nhưng điểm cross-validation cũng giảm (C = 1: chênh 0,06, CV 0,8946). Nhóm giữ lựa chọn theo cross-validation và ghi rõ cảnh báo ở đây.

### 4. Đầu ra

- Model: `outputs/baseline/tfidf_logreg.joblib` (4,5 MB)
- Báo cáo đầy đủ: `outputs/reports/baseline_report.json`
- **Điểm trên tập test chưa được tính.** Tập test chỉ chạy một lần ở bước 6, cho cả baseline và DistilBERT cùng lúc.

## Phần B – Fine-tune DistilBERT (bước 5)

Code: [src/lab3/train.py](../src/lab3/train.py), giai đoạn `--stage train` trong [src/02_train_finetune.py](../src/02_train_finetune.py).

### 1. Ứng với 7 bước của đề

| Bước của đề | Làm gì | Hàm |
|---|---|---|
| 1. Cài transformers, datasets, evaluate | `requirements.txt` | — |
| 2. Load dataset | Đọc `data/splits/train.csv`, `validation.csv` (đã chia, làm sạch, loại trùng ở bước 3). **Không đọc `test.csv`** | `load_split` |
| 3. Load model pretrained và tokenizer | `distilbert/distilbert-base-uncased` + lớp phân loại 2 nhãn mới (khởi tạo ngẫu nhiên, nên transformers in cảnh báo "newly initialized", đó là điều mong đợi) | `load_base_model` |
| 4. Tiền xử lý | Tokenize, cắt ở `max_length`, **pad theo từng batch** (`DataCollatorWithPadding`) thay vì pad cố định | `frame_to_dataset` |
| 5. Khai báo training arguments | Xem bảng ở mục 2 | `build_training_arguments` |
| 6. Tạo Trainer và fine-tune | `Trainer` + `EarlyStoppingCallback`, metric tính bằng thư viện `evaluate` | `run_training` |
| 7. Đánh giá | Model tốt nhất được chấm trên validation và trên cả tập train (để so). **Tập test chấm ở bước 6** | `run_training` |

### 2. Siêu tham số

| Tham số | Giá trị | Lý do |
|---|---|---|
| `max_length` | 256 | 58,8% review train đọc trọn. 512 không chạy được trên máy 8 GB (xem mục 3) |
| Epoch | 2 | Giống blog tham khảo. Fine-tune thường hội tụ sau 2–3 epoch |
| Learning rate | 2e-5, giảm tuyến tính, 10% bước đầu tăng dần (warmup) | Learning rate nhỏ để không phá kiến thức có sẵn của model pretrained |
| Batch | 16 | Vừa bộ nhớ GPU M1 với `max_length` 256 (đã đo) |
| Optimizer | AdamW (mặc định của Trainer), `weight_decay=0.01` | Regularization L2 kiểu AdamW |
| Gradient clipping | `max_grad_norm=1.0` | Chống gradient bùng nổ (bài giảng mục 7) |
| Dropout | 0,1 trong các lớp Transformer, 0,2 ở đầu phân loại (mặc định của DistilBERT) | Regularization |
| Đánh giá validation | Mỗi nửa epoch (697 bước), 4 lần tất cả | Để early stopping có tác dụng ngay trong 2 epoch |
| Early stopping | Dừng nếu 2 lần đánh giá liên tiếp accuracy không tăng; cuối cùng nạp lại checkpoint tốt nhất | Chọn model theo validation, không theo test |
| Checkpoint | Giữ tối đa 2 (tốt nhất + mới nhất) | Mỗi checkpoint khoảng 800 MB |
| Seed | 42 (model và thứ tự dữ liệu) | Tái lập kết quả |

### 3. Chạy thử đo tốc độ (09/11/2026, Apple M1, 8 GB RAM)

Lệnh `python src/02_train_finetune.py --stage train --smoke` train 30 bước trên 480 review mẫu, bỏ qua 5 bước đầu (khởi động) khi đo, rồi ước tính lần chạy thật:

| Cấu hình | Tốc độ | Ước tính lần chạy thật | Kết luận |
|---|---|---|---|
| `max_length=256`, batch 16 | 0,72 bước/giây (train), 42 review/giây (đánh giá) | Train 2.788 bước: 1 giờ 5 phút. Đánh giá validation 4 lần: 4 phút. Chấm lại tập train: 9 phút. **Tổng khoảng 1 giờ 17 phút** | **Chọn.** Nằm trong giới hạn 3 giờ |
| `max_length=512`, batch 8 × 2 | Bước 1–3: khoảng 3,4 giây/bước, từ bước 4 tăng lên 34 → 120 giây/bước | Không ước tính được | **Không chạy được.** Hết RAM: máy phải dùng tới 7,3 GB swap. Đã dừng sau 10 phút |

Thời gian thật có thể lâu hơn ước tính nếu máy nóng lên (chip tự giảm tốc độ) hoặc đang mở nhiều ứng dụng khác chiếm RAM.

### 4. Kết quả lần train đầy đủ (09/11/2026)

Chạy bằng `caffeinate -i python src/02_train_finetune.py --stage train` trên GPU M1. Mất **1 giờ 11 phút** (train 63,5 phút, phần còn lại là đánh giá và lưu model). Chạy đủ 2.788/2.788 bước, early stopping không kích hoạt.

**Validation qua từng lần đánh giá:**

| Bước | Epoch | Accuracy | F1 | Loss validation |
|---|---|---|---|---|
| 697 | 0,5 | 0,8940 | 0,8957 | 0,260 |
| 1.394 | 1,0 | 0,9049 | 0,9069 | **0,235** |
| 2.091 | 1,5 | 0,9053 | 0,9039 | 0,301 |
| 2.788 | 2,0 | **0,9073** | **0,9081** | 0,291 |

Loss train giảm từ 0,696 (bước 50) xuống 0,168 (bước 2.750). Checkpoint tốt nhất theo accuracy validation là **bước 2.788**, được lưu vào `outputs/final_model/` (256 MB).

**Model tốt nhất, so train với validation:**

| Tập | Accuracy | Precision | Recall | F1 | Loss |
|---|---|---|---|---|---|
| train (22.297) | 0,9692 | 0,9658 | 0,9731 | 0,9694 | 0,102 |
| validation (2.482) | **0,9073** | 0,9038 | 0,9125 | **0,9081** | 0,291 |

**Overfitting ở mức nhẹ.** Accuracy train cao hơn validation 0,062, vượt ngưỡng 0,05 nên code in cảnh báo. Dấu hiệu thứ hai: sang epoch 2, loss validation tăng (0,235 → 0,291) trong khi accuracy vẫn nhích lên, nghĩa là model tự tin hơn mức cần thiết ở những câu đoán sai. Overfitting vẫn ở mức nhẹ vì:
- accuracy validation không giảm qua các lần đánh giá;
- khoảng chênh 0,062 nhỏ hơn của baseline (0,090);
- model cuối cùng được chọn theo validation, không lấy đại checkpoint cuối.

### 5. So với baseline và nguyên nhân

Trên validation, DistilBERT (0,9073) **ngang baseline** TF-IDF + LogisticRegression (0,9098). Nhóm kiểm chứng giả thuyết "do cắt review ở 256 token" bằng cách tách validation theo độ dài:

| Nhóm review validation | Số review | DistilBERT | Baseline |
|---|---|---|---|
| Không bị cắt (≤ 256 token) | 1.456 | **0,9210** | 0,9011 |
| Bị cắt (> 256 token) | 1.026 | 0,8879 | **0,9220** |
| Tất cả | 2.482 | 0,9073 | 0,9098 |

**Kết luận:** khi đọc trọn review, DistilBERT tốt hơn baseline 2 điểm. Với review dài hơn 256 token, DistilBERT chỉ thấy phần đầu, mất phần cuối là nơi người viết thường chốt ý, nên thua baseline 3,4 điểm. Baseline TF-IDF thì luôn đọc trọn review. Giới hạn 256 token là do máy 8 GB RAM không chạy được 512 (mục 3).

Hướng cải thiện mà không tốn thêm bộ nhớ: giữ **phần đầu + phần cuối** của review (vd 128 token đầu + 128 token cuối) thay vì chỉ giữ phần đầu. Cách này đã được chứng minh hiệu quả trên IMDb (Sun và cộng sự, 2019, *How to Fine-Tune BERT for Text Classification?*). Muốn áp dụng thì phải train lại khoảng 1 giờ 10 phút.

Tập test **chưa** được dùng. Điểm test của cả hai model sẽ chấm ở bước 6.

Đầu ra: `outputs/final_model/` (model + tokenizer), `outputs/reports/train_report.json` (siêu tham số, toàn bộ lịch sử loss / metric để vẽ đồ thị ở bước 6), `outputs/reports/train_console.log` (log terminal). `outputs/checkpoints/` (1,5 GB) chỉ dùng để chạy tiếp, không nén vào zip.
