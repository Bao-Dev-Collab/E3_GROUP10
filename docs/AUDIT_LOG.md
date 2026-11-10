# Nhật ký kiểm toán

Mỗi bước đi qua 5 giai đoạn:

| Giai đoạn | Ý nghĩa |
|---|---|
| Description | Mô tả việc cần làm và tiêu chí hoàn thành |
| Gen | Tạo code / file |
| Smoke | Chạy thử nhanh để bắt lỗi |
| Run | Chạy thật |
| Check | Kiểm tra kết quả, nhóm duyệt |

---

## Bước 1 – Khởi tạo cấu trúc, môi trường, tải dữ liệu (09/11/2026)

| Giai đoạn | Nội dung | Kết quả |
|---|---|---|
| Description | Tạo cấu trúc thư mục theo project mẫu, thêm `data/`, `tests/`, `src/lab3/`. Tạo môi trường Python. Tải IMDb train + test vào `data/raw/` | — |
| Gen | Tạo cây thư mục, các file khung trong `docs/`, `src/`, `notebooks/`, `README.md`, `RULES.md`, `REPORT.md`, `AI_Usage.md` | Xong |
| Smoke | Không áp dụng (chưa có code) | — |
| Run | Tạo `.venv` (Python 3.13.2), cài thư viện, ghi phiên bản vào `requirements.txt`. Tải `train/0000.parquet`, `test/0000.parquet` từ `stanfordnlp/imdb` (nhánh `refs/convert/parquet`), lưu thành CSV | Xong. MPS (GPU của M1) dùng được |
| Check | `imdb_train.csv` và `imdb_test.csv`: mỗi file 25.000 dòng, cột `text`, `label`, nhãn 0 và 1 mỗi loại 12.500, không có review rỗng hay thiếu. Đọc lại CSV ra đúng y dữ liệu gốc. Chạy lại bước chuyển đổi ra file giống hệt (cùng SHA-256). Phát hiện cả 2 file đều sắp xếp theo nhãn | Nhóm đã duyệt |
| Bổ sung | Theo yêu cầu của nhóm: ghi nguồn, cách tải, cách chuyển đổi, kết quả kiểm tra và kế hoạch xử lý vào `docs/data.md` | Xong |

---

## Bước 2 – Bài 1: phân tích cảm xúc (09/11/2026)

| Giai đoạn | Nội dung | Kết quả |
|---|---|---|
| Description | Làm đủ 4 bước của Exercise 1: dùng model `distilbert/distilbert-base-uncased-finetuned-sst-2-english`, tokenize câu mẫu, tự tính logits → softmax → nhãn, đối chiếu với `pipeline()`. Có test pytest theo `RULES.md` | — |
| Gen | `src/lab3/config.py`, `src/lab3/utils.py`, `src/lab3/sentiment.py`, `src/01_sentiment_analysis.py`, `pytest.ini`, 4 file test trong `tests/` | Xong |
| Smoke | Chạy thử API của transformers 5.22.2. Phát hiện crash **Bus error** (mã thoát 138) khi chạy model trên CPU với 1 câu. Khoanh vùng: không do attention, không do Linear thường; do trọng số được nạp bằng memory-map từ file chỉ đọc. Chép trọng số ra bộ nhớ thường thì hết lỗi | Đã sửa bằng `_materialize_parameters` |
| Run | `python src/01_sentiment_analysis.py` chạy trên MPS: 3 câu mẫu ra POSITIVE / NEGATIVE / POSITIVE, sai lệch so với `pipeline()` là 1,9 × 10⁻¹⁰. Chạy với `--device cpu` cũng thành công | Xong |
| Check | `python -m pytest`: **112/112 test pass** (46 giây). Tắt bản sửa lỗi CPU thì lỗi quay lại (mã 138), chứng tỏ test chống tái phát có tác dụng | Chờ nhóm duyệt |

Các test case tự thêm ngoài ví dụ (gồm trường hợp biên):

- câu có emoji, tiếng Việt, tiếng Nhật
- câu rất dài (2.000 từ), bị cắt đúng 512 token
- `max_length` nhỏ hơn số token đặc biệt: tokenizer âm thầm bỏ qua việc cắt câu, nên code chặn trước
- kết quả khi chạy cả batch phải bằng kết quả khi chạy từng câu (pad không làm đổi kết quả)
- logits rất lớn (1000) hoặc rất âm (−1000) không gây tràn số
- logits bằng nhau thì chọn nhãn có chỉ số nhỏ nhất
- chữ hoa và chữ thường cho cùng token (model uncased)
- model hơn 2 nhãn
- chạy CPU với 1 câu (chống tái phát Bus error)

Input sai đã test: `None`, chuỗi rỗng, chuỗi chỉ có khoảng trắng, list rỗng, sai kiểu (số, bytes, dict, bool), `max_length` âm / bằng 0 / quá lớn / không phải số nguyên, thư mục model không tồn tại, model không có trên Hub, thiết bị không hợp lệ, đường dẫn lưu file rỗng hoặc là thư mục.

### Bước 2 – Rà soát lại theo yêu cầu của nhóm (09/11/2026)

| Giai đoạn | Nội dung | Kết quả |
|---|---|---|
| Description | Rà lỗi lớn trong code Bài 1. Chỉ sửa đúng chỗ gây lỗi, tìm nguyên nhân gốc, không vá tạm | — |
| Check | Chẩn đoán ở bước Smoke phía trên **chưa đúng gốc**. Trọng số nạp bằng memory-map nằm ở địa chỉ chia 4 dư 2, vì dữ liệu trong `model.safetensors` bắt đầu ở byte 12.518. Thí nghiệm tách yếu tố: căn hàng đúng + vùng nhớ chỉ đọc → chạy bình thường; lệch căn hàng + vùng nhớ ghi được → crash (mã 138). Vậy nguyên nhân là lệch căn hàng, không phải bộ nhớ chỉ đọc | Đã xác định |
| Gen | Bỏ hàm `_materialize_parameters` (chép trọng số sau khi nạp, có dùng thuộc tính nội bộ `_buffers`). Thay bằng tham số chính thức `disable_mmap=True` khi nạp model. Thêm test `test_load_on_cpu_gives_aligned_weights` | Xong |
| Run | Script chạy được trên MPS và trên CPU với 1 câu. Nạp lại không có `disable_mmap` thì 104 tensor bị lệch, nên test mới phát hiện được lỗi cũ | Xong |
| Check | `python -m pytest`: **113/113 test pass** | Nhóm đã duyệt |

---

## Bước 3 – Bài 2: chuẩn bị dữ liệu (09/11/2026)

| Giai đoạn | Nội dung | Kết quả |
|---|---|---|
| Description | Chia train/validation/test **trước** mọi bước xử lý; làm sạch `<br />`; loại review trùng giữa các tập (test giữ nguyên); thống kê độ dài token chỉ trên train; lưu `data/splits/` | — |
| Gen | `src/lab3/data.py`, giai đoạn `--stage data` trong `src/02_train_finetune.py`, cấu hình Bài 2 trong `config.py`, `tests/test_data.py`, `tests/test_data_script.py` | Xong |
| Smoke | Khảo sát trước khi viết code: 123 review có ở cả train và test (rò rỉ dữ liệu), khoảng 190 dòng trùng trong train, 390 dòng trùng trong test, không nhóm trùng nào có nhãn mâu thuẫn. Thẻ HTML gần như chỉ có `<br />` | Đã đưa vào thiết kế |
| Run | `python src/02_train_finetune.py --stage data` (15 giây): train 22.297 / validation 2.482 / test 25.000. Đã loại 107 + 16 review trùng với test. Độ dài token trên train: trung vị 222, 41,2% review dài hơn 256 token | Xong |
| Check | 207/207 test pass. Sau đó tự rà lỗi (xem mục dưới) | Nhóm đã duyệt |

### Bước 3 – Tự rà lỗi sau khi code

| # | Lỗi | Nguyên nhân gốc rễ | Cách sửa |
|---|---|---|---|
| 1 | `split_train_validation` có thể trả về tập **thiếu hẳn một lớp** mà không báo lỗi, dù docstring cam kết đủ lớp. Tái hiện: 98/2 mẫu với `val_size=0.02`, 95/5 với `0.1` (validation mất lớp 1); 8/2 với `0.8` (train mất lớp 1) | Bước kiểm tra trước khi chia chỉ so **tổng** số mẫu validation với số lớp. Nhưng stratify chia số mẫu theo tỉ lệ **từng lớp** rồi làm tròn, nên lớp thiểu số có thể bị làm tròn về 0 ở một bên | Kiểm tra trên **kết quả thật** sau khi chia: tập nào thiếu lớp thì báo `ValueError` kèm hướng xử lý. Thêm 3 test cho 3 trường hợp trên |
| 2 | File CSV ghi ra **khác nhau theo hệ điều hành**: cùng dữ liệu nhưng trên Windows sẽ khác byte và khác SHA-256, làm hỏng việc kiểm tra tái lập kết quả | `DataFrame.to_csv` mặc định xuống dòng theo `os.linesep` (`\r\n` trên Windows, `\n` trên macOS/Linux). Đã tái hiện bằng cách giả lập Windows | Ghi rõ `lineterminator="\n"` trong `save_split` và trong lệnh chuyển đổi ở `docs/data.md`. Thêm test giả lập Windows. Trên macOS các file cũ giữ nguyên từng byte (đã so SHA-256) |

Sau khi sửa: **211/211 test pass**. Chạy lại giai đoạn data ra 3 file giống hệt trước khi sửa.

Hạn chế ghi nhận, không phải lỗi (đã ghi ở `docs/data.md` mục 9): validation có thể chung phim với train vì bản IMDb trên Hugging Face không có mã phim; tập test giữ 201 bản trùng nội bộ để so sánh được với các công trình khác.

---

## Bước 4 – Bài 2: baseline TF-IDF + LogisticRegression (09/11/2026)

| Giai đoạn | Nội dung | Kết quả |
|---|---|---|
| Description | Baseline để so với DistilBERT. TF-IDF nằm trong sklearn `Pipeline` (không rò rỉ); chọn tham số bằng `StratifiedKFold` 5 fold chỉ trên train; báo cáo train và validation, cảnh báo nếu chênh lệch lớn; không đụng tới test | — |
| Gen | `src/lab3/evaluate.py` (metrics, cảnh báo overfitting), `src/lab3/baseline.py`, `load_split` trong `data.py` (đọc tập đã chia, giữ nguyên review "NA"), giai đoạn `--stage baseline` trong `02_train_finetune.py`, 4 file test mới và `tests/helpers.py` | Xong |
| Smoke | Đo trước trên data thật: mỗi lần train 2–8 giây, nên lưới 8 bộ tham số × 5 fold là khả thi | — |
| Run | `python src/02_train_finetune.py --stage baseline` (3 phút). Tốt nhất: 1–2 gram, C = 16, CV accuracy 0,9049 ± 0,0041. Validation: accuracy 0,9098, F1 0,9106. Train: 0,9997, nên **cảnh báo overfitting** (chênh 0,090 > 0,05). Đã giải thích trong `docs/training.md` | Xong |
| Check | **275/275 test pass** (gồm toàn bộ test các bước trước) | Nhóm đã duyệt |

### Bước 4 – Tự rà lỗi sau khi code

| Nghi vấn | Kiểm chứng | Kết luận |
|---|---|---|
| C = 16 tốt nhất nhưng nằm ở mép lưới, có thể giá trị tối ưu thật nằm ngoài lưới | Chạy thêm 5-fold CV (chỉ trên train) với C = 64 và 256: CV accuracy 0,9030 và 0,9026, đều thấp hơn 0,9049 | **Không phải lỗi.** 16 là đỉnh. Không sửa code, chỉ ghi kết quả kiểm chứng vào `docs/training.md` |
| Rò rỉ dữ liệu trong cross-validation | TF-IDF nằm trong `Pipeline`, `GridSearchCV` nhận cả Pipeline (có test kiểm tra). Giai đoạn baseline chạy được khi thư mục không có `test.csv` (có test kiểm tra) | Không có rò rỉ |
| Điểm train ≈ 1,0 (cảnh báo overfitting) | Với 100.000 đặc trưng và 22.297 review, mô hình tuyến tính tách được hết tập train. CV và validation khớp nhau (0,905 và 0,910) | Đúng hành vi mong đợi, cảnh báo được giữ và giải thích |
| Sửa `data.py` (tách hàm đọc CSV dùng chung) có làm hỏng bước 3 không | Toàn bộ test bước 3 vẫn pass | Không ảnh hưởng |

Không phát hiện lỗi lớn cần sửa trong bước 4.

---

## Bước 5 – Bài 2: fine-tune DistilBERT (09/11/2026)

| Giai đoạn | Nội dung | Kết quả |
|---|---|---|
| Description | Làm bước 3–7 của đề bằng `Trainer`: tải `distilbert-base-uncased`, tokenize (pad theo batch), `TrainingArguments`, fine-tune với early stopping theo validation, đánh giá trên validation và train. Không dùng test. Có `--smoke` để đo tốc độ, `--resume` để chạy tiếp | — |
| Gen | `src/lab3/train.py`, giai đoạn `--stage train`, cấu hình Bài 2 trong `config.py`, `tests/test_train.py`, `tests/test_train_script.py`, fixture model DistilBERT tí hon trong `conftest.py` | Xong |
| Smoke | Chạy thử trên GPU M1 với model thật: `max_length=256`, batch 16 chạy được, 0,72 bước/giây, **ước tính 1 giờ 17 phút**. `max_length=512` (batch 8 × 2) hết RAM: dùng 7,3 GB swap, mỗi bước chậm từ 3 giây lên 120 giây; đã dừng | Chọn 256 |
| Run | Train đầy đủ chạy nền trên GPU M1: **1 giờ 11 phút**, 2.788/2.788 bước. Model tốt nhất (bước 2.788): validation accuracy **0,9073**, F1 0,9081; train accuracy 0,9692; chênh 0,062, cảnh báo overfitting nhẹ. Model nạp lại được trên cả CPU lẫn MPS | Xong |
| Check | **345/345 test pass**, gồm train thật end-to-end bằng model tí hon trên CPU, chạy tiếp từ checkpoint, nạp lại model đã lưu và dự đoán | — |

### Bước 5 – Tự rà lỗi sau khi code

| # | Lỗi | Nguyên nhân gốc rễ | Cách sửa |
|---|---|---|---|
| 1 | Dữ liệu đưa vào model có thêm cột `token_type_ids` | transformers 5 nạp tokenizer của DistilBERT dưới dạng `BertTokenizer` (`model_input_names` gồm `token_type_ids`). Hàm `forward` của DistilBERT không có tham số này nên cột bị âm thầm bỏ qua | Tokenize với `return_token_type_ids=False`; test kiểm tra Dataset chỉ có `input_ids`, `attention_mask`, `labels` |
| 2 | Log in cảnh báo sai "early stopping is disabled" sau khi train xong | `EarlyStoppingCallback` phản ứng với **mọi** lần `evaluate`, kể cả lần chấm tập train (tên metric có tiền tố `train_` chứ không phải `eval_`) | Gỡ callback early stopping trước khi chấm tập train; thêm test kiểm tra log không còn cảnh báo |
| 3 | Test `--resume` ban đầu kỳ vọng sai (cho rằng số bước giữ nguyên) | Lần chạy đầu dừng sớm ở bước 15/20. Chạy tiếp thì `Trainer` đi tiếp từ checkpoint-15 tới bước 20, đúng hành vi của chạy tiếp | Sửa test để kiểm tra đúng tính chất: không bước nào bị train lại, số bước không vượt kế hoạch |
| 4 | `max_length=512` không dùng được | Giới hạn phần cứng (8 GB RAM), không phải lỗi code | Ghi vào `docs/training.md`; giữ 256 |

### Bước 5 – Phân tích sau khi train

| Nghi vấn | Kiểm chứng (chỉ trên validation) | Kết luận |
|---|---|---|
| DistilBERT (0,9073) không hơn baseline (0,9098), có phải lỗi không | Tách validation theo độ dài. Review ≤ 256 token: DistilBERT 0,921 so với baseline 0,901. Review > 256 token: DistilBERT 0,888 so với baseline 0,922 | **Không phải lỗi code.** Nguyên nhân là `max_length=256` cắt mất phần cuối của 41% số review; giới hạn này do RAM 8 GB. Đề xuất cắt giữ đầu + cuối (cần train lại), chờ nhóm quyết định |

---

## Bước 6 – Bài 2: đánh giá trên tập test (09/11/2026)

| Giai đoạn | Nội dung | Kết quả |
|---|---|---|
| Description | Chấm DistilBERT và baseline trên 25.000 review test, **một lần duy nhất**, sau khi đã chốt model. Confusion matrix, đồ thị quá trình train, phân tích theo độ dài, so sánh hai model, ví dụ đoán sai | — |
| Gen | `src/lab3/final_evaluation.py`, `src/lab3/plots.py`, `confusion_counts` trong `evaluate.py`, `load_json` trong `utils.py`, giai đoạn `--stage evaluate`, 3 file test mới và test bổ sung | Xong |
| Smoke | Chạy đầy đủ bằng model tí hon trên CPU qua pytest (cả script) trước khi chấm test thật. 1 test fail do chính test gọi thiếu tham số `max_length`, đã sửa test (code không lỗi) | Xong |
| Run | `python src/02_train_finetune.py --stage evaluate` (khoảng 10 phút, GPU M1). DistilBERT: accuracy **0,9145**, F1 **0,9153**. Baseline: 0,9032 / 0,9032 | Xong |
| Check | **391/391 test pass**. Accuracy tính lại từ confusion matrix khớp (22.862/25.000 và 22.579/25.000). Đã xem 2 file ảnh: hiển thị đúng, tiếng Việt không lỗi font | Nhóm đã duyệt |

### Bước 6 – Tự rà lỗi sau khi code

| Nghi vấn | Kiểm chứng | Kết luận |
|---|---|---|
| Số liệu báo cáo có nhất quán không | Accuracy trong `metrics` = đường chéo confusion matrix / 25.000 = dòng "Tất cả" của bảng theo độ dài = accuracy tính lại từ `test_predictions.csv` (có test tự động kiểm tra cả 3) | Nhất quán |
| `max_length` khi chấm có khớp lúc train không | Đọc từ `train_report.json` (256), không lấy từ tham số dòng lệnh | Khớp |
| Một review xuất hiện 2 lần trong danh sách lỗi | Do tập test giữ 201 bản trùng nội bộ (quyết định ở bước 3) | Không phải lỗi tính toán. Ghi chú trong `docs/evaluation.md` |
| Trên validation hai model ngang nhau, trên test DistilBERT hơn 1,1 điểm | Validation có thể chung phim với train (hạn chế đã ghi ở bước 3), có lợi cho TF-IDF | Giả thuyết hợp lý, chưa kiểm chứng được vì không có mã phim. Ghi vào `docs/evaluation.md` mục 8 |

Không phát hiện lỗi lớn cần sửa trong bước 6.

---

## Bước 7 – Dự đoán review mới (09/11/2026)

| Giai đoạn | Nội dung | Kết quả |
|---|---|---|
| Description | Script dự đoán review mới bằng model đã fine-tune. Nhập bằng `--text`, `--file` hoặc `--interactive`; `max_length` lấy đúng giá trị lúc train; ghi rõ review nào bị cắt | — |
| Gen | `src/lab3/predict.py`, `src/03_predict_finetuned.py`, câu mẫu trong `config.py`, `tests/test_predict.py`, `tests/test_predict_script.py`, `docs/demo.md` | Xong |
| Smoke | Chạy thử bằng model tí hon trên CPU qua pytest (cả 3 cách nhập, các lỗi đầu vào) | Pass |
| Run | Model thật: đúng 4/5 câu mẫu (đúng cả câu phủ định "not bad at all"); sai câu mỉa mai nhưng độ tin cậy chỉ 58,6%. Chạy được trên MPS và CPU | Xong |
| Check | **425/425 test pass** | Nhóm đã duyệt |

### Bước 7 – Tự rà lỗi sau khi code

| # | Lỗi | Nguyên nhân gốc rễ | Cách sửa |
|---|---|---|---|
| 1 | Chế độ `--interactive` đánh số câu nào cũng là "1." | `print_results` tự đánh số lại từ 1 cho mỗi danh sách nó nhận, mà chế độ hỏi–đáp mỗi lần chỉ truyền 1 kết quả | Thêm tham số `start` (số thứ tự bắt đầu); chế độ hỏi–đáp truyền số thứ tự thật. Thêm test kiểm tra đánh số 1, 2 |

---

## Bước 8 – Notebook tổng hợp (09/11/2026)

| Giai đoạn | Nội dung | Kết quả |
|---|---|---|
| Description | Một notebook trình bày cả 2 bài, gọi đúng các hàm trong `src/lab3/` và đọc kết quả đã lưu, **không train lại** | — |
| Gen | `notebooks/Practice3_HuggingFace_Final.ipynb` (23 ô), bản `.html`, hàm `mcnemar_test` trong `evaluate.py`, `tests/test_notebook.py`, thêm test McNemar | Xong |
| Run | Chạy toàn bộ notebook bằng `nbclient` với model thật trên MPS: 10 ô code, **0 ô lỗi**, 12 giây. Lưu kèm kết quả để mở ra là xem được ngay | Xong |
| Check | Kiểm định McNemar giữa DistilBERT và baseline trên test: χ² = 29,86, p ≈ 4,5 × 10⁻⁸ (khác biệt có ý nghĩa thống kê) | Chờ nhóm duyệt |

## Bước 9 – Báo cáo và đóng gói (09–10/11/2026)

| Giai đoạn | Nội dung | Kết quả |
|---|---|---|
| Description | Hoàn thiện tài liệu, báo cáo tổng hợp, tuyên bố sử dụng AI, đưa bài lên GitHub | — |
| Gen | `REPORT.md`, `AI_Usage.md`, `docs/MONITORS_DIAGRAM.md` (3 sơ đồ Mermaid), phần kiến trúc DistilBERT trong `docs/model.md`, McNemar trong `docs/evaluation.md`, `README.md` (bảng kết quả, hướng dẫn), `docs/plan-progress-status.html`, `.gitignore` | Xong |
| Check | **440/440 test pass** (5 phút 10 giây). Sửa một chỗ sai trong `docs/training.md`: dropout của đầu phân loại là 0,2 (không phải 0,1), lấy từ `config.json` của model | Xong |
| Run | Đưa lên GitHub `Bao-Dev-Collab/E3_GROUP10`, nối tiếp commit "Initial commit" có sẵn. Không đẩy `.venv/`, `outputs/checkpoints/` (1,5 GB) và `outputs/final_model/` (255 MB, vượt giới hạn 100 MB/file của GitHub) | Xem lịch sử commit trên GitHub |
