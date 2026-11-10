# Mô hình

## Bài 1 – Model phân tích cảm xúc đã train sẵn

### 1. Model sử dụng

| Hạng mục | Thông tin |
|---|---|
| Id trên Hub | [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) |
| Kiến trúc | `DistilBertForSequenceClassification` |
| Số tham số | 66.955.010 |
| Nhãn | `0` = NEGATIVE, `1` = POSITIVE |
| Độ dài tối đa | 512 token mỗi câu |
| Nguồn gốc | `distilbert-base-uncased` đã được fine-tune sẵn trên SST-2 (Stanford Sentiment Treebank, câu review phim, 2 nhãn) |
| "uncased" | Tokenizer đổi mọi chữ thành chữ thường, nên "GREAT" và "great" cho ra cùng token |

Lý do chọn: đây là model mặc định của `pipeline("sentiment-analysis")`, cũng là model mà bài blog tham khảo của đề dùng, và cùng họ DistilBERT với model sẽ fine-tune ở Bài 2.

### 2. Luồng xử lý theo 4 bước của đề

| Bước của đề | Làm gì | Hàm trong code |
|---|---|---|
| 1. Cài transformers | Cài qua `requirements.txt`; script in ra phiên bản đang dùng | — |
| 2. Dùng model đã train sẵn | Tải tokenizer và model từ Hub, đặt chế độ `eval` | `load_sentiment_model` |
| 3. Tokenize câu mẫu | Câu → token → `input_ids`, `attention_mask` | `tokenize_texts` |
| 4. Phân tích cảm xúc | `input_ids` → model → logits → softmax → nhãn. Đối chiếu với `pipeline()` | `predict_sentiment`, `predict_with_pipeline` |

Code nằm trong [src/lab3/sentiment.py](../src/lab3/sentiment.py). Script chạy là [src/01_sentiment_analysis.py](../src/01_sentiment_analysis.py).

### 3. Tokenization câu mẫu (kết quả thật)

Câu: *"I absolutely loved this movie, the cinematography was breathtaking!"*

| Vị trí | Token | input_id | attention_mask |
|---|---|---|---|
| 0 | `[CLS]` | 101 | 1 |
| 1 | `i` | 1045 | 1 |
| 2 | `absolutely` | 7078 | 1 |
| 3 | `loved` | 3866 | 1 |
| 4 | `this` | 2023 | 1 |
| 5 | `movie` | 3185 | 1 |
| 6 | `,` | 1010 | 1 |
| 7 | `the` | 1996 | 1 |
| 8 | `cinematography` | 16434 | 1 |
| 9 | `was` | 2001 | 1 |
| 10 | `breath` | 3052 | 1 |
| 11 | `##taking` | 17904 | 1 |
| 12 | `!` | 999 | 1 |
| 13 | `[SEP]` | 102 | 1 |

Nhận xét:
- `[CLS]` luôn đứng đầu câu. Lớp phân loại dùng vector tại vị trí này để quyết định nhãn.
- `[SEP]` đánh dấu hết câu.
- "breathtaking" không có sẵn trong từ điển nên bị tách thành `breath` + `##taking`. Dấu `##` nghĩa là mảnh này nối vào từ đứng trước (cách tách WordPiece).
- `attention_mask` = 1 là token thật. Khi nhiều câu chạy chung một batch, câu ngắn được thêm `[PAD]` với `attention_mask` = 0 để model bỏ qua.

### 4. Từ logits ra nhãn

Model trả về 2 số (logits), mỗi nhãn một số. Softmax đổi logits thành xác suất:

$$p_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$

Trừ `max(z)` không làm đổi kết quả nhưng tránh tràn số khi logits lớn. Code tự cài đặt công thức này (hàm `softmax`) thay vì gọi sẵn của PyTorch, và test kiểm tra kết quả khớp với `torch.softmax`.

Ví dụ với câu mẫu: logits = [-4,3622; 4,7191], suy ra P(POSITIVE) = 1 / (1 + e^(−9,0813)) ≈ 0,9999.

### 5. Kết quả 3 câu mẫu

| Câu | Logits [NEG, POS] | Nhãn | Độ tin cậy |
|---|---|---|---|
| I absolutely loved this movie, the cinematography was breathtaking! | [-4,3622; 4,7191] | POSITIVE | 0,9999 |
| The plot was predictable and the acting felt painfully wooden. | [4,7510; -3,8116] | NEGATIVE | 0,9998 |
| The film was not bad at all. | [-2,9476; 3,0546] | POSITIVE | 0,9975 |

Câu thứ ba có chữ "bad" nhưng model vẫn hiểu đúng là tích cực nhờ cụm phủ định "not bad at all". Độ tin cậy thấp hơn hai câu đầu.

**Đối chiếu với `pipeline("sentiment-analysis")`:** nhãn trùng nhau ở cả 3 câu, sai lệch xác suất lớn nhất là 1,9 × 10⁻¹⁰ (ngưỡng cho phép 10⁻⁴). Như vậy cách tự làm từng bước cho ra đúng kết quả của pipeline.

Kết quả đầy đủ được lưu ở `outputs/reports/exercise1_sentiment.json`.

### 6. Lưu ý kỹ thuật: lỗi crash trên CPU

- **Hiện tượng:** chạy model trên CPU với 1 câu thì Python bị tắt đột ngột (Bus error, mã thoát 138). Chạy 2 câu trở lên, hoặc chạy trên GPU của M1 (MPS), thì không bị.
- **Nguyên nhân gốc rễ:** trọng số bị **lệch căn hàng bộ nhớ**. Chuỗi nguyên nhân:
  1. Trong file `model.safetensors` của model này, phần dữ liệu bắt đầu ở byte 12.518. Số này không chia hết cho 4, trong khi mỗi số float32 chiếm 4 byte.
  2. transformers 5 trên macOS/Linux mặc định **ánh xạ thẳng file vào bộ nhớ** (memory-map) để không phải chép. Vì vậy mọi tensor trọng số nằm ở địa chỉ chia 4 dư 2.
  3. Khi tính cho 1 câu, PyTorch trên CPU ARM64 dùng phép nhân ma trận với vector. Phép này đòi hỏi dữ liệu căn hàng đúng, gặp địa chỉ lệch thì CPU báo lỗi căn hàng (Bus error). Với 2 câu trở lên, PyTorch dùng phép nhân ma trận với ma trận, phép này chịu được địa chỉ lệch. Trên MPS, trọng số được chép sang bộ nhớ GPU nên địa chỉ tự căn hàng.
- **Đã loại trừ:** đặt trọng số căn hàng đúng trong vùng nhớ *chỉ đọc* thì chạy bình thường. Ngược lại, đặt trọng số lệch căn hàng trong vùng nhớ *ghi được* vẫn crash. Vậy nguyên nhân là lệch căn hàng, không phải bộ nhớ chỉ đọc.
- **Cách sửa:** nạp model với tham số chính thức `disable_mmap=True` của transformers. File được đọc hẳn vào RAM, PyTorch tự cấp phát vùng nhớ căn hàng đúng. Không chép lại trọng số sau khi nạp.
- **Kiểm chứng:**
  - `test_load_on_cpu_gives_aligned_weights` kiểm tra không còn tensor nào lệch căn hàng
  - `test_script_on_cpu_single_sentence` chạy CPU với 1 câu trong tiến trình riêng, để nếu lỗi tái phát thì chỉ test đó fail chứ không làm sập cả phiên test

## Bài 2 – Base model để fine-tune

### 1. DistilBERT là gì

`distilbert/distilbert-base-uncased` là phiên bản thu gọn của BERT-base, tạo bằng **chưng cất tri thức** (knowledge distillation; Sanh và cộng sự, 2019):
- Một model nhỏ ("học trò", 6 lớp) được train để bắt chước đầu ra của BERT-base ("thầy", 12 lớp), thay vì chỉ học từ nhãn.
- Kết quả: nhỏ hơn BERT-base khoảng 40%, chạy nhanh hơn khoảng 60%, giữ được khoảng 97% khả năng hiểu ngôn ngữ (đo trên bộ GLUE).

Nhỏ và nhanh như vậy nên DistilBERT phù hợp để fine-tune trên máy M1 8 GB RAM.

| Thông số (từ `outputs/final_model/config.json`) | Giá trị |
|---|---|
| Số lớp Transformer | 6 |
| Kích thước vector ẩn | 768 |
| Số attention head | 12 |
| Kích thước lớp feed-forward | 3.072, hàm kích hoạt GELU |
| Từ điển | 30.522 token WordPiece, không phân biệt hoa thường |
| Độ dài tối đa | 512 token (nhóm dùng 256, xem [training.md](training.md)) |
| Dropout | 0,1 trong các lớp Transformer và attention; 0,2 ở đầu phân loại |
| Tổng số tham số (có đầu phân loại) | 66.955.010 |

### 2. Đầu phân loại

```text
review -> tokenizer -> [CLS] t1 t2 ... [SEP] -> 6 lớp Transformer -> vector của [CLS] (768)
       -> Linear 768→768 -> ReLU -> Dropout 0,2 -> Linear 768→2 -> logits [NEGATIVE, POSITIVE]
```

Hai lớp `Linear` cuối (`pre_classifier`, `classifier`) **không có trong model gốc**, nên được khởi tạo ngẫu nhiên. Khi tải model, transformers in cảnh báo "newly initialized", và đó chính là phần cần train. Khi fine-tune, **toàn bộ** model được cập nhật (không đóng băng lớp nào), với learning rate nhỏ (2e-5) để không phá kiến thức có sẵn.

### 3. Khác gì với model của Bài 1

| | Bài 1 | Bài 2 |
|---|---|---|
| Model | `distilbert-base-uncased-finetuned-sst-2-english` | `distilbert-base-uncased` |
| Đã biết phân loại cảm xúc chưa | Rồi (fine-tune sẵn trên SST-2: các câu ngắn trích từ review phim) | Chưa, đầu phân loại khởi tạo ngẫu nhiên |
| Nhóm làm gì | Chỉ dùng để dự đoán | Fine-tune trên 22.297 review IMDb |

Bài 2 cố ý bắt đầu từ model **chưa** biết phân loại cảm xúc, để kết quả thể hiện đúng hiệu quả của việc fine-tune.
