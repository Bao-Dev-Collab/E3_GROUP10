# Dữ liệu

Tài liệu này ghi lại data lấy từ đâu và đã xử lý thế nào. Mục 1–5 làm ở bước 1 (tải data). Mục 6–9 làm ở bước 3 (chia, làm sạch, loại trùng, thống kê).

## 1. Nguồn dữ liệu

| Hạng mục | Thông tin |
|---|---|
| Tên | Large Movie Review Dataset (IMDb) |
| Đơn vị công bố | Đại học Stanford |
| Trang gốc | https://ai.stanford.edu/~amaas/data/sentiment/ (file `aclImdb_v1.tar.gz`) |
| Bản nhóm dùng | Hugging Face Hub, dataset [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb), cấu hình `plain_text` |
| Liên hệ với đề | Bài blog tham khảo của đề (https://huggingface.co/blog/sentiment-analysis-python) dùng `load_dataset("imdb")`, trỏ về đúng dataset này |
| Nhiệm vụ | Phân loại cảm xúc nhị phân của review phim: `0` = tiêu cực (neg), `1` = tích cực (pos) |

**Trích dẫn:**

> Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning Word Vectors for Sentiment Analysis*. Proceedings of ACL 2011.

Dataset trên Hub có 3 tập. Nhóm chỉ lấy 2 tập có nhãn:

| Tập | Số review | Có nhãn | Nhóm dùng |
|---|---|---|---|
| `train` | 25.000 | Có | Có |
| `test` | 25.000 | Có | Có |
| `unsupervised` | 50.000 | Không | Không. Bài 2 là học có giám sát nên không cần |

## 2. Cách tải (ngày 09/11/2026)

Tải bản parquet mà Hugging Face tự chuyển đổi từ dataset gốc (nhánh `refs/convert/parquet`):

| Tập | Link | Dung lượng |
|---|---|---|
| train | https://huggingface.co/datasets/stanfordnlp/imdb/resolve/refs%2Fconvert%2Fparquet/plain_text/train/0000.parquet | 20.979.968 byte |
| test | https://huggingface.co/datasets/stanfordnlp/imdb/resolve/refs%2Fconvert%2Fparquet/plain_text/test/0000.parquet | 20.470.363 byte |

Lệnh đã chạy (từ thư mục `lab3_DL/`):

```bash
mkdir -p tmp_parquet
BASE="https://huggingface.co/datasets/stanfordnlp/imdb/resolve/refs%2Fconvert%2Fparquet/plain_text"
curl -L -o tmp_parquet/train.parquet "$BASE/train/0000.parquet"
curl -L -o tmp_parquet/test.parquet  "$BASE/test/0000.parquet"
```

File parquet chỉ là file tạm, không giữ trong project.

## 3. Chuyển sang CSV

Lý do chọn CSV: mở xem trực tiếp được trong VS Code hoặc Excel, đọc được bằng pandas trên mọi hệ điều hành.

```python
import pandas as pd

for split in ["train", "test"]:
    df = pd.read_parquet(f"tmp_parquet/{split}.parquet")
    df.to_csv(f"data/raw/imdb_{split}.csv", index=False, lineterminator="\n")
```

`lineterminator="\n"` bắt buộc phải có. Nếu bỏ, pandas xuống dòng theo hệ điều hành (Windows dùng `\r\n`), nên file tạo trên Windows sẽ khác từng byte và mã SHA-256 bên dưới không còn khớp. File trong project được tạo trên macOS, nơi mặc định đã là `\n`, nên không bị ảnh hưởng.

Bước này **chỉ đổi định dạng file**. Không sửa, xoá hay sắp xếp lại bất kỳ dòng nào. Kết quả:

| File | Số dòng | Cột | Dung lượng |
|---|---|---|---|
| `data/raw/imdb_train.csv` | 25.000 | `text`, `label` | 33.322.167 byte |
| `data/raw/imdb_test.csv` | 25.000 | `text`, `label` | 32.540.147 byte |

## 4. Kiểm tra sau khi tải

| Kiểm tra | train | test |
|---|---|---|
| Số dòng | 25.000 | 25.000 |
| Nhãn 0 / nhãn 1 | 12.500 / 12.500 | 12.500 / 12.500 |
| Review thiếu (`NaN`) | 0 | 0 |
| Review rỗng hoặc chỉ có khoảng trắng | 0 | 0 |
| Đọc lại CSV ra đúng y dữ liệu parquet | Đúng | Đúng |
| Chạy lại bước chuyển đổi ra file giống hệt | Đúng | Đúng |

**Mã SHA-256** để kiểm tra bản data của mỗi thành viên có giống nhau không:

```text
0ee7df69245ca053db679b6c092801cc46d87ba4406798c9c7e09f3f292c6fa7  data/raw/imdb_train.csv
b0dc66ba545c6c48b55b138f96dfb2beccbcc0bb99abd24fd2ec83110badb08a  data/raw/imdb_test.csv
```

Cách kiểm tra: `shasum -a 256 data/raw/*.csv` (macOS/Linux) hoặc `certutil -hashfile data\raw\imdb_train.csv SHA256` (Windows).

## 5. Đặc điểm data cần lưu ý khi xử lý

| Đặc điểm | Bằng chứng | Hệ quả |
|---|---|---|
| **Các dòng đã được sắp xếp theo nhãn** | Ở cả train và test, 12.500 dòng đầu toàn nhãn 0, 12.500 dòng sau toàn nhãn 1 | Không được cắt một đoạn liên tiếp để làm validation. Phải xáo trộn và chia có stratify |
| Review có thẻ HTML `<br />` | 61/100 review đầu của tập train | Cần làm sạch |
| Review dài | 100 review đầu của tập train: trung bình khoảng 242 từ, dài nhất 888 từ | Phải cắt bớt khi tokenize; chọn `max_length` theo thống kê |
| Hai nhãn cân bằng | 50% / 50% ở cả hai tập | Accuracy là thước đo phù hợp, vẫn báo cáo thêm F1 |

## 6. Quy trình xử lý (bước 3)

Code: [src/lab3/data.py](../src/lab3/data.py). Chạy bằng lệnh:

```bash
python src/02_train_finetune.py --stage data
```

Thứ tự xử lý tuân theo [RULES.md](../RULES.md): **chia dữ liệu trước mọi bước xử lý khác**.

| # | Bước | Cách làm | Hàm |
|---|---|---|---|
| 1 | Đọc và kiểm tra | Kiểm tra đủ cột, không có review rỗng, nhãn là số nguyên thuộc {0, 1}. Các chuỗi như "NA" hay "null" được giữ nguyên là review, không bị pandas đổi thành giá trị thiếu | `load_raw_csv` |
| 2 | **Chia dữ liệu** | `imdb_test.csv` giữ nguyên làm **test**. `imdb_train.csv` được xáo trộn rồi chia có stratify 90/10 (seed 42) thành **train** và **validation**. Cả hai tập phải có đủ 2 lớp | `split_train_validation` |
| 3 | Kiểm tra nhãn | Nhãn của validation và test phải đều đã có trong train | `prepare_data` |
| 4 | Làm sạch | Thay `<br />` (và các biến thể `<br>`, `<br/>`, `<BR>`) bằng khoảng trắng, gộp khoảng trắng thừa. Hàm xử lý từng review riêng lẻ, không học gì từ dữ liệu. Các thẻ HTML khác rất hiếm nên giữ nguyên | `clean_text` |
| 5 | Loại trùng | Hai review được coi là trùng nếu giống nhau sau khi làm sạch và đổi hết về chữ thường (model không phân biệt hoa/thường). Thứ tự ưu tiên giữ lại: test > validation > train | `remove_duplicates` |
| 6 | Thống kê độ dài token | **Chỉ trên train**, dùng tokenizer của `distilbert/distilbert-base-uncased` | `token_length_stats` |
| 7 | Lưu | `data/splits/train.csv`, `validation.csv`, `test.csv` gồm các cột `source_split`, `source_index` (để truy lại dòng gốc), `text` (đã làm sạch), `label` | `save_split` |

Báo cáo đầy đủ: `outputs/reports/data_report.json`.

## 7. Kết quả

**Loại trùng:**

| Loại | Số dòng |
|---|---|
| Train trùng với test, bỏ khỏi train | 107 |
| Validation trùng với test, bỏ khỏi validation | 16 |
| Train trùng với validation, bỏ khỏi train | 20 |
| Trùng trong nội bộ train, giữ bản đầu | 76 |
| Trùng trong nội bộ validation, giữ bản đầu | 2 |
| Nhóm trùng có nhãn mâu thuẫn | 0 |
| Trùng trong nội bộ test (giữ nguyên, chỉ đếm) | 201 |

Tổng cộng 123 review (107 + 16) xuất hiện ở cả tập train gốc và tập test. Nếu không loại, model sẽ được chấm điểm trên chính những review nó đã học, và điểm test sẽ cao hơn thực tế.

**Các tập sau xử lý:**

| Tập | Số review | Nhãn 0 | Nhãn 1 |
|---|---|---|---|
| train | 22.297 | 11.100 | 11.197 |
| validation | 2.482 | 1.236 | 1.246 |
| test | 25.000 | 12.500 | 12.500 |

Mã SHA-256:

```text
b65b83e242b44a4df9bac95cdf27fb112186759816dfb4b309574f8391d2de5b  data/splits/train.csv
ad342b1d210bf9de5474bae1ba282b5fb69f1adba716a36970505f1d33a5bc4f  data/splits/validation.csv
a81e7391d7a66ed215a577641e434570686178d21fa52998f466de619ef78396  data/splits/test.csv
```

Chạy lại lệnh ở mục 6 sẽ ra đúng 3 file này (cùng seed, cùng SHA-256, trên mọi hệ điều hành).

## 8. Độ dài token và lựa chọn `max_length`

Thống kê trên 22.297 review của tập train (tính cả `[CLS]` và `[SEP]`):

| Chỉ số | Số token |
|---|---|
| Trung bình | 298 |
| Trung vị | 222 |
| 90% review ngắn hơn | 584 |
| 95% review ngắn hơn | 762 |
| Dài nhất | 3.047 |

| `max_length` | Tỉ lệ review bị cắt bớt |
|---|---|
| 128 | 87,9% |
| 256 | 41,2% |
| 512 (giới hạn của DistilBERT) | 13,4% |

Nhận xét:
- Với 256, hơn một nửa số review (58,8%) được đọc trọn. Những review bị cắt vẫn giữ được 256 token đầu.
- 512 giảm số review bị cắt xuống 13,4%, nhưng thời gian train tăng khoảng 2–3 lần.
- Lựa chọn cuối cùng sẽ được đưa ra ở bước 5, sau khi chạy thử đo tốc độ thật trên máy, để vừa giới hạn 2–3 tiếng.

## 9. Hạn chế đã biết

- **Có thể cùng một bộ phim xuất hiện ở cả train và validation.** Bộ IMDb gốc đảm bảo train và test không chung phim nào. Nhưng bản trên Hugging Face không có mã phim, nên validation được chia ngẫu nhiên. Vì vậy điểm validation có thể cao hơn test một chút (model có thể dựa vào tên diễn viên hay tên phim). Tập test vẫn sạch, nên kết quả cuối cùng báo cáo trên test.
- **Tập test còn 201 bản trùng trong nội bộ.** Nhóm giữ nguyên để kết quả so sánh được với các công trình khác dùng IMDb.
