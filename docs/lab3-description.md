# Đề bài Lab 3 – Get started with Hugging Face

## 1. Đề bài gốc (Practice 3)

### Exercise 1: Sentiment Analysis with Hugging Face

1. Install the Hugging Face transformers library.
2. Use a pre-trained sentiment analysis model from the Hugging Face Hub.
3. Tokenize a sample sentence.
4. Perform sentiment analysis on the sentence.

### Exercise 2: Finetuning a Pretrained Model for Binary Text Classification

1. Install the necessary Hugging Face libraries (transformers, datasets, evaluate).
2. Load a simple dataset for binary text classification.
3. Load a pretrained model and its tokenizer.
4. Preprocess the dataset to be suitable for the model.
5. Define training arguments.
6. Create a Trainer object and finetune the model.
7. Evaluate the finetuned model.

### Tài liệu tham khảo của đề

| Link | Nội dung (kiểm tra ngày 09/11/2026) |
|---|---|
| https://huggingface.co/docs/transformers/en/training#fine-tune-a-pretrained-model | Trang đã đổi nội dung: hướng dẫn fine-tune mô hình sinh văn bản (Qwen3), không phải phân loại |
| https://huggingface.co/blog/sentiment-analysis-python | Fine-tune `distilbert-base-uncased` trên IMDb (3.000 mẫu train, 300 mẫu test), accuracy 88%, F1 89% |
| https://www.kaggle.com/code/gauravduttakiit/sentiment-analysis-using-hugging-face | Trang không còn tồn tại |
| https://www.kaggle.com/code/neerajmohan/fine-tuning-bert-for-text-classification | Fine-tune BERT trên bộ Disaster Tweets (cuộc thi Kaggle) |

## 2. Yêu cầu theo cách hiểu của nhóm

- **Bài 1:** dùng model đã train sẵn để phân tích cảm xúc, không train. Phải cho thấy câu được tokenize thành gì (tokens, `input_ids`, `attention_mask`), rồi tự tính logits → softmax → nhãn và đối chiếu với `pipeline()`.
- **Bài 2:** fine-tune một model pretrained cho bài toán phân loại nhị phân, làm đủ 7 bước của đề, rồi đánh giá model sau khi fine-tune.

## 3. Các quyết định của nhóm

| Hạng mục | Lựa chọn | Lý do |
|---|---|---|
| Dataset Bài 2 | IMDb (`stanfordnlp/imdb`) | Nhị phân, cùng chủ đề cảm xúc với Bài 1, là dataset của blog tham khảo, tập test có nhãn (25.000 review) |
| Model Bài 1 | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | Model sentiment phổ biến nhất trên Hub, là model mặc định của `pipeline("sentiment-analysis")` |
| Base model Bài 2 | `distilbert/distilbert-base-uncased` | Giống blog tham khảo, nhẹ hơn BERT, phù hợp máy 8 GB |
| Dữ liệu train | Toàn bộ, không lấy tập con | Yêu cầu của nhóm |
| Thời gian train | 2–3 tiếng trên Apple M1 (8 GB RAM) | Yêu cầu của nhóm. Cấu hình dự kiến: `max_length=256`, 2 epoch |
| Hình thức nộp | Project VS Code (file `.py` + file `.md`), nén zip | Yêu cầu của giảng viên |

Quy tắc khoa học áp dụng cho toàn bộ bài: xem [RULES.md](../RULES.md).
