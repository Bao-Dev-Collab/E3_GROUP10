# Tiến độ dự án

Mỗi bước làm xong sẽ dừng lại để nhóm kiểm tra trước khi sang bước tiếp theo.

| # | Bước | Sản phẩm | Trạng thái |
|---|---|---|---|
| 1 | Khởi tạo cấu trúc thư mục, môi trường, tải dữ liệu | Cây thư mục, `.venv`, `requirements.txt`, `data/raw/` | Xong, nhóm đã duyệt |
| 2 | Bài 1: phân tích cảm xúc | `src/01_sentiment_analysis.py`, test, phần Bài 1 trong `docs/model.md` | Xong, nhóm đã duyệt |
| 3 | Bài 2 – dữ liệu: chia, làm sạch, kiểm tra trùng | `src/lab3/data.py`, test, `docs/data.md` | Xong, nhóm đã duyệt |
| 4 | Bài 2 – baseline TF-IDF + LogisticRegression (5-fold) | `src/lab3/baseline.py`, test, phần A của `docs/training.md` | Xong, nhóm đã duyệt |
| 5 | Bài 2 – fine-tune DistilBERT (chạy thử → chạy đầy đủ) | `src/lab3/train.py`, `--stage train`, test, phần B của `docs/training.md` | Xong (1 giờ 11 phút, validation accuracy 0,9073), nhóm đã duyệt |
| 6 | Bài 2 – đánh giá | `src/lab3/final_evaluation.py`, `src/lab3/plots.py`, `--stage evaluate`, test, `outputs/reports/`, `docs/evaluation.md` | Xong (test accuracy 0,9145), nhóm đã duyệt |
| 7 | Dự đoán câu mới | `src/lab3/predict.py`, `src/03_predict_finetuned.py`, test, `docs/demo.md` | Xong, nhóm đã duyệt |
| 8 | Notebook tổng hợp | `notebooks/Practice3_HuggingFace_Final.ipynb` (đã chạy sẵn kết quả) + bản `.html`, `tests/test_notebook.py` | Xong, chờ nhóm duyệt |
| 9 | Báo cáo và đóng gói | `REPORT.md`, `AI_Usage.md`, `README.md`, `docs/MONITORS_DIAGRAM.md`, bản HTML của file này, file zip | Xong, chờ nhóm duyệt |
