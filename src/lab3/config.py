"""Cấu hình dùng chung cho toàn bộ Lab 3.

Mọi đường dẫn đều tính từ thư mục gốc của project (lab3_DL/), nên chạy script
từ thư mục nào hay chạy trong notebook cũng ghi ra cùng một chỗ.
"""

from pathlib import Path

# src/lab3/config.py -> parents[0] = src/lab3, parents[1] = src, parents[2] = lab3_DL
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
FINAL_MODEL_DIR = OUTPUTS_DIR / "final_model"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# ---------------------------------------------------------------------------
# Bài 1: phân tích cảm xúc bằng model đã train sẵn
# ---------------------------------------------------------------------------
SENTIMENT_MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# Câu mẫu mặc định khi không truyền --text. Câu đầu tiên được phân tích tokenization chi tiết.
EXERCISE1_SAMPLE_TEXTS = (
    "I absolutely loved this movie, the cinematography was breathtaking!",
    "The plot was predictable and the acting felt painfully wooden.",
    "The film was not bad at all.",
)

# DistilBERT nhận tối đa 512 token mỗi câu.
EXERCISE1_MAX_LENGTH = 512

# Sai lệch xác suất tối đa cho phép giữa cách tự tính và pipeline().
EXERCISE1_PROBABILITY_TOLERANCE = 1e-4

EXERCISE1_REPORT_PATH = REPORTS_DIR / "exercise1_sentiment.json"

# ---------------------------------------------------------------------------
# Bài 2: dữ liệu
# ---------------------------------------------------------------------------
SEED = 42

BASE_MODEL_NAME = "distilbert/distilbert-base-uncased"

# Nhãn của IMDb: 0 = tiêu cực, 1 = tích cực.
ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}

RAW_TRAIN_PATH = RAW_DATA_DIR / "imdb_train.csv"
RAW_TEST_PATH = RAW_DATA_DIR / "imdb_test.csv"

# Tỉ lệ tách validation từ tập train gốc (25.000 x 0.1 = 2.500 review).
VAL_SIZE = 0.1

# Các mốc max_length để thống kê tỉ lệ review bị cắt (chỉ tính trên tập train).
TOKEN_LENGTH_THRESHOLDS = (128, 256, 512)

DATA_REPORT_PATH = REPORTS_DIR / "data_report.json"

TRAIN_SPLIT_PATH = SPLITS_DIR / "train.csv"
VALIDATION_SPLIT_PATH = SPLITS_DIR / "validation.csv"
TEST_SPLIT_PATH = SPLITS_DIR / "test.csv"

# Train cao hơn validation quá mức này (tính theo accuracy, thang 0–1) thì cảnh báo overfitting.
OVERFIT_GAP_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Bài 2: baseline TF-IDF + LogisticRegression
# ---------------------------------------------------------------------------
CV_FOLDS = 5

# Lưới tham số chọn bằng cross-validation trên tập train (không dùng validation/test để chọn).
BASELINE_PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.25, 1.0, 4.0, 16.0],
}

# Giới hạn số từ / cụm từ của TF-IDF để file model nhỏ gọn khi nén nộp bài.
BASELINE_MAX_FEATURES = 100_000

# Số tiến trình chạy song song khi cross-validation (máy 8 GB RAM nên để 2).
BASELINE_N_JOBS = 2

BASELINE_MODEL_PATH = OUTPUTS_DIR / "baseline" / "tfidf_logreg.joblib"
BASELINE_REPORT_PATH = REPORTS_DIR / "baseline_report.json"

# ---------------------------------------------------------------------------
# Bài 2: fine-tune DistilBERT
# ---------------------------------------------------------------------------
TRAIN_MAX_LENGTH = 256          # 58,8% review train đọc trọn (xem docs/data.md mục 8)
NUM_EPOCHS = 2
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1
WEIGHT_DECAY = 0.01             # regularization L2 kiểu AdamW
WARMUP_RATIO = 0.1              # 10% số bước đầu tăng dần learning rate
MAX_GRAD_NORM = 1.0             # gradient clipping
EVALS_PER_EPOCH = 2             # đánh giá validation mỗi nửa epoch để early stopping có tác dụng
EARLY_STOPPING_PATIENCE = 2     # dừng nếu 2 lần đánh giá liên tiếp không cải thiện
METRIC_FOR_BEST_MODEL = "accuracy"
LOGGING_STEPS = 50
SAVE_TOTAL_LIMIT = 2            # giữ checkpoint tốt nhất + mới nhất

# Chạy thử (--smoke): train vài bước trên mẫu nhỏ để bắt lỗi và đo tốc độ, không ghi đè model thật.
SMOKE_TRAIN_SAMPLES = 480
SMOKE_VALIDATION_SAMPLES = 256
SMOKE_MAX_STEPS = 30
SMOKE_WARMUP_STEPS_IGNORED = 5  # bỏ qua vài bước đầu (khởi động) khi đo tốc độ
TIME_BUDGET_HOURS = 3.0

TRAIN_REPORT_PATH = REPORTS_DIR / "train_report.json"
SMOKE_REPORT_PATH = REPORTS_DIR / "smoke_report.json"

# ---------------------------------------------------------------------------
# Bài 2: đánh giá cuối trên tập test (chạy một lần, sau khi đã chốt model)
# ---------------------------------------------------------------------------
EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_report.json"
CONFUSION_MATRIX_FILENAME = "confusion_matrices.png"
TRAINING_CURVES_FILENAME = "training_curves.png"
TEST_PREDICTIONS_FILENAME = "test_predictions.csv"
ERROR_EXAMPLES_PER_TYPE = 5     # số ví dụ đoán sai tự tin nhất cho mỗi loại lỗi
ERROR_PREVIEW_CHARS = 300       # chỉ trích phần đầu review trong báo cáo

# ---------------------------------------------------------------------------
# Dự đoán câu mới bằng model đã fine-tune (03_predict_finetuned.py)
# ---------------------------------------------------------------------------
# Câu mẫu khi không truyền --text / --file: có câu rõ ràng, câu phủ định, câu mỉa mai, câu khen chê lẫn lộn.
DEMO_TEXTS = (
    "One of the best films I have seen in years. The acting and the soundtrack are wonderful.",
    "A complete waste of two hours. The plot makes no sense and the dialogue is painful.",
    "It's not a bad movie at all, I actually enjoyed it more than I expected.",
    "Oh great, another sequel nobody asked for. Just what we needed.",
    "The first half is slow and boring, but the ending is absolutely brilliant.",
)
PREDICT_BATCH_SIZE = 32
