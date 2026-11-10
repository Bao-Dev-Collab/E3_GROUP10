"""Bài 1: phân tích cảm xúc bằng model đã train sẵn trên Hugging Face Hub.

Các bước theo đề:
1. Cài thư viện transformers (xem requirements.txt).
2. Dùng một model phân tích cảm xúc đã train sẵn trên Hub.
3. Tokenize một câu mẫu.
4. Phân tích cảm xúc của câu: tự tính logits -> softmax -> nhãn, rồi đối chiếu với pipeline().

Cách chạy (từ thư mục lab3_DL/):
    python src/01_sentiment_analysis.py
    python src/01_sentiment_analysis.py --text "This movie is great" --text "I hated it"
    python src/01_sentiment_analysis.py --device cpu
"""

import argparse
import sys
from pathlib import Path

# Cho phép chạy trực tiếp file này mà vẫn import được package lab3 trong src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import transformers  # noqa: E402

from lab3 import config  # noqa: E402
from lab3.sentiment import (  # noqa: E402
    load_sentiment_model,
    max_probability_difference,
    predict_sentiment,
    predict_with_pipeline,
    tokenize_texts,
    validate_texts,
)
from lab3.utils import get_device, save_json  # noqa: E402

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_INPUT_ERROR = 2


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh.

    Giả định: argv = None nghĩa là đọc từ sys.argv; argparse tự báo lỗi (mã thoát 2) khi sai kiểu.
    """
    parser = argparse.ArgumentParser(description="Bài 1: phân tích cảm xúc bằng model đã train sẵn.")
    parser.add_argument(
        "--text",
        action="append",
        help="Câu cần phân tích. Lặp lại để truyền nhiều câu. Mặc định: các câu mẫu trong config.",
    )
    parser.add_argument("--model", default=config.SENTIMENT_MODEL_NAME, help="Id model trên Hub hoặc thư mục local.")
    parser.add_argument("--max-length", type=int, default=config.EXERCISE1_MAX_LENGTH, help="Số token tối đa mỗi câu.")
    parser.add_argument("--device", default=None, help="cuda, mps hoặc cpu. Mặc định: tự chọn.")
    parser.add_argument("--output", type=Path, default=config.EXERCISE1_REPORT_PATH, help="File JSON lưu kết quả.")
    return parser.parse_args(argv)


def print_header(title: str) -> None:
    """In tiêu đề một bước. Giả định: title là chuỗi ngắn trên một dòng."""
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def describe_tokenization(tokenized: dict) -> dict:
    """Lấy chi tiết tokenization của câu đầu tiên để in và lưu báo cáo.

    Giả định: tokenized là kết quả của tokenize_texts; chỉ phân tích câu đầu tiên.
    """
    mask = tokenized["encoding"]["attention_mask"][0]
    ids = tokenized["encoding"]["input_ids"][0][mask.bool()]
    return {
        "text": tokenized["texts"][0],
        "tokens": tokenized["tokens"][0],
        "input_ids": [int(value) for value in ids],
        "attention_mask": [int(value) for value in mask[mask.bool()]],
    }


def print_tokenization(detail: dict) -> None:
    """In bảng token / input_id / attention_mask. Giả định: detail do describe_tokenization tạo."""
    print(f'Câu mẫu: "{detail["text"]}"\n')
    print(f"{'Vị trí':>6}  {'Token':<20}{'input_id':>9}  {'attention_mask':>14}")
    for position, (token, token_id, mask) in enumerate(
        zip(detail["tokens"], detail["input_ids"], detail["attention_mask"])
    ):
        print(f"{position:>6}  {token:<20}{token_id:>9}  {mask:>14}")
    print(f"\nTổng số token: {len(detail['tokens'])} (đã gồm token đặc biệt ở đầu và cuối câu).")
    subwords = [token for token in detail["tokens"] if token.startswith("##")]
    if subwords:
        print(f"Token bắt đầu bằng '##' là mảnh nối vào từ đứng trước (WordPiece): {subwords}")


def print_predictions(title: str, predictions: list[dict]) -> None:
    """In nhãn, độ tin cậy và xác suất của từng câu. Giả định: mỗi phần tử có text, label, probabilities."""
    print(f"\n{title}")
    for index, prediction in enumerate(predictions, start=1):
        probabilities = ", ".join(f"{label}={value:.4f}" for label, value in prediction["probabilities"].items())
        print(f'  {index}. "{prediction["text"]}"')
        if "logits" in prediction:
            logits = ", ".join(f"{value:.4f}" for value in prediction["logits"])
            print(f"     logits = [{logits}]")
        print(f"     -> {prediction['label']} (độ tin cậy {prediction['score']:.4f}) | {probabilities}")


def main(argv: list[str] | None = None) -> int:
    """Chạy Bài 1 và lưu kết quả ra JSON.

    Giả định:
    - Lỗi do đầu vào (câu rỗng, max_length sai, model không tồn tại...) được in rõ ra stderr, trả mã 2.
    - Nếu cách tự tính và pipeline() cho kết quả lệch quá ngưỡng, in cảnh báo và trả mã 1.
    """
    args = parse_args(argv)
    try:
        texts = validate_texts(args.text if args.text else list(config.EXERCISE1_SAMPLE_TEXTS))
        device = get_device(args.device)

        print_header("Bước 1 – Thư viện")
        print(f"transformers {transformers.__version__} | torch {torch.__version__} | thiết bị: {device}")

        print_header("Bước 2 – Tải model đã train sẵn")
        tokenizer, model = load_sentiment_model(args.model, device)
        num_parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"Model: {args.model}")
        print(f"Kiến trúc: {type(model).__name__} | số tham số: {num_parameters:,}")
        print(f"Nhãn: {model.config.id2label} | tối đa {tokenizer.model_max_length} token mỗi câu")

        print_header("Bước 3 – Tokenize câu mẫu")
        detail = describe_tokenization(tokenize_texts(tokenizer, texts[0], args.max_length))
        print_tokenization(detail)

        print_header("Bước 4 – Phân tích cảm xúc")
        manual = predict_sentiment(model, tokenizer, texts, args.max_length)
        print_predictions("Cách tự làm (tokenizer -> model -> logits -> softmax tự cài đặt):", manual)
        piped = predict_with_pipeline(model, tokenizer, texts, args.max_length)
        print_predictions('Cách dùng pipeline("sentiment-analysis"):', piped)

        difference = max_probability_difference(manual, piped)
        labels_match = all(left["label"] == right["label"] for left, right in zip(manual, piped))
        consistent = labels_match and difference <= config.EXERCISE1_PROBABILITY_TOLERANCE
        print(f"\nSai lệch xác suất lớn nhất giữa 2 cách: {difference:.2e} (ngưỡng {config.EXERCISE1_PROBABILITY_TOLERANCE:.0e})")
        print(f"Nhãn 2 cách trùng nhau: {'có' if labels_match else 'KHÔNG'}")

        report = {
            "model": args.model,
            "device": str(device),
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "max_length": args.max_length,
            "model_info": {
                "architecture": type(model).__name__,
                "num_parameters": num_parameters,
                "id2label": model.config.id2label,
                "model_max_length": tokenizer.model_max_length,
            },
            "tokenization_example": detail,
            "predictions_manual": manual,
            "predictions_pipeline": piped,
            "max_probability_difference": difference,
            "labels_match": labels_match,
            "tolerance": config.EXERCISE1_PROBABILITY_TOLERANCE,
        }
        saved = save_json(report, args.output)
        print(f"\nĐã lưu kết quả: {saved}")
    except (TypeError, ValueError, FileNotFoundError, IsADirectoryError, RuntimeError) as error:
        print(f"Lỗi: {error}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    if not consistent:
        print("CẢNH BÁO: hai cách cho kết quả lệch nhau quá ngưỡng.", file=sys.stderr)
        return EXIT_MISMATCH
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
