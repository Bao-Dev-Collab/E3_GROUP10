"""Dự đoán cảm xúc (tích cực / tiêu cực) cho review mới bằng model DistilBERT đã fine-tune ở Bài 2.

Cách chạy (từ thư mục lab3_DL/, sau khi đã chạy --stage train):
    python src/03_predict_finetuned.py                                   # các câu mẫu có sẵn
    python src/03_predict_finetuned.py --text "Great movie!" --text "So boring."
    python src/03_predict_finetuned.py --file reviews.txt                # mỗi dòng một review
    python src/03_predict_finetuned.py --interactive                     # gõ từng câu, Enter để dự đoán
"""

import argparse
import sys
from pathlib import Path

# Cho phép chạy trực tiếp file này mà vẫn import được package lab3 trong src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab3 import config  # noqa: E402
from lab3.predict import predict_texts, read_texts_file, resolve_max_length  # noqa: E402
from lab3.sentiment import load_sentiment_model  # noqa: E402
from lab3.utils import get_device, save_json  # noqa: E402

EXIT_OK = 0
EXIT_INPUT_ERROR = 2


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh. Giả định: argv = None nghĩa là đọc từ sys.argv."""
    parser = argparse.ArgumentParser(description="Dự đoán cảm xúc review bằng model DistilBERT đã fine-tune.")
    parser.add_argument("--text", action="append", help="Review cần dự đoán. Lặp lại để truyền nhiều câu.")
    parser.add_argument("--file", type=Path, help="File .txt, mỗi dòng một review.")
    parser.add_argument("--interactive", action="store_true", help="Gõ từng review; dòng trống để thoát.")
    parser.add_argument("--model-dir", type=Path, default=config.FINAL_MODEL_DIR, help="Thư mục model đã fine-tune.")
    parser.add_argument("--train-report", type=Path, default=config.TRAIN_REPORT_PATH, help="Lấy max_length lúc train.")
    parser.add_argument("--max-length", type=int, default=None, help="Ghi đè max_length (mặc định: như lúc train).")
    parser.add_argument("--device", default=None, help="cuda, mps hoặc cpu. Mặc định: tự chọn.")
    parser.add_argument("--output", type=Path, default=None, help="Lưu kết quả ra file JSON (tuỳ chọn).")
    return parser.parse_args(argv)


def print_results(results: list[dict], max_length: int, start: int = 1) -> None:
    """In bảng kết quả. Giả định: results do predict_texts tạo; start là số thứ tự của kết quả đầu tiên."""
    for index, result in enumerate(results, start=start):
        note = f", bị cắt còn {max_length}/{result['num_tokens']} token" if result["truncated"] else ""
        preview = result["text"] if len(result["text"]) <= 100 else result["text"][:100] + "…"
        print(f'{index}. {result["label"]:<8} (độ tin cậy {result["confidence"]:.1%}{note})')
        print(f'   "{preview}"')


def interactive_loop(model, tokenizer, max_length: int) -> list[dict]:
    """Đọc từng dòng từ bàn phím (stdin) và dự đoán ngay.

    Giả định: dòng trống hoặc hết dữ liệu vào (Ctrl+D) thì dừng.
    """
    print("Gõ một review tiếng Anh rồi nhấn Enter (dòng trống để thoát):")
    collected = []
    for line in sys.stdin:
        text = line.rstrip("\n")
        if not text.strip():
            break
        result = predict_texts(model, tokenizer, [text], max_length, 1)[0]
        collected.append(result)
        print_results([result], max_length, start=len(collected))
    return collected


def main(argv: list[str] | None = None) -> int:
    """Chạy dự đoán.

    Giả định:
    - Có thể kết hợp --text và --file; không truyền gì (và không --interactive) thì dùng câu mẫu trong config.
    - Lỗi do đầu vào (model chưa train, file không có, câu rỗng...) được in rõ ra stderr, trả mã 2.
    """
    args = parse_args(argv)
    try:
        if args.interactive and (args.text or args.file):
            raise ValueError("--interactive không dùng chung với --text hay --file.")
        if not args.model_dir.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục model '{args.model_dir}'. Hãy chạy src/02_train_finetune.py --stage train trước."
            )
        max_length = resolve_max_length(args.max_length, args.train_report)
        texts: list[str] = list(args.text or [])
        if args.file is not None:
            texts += read_texts_file(args.file)
        device = get_device(args.device)
        tokenizer, model = load_sentiment_model(str(args.model_dir), device)
        print(f"Model: {args.model_dir} | thiết bị: {device} | max_length: {max_length}\n")

        if args.interactive:
            results = interactive_loop(model, tokenizer, max_length)
        else:
            results = predict_texts(model, tokenizer, texts or list(config.DEMO_TEXTS), max_length,
                                    config.PREDICT_BATCH_SIZE)
            print_results(results, max_length)

        if args.output is not None:
            saved = save_json({"model_dir": str(args.model_dir), "max_length": max_length, "predictions": results},
                              args.output)
            print(f"\nĐã lưu kết quả: {saved}")
    except (TypeError, ValueError, FileNotFoundError, IsADirectoryError, RuntimeError) as error:
        print(f"Lỗi: {error}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
