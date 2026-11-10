"""Dự đoán cảm xúc cho review mới bằng model DistilBERT đã fine-tune ở Bài 2."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from lab3.final_evaluation import token_lengths
from lab3.sentiment import predict_sentiment, validate_texts
from lab3.utils import load_json


def read_texts_file(path: str | Path) -> list[str]:
    """Đọc file .txt, mỗi dòng là một review.

    Giả định:
    - File mã hoá UTF-8. Dòng trống hoặc chỉ có khoảng trắng bị bỏ qua.
    - Giữ nguyên nội dung từng dòng (chỉ bỏ ký tự xuống dòng ở cuối).
    - File phải có ít nhất 1 dòng có nội dung.
    """
    if path is None or isinstance(path, bool) or not isinstance(path, (str, Path)):
        raise TypeError(f"path phải là str hoặc Path, nhận được {type(path).__name__}.")
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Không tìm thấy file '{target}'.")
    if target.is_dir():
        raise IsADirectoryError(f"'{target}' là thư mục, cần đường dẫn tới file .txt.")
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"File '{target}' không phải văn bản UTF-8.") from error
    texts = [line for line in lines if line.strip()]
    if not texts:
        raise ValueError(f"File '{target}' không có dòng nào có nội dung.")
    return texts


def resolve_max_length(override: int | None, train_report_path: str | Path) -> int:
    """Chọn max_length khi dự đoán: mặc định dùng đúng giá trị lúc train.

    Giả định:
    - override = None: đọc hyperparameters.max_length trong báo cáo train.
    - override là số nguyên dương: dùng giá trị đó (vd khi chạy model khác).
    """
    if override is not None:
        if isinstance(override, bool) or not isinstance(override, int):
            raise TypeError(f"max_length phải là số nguyên, nhận được {type(override).__name__}.")
        if override < 1:
            raise ValueError(f"max_length phải >= 1, nhận được {override}.")
        return override
    try:
        report = load_json(train_report_path)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Không tìm thấy báo cáo train '{train_report_path}' để lấy max_length. "
            "Chạy --stage train trước, hoặc truyền --max-length."
        ) from error
    value = report.get("hyperparameters", {}).get("max_length") if isinstance(report, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Báo cáo train '{train_report_path}' không có max_length hợp lệ.")
    return value


def predict_texts(
    model: torch.nn.Module, tokenizer: Any, texts: Sequence[str], max_length: int, batch_size: int
) -> list[dict[str, Any]]:
    """Dự đoán từng review, theo đúng thứ tự đầu vào.

    Giả định:
    - model, tokenizer tải bằng load_sentiment_model từ thư mục model đã fine-tune.
    - Review dài hơn max_length token bị cắt giữ phần đầu (giống lúc train); kết quả ghi rõ có bị cắt không.
    - batch_size chỉ ảnh hưởng tốc độ, không ảnh hưởng kết quả.
    """
    items = validate_texts(texts)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(f"batch_size phải là số nguyên, nhận được {type(batch_size).__name__}.")
    if batch_size < 1:
        raise ValueError(f"batch_size phải >= 1, nhận được {batch_size}.")

    full_lengths = token_lengths(tokenizer, items)
    results = []
    for start in range(0, len(items), batch_size):
        for prediction in predict_sentiment(model, tokenizer, items[start:start + batch_size], max_length):
            index = len(results)
            results.append(
                {
                    "text": prediction["text"],
                    "label": prediction["label"],
                    "confidence": prediction["score"],
                    "probabilities": prediction["probabilities"],
                    "num_tokens": int(full_lengths[index]),
                    "truncated": bool(full_lengths[index] > max_length),
                }
            )
    return results
