"""Bài 2 – đánh giá cuối: chấm DistilBERT và baseline trên tập TEST.

Tập test chỉ được dùng ở đây, sau khi mọi lựa chọn (tham số baseline, max_length, checkpoint)
đã chốt bằng train / validation. Kết quả ở đây chỉ để báo cáo, không dùng để chỉnh model.
"""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from lab3.baseline import load_model
from lab3.evaluate import as_label_list, classification_metrics, confusion_counts
from lab3.plots import plot_confusion_matrices, plot_training_curves
from lab3.sentiment import load_sentiment_model, predict_sentiment


def _positive_int(value: Any, name: str) -> int:
    """Giả định: value là số nguyên dương (không nhận bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} phải là số nguyên, nhận được {type(value).__name__}.")
    if value < 1:
        raise ValueError(f"{name} phải >= 1, nhận được {value}.")
    return int(value)


def _texts(texts: Any) -> list[str]:
    """Giả định: texts là dãy (list, tuple, Series) các chuỗi có nội dung, không rỗng."""
    if texts is None or isinstance(texts, (str, bytes)) or not isinstance(texts, (Sequence, pd.Series, np.ndarray)):
        raise TypeError(f"texts phải là một dãy chuỗi, nhận được {type(texts).__name__}.")
    items = list(texts)
    if not items:
        raise ValueError("texts rỗng.")
    if any(not isinstance(text, str) for text in items):
        raise TypeError("Mọi phần tử của texts phải là str.")
    return items


def predict_in_batches(
    model: torch.nn.Module, tokenizer: Any, texts: Sequence[str], max_length: int, batch_size: int,
    progress_every: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Dự đoán nhiều review theo từng batch.

    Giả định:
    - model, tokenizer tải bằng load_sentiment_model; max_length giống lúc train.
    - Kết quả không phụ thuộc batch_size (pad trong batch không làm đổi kết quả, đã có test ở Bài 1).
    - progress_every > 0: in tiến độ sau mỗi progress_every batch.
    Trả về (nhãn dự đoán dạng số, ma trận xác suất số câu × số lớp theo thứ tự id của model).
    """
    items = _texts(texts)
    size = _positive_int(batch_size, "batch_size")
    num_batches = math.ceil(len(items) / size)
    predictions, probabilities = [], []
    for index in range(num_batches):
        chunk = items[index * size:(index + 1) * size]
        for result in predict_sentiment(model, tokenizer, chunk, max_length):
            predictions.append(result["label_id"])
            probabilities.append(list(result["probabilities"].values()))
        if progress_every and (index + 1) % progress_every == 0:
            print(f"  Đã dự đoán {min((index + 1) * size, len(items)):,}/{len(items):,} review", flush=True)
    return np.array(predictions, dtype=int), np.array(probabilities, dtype=float)


def token_lengths(tokenizer: Any, texts: Sequence[str]) -> np.ndarray:
    """Số token của mỗi review khi KHÔNG cắt (tính cả token đặc biệt). Giả định: tokenizer của Hugging Face."""
    items = _texts(texts)
    encoded = tokenizer(items, add_special_tokens=True, truncation=False, verbose=False)
    return np.array([len(ids) for ids in encoded["input_ids"]], dtype=int)


def accuracy_by_length(
    y_true: Any, predictions: dict[str, Any], lengths: Any, max_length: int
) -> list[dict[str, Any]]:
    """Accuracy của từng model trên nhóm review bị cắt / không bị cắt ở max_length.

    Giả định: mọi dãy cùng độ dài; review dài hơn max_length token là "bị cắt" (độ dài bằng đúng
    max_length vẫn đọc trọn). Nhóm rỗng có accuracy = None.
    """
    truth = np.array(as_label_list(y_true, "y_true"))
    limit = _positive_int(max_length, "max_length")
    token_counts = np.asarray(lengths)
    if not isinstance(predictions, dict) or not predictions:
        raise ValueError("predictions phải là dict không rỗng {tên model: nhãn dự đoán}.")
    if token_counts.shape != truth.shape:
        raise ValueError("lengths phải cùng độ dài với y_true.")
    guesses = {name: np.array(as_label_list(values, name)) for name, values in predictions.items()}
    for name, values in guesses.items():
        if values.shape != truth.shape:
            raise ValueError(f"Dự đoán của '{name}' khác độ dài với y_true.")

    groups = [
        (f"Không bị cắt (≤ {limit} token)", token_counts <= limit),
        (f"Bị cắt (> {limit} token)", token_counts > limit),
        ("Tất cả", np.ones_like(truth, dtype=bool)),
    ]
    rows = []
    for group, mask in groups:
        count = int(mask.sum())
        row: dict[str, Any] = {"group": group, "count": count}
        for name, values in guesses.items():
            row[name] = float(np.mean(values[mask] == truth[mask])) if count else None
        rows.append(row)
    return rows


def most_confident_errors(
    texts: Sequence[str], y_true: Any, y_pred: Any, confidence: Any, id2label: dict[int, str],
    per_type: int, preview_chars: int,
) -> dict[str, list[dict[str, Any]]]:
    """Các review model đoán sai mà lại tự tin nhất, chia theo loại lỗi (nhãn thật -> nhãn đoán).

    Giả định: mọi dãy cùng độ dài; confidence là xác suất của nhãn được đoán (0..1); chỉ trích
    preview_chars ký tự đầu của review. Loại lỗi nào ít hơn per_type ví dụ thì trả về hết số có.
    """
    items = _texts(texts)
    truth = np.array(as_label_list(y_true, "y_true"))
    guess = np.array(as_label_list(y_pred, "y_pred"))
    scores = np.asarray(confidence, dtype=float)
    limit = _positive_int(per_type, "per_type")
    chars = _positive_int(preview_chars, "preview_chars")
    if not (len(items) == truth.size == guess.size == scores.size):
        raise ValueError("texts, y_true, y_pred, confidence phải cùng độ dài.")

    result: dict[str, list[dict[str, Any]]] = {}
    for actual in sorted(id2label):
        for predicted in sorted(id2label):
            if actual == predicted:
                continue
            indices = np.where((truth == actual) & (guess == predicted))[0]
            ordered = indices[np.argsort(-scores[indices], kind="stable")][:limit]
            key = f"{id2label[actual]} -> {id2label[predicted]}"
            result[key] = [
                {
                    "row": int(i),
                    "confidence": float(scores[i]),
                    "text_preview": items[i][:chars] + ("…" if len(items[i]) > chars else ""),
                }
                for i in ordered
            ]
    return result


def run_final_evaluation(
    test: pd.DataFrame,
    *,
    final_model_dir: str | Path,
    baseline_model_path: str | Path,
    train_report: dict[str, Any],
    id2label: dict[int, str],
    device: torch.device,
    batch_size: int,
    output_dir: str | Path,
    figure_names: tuple[str, str],
    predictions_name: str,
    errors_per_type: int,
    preview_chars: int,
) -> dict[str, Any]:
    """Chấm DistilBERT và baseline trên tập test, lưu báo cáo, đồ thị, bảng dự đoán.

    Giả định:
    - test có cột source_index, text, label (data/splits/test.csv).
    - max_length lấy từ train_report (đúng giá trị đã dùng khi train), không chọn lại ở đây.
    - Không chỉnh model nào dựa trên kết quả này.
    """
    if not isinstance(test, pd.DataFrame):
        raise TypeError(f"test phải là pandas.DataFrame, nhận được {type(test).__name__}.")
    missing = [column for column in ("source_index", "text", "label") if column not in test.columns]
    if missing:
        raise ValueError(f"test thiếu cột: {missing}.")
    if not isinstance(train_report, dict) or "hyperparameters" not in train_report or "log_history" not in train_report:
        raise ValueError("train_report phải là báo cáo của giai đoạn train (có hyperparameters, log_history).")
    max_length = _positive_int(train_report["hyperparameters"].get("max_length"), "max_length trong train_report")
    out_dir = Path(output_dir)

    texts, labels = test["text"].tolist(), test["label"].to_numpy()
    tokenizer, model = load_sentiment_model(str(final_model_dir), device)
    bert_pred, bert_probs = predict_in_batches(model, tokenizer, texts, max_length, batch_size, progress_every=50)
    baseline = load_model(baseline_model_path)
    base_pred = baseline.predict(texts).astype(int)
    base_probs = baseline.predict_proba(texts)

    label_ids = sorted(id2label)
    names = [id2label[i] for i in label_ids]
    metrics = {
        "DistilBERT": classification_metrics(labels, bert_pred),
        "Baseline TF-IDF + LogReg": classification_metrics(labels, base_pred),
    }
    matrices = {
        "DistilBERT": confusion_counts(labels, bert_pred, label_ids),
        "Baseline TF-IDF + LogReg": confusion_counts(labels, base_pred, label_ids),
    }
    lengths = token_lengths(tokenizer, texts)
    by_length = accuracy_by_length(
        labels, {"DistilBERT": bert_pred, "Baseline TF-IDF + LogReg": base_pred}, lengths, max_length
    )
    bert_confidence = bert_probs[np.arange(len(bert_pred)), bert_pred]
    errors = most_confident_errors(texts, labels, bert_pred, bert_confidence, id2label, errors_per_type, preview_chars)
    for examples in errors.values():
        for example in examples:
            example["source_index"] = int(test["source_index"].iloc[example["row"]])
            example["token_length"] = int(lengths[example["row"]])
            example["baseline_prediction"] = id2label[int(base_pred[example["row"]])]

    both_right = (bert_pred == labels) & (base_pred == labels)
    both_wrong = (bert_pred != labels) & (base_pred != labels)
    agreement = {
        "both_correct": int(both_right.sum()),
        "only_distilbert_correct": int(((bert_pred == labels) & (base_pred != labels)).sum()),
        "only_baseline_correct": int(((bert_pred != labels) & (base_pred == labels)).sum()),
        "both_wrong": int(both_wrong.sum()),
    }

    confusion_path = plot_confusion_matrices(matrices, names, out_dir / figure_names[0])
    curves_path = plot_training_curves(train_report["log_history"], out_dir / figure_names[1])
    positive = label_ids[-1]
    table = pd.DataFrame({
        "source_index": test["source_index"].to_numpy(),
        "label": labels,
        "token_length": lengths,
        "distilbert_prediction": bert_pred,
        f"distilbert_prob_{id2label[positive]}": bert_probs[:, label_ids.index(positive)],
        "baseline_prediction": base_pred,
        f"baseline_prob_{id2label[positive]}": base_probs[:, list(baseline.classes_).index(positive)],
    })
    predictions_path = out_dir / predictions_name
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(predictions_path, index=False, lineterminator="\n")

    return {
        "n_test": int(len(test)),
        "max_length": max_length,
        "device": str(device),
        "metrics": metrics,
        "confusion_matrices": {"labels": names, **matrices},
        "accuracy_by_length": by_length,
        "agreement": agreement,
        "most_confident_errors_distilbert": errors,
        "files": {
            "confusion_matrices": str(confusion_path),
            "training_curves": str(curves_path),
            "test_predictions": str(predictions_path),
        },
    }
