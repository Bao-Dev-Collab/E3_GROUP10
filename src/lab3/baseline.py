"""Bài 2 – baseline: TF-IDF + LogisticRegression, chọn tham số bằng cross-validation.

Baseline là mốc để so với DistilBERT ở bước đánh giá. Chỉ dùng tập train (chọn tham số bằng
StratifiedKFold) và tập validation (báo cáo). Tập test KHÔNG được dùng ở đây.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from lab3.evaluate import as_label_list, classification_metrics, overfitting_gap

_MAX_SEED = 2**32 - 1


def _check_seed(seed: Any) -> int:
    """Giả định: seed là số nguyên 0..2^32-1 (giới hạn của numpy)."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed phải là số nguyên, nhận được {type(seed).__name__}.")
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed phải nằm trong khoảng 0..{_MAX_SEED}, nhận được {seed}.")
    return int(seed)


def _check_positive_int(value: Any, name: str, minimum: int = 1) -> int:
    """Giả định: value là số nguyên (không nhận bool) và >= minimum."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} phải là số nguyên, nhận được {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} phải >= {minimum}, nhận được {value}.")
    return int(value)


def build_baseline_pipeline(seed: int, max_features: int) -> Pipeline:
    """Tạo Pipeline TF-IDF -> LogisticRegression.

    Giả định:
    - TF-IDF nằm TRONG Pipeline: mỗi lần fit (kể cả trong từng fold của cross-validation) chỉ học
      từ vựng và trọng số IDF trên phần dữ liệu train của lần đó, nên không rò rỉ sang phần kiểm tra.
    - Đổi chữ thường, bỏ từ chỉ xuất hiện trong 1 review (min_df=2), dùng log cho tần suất (sublinear_tf),
      giữ tối đa max_features từ / cụm từ phổ biến nhất.
    - LogisticRegression có regularization L2 (C nhỏ thì phạt mạnh), solver liblinear, random_state = seed.
    """
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, min_df=2, sublinear_tf=True,
                                      max_features=_check_positive_int(max_features, "max_features"))),
            ("clf", LogisticRegression(max_iter=1000, solver="liblinear", random_state=_check_seed(seed))),
        ]
    )


def _check_texts_labels(texts: Any, labels: Any) -> tuple[list[str], list[int]]:
    """Giả định: texts là dãy chuỗi có nội dung, labels là dãy số nguyên cùng độ dài, không rỗng."""
    if texts is None or isinstance(texts, (str, bytes)) or not isinstance(texts, (Sequence, pd.Series, np.ndarray)):
        raise TypeError(f"texts phải là một dãy chuỗi, nhận được {type(texts).__name__}.")
    items = list(texts)
    if not items:
        raise ValueError("texts rỗng.")
    for index, text in enumerate(items):
        if not isinstance(text, str):
            raise TypeError(f"texts[{index}] phải là str, nhận được {type(text).__name__}.")
        if not text.strip():
            raise ValueError(f"texts[{index}] rỗng.")
    targets = as_label_list(labels, "labels")
    if len(items) != len(targets):
        raise ValueError(f"texts và labels khác độ dài: {len(items)} và {len(targets)}.")
    return items, targets


def tune_baseline(
    texts: Sequence[str],
    labels: Sequence[int],
    param_grid: dict[str, list],
    n_splits: int,
    seed: int,
    n_jobs: int,
    max_features: int,
) -> GridSearchCV:
    """Chọn tham số baseline bằng StratifiedKFold trên tập TRAIN, rồi train lại trên toàn bộ train.

    Giả định:
    - Chỉ truyền dữ liệu train vào đây (không truyền validation/test).
    - param_grid là dict không rỗng {tên tham số trong Pipeline: list giá trị không rỗng}.
    - 2 <= n_splits <= số mẫu của lớp ít nhất; các fold được xáo trộn với seed.
    - Chọn bộ tham số có accuracy trung bình các fold cao nhất (dữ liệu gần cân bằng); F1 cũng được ghi lại.
    - n_jobs là số tiến trình chạy song song (>= 1).
    """
    items, targets = _check_texts_labels(texts, labels)
    if not isinstance(param_grid, dict):
        raise TypeError(f"param_grid phải là dict, nhận được {type(param_grid).__name__}.")
    if not param_grid or any(not isinstance(values, list) or not values for values in param_grid.values()):
        raise ValueError("param_grid phải có ít nhất 1 tham số, mỗi tham số là list không rỗng.")
    folds = _check_positive_int(n_splits, "n_splits", minimum=2)
    smallest_class = int(pd.Series(targets).value_counts().min())
    if len(set(targets)) < 2:
        raise ValueError("Cần ít nhất 2 lớp để cross-validation có stratify.")
    if folds > smallest_class:
        raise ValueError(f"n_splits={folds} lớn hơn số mẫu của lớp ít nhất ({smallest_class}).")
    jobs = _check_positive_int(n_jobs, "n_jobs")

    search = GridSearchCV(
        build_baseline_pipeline(seed, max_features),
        param_grid,
        scoring={"accuracy": "accuracy", "f1": "f1"},
        refit="accuracy",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=_check_seed(seed)),
        return_train_score=True,
        n_jobs=jobs,
        error_score="raise",
    )
    search.fit(items, targets)
    return search


def _json_safe(value: Any) -> Any:
    """Đổi tuple / kiểu numpy sang kiểu ghi được ra JSON. Giả định: value là giá trị tham số đơn giản."""
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize_cv(search: GridSearchCV) -> list[dict[str, Any]]:
    """Bảng kết quả cross-validation, xếp theo thứ hạng accuracy.

    Giả định: search đã fit với scoring gồm accuracy, f1 và return_train_score=True.
    """
    results = search.cv_results_
    rows = []
    for index, params in enumerate(results["params"]):
        rows.append(
            {
                "rank": int(results["rank_test_accuracy"][index]),
                "params": {key: _json_safe(value) for key, value in params.items()},
                "cv_accuracy_mean": float(results["mean_test_accuracy"][index]),
                "cv_accuracy_std": float(results["std_test_accuracy"][index]),
                "cv_f1_mean": float(results["mean_test_f1"][index]),
                "train_accuracy_mean": float(results["mean_train_accuracy"][index]),
            }
        )
    return sorted(rows, key=lambda row: (row["rank"], -row["cv_accuracy_mean"]))


def run_baseline(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    param_grid: dict[str, list],
    n_splits: int,
    seed: int,
    gap_threshold: float,
    n_jobs: int,
    max_features: int,
) -> tuple[Pipeline, dict[str, Any]]:
    """Chọn tham số trên train, rồi báo cáo điểm trên cả train và validation.

    Giả định:
    - train, validation có cột text, label (đã chia, làm sạch, loại trùng ở bước dữ liệu).
    - Validation chỉ dùng để báo cáo, không dùng để chọn tham số. Không có tập test ở đây.
    - Overfitting được cảnh báo khi accuracy train − accuracy validation > gap_threshold.
    """
    for name, frame in (("train", train), ("validation", validation)):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} phải là pandas.DataFrame, nhận được {type(frame).__name__}.")
        missing = [column for column in ("text", "label") if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} thiếu cột: {missing}.")

    search = tune_baseline(
        train["text"], train["label"], param_grid, n_splits, seed, n_jobs, max_features
    )
    model = search.best_estimator_
    train_metrics = classification_metrics(train["label"], model.predict(train["text"].tolist()))
    validation_texts, validation_labels = _check_texts_labels(validation["text"], validation["label"])
    validation_metrics = classification_metrics(validation_labels, model.predict(validation_texts))

    report = {
        "seed": int(seed),
        "cv_folds": int(n_splits),
        "max_features": int(max_features),
        "param_grid": {key: [_json_safe(value) for value in values] for key, values in param_grid.items()},
        "cv_results": summarize_cv(search),
        "best_params": {key: _json_safe(value) for key, value in search.best_params_.items()},
        "best_cv_accuracy": float(search.best_score_),
        "vocabulary_size": int(len(model.named_steps["tfidf"].vocabulary_)),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "overfitting": overfitting_gap(train_metrics["accuracy"], validation_metrics["accuracy"], gap_threshold),
    }
    return model, report


def save_model(model: Pipeline, path: str | Path) -> Path:
    """Lưu Pipeline đã train ra file joblib.

    Giả định: model là sklearn Pipeline đã fit; path trỏ tới một file, thư mục cha tự tạo nếu chưa có.
    """
    if not isinstance(model, Pipeline):
        raise TypeError(f"model phải là sklearn Pipeline, nhận được {type(model).__name__}.")
    if path is None or isinstance(path, bool) or not isinstance(path, (str, Path)):
        raise TypeError(f"path phải là str hoặc Path, nhận được {type(path).__name__}.")
    target = Path(path)
    if target.is_dir():
        raise IsADirectoryError(f"'{target}' là thư mục, cần đường dẫn tới một file.")
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)
    return target


def load_model(path: str | Path) -> Pipeline:
    """Nạp Pipeline baseline đã lưu.

    Giả định: file do chính save_model của project tạo ra. joblib dùng pickle, chỉ nạp file tin cậy.
    """
    if path is None or isinstance(path, bool) or not isinstance(path, (str, Path)):
        raise TypeError(f"path phải là str hoặc Path, nhận được {type(path).__name__}.")
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Không tìm thấy file model '{target}'.")
    if target.is_dir():
        raise IsADirectoryError(f"'{target}' là thư mục, cần đường dẫn tới file model.")
    model = joblib.load(target)
    if not isinstance(model, Pipeline):
        raise TypeError(f"File '{target}' không chứa sklearn Pipeline.")
    return model
