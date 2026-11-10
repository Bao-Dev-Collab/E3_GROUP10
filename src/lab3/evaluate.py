"""Các hàm đánh giá dùng chung cho baseline và DistilBERT."""

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def as_label_list(values: Any, name: str) -> list[int]:
    """Chuyển dãy nhãn thành list[int].

    Giả định: values là dãy (list, tuple, numpy array, pandas Series...) các số nguyên; không nhận bool, số thực, chuỗi.
    """
    if values is None or isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{name} phải là một dãy số nguyên, nhận được {type(values).__name__}.")
    items = list(values)
    for index, value in enumerate(items):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name}[{index}] phải là số nguyên, nhận được {type(value).__name__}.")
    return [int(value) for value in items]


def classification_metrics(y_true: Any, y_pred: Any, positive_label: int = 1) -> dict[str, float]:
    """Accuracy, precision, recall, F1 cho bài toán phân loại nhị phân.

    Giả định:
    - y_true và y_pred cùng độ dài, không rỗng, gồm tối đa 2 nhãn số nguyên (tính cả positive_label).
    - precision, recall, F1 tính cho nhãn dương positive_label.
    - Nếu model không đoán nhãn dương lần nào, precision được tính là 0 thay vì báo lỗi chia cho 0.
    """
    truth = as_label_list(y_true, "y_true")
    predicted = as_label_list(y_pred, "y_pred")
    if isinstance(positive_label, bool) or not isinstance(positive_label, (int, np.integer)):
        raise TypeError(f"positive_label phải là số nguyên, nhận được {type(positive_label).__name__}.")
    if not truth:
        raise ValueError("y_true rỗng.")
    if len(truth) != len(predicted):
        raise ValueError(f"y_true và y_pred khác độ dài: {len(truth)} và {len(predicted)}.")
    labels = set(truth) | set(predicted) | {int(positive_label)}
    if len(labels) > 2:
        raise ValueError(f"Chỉ hỗ trợ phân loại nhị phân, nhận được các nhãn {sorted(labels)}.")

    options = {"pos_label": int(positive_label), "average": "binary", "zero_division": 0}
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, **options)),
        "recall": float(recall_score(truth, predicted, **options)),
        "f1": float(f1_score(truth, predicted, **options)),
        "count": len(truth),
    }


def confusion_counts(y_true: Any, y_pred: Any, labels: list[int]) -> list[list[int]]:
    """Ma trận nhầm lẫn: dòng = nhãn thật, cột = nhãn dự đoán, theo đúng thứ tự labels.

    Giả định: y_true, y_pred là dãy số nguyên cùng độ dài, không rỗng; mọi nhãn đều nằm trong labels
    (labels không trùng nhau).
    """
    truth = as_label_list(y_true, "y_true")
    predicted = as_label_list(y_pred, "y_pred")
    order = as_label_list(labels, "labels")
    if not truth:
        raise ValueError("y_true rỗng.")
    if len(truth) != len(predicted):
        raise ValueError(f"y_true và y_pred khác độ dài: {len(truth)} và {len(predicted)}.")
    if len(set(order)) != len(order) or not order:
        raise ValueError("labels phải khác rỗng và không trùng nhau.")
    unknown = sorted((set(truth) | set(predicted)) - set(order))
    if unknown:
        raise ValueError(f"Có nhãn không nằm trong labels: {unknown}.")
    position = {label: index for index, label in enumerate(order)}
    matrix = [[0] * len(order) for _ in order]
    for actual, guess in zip(truth, predicted):
        matrix[position[actual]][position[guess]] += 1
    return matrix


def _check_score(value: Any, name: str) -> float:
    """Giả định: điểm là số thực hữu hạn trong [0, 1]."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} phải là số, nhận được {type(value).__name__}.")
    if math.isnan(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} phải nằm trong [0, 1], nhận được {value}.")
    return float(value)


def overfitting_gap(train_score: float, validation_score: float, threshold: float) -> dict[str, Any]:
    """So điểm train với validation và cảnh báo nếu chênh lệch lớn.

    Giả định:
    - Hai điểm cùng một thước đo (vd accuracy), thang 0–1, càng cao càng tốt.
    - Cảnh báo khi train − validation > threshold (bằng đúng ngưỡng thì không cảnh báo).
    - Validation cao hơn train (gap âm) không phải overfitting.
    """
    train = _check_score(train_score, "train_score")
    validation = _check_score(validation_score, "validation_score")
    limit = _check_score(threshold, "threshold")
    gap = train - validation
    return {
        "train": train,
        "validation": validation,
        "gap": gap,
        "threshold": limit,
        "overfitting_warning": gap > limit,
    }


def mcnemar_test(only_first_correct: int, only_second_correct: int) -> dict[str, float]:
    """Kiểm định McNemar: hai model chấm trên CÙNG một tập có khác nhau thật không, hay chỉ do ngẫu nhiên.

    Giả định:
    - only_first_correct: số mẫu chỉ model thứ nhất đúng; only_second_correct: số mẫu chỉ model thứ hai đúng.
      Các mẫu cả hai cùng đúng hoặc cùng sai không ảnh hưởng tới kiểm định.
    - Giả thuyết H0: hai model tốt như nhau, tức mỗi mẫu "bất đồng" có 50% nghiêng về mỗi bên.
    - Trả về thống kê chi-bình-phương có hiệu chỉnh liên tục và p-value của nó, kèm p-value của kiểm định
      nhị thức chính xác (đáng tin hơn khi số mẫu bất đồng nhỏ).
    """
    counts = []
    for name, value in (("only_first_correct", only_first_correct), ("only_second_correct", only_second_correct)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} phải là số nguyên, nhận được {type(value).__name__}.")
        if value < 0:
            raise ValueError(f"{name} không được âm, nhận được {value}.")
        counts.append(int(value))
    first, second = counts
    discordant = first + second
    if discordant == 0:
        raise ValueError("Hai model không bất đồng ở mẫu nào nên không kiểm định được.")
    statistic = (abs(first - second) - 1) ** 2 / discordant if first != second else 0.0
    return {
        "only_first_correct": first,
        "only_second_correct": second,
        "chi2": float(statistic),
        "p_value_chi2": float(chi2.sf(statistic, df=1)),
        "p_value_exact": float(binomtest(first, discordant, 0.5).pvalue),
    }
