"""Test cho lab3.evaluate: metrics phân loại nhị phân và cảnh báo overfitting."""

import numpy as np
import pandas as pd
import pytest

from lab3.evaluate import classification_metrics, overfitting_gap


# ===================================================== classification_metrics
def test_metrics_perfect_predictions():
    metrics = classification_metrics([0, 1, 1, 0], [0, 1, 1, 0])
    assert metrics == {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "count": 4}


def test_metrics_known_values():
    # 1 dương đúng, 1 dương bị đoán sai thành âm, 2 âm đúng.
    metrics = classification_metrics([1, 1, 0, 0], [1, 0, 0, 0])
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_metrics_model_never_predicts_positive_does_not_crash():
    metrics = classification_metrics([1, 0, 0, 1], [0, 0, 0, 0])
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_metrics_only_one_class_present():
    assert classification_metrics([0, 0], [0, 0])["accuracy"] == 1.0


def test_metrics_accepts_numpy_and_pandas():
    metrics = classification_metrics(np.array([0, 1, 1]), pd.Series([0, 1, 0]))
    assert metrics["accuracy"] == pytest.approx(2 / 3)


def test_metrics_custom_positive_label():
    metrics = classification_metrics([1, 1, 0, 0], [1, 0, 0, 0], positive_label=0)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)


@pytest.mark.parametrize("y_true, y_pred", [([0, 1], [0]), ([], [])])
def test_metrics_wrong_length_or_empty_raises(y_true, y_pred):
    with pytest.raises(ValueError):
        classification_metrics(y_true, y_pred)


@pytest.mark.parametrize("bad", [None, "0101", 5, [0, 1.0], [True, False], [0, "1"]])
def test_metrics_wrong_type_raises(bad):
    with pytest.raises(TypeError):
        classification_metrics(bad, [0, 1])


def test_metrics_more_than_two_labels_raises():
    with pytest.raises(ValueError, match="nhị phân"):
        classification_metrics([0, 1, 2], [0, 1, 2])


@pytest.mark.parametrize("label", ["1", 1.0, True, None])
def test_metrics_positive_label_wrong_type_raises(label):
    with pytest.raises(TypeError):
        classification_metrics([0, 1], [0, 1], positive_label=label)


# ============================================================ overfitting_gap
def test_gap_warns_when_train_much_higher():
    result = overfitting_gap(0.99, 0.85, 0.05)
    assert result["gap"] == pytest.approx(0.14)
    assert result["overfitting_warning"] is True


def test_gap_equal_to_threshold_does_not_warn():
    assert overfitting_gap(0.75, 0.5, 0.25)["overfitting_warning"] is False


def test_gap_validation_higher_than_train_does_not_warn():
    result = overfitting_gap(0.8, 0.9, 0.05)
    assert result["gap"] == pytest.approx(-0.1)
    assert result["overfitting_warning"] is False


@pytest.mark.parametrize("train, validation, threshold", [(1.2, 0.5, 0.05), (0.9, -0.1, 0.05), (float("nan"), 0.5, 0.05), (0.9, 0.8, -0.01)])
def test_gap_out_of_range_raises(train, validation, threshold):
    with pytest.raises(ValueError):
        overfitting_gap(train, validation, threshold)


@pytest.mark.parametrize("bad", ["0.9", None, True, [0.9]])
def test_gap_wrong_type_raises(bad):
    with pytest.raises(TypeError):
        overfitting_gap(bad, 0.8, 0.05)


# ============================================================ confusion_counts
from lab3.evaluate import confusion_counts  # noqa: E402


def test_confusion_counts_known_matrix():
    # dòng = nhãn thật, cột = dự đoán
    assert confusion_counts([0, 0, 1, 1, 1], [0, 1, 1, 1, 0], [0, 1]) == [[1, 1], [1, 2]]


def test_confusion_counts_follows_label_order():
    assert confusion_counts([0, 1, 1], [0, 1, 0], [1, 0]) == [[1, 1], [0, 1]]


def test_confusion_counts_total_equals_samples():
    matrix = confusion_counts([0, 1] * 10, [1, 1] * 10, [0, 1])
    assert sum(map(sum, matrix)) == 20


@pytest.mark.parametrize("y_true, y_pred, labels", [
    ([0, 2], [0, 1], [0, 1]),     # nhãn lạ
    ([0, 1], [0, 1], [0, 0]),     # labels trùng
    ([0, 1], [0], [0, 1]),        # khác độ dài
    ([], [], [0, 1]),             # rỗng
    ([0, 1], [0, 1], []),         # labels rỗng
])
def test_confusion_counts_invalid(y_true, y_pred, labels):
    with pytest.raises(ValueError):
        confusion_counts(y_true, y_pred, labels)


# ================================================================ mcnemar_test
from lab3.evaluate import mcnemar_test  # noqa: E402


def test_mcnemar_known_value():
    # (|1473 - 1190| - 1)^2 / (1473 + 1190) = 282^2 / 2663
    result = mcnemar_test(1473, 1190)
    assert result["chi2"] == pytest.approx(282**2 / 2663)
    assert result["p_value_chi2"] < 1e-6 and result["p_value_exact"] < 1e-6


def test_mcnemar_equal_counts_means_no_difference():
    result = mcnemar_test(50, 50)
    assert result["chi2"] == 0.0
    assert result["p_value_chi2"] == pytest.approx(1.0)
    assert result["p_value_exact"] == pytest.approx(1.0)


def test_mcnemar_is_symmetric():
    assert mcnemar_test(30, 10)["p_value_exact"] == pytest.approx(mcnemar_test(10, 30)["p_value_exact"])


def test_mcnemar_small_counts():
    result = mcnemar_test(1, 0)
    assert result["p_value_exact"] == pytest.approx(1.0)


@pytest.mark.parametrize("a, b, error", [(0, 0, ValueError), (-1, 5, ValueError), (1.5, 2, TypeError), ("3", 2, TypeError), (True, 2, TypeError), (None, 2, TypeError)])
def test_mcnemar_invalid(a, b, error):
    with pytest.raises(error):
        mcnemar_test(a, b)
