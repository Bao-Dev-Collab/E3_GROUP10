"""Test cho lab3.plots: vẽ ra file PNG thật, kiểm tra đầu vào sai."""

import pytest

from lab3.plots import plot_confusion_matrices, plot_training_curves

PNG_SIGNATURE = b"\x89PNG"
LOG = [
    {"loss": 0.69, "step": 50}, {"loss": 0.40, "step": 100},
    {"eval_loss": 0.30, "eval_accuracy": 0.88, "step": 100},
    {"loss": 0.25, "step": 150},
    {"eval_loss": 0.28, "eval_accuracy": 0.90, "step": 200},
    {"eval_loss": 0.28, "eval_accuracy": 0.90, "step": 200},  # Trainer đánh giá lại model tốt nhất ở cuối
    {"train_runtime": 10.0, "step": 200},
]


def test_confusion_png_is_created(tmp_path):
    path = plot_confusion_matrices({"A": [[5, 1], [2, 7]], "B": [[6, 0], [0, 9]]}, ["NEG", "POS"], tmp_path / "x" / "cm.png")
    assert path.read_bytes()[:4] == PNG_SIGNATURE


def test_confusion_handles_empty_row(tmp_path):
    path = plot_confusion_matrices({"A": [[0, 0], [3, 4]]}, ["NEG", "POS"], tmp_path / "cm.png")
    assert path.exists()


@pytest.mark.parametrize("matrices", [{}, {"A": [[1, 2]]}, {"A": [[1, 2], [3]]}, {"A": [[1, -1], [0, 2]]}, None])
def test_confusion_invalid_matrices(tmp_path, matrices):
    with pytest.raises(ValueError):
        plot_confusion_matrices(matrices, ["NEG", "POS"], tmp_path / "cm.png")


def test_confusion_invalid_path(tmp_path):
    with pytest.raises(IsADirectoryError):
        plot_confusion_matrices({"A": [[1, 0], [0, 1]]}, ["NEG", "POS"], tmp_path)
    with pytest.raises(TypeError):
        plot_confusion_matrices({"A": [[1, 0], [0, 1]]}, ["NEG", "POS"], None)


def test_training_curves_png_is_created(tmp_path):
    assert plot_training_curves(LOG, tmp_path / "curves.png").read_bytes()[:4] == PNG_SIGNATURE


@pytest.mark.parametrize("history", [[], [{"loss": 0.5, "step": 1}], [{"eval_loss": 0.3, "eval_accuracy": 0.9, "step": 1}]])
def test_training_curves_need_train_and_eval_entries(tmp_path, history):
    with pytest.raises(ValueError):
        plot_training_curves(history, tmp_path / "curves.png")


def test_training_curves_wrong_type(tmp_path):
    with pytest.raises(TypeError):
        plot_training_curves({"loss": 0.5}, tmp_path / "curves.png")
