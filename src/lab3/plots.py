"""Vẽ đồ thị cho báo cáo: confusion matrix và quá trình train."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # vẽ ra file, không cần cửa sổ hiển thị
import matplotlib.pyplot as plt  # noqa: E402


def _target(path: Any) -> Path:
    """Giả định: path là str hoặc Path tới file ảnh; thư mục cha tự tạo nếu chưa có."""
    if path is None or isinstance(path, bool) or not isinstance(path, (str, Path)):
        raise TypeError(f"path phải là str hoặc Path, nhận được {type(path).__name__}.")
    target = Path(path)
    if target.is_dir():
        raise IsADirectoryError(f"'{target}' là thư mục, cần đường dẫn tới file ảnh.")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def plot_confusion_matrices(matrices: dict[str, list[list[int]]], label_names: list[str], path: str | Path) -> Path:
    """Vẽ các confusion matrix cạnh nhau, mỗi ô ghi số lượng và tỉ lệ theo dòng (theo nhãn thật).

    Giả định: matrices là dict {tên model: ma trận vuông C×C số nguyên không âm}, C = số tên nhãn.
    """
    if not isinstance(matrices, dict) or not matrices:
        raise ValueError("matrices phải là dict không rỗng {tên model: ma trận}.")
    size = len(label_names)
    for name, matrix in matrices.items():
        if len(matrix) != size or any(len(row) != size for row in matrix):
            raise ValueError(f"Ma trận của '{name}' phải có kích thước {size}×{size}.")
        if any(value < 0 for row in matrix for value in row):
            raise ValueError(f"Ma trận của '{name}' có số âm.")
    target = _target(path)

    figure, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4.4), squeeze=False)
    for axis, (name, matrix) in zip(axes[0], matrices.items()):
        image = axis.imshow(matrix, cmap="Blues")
        for row, values in enumerate(matrix):
            total = sum(values) or 1
            for column, value in enumerate(values):
                dark = value > max(max(r) for r in matrix) / 2
                axis.text(column, row, f"{value:,}\n({value / total:.1%})", ha="center", va="center",
                          color="white" if dark else "black", fontsize=11)
        axis.set_xticks(range(size), label_names)
        axis.set_yticks(range(size), label_names)
        axis.set_xlabel("Dự đoán")
        axis.set_ylabel("Nhãn thật")
        axis.set_title(name)
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target


def plot_training_curves(log_history: list[dict[str, Any]], path: str | Path) -> Path:
    """Vẽ loss train, loss validation và accuracy validation theo số bước.

    Giả định:
    - log_history là trainer.state.log_history: các dict có "step" và "loss" (train) hoặc
      "eval_loss", "eval_accuracy" (validation).
    - Nếu có nhiều lần đánh giá cùng một bước (Trainer đánh giá lại model tốt nhất ở cuối), chỉ giữ lần đầu.
    """
    if not isinstance(log_history, list):
        raise TypeError(f"log_history phải là list, nhận được {type(log_history).__name__}.")
    train_points = [(entry["step"], entry["loss"]) for entry in log_history if isinstance(entry, dict) and "loss" in entry]
    evaluation: dict[int, tuple[float, float]] = {}
    for entry in log_history:
        if isinstance(entry, dict) and "eval_loss" in entry and "eval_accuracy" in entry:
            evaluation.setdefault(entry["step"], (entry["eval_loss"], entry["eval_accuracy"]))
    if not train_points or not evaluation:
        raise ValueError("log_history cần có ít nhất 1 mục loss train và 1 mục đánh giá validation.")
    target = _target(path)

    eval_steps = sorted(evaluation)
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axis.plot(*zip(*train_points), label="Loss train", alpha=0.8)
    loss_axis.plot(eval_steps, [evaluation[s][0] for s in eval_steps], "o-", label="Loss validation")
    loss_axis.set_xlabel("Bước")
    loss_axis.set_ylabel("Loss")
    loss_axis.set_title("Loss theo số bước")
    loss_axis.legend()
    accuracy_axis.plot(eval_steps, [evaluation[s][1] for s in eval_steps], "o-", color="tab:green")
    for step in eval_steps:
        accuracy_axis.annotate(f"{evaluation[step][1]:.4f}", (step, evaluation[step][1]),
                               textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
    accuracy_axis.set_xlabel("Bước")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_title("Accuracy validation")
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target
