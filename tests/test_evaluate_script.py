"""Chạy thử giai đoạn evaluate của src/02_train_finetune.py (tiến trình riêng, model tí hon, CPU)."""

import json
import subprocess
import sys

import pytest

from helpers import synthetic_reviews
from lab3 import config
from lab3.baseline import save_model, tune_baseline
from lab3.data import save_split
from lab3.utils import save_json

pytestmark = pytest.mark.slow

SCRIPT = config.PROJECT_ROOT / "src" / "02_train_finetune.py"


def run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=config.PROJECT_ROOT, timeout=600
    )


@pytest.fixture
def workspace(tmp_path):
    test = synthetic_reviews(8, seed=11)
    save_split(test.assign(source_split="test", source_index=range(len(test))), tmp_path / "splits" / "test.csv")
    train = synthetic_reviews(20, seed=12)
    save_model(tune_baseline(train["text"], train["label"], {"clf__C": [1.0]}, 2, 42, 1, 500).best_estimator_,
               tmp_path / "baseline.joblib")
    save_json({"hyperparameters": {"max_length": 32},
               "log_history": [{"loss": 0.7, "step": 1}, {"eval_loss": 0.6, "eval_accuracy": 0.5, "step": 2}]},
              tmp_path / "train_report.json")
    return tmp_path


def evaluate_args(workspace, model_dir):
    return [
        "--stage", "evaluate", "--device", "cpu",
        "--splits-dir", str(workspace / "splits"),
        "--final-model-dir", str(model_dir),
        "--baseline-model", str(workspace / "baseline.joblib"),
        "--train-report", str(workspace / "train_report.json"),
        "--report", str(workspace / "reports" / "evaluation.json"),
    ]


def test_evaluate_stage_writes_report_figures_and_predictions(workspace, tiny_model_dir):
    result = run_script(*evaluate_args(workspace, tiny_model_dir))
    assert result.returncode == 0, result.stderr
    assert "KẾT QUẢ TRÊN TẬP TEST" in result.stdout
    report = json.loads((workspace / "reports" / "evaluation.json").read_text(encoding="utf-8"))
    assert report["n_test"] == 16
    for name in (config.CONFUSION_MATRIX_FILENAME, config.TRAINING_CURVES_FILENAME, config.TEST_PREDICTIONS_FILENAME):
        assert (workspace / "reports" / name).exists()


def test_evaluate_stage_without_trained_model_exit_2(workspace):
    result = run_script(*evaluate_args(workspace, workspace / "khong_co_model"))
    assert result.returncode == 2
    assert "--stage train" in result.stderr


def test_evaluate_stage_without_train_report_exit_2(workspace, tiny_model_dir):
    (workspace / "train_report.json").unlink()
    result = run_script(*evaluate_args(workspace, tiny_model_dir))
    assert result.returncode == 2
    assert "Không tìm thấy" in result.stderr
