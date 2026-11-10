"""Chạy thử giai đoạn train của src/02_train_finetune.py như người dùng thật, với model tí hon trên CPU."""

import json
import subprocess
import sys

import pytest

from helpers import synthetic_reviews
from lab3 import config
from lab3.data import save_split

pytestmark = pytest.mark.slow

SCRIPT = config.PROJECT_ROOT / "src" / "02_train_finetune.py"


def run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=config.PROJECT_ROOT, timeout=900
    )


@pytest.fixture
def workspace(tmp_path):
    splits = tmp_path / "splits"
    for name, frame in (("train", synthetic_reviews(24, seed=7)), ("validation", synthetic_reviews(6, seed=8))):
        save_split(frame.assign(source_split="train", source_index=range(len(frame))), splits / f"{name}.csv")
    return tmp_path


def train_args(workspace, tiny_model_dir, *extra):
    return [
        "--stage", "train",
        "--model", str(tiny_model_dir),
        "--device", "cpu",
        "--splits-dir", str(workspace / "splits"),
        "--checkpoints-dir", str(workspace / "ck"),
        "--final-model-dir", str(workspace / "final"),
        "--report", str(workspace / "report.json"),
        "--max-length", "32",
        "--batch-size", "8",
        *extra,
    ]


def test_smoke_prints_estimate_and_saves_nothing(workspace, tiny_model_dir):
    result = run_script(*train_args(workspace, tiny_model_dir, "--smoke"))
    assert result.returncode == 0, result.stderr
    assert "TỔNG" in result.stdout
    report = json.loads((workspace / "report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "smoke"
    assert report["estimate_full_run"]["total_seconds"] > 0
    assert not (workspace / "final").exists()
    assert not (workspace / "ck").exists()


def test_full_run_then_rerun_rules(workspace, tiny_model_dir):
    first = run_script(*train_args(workspace, tiny_model_dir, "--epochs", "1"))
    assert first.returncode == 0, first.stderr
    assert (workspace / "final" / "model.safetensors").exists()
    report = json.loads((workspace / "report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "full" and "train_metrics" in report

    again = run_script(*train_args(workspace, tiny_model_dir, "--epochs", "1"))
    assert again.returncode == 2 and "--resume" in again.stderr

    resumed = run_script(*train_args(workspace, tiny_model_dir, "--epochs", "1", "--resume"))
    assert resumed.returncode == 0, resumed.stderr

    overwritten = run_script(*train_args(workspace, tiny_model_dir, "--epochs", "1", "--overwrite"))
    assert overwritten.returncode == 0, overwritten.stderr


@pytest.mark.parametrize("extra, message", [
    (["--batch-size", "0"], "batch_size"),
    (["--max-length", "1"], "max_length"),
    (["--smoke", "--resume"], "--smoke"),
    (["--learning-rate", "-1"], "learning_rate"),
    (["--device", "tpu"], "không hợp lệ"),
])
def test_invalid_settings_exit_2(workspace, tiny_model_dir, extra, message):
    result = run_script(*train_args(workspace, tiny_model_dir, *extra))
    assert result.returncode == 2
    assert message in result.stderr


def test_missing_splits_exit_2(tmp_path, tiny_model_dir):
    result = run_script(*train_args(tmp_path, tiny_model_dir))
    assert result.returncode == 2
    assert "Không tìm thấy" in result.stderr
