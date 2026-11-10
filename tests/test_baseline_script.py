"""Chạy thử giai đoạn baseline của src/02_train_finetune.py (tiến trình riêng, dữ liệu tổng hợp nhỏ)."""

import json
import subprocess
import sys

import pytest

from lab3 import config
from lab3.data import save_split
from helpers import synthetic_reviews as synthetic

SCRIPT = config.PROJECT_ROOT / "src" / "02_train_finetune.py"


def run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=config.PROJECT_ROOT, timeout=600
    )


@pytest.fixture
def splits_without_test(tmp_path):
    """Chỉ có train.csv và validation.csv: giai đoạn baseline không được cần tới test.csv."""
    folder = tmp_path / "splits"
    for name, frame in (("train", synthetic(40, seed=5)), ("validation", synthetic(10, seed=6))):
        data = frame.assign(source_split="train", source_index=range(len(frame)))
        save_split(data, folder / f"{name}.csv")
    return folder


def baseline_args(splits, tmp_path, *extra):
    return [
        "--stage", "baseline",
        "--splits-dir", str(splits),
        "--report", str(tmp_path / "report.json"),
        "--baseline-model", str(tmp_path / "model.joblib"),
        "--cv-folds", "3",
        "--n-jobs", "1",
        *extra,
    ]


def test_baseline_stage_runs_without_test_file(splits_without_test, tmp_path):
    result = run_script(*baseline_args(splits_without_test, tmp_path))
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["cv_folds"] == 3
    assert len(report["cv_results"]) == len(config.BASELINE_PARAM_GRID["tfidf__ngram_range"]) * len(
        config.BASELINE_PARAM_GRID["clf__C"]
    )
    assert (tmp_path / "model.joblib").exists()


@pytest.mark.parametrize("extra, message", [(["--cv-folds", "1"], "n_splits"), (["--n-jobs", "0"], "n_jobs")])
def test_baseline_stage_invalid_settings_exit_2(splits_without_test, tmp_path, extra, message):
    result = run_script(*baseline_args(splits_without_test, tmp_path, *extra))
    assert result.returncode == 2
    assert message in result.stderr
    assert not (tmp_path / "model.joblib").exists()


def test_baseline_stage_missing_splits_exit_2(tmp_path):
    result = run_script(*baseline_args(tmp_path / "khong_co", tmp_path))
    assert result.returncode == 2
    assert "Không tìm thấy" in result.stderr
