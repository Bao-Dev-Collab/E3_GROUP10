"""Chạy thử giai đoạn data của src/02_train_finetune.py như người dùng thật (tiến trình riêng)."""

import json
import subprocess
import sys

import pandas as pd
import pytest

from lab3 import config

pytestmark = pytest.mark.slow

SCRIPT = config.PROJECT_ROOT / "src" / "02_train_finetune.py"


def run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=config.PROJECT_ROOT, timeout=600
    )


@pytest.fixture
def small_raw(tmp_path):
    train = pd.DataFrame({"text": [f"review {i}<br />text" for i in range(40)], "label": [0] * 20 + [1] * 20})
    test = pd.DataFrame({"text": [f"test {i}" for i in range(10)], "label": [0, 1] * 5})
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    return tmp_path


def data_args(folder, *extra):
    return [
        "--stage", "data",
        "--raw-train", str(folder / "train.csv"),
        "--raw-test", str(folder / "test.csv"),
        "--splits-dir", str(folder / "splits"),
        "--report", str(folder / "report.json"),
        *extra,
    ]


def test_data_stage_writes_splits_and_report(small_raw):
    result = run_script(*data_args(small_raw))
    assert result.returncode == 0, result.stderr
    report = json.loads((small_raw / "report.json").read_text(encoding="utf-8"))
    assert report["final_rows"] == {"train": 36, "validation": 4, "test": 10}
    for name in ("train", "validation", "test"):
        assert (small_raw / "splits" / f"{name}.csv").exists()


@pytest.mark.parametrize("extra, message", [(["--val-size", "1.5"], "val_size"), (["--seed", "-3"], "seed")])
def test_data_stage_invalid_numbers_exit_2(small_raw, extra, message):
    result = run_script(*data_args(small_raw, *extra))
    assert result.returncode == 2
    assert message in result.stderr
    assert not (small_raw / "report.json").exists()


def test_data_stage_missing_raw_file_exit_2(tmp_path):
    result = run_script(*data_args(tmp_path))
    assert result.returncode == 2
    assert "Không tìm thấy" in result.stderr


def test_unknown_stage_is_rejected_by_argparse():
    result = run_script("--stage", "xyz")
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
