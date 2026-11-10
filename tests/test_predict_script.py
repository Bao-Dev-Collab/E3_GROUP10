"""Chạy thử src/03_predict_finetuned.py như người dùng thật (tiến trình riêng, model tí hon, CPU)."""

import json
import subprocess
import sys

import pytest

from lab3 import config

pytestmark = pytest.mark.slow

SCRIPT = config.PROJECT_ROOT / "src" / "03_predict_finetuned.py"


def run_script(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], input=stdin, capture_output=True, text=True,
        cwd=config.PROJECT_ROOT, timeout=600,
    )


def base_args(model_dir):
    return ["--model-dir", str(model_dir), "--max-length", "32", "--device", "cpu"]


def test_texts_from_command_line_saved_in_order(tiny_model_dir, tmp_path):
    output = tmp_path / "pred.json"
    result = run_script(*base_args(tiny_model_dir), "--text", "Loved it", "--text", "Hated it", "--output", str(output))
    assert result.returncode == 0, result.stderr
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert [p["text"] for p in saved["predictions"]] == ["Loved it", "Hated it"]
    assert saved["max_length"] == 32


def test_texts_from_file_and_command_line_combined(tiny_model_dir, tmp_path):
    reviews = tmp_path / "reviews.txt"
    reviews.write_text("from file one\n\nfrom file two\n", encoding="utf-8")
    output = tmp_path / "pred.json"
    result = run_script(*base_args(tiny_model_dir), "--text", "from cli", "--file", str(reviews), "--output", str(output))
    assert result.returncode == 0, result.stderr
    texts = [p["text"] for p in json.loads(output.read_text(encoding="utf-8"))["predictions"]]
    assert texts == ["from cli", "from file one", "from file two"]


def test_default_demo_sentences(tiny_model_dir):
    result = run_script(*base_args(tiny_model_dir))
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("độ tin cậy") == len(config.DEMO_TEXTS)


def test_interactive_mode_stops_at_blank_line(tiny_model_dir):
    result = run_script(*base_args(tiny_model_dir), "--interactive", stdin="good movie\nbad movie\n\nnot read\n")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("độ tin cậy") == 2
    numbers = [line.split(".")[0] for line in result.stdout.splitlines() if "độ tin cậy" in line]
    assert numbers == ["1", "2"]  # đánh số nối tiếp, không bắt đầu lại từ 1 mỗi câu


def test_max_length_read_from_train_report(tiny_model_dir, tmp_path):
    report = tmp_path / "train_report.json"
    report.write_text(json.dumps({"hyperparameters": {"max_length": 24}}), encoding="utf-8")
    result = run_script("--model-dir", str(tiny_model_dir), "--train-report", str(report), "--device", "cpu", "--text", "ok")
    assert result.returncode == 0, result.stderr
    assert "max_length: 24" in result.stdout


@pytest.mark.parametrize("extra, message", [
    (["--text", "   "], "rỗng"),
    (["--file", "khong_co.txt"], "Không tìm thấy"),
    (["--interactive", "--text", "x"], "--interactive"),
    (["--max-length", "0"], "max_length"),
])
def test_invalid_input_exit_2(tiny_model_dir, extra, message):
    args = ["--model-dir", str(tiny_model_dir), "--device", "cpu"]
    if "--max-length" not in extra:
        args += ["--max-length", "32"]
    result = run_script(*args, *extra)
    assert result.returncode == 2
    assert message in result.stderr


def test_missing_model_tells_to_train_first(tmp_path):
    result = run_script("--model-dir", str(tmp_path / "chua_train"), "--max-length", "32")
    assert result.returncode == 2
    assert "--stage train" in result.stderr
