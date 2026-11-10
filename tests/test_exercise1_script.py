"""Chạy thử script src/01_sentiment_analysis.py như người dùng thật (chạy trong tiến trình riêng).

Chạy tiến trình riêng còn giúp: nếu PyTorch bị crash cứng (vd Bus error trên CPU),
chỉ test đó báo fail chứ không làm sập cả phiên pytest.
"""

import json
import subprocess
import sys

import pytest

from lab3 import config

pytestmark = pytest.mark.slow

SCRIPT = config.PROJECT_ROOT / "src" / "01_sentiment_analysis.py"


def run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=config.PROJECT_ROOT,
        timeout=600,
    )


def test_script_default_run_writes_report(tmp_path):
    output = tmp_path / "bai1.json"
    result = run_script("--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["predictions_manual"]) == len(config.EXERCISE1_SAMPLE_TEXTS)
    assert report["labels_match"] is True
    assert report["max_probability_difference"] <= config.EXERCISE1_PROBABILITY_TOLERANCE
    assert report["tokenization_example"]["text"] == config.EXERCISE1_SAMPLE_TEXTS[0]


def test_script_on_cpu_single_sentence(tmp_path):
    """Chống tái phát lỗi Bus error khi chạy CPU với batch 1 (xem docstring của load_sentiment_model)."""
    output = tmp_path / "cpu.json"
    result = run_script("--device", "cpu", "--text", "What a lovely surprise.", "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["device"] == "cpu"
    assert len(report["predictions_manual"]) == 1


def test_script_custom_texts_keep_order(tmp_path):
    output = tmp_path / "custom.json"
    texts = ["Second best film this year.", "I want my money back."]
    result = run_script("--text", texts[0], "--text", texts[1], "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert [p["text"] for p in report["predictions_manual"]] == texts


@pytest.mark.parametrize(
    "args, message",
    [
        (["--text", "   "], "rỗng"),
        (["--max-length", "1"], "max_length"),
        (["--max-length", "10000"], "max_length"),
        (["--device", "tpu"], "không hợp lệ"),
        (["--model", "./khong_ton_tai_model"], "Không tìm thấy"),
    ],
)
def test_script_invalid_input_exits_with_code_2(args, message, tmp_path):
    result = run_script(*args, "--output", str(tmp_path / "x.json"))
    assert result.returncode == 2
    assert message in result.stderr
    assert not (tmp_path / "x.json").exists()


def test_script_non_integer_max_length_is_rejected_by_argparse():
    result = run_script("--max-length", "abc")
    assert result.returncode == 2
    assert "invalid int value" in result.stderr
