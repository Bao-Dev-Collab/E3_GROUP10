"""Test cho lab3.predict. Phần slow dùng model DistilBERT tí hon trên CPU."""

import pytest
import torch

from lab3.predict import predict_texts, read_texts_file, resolve_max_length
from lab3.sentiment import load_sentiment_model
from lab3.utils import save_json


# ============================================================ read_texts_file
def test_read_texts_file_skips_blank_lines_and_keeps_content(tmp_path):
    path = tmp_path / "reviews.txt"
    path.write_text("Great film!\n\n   \n  Phim hay 😍 \nBad, really bad.\n", encoding="utf-8")
    assert read_texts_file(path) == ["Great film!", "  Phim hay 😍 ", "Bad, really bad."]


def test_read_texts_file_without_trailing_newline(tmp_path):
    path = tmp_path / "one.txt"
    path.write_text("only one review", encoding="utf-8")
    assert read_texts_file(path) == ["only one review"]


def test_read_texts_file_only_blank_lines_raises(tmp_path):
    path = tmp_path / "blank.txt"
    path.write_text("\n  \n\t\n", encoding="utf-8")
    with pytest.raises(ValueError, match="không có dòng"):
        read_texts_file(path)


def test_read_texts_file_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_texts_file(tmp_path / "missing.txt")
    with pytest.raises(IsADirectoryError):
        read_texts_file(tmp_path)
    with pytest.raises(TypeError):
        read_texts_file(None)
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(ValueError, match="UTF-8"):
        read_texts_file(binary)


# ========================================================== resolve_max_length
def test_resolve_max_length_reads_train_report(tmp_path):
    report = save_json({"hyperparameters": {"max_length": 256}}, tmp_path / "train.json")
    assert resolve_max_length(None, report) == 256


def test_resolve_max_length_override_wins(tmp_path):
    assert resolve_max_length(128, tmp_path / "khong_can.json") == 128


@pytest.mark.parametrize("override, error", [(0, ValueError), (-5, ValueError), (1.5, TypeError), ("256", TypeError), (True, TypeError)])
def test_resolve_max_length_invalid_override(tmp_path, override, error):
    with pytest.raises(error):
        resolve_max_length(override, tmp_path / "x.json")


def test_resolve_max_length_missing_report_explains_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="--max-length"):
        resolve_max_length(None, tmp_path / "missing.json")


@pytest.mark.parametrize("content", [{}, {"hyperparameters": {}}, {"hyperparameters": {"max_length": 0}}, [1, 2]])
def test_resolve_max_length_bad_report(tmp_path, content):
    report = save_json(content, tmp_path / "bad.json")
    with pytest.raises(ValueError):
        resolve_max_length(None, report)


# ============================================================ predict (slow)
@pytest.fixture(scope="module")
def tiny_bundle(tiny_model_dir):
    return load_sentiment_model(str(tiny_model_dir), torch.device("cpu"))


@pytest.mark.slow
def test_predict_texts_output_and_order(tiny_bundle):
    tokenizer, model = tiny_bundle
    texts = ["good film", "bad film", "word " * 100]
    results = predict_texts(model, tokenizer, texts, 32, 2)
    assert [r["text"] for r in results] == texts
    for result in results:
        assert result["label"] in ("NEGATIVE", "POSITIVE")
        assert result["confidence"] == pytest.approx(max(result["probabilities"].values()))
        assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert [r["truncated"] for r in results] == [False, False, True]
    assert results[2]["num_tokens"] == 102


@pytest.mark.slow
def test_predict_texts_length_equal_to_limit_is_not_truncated(tiny_bundle):
    tokenizer, model = tiny_bundle
    text = "word " * 30  # 30 token + [CLS] + [SEP] = 32
    assert predict_texts(model, tokenizer, [text], 32, 1)[0]["truncated"] is False


@pytest.mark.slow
def test_predict_texts_batch_size_does_not_change_results(tiny_bundle):
    tokenizer, model = tiny_bundle
    texts = ["a good film", "terrible", "fine I guess", "word " * 60]
    one = predict_texts(model, tokenizer, texts, 32, 1)
    many = predict_texts(model, tokenizer, texts, 32, 16)
    assert [r["label"] for r in one] == [r["label"] for r in many]
    assert all(abs(a["confidence"] - b["confidence"]) < 1e-5 for a, b in zip(one, many))


@pytest.mark.slow
@pytest.mark.parametrize("texts, batch_size, error", [
    ([], 4, ValueError), (["  "], 4, ValueError), (None, 4, TypeError), (["ok"], 0, ValueError), (["ok"], 2.0, TypeError),
])
def test_predict_texts_invalid(tiny_bundle, texts, batch_size, error):
    tokenizer, model = tiny_bundle
    with pytest.raises(error):
        predict_texts(model, tokenizer, texts, 32, batch_size)
