"""Test cho lab3.final_evaluation.

Phần đầu không cần model. Phần slow dùng model DistilBERT tí hon (tiny_model_dir) và baseline nhỏ.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from helpers import synthetic_reviews
from lab3 import config
from lab3.baseline import save_model, tune_baseline
from lab3.final_evaluation import (
    accuracy_by_length,
    most_confident_errors,
    predict_in_batches,
    run_final_evaluation,
    token_lengths,
)
from lab3.sentiment import load_sentiment_model

LABELS = config.ID2LABEL


# ========================================================== accuracy_by_length
def test_accuracy_by_length_known_values():
    rows = accuracy_by_length([0, 1, 1, 0], {"A": [0, 1, 0, 0], "B": [1, 1, 1, 0]}, [10, 40, 50, 20], 32)
    uncut, cut, total = rows
    assert (uncut["count"], uncut["A"], uncut["B"]) == (2, 1.0, 0.5)
    assert (cut["count"], cut["A"], cut["B"]) == (2, 0.5, 1.0)
    assert (total["count"], total["A"], total["B"]) == (4, 0.75, 0.75)


def test_accuracy_by_length_equal_to_limit_is_not_cut():
    rows = accuracy_by_length([1], {"A": [1]}, [32], 32)
    assert rows[0]["count"] == 1 and rows[1]["count"] == 0


def test_accuracy_by_length_empty_group_is_none():
    assert accuracy_by_length([1, 0], {"A": [1, 0]}, [5, 6], 32)[1]["A"] is None


@pytest.mark.parametrize("predictions, lengths, max_length", [
    ({}, [1, 2], 32),
    ({"A": [0, 1]}, [1], 32),
    ({"A": [0]}, [1, 2], 32),
    ({"A": [0, 1]}, [1, 2], 0),
])
def test_accuracy_by_length_invalid(predictions, lengths, max_length):
    with pytest.raises(ValueError):
        accuracy_by_length([0, 1], predictions, lengths, max_length)


# ======================================================== most_confident_errors
TEXTS = ["a" * 10, "b" * 400, "c", "d", "e", "f"]
TRUTH = [0, 0, 0, 1, 1, 1]
GUESS = [1, 1, 0, 0, 0, 1]
CONF = [0.6, 0.95, 0.9, 0.7, 0.99, 0.8]


def test_most_confident_errors_sorted_and_grouped():
    errors = most_confident_errors(TEXTS, TRUTH, GUESS, CONF, LABELS, 5, 300)
    assert [e["row"] for e in errors["NEGATIVE -> POSITIVE"]] == [1, 0]
    assert [e["row"] for e in errors["POSITIVE -> NEGATIVE"]] == [4, 3]


def test_most_confident_errors_limit_and_preview():
    errors = most_confident_errors(TEXTS, TRUTH, GUESS, CONF, LABELS, 1, 300)
    top = errors["NEGATIVE -> POSITIVE"]
    assert len(top) == 1 and top[0]["row"] == 1
    assert top[0]["text_preview"] == "b" * 300 + "…"
    assert most_confident_errors(TEXTS, TRUTH, GUESS, CONF, LABELS, 5, 300)["NEGATIVE -> POSITIVE"][1]["text_preview"] == "a" * 10


def test_most_confident_errors_none_when_all_correct():
    errors = most_confident_errors(["x", "y"], [0, 1], [0, 1], [0.9, 0.9], LABELS, 3, 50)
    assert errors == {"NEGATIVE -> POSITIVE": [], "POSITIVE -> NEGATIVE": []}


@pytest.mark.parametrize("args, error", [
    ((TEXTS, TRUTH, GUESS, CONF[:3], LABELS, 5, 300), ValueError),
    ((TEXTS, TRUTH, GUESS, CONF, LABELS, 0, 300), ValueError),
    ((TEXTS, TRUTH, GUESS, CONF, LABELS, 5, -1), ValueError),
    ((None, TRUTH, GUESS, CONF, LABELS, 5, 300), TypeError),
    (([], [], [], [], LABELS, 5, 300), ValueError),
])
def test_most_confident_errors_invalid(args, error):
    with pytest.raises(error):
        most_confident_errors(*args)


# ================================================== run_final_evaluation (input)
REPORT = {"hyperparameters": {"max_length": 32}, "log_history": []}


def evaluate_kwargs(tmp_path, **overrides):
    options = dict(
        final_model_dir=tmp_path, baseline_model_path=tmp_path / "b.joblib", train_report=REPORT, id2label=LABELS,
        device=torch.device("cpu"), batch_size=8, output_dir=tmp_path, figure_names=("cm.png", "curves.png"),
        predictions_name="pred.csv", errors_per_type=3, preview_chars=100,
    )
    options.update(overrides)
    return options


def test_run_final_evaluation_rejects_bad_inputs_before_loading_models(tmp_path):
    good = synthetic_reviews(2, seed=0).assign(source_index=range(4))
    with pytest.raises(TypeError):
        run_final_evaluation("test", **evaluate_kwargs(tmp_path))
    with pytest.raises(ValueError, match="thiếu cột"):
        run_final_evaluation(good.drop(columns="source_index"), **evaluate_kwargs(tmp_path))
    with pytest.raises(ValueError, match="train_report"):
        run_final_evaluation(good, **evaluate_kwargs(tmp_path, train_report={"log_history": []}))
    with pytest.raises(ValueError, match="max_length"):
        run_final_evaluation(good, **evaluate_kwargs(tmp_path, train_report={"hyperparameters": {"max_length": 0}, "log_history": []}))


# ============================================================ cần model (slow)
@pytest.mark.slow
def test_predict_in_batches_does_not_depend_on_batch_size(tiny_model_dir):
    tokenizer, model = load_sentiment_model(str(tiny_model_dir), torch.device("cpu"))
    texts = synthetic_reviews(6, seed=1)["text"].tolist() + ["word " * 100]
    one_pred, one_prob = predict_in_batches(model, tokenizer, texts, 32, 1)
    all_pred, all_prob = predict_in_batches(model, tokenizer, texts, 32, 64)
    assert one_pred.tolist() == all_pred.tolist()
    assert np.allclose(one_prob, all_prob, atol=1e-5)
    assert np.allclose(all_prob.sum(axis=1), 1.0)
    assert all_prob.shape == (len(texts), 2)


@pytest.mark.slow
def test_predict_in_batches_invalid(tiny_model_dir):
    tokenizer, model = load_sentiment_model(str(tiny_model_dir), torch.device("cpu"))
    with pytest.raises(ValueError):
        predict_in_batches(model, tokenizer, ["good"], 32, 0)
    with pytest.raises(ValueError):
        predict_in_batches(model, tokenizer, [], 32, 8)


@pytest.mark.slow
def test_token_lengths_count_without_truncation(tiny_model_dir):
    tokenizer, _ = load_sentiment_model(str(tiny_model_dir), torch.device("cpu"))
    lengths = token_lengths(tokenizer, ["good film", "word " * 700])
    assert lengths[0] == 4  # [CLS] good film [SEP]
    assert lengths[1] == 702


@pytest.fixture
def evaluation_inputs(tmp_path, tiny_model_dir):
    train = synthetic_reviews(30, seed=1)
    baseline = tune_baseline(train["text"], train["label"], {"clf__C": [1.0]}, 2, 42, 1, 1000).best_estimator_
    baseline_path = save_model(baseline, tmp_path / "baseline.joblib")
    test = synthetic_reviews(10, seed=9).assign(source_index=range(100, 120))
    report = {
        "hyperparameters": {"max_length": 32},
        "log_history": [{"loss": 0.7, "step": 1}, {"eval_loss": 0.6, "eval_accuracy": 0.5, "step": 2}],
    }
    return test, baseline_path, report


@pytest.mark.slow
def test_run_final_evaluation_end_to_end(tmp_path, tiny_model_dir, evaluation_inputs):
    test, baseline_path, train_report = evaluation_inputs
    report = run_final_evaluation(test, **evaluate_kwargs(
        tmp_path / "out", final_model_dir=tiny_model_dir, baseline_model_path=baseline_path, train_report=train_report,
    ))
    n = len(test)
    assert report["n_test"] == n and report["max_length"] == 32
    for name in ("DistilBERT", "Baseline TF-IDF + LogReg"):
        matrix = report["confusion_matrices"][name]
        assert sum(map(sum, matrix)) == n
        assert report["metrics"][name]["accuracy"] == pytest.approx((matrix[0][0] + matrix[1][1]) / n)
        assert report["accuracy_by_length"][-1][name] == pytest.approx(report["metrics"][name]["accuracy"])
    assert sum(report["agreement"].values()) == n

    table = pd.read_csv(report["files"]["test_predictions"])
    assert len(table) == n
    assert table["source_index"].tolist() == list(range(100, 120))
    assert np.mean(table["distilbert_prediction"] == table["label"]) == pytest.approx(report["metrics"]["DistilBERT"]["accuracy"])
    for key in ("confusion_matrices", "training_curves"):
        assert open(report["files"][key], "rb").read(4) == b"\x89PNG"
    for examples in report["most_confident_errors_distilbert"].values():
        for example in examples:
            assert 100 <= example["source_index"] < 120
