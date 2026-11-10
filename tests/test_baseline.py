"""Test cho lab3.baseline với dữ liệu tổng hợp nhỏ (chạy vài giây)."""

import joblib
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from helpers import synthetic_reviews as synthetic
from lab3.baseline import build_baseline_pipeline, load_model, run_baseline, save_model, tune_baseline

SMALL_GRID = {"tfidf__ngram_range": [(1, 1)], "clf__C": [0.1, 1.0]}


@pytest.fixture(scope="module")
def train_frame():
    return synthetic(60, seed=1)


@pytest.fixture(scope="module")
def validation_frame():
    return synthetic(15, seed=2)


def tune(frame, **overrides):
    options = {"param_grid": SMALL_GRID, "n_splits": 3, "seed": 42, "n_jobs": 1, "max_features": 1000}
    options.update(overrides)
    return tune_baseline(frame["text"], frame["label"], **options)


# ===================================================== build_baseline_pipeline
def test_pipeline_puts_tfidf_inside_so_each_fold_refits_it():
    pipeline = build_baseline_pipeline(seed=3, max_features=500)
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["tfidf", "clf"]
    assert isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)
    assert pipeline.named_steps["tfidf"].max_features == 500
    assert pipeline.named_steps["clf"].random_state == 3


@pytest.mark.parametrize("seed, max_features, error", [(-1, 10, ValueError), (1.5, 10, TypeError), (1, 0, ValueError), (1, "10", TypeError), (True, 10, TypeError)])
def test_pipeline_invalid_arguments_raise(seed, max_features, error):
    with pytest.raises(error):
        build_baseline_pipeline(seed, max_features)


# =============================================================== tune_baseline
def test_tune_learns_separable_data(train_frame):
    search = tune(train_frame)
    assert search.best_score_ >= 0.9
    assert len(search.cv_results_["params"]) == 2
    assert isinstance(search.estimator, Pipeline)  # CV nhận cả Pipeline, không nhận ma trận đã vector hoá sẵn


def test_tune_is_reproducible(train_frame):
    first = tune(train_frame).cv_results_["mean_test_accuracy"].tolist()
    second = tune(train_frame).cv_results_["mean_test_accuracy"].tolist()
    assert first == second


def test_tune_never_sees_words_outside_train(train_frame):
    search = tune(train_frame)
    assert "zzzunseenword" not in search.best_estimator_.named_steps["tfidf"].vocabulary_


@pytest.mark.parametrize("overrides, error", [
    ({"n_splits": 1}, ValueError),
    ({"n_splits": 61}, ValueError),
    ({"n_splits": 3.0}, TypeError),
    ({"n_jobs": 0}, ValueError),
    ({"param_grid": {}}, ValueError),
    ({"param_grid": {"clf__C": []}}, ValueError),
    ({"param_grid": [("clf__C", [1.0])]}, TypeError),
])
def test_tune_invalid_settings_raise(train_frame, overrides, error):
    with pytest.raises(error):
        tune(train_frame, **overrides)


@pytest.mark.parametrize("texts, labels, error", [
    (["good film", "bad film"], [1], ValueError),
    (["good film", None], [1, 0], TypeError),
    ([], [], ValueError),
    (["good film", "   "], [1, 0], ValueError),
    ("good film", [1], TypeError),
    (["a b", "c d", "e f"], [1, 1, 1], ValueError),
])
def test_tune_invalid_data_raises(texts, labels, error):
    with pytest.raises(error):
        tune_baseline(texts, labels, SMALL_GRID, 2, 42, 1, 100)


# ================================================================ run_baseline
def test_run_baseline_reports_train_and_validation_only(train_frame, validation_frame):
    model, report = run_baseline(train_frame, validation_frame, SMALL_GRID, 3, 42, 0.05, 1, 1000)
    assert isinstance(model, Pipeline)
    assert {"train_metrics", "validation_metrics", "overfitting", "cv_results", "best_params"} <= set(report)
    assert not any("test" in key for key in report)
    assert report["validation_metrics"]["count"] == len(validation_frame)
    assert report["train_metrics"]["count"] == len(train_frame)
    assert report["overfitting"]["gap"] == pytest.approx(
        report["train_metrics"]["accuracy"] - report["validation_metrics"]["accuracy"]
    )
    assert report["best_params"]["tfidf__ngram_range"] == [1, 1]  # tuple đã đổi sang list để ghi JSON


def test_run_baseline_invalid_frames_raise(train_frame):
    with pytest.raises(TypeError):
        run_baseline(train_frame, "validation", SMALL_GRID, 3, 42, 0.05, 1, 1000)
    with pytest.raises(ValueError, match="thiếu cột"):
        run_baseline(train_frame.drop(columns="label"), train_frame, SMALL_GRID, 3, 42, 0.05, 1, 1000)


# =========================================================== save / load model
def test_save_and_load_model_round_trip(tmp_path, train_frame, validation_frame):
    model = tune(train_frame).best_estimator_
    path = save_model(model, tmp_path / "nested" / "model.joblib")
    loaded = load_model(path)
    texts = validation_frame["text"].tolist()
    assert loaded.predict(texts).tolist() == model.predict(texts).tolist()


def test_load_model_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib")
    with pytest.raises(IsADirectoryError):
        load_model(tmp_path)
    with pytest.raises(TypeError):
        load_model(None)
    joblib.dump({"not": "a pipeline"}, tmp_path / "dict.joblib")
    with pytest.raises(TypeError, match="Pipeline"):
        load_model(tmp_path / "dict.joblib")


def test_save_model_errors(tmp_path, train_frame):
    model = tune(train_frame).best_estimator_
    with pytest.raises(IsADirectoryError):
        save_model(model, tmp_path)
    with pytest.raises(TypeError):
        save_model("not a model", tmp_path / "m.joblib")
    with pytest.raises(TypeError):
        save_model(model, 123)
