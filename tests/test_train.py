"""Test cho lab3.train.

Phần đầu chạy nhanh, không cần model. Phần đánh dấu slow dùng model DistilBERT tí hon (tiny_model_dir)
để chạy thật toàn bộ Trainer trên CPU trong vài giây.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from helpers import synthetic_reviews
from lab3 import config
from lab3.evaluate import classification_metrics
from lab3.sentiment import load_sentiment_model, predict_sentiment
from lab3.train import (
    StepTimer,
    build_training_arguments,
    estimate_full_run,
    eval_interval,
    frame_to_dataset,
    load_base_model,
    make_compute_metrics,
    prepare_checkpoint_dir,
    resolve_model_source,
    run_training,
    steps_per_epoch,
)


# ======================================================== số bước, ước tính
@pytest.mark.parametrize("n, batch, accum, expected", [(22297, 16, 1, 1394), (10, 16, 1, 1), (100, 8, 2, 7), (16, 16, 1, 1), (17, 16, 1, 2)])
def test_steps_per_epoch(n, batch, accum, expected):
    assert steps_per_epoch(n, batch, accum) == expected


@pytest.mark.parametrize("args, error", [((0, 16, 1), ValueError), ((100, 0, 1), ValueError), ((100, 16, -1), ValueError), ((100.0, 16, 1), TypeError), ((100, True, 1), TypeError)])
def test_steps_per_epoch_invalid(args, error):
    with pytest.raises(error):
        steps_per_epoch(*args)


@pytest.mark.parametrize("n, batch, accum, per_epoch, expected", [(22297, 16, 1, 2, 697), (5, 16, 1, 2, 1), (100, 10, 1, 3, 3)])
def test_eval_interval(n, batch, accum, per_epoch, expected):
    assert eval_interval(n, batch, accum, per_epoch) == expected


def test_eval_interval_invalid():
    with pytest.raises(ValueError):
        eval_interval(100, 10, 1, 0)


def test_estimate_full_run_known_values():
    # 1000 review / batch 10 = 100 bước mỗi epoch, 2 epoch = 200 bước, 2 bước/giây -> 100 giây.
    estimate = estimate_full_run(2.0, 100.0, 1000, 200, 10, 1, 2, 2, True)
    assert estimate["total_update_steps"] == 200
    assert estimate["train_seconds"] == pytest.approx(100.0)
    assert estimate["validation_seconds"] == pytest.approx(8.0)  # 4 lần × 200 review / 100 review/giây
    assert estimate["train_eval_seconds"] == pytest.approx(10.0)
    assert estimate["total_seconds"] == pytest.approx(118.0)


def test_estimate_full_run_without_train_eval():
    assert estimate_full_run(2.0, 100.0, 1000, 200, 10, 1, 2, 2, False)["total_seconds"] == pytest.approx(108.0)


@pytest.mark.parametrize("speed, error", [(0, ValueError), (-1.0, ValueError), (float("nan"), ValueError), (float("inf"), ValueError), ("2", TypeError), (None, TypeError)])
def test_estimate_full_run_invalid_speed(speed, error):
    with pytest.raises(error):
        estimate_full_run(speed, 100.0, 1000, 200, 10, 1, 2, 2, True)


def test_estimate_full_run_include_flag_must_be_bool():
    with pytest.raises(TypeError):
        estimate_full_run(2.0, 100.0, 1000, 200, 10, 1, 2, 2, "yes")


# ================================================= build_training_arguments
def make_arguments(tmp_path, **overrides):
    options = dict(
        num_epochs=2, learning_rate=2e-5, train_batch_size=16, eval_batch_size=64, gradient_accumulation_steps=1,
        weight_decay=0.01, warmup_ratio=0.1, max_grad_norm=1.0, eval_steps=697, logging_steps=50,
        save_total_limit=2, metric_for_best_model="accuracy", seed=42, use_cpu=True,
    )
    options.update(overrides)
    return build_training_arguments(tmp_path / "ck", **options)


def test_training_arguments_follow_the_plan(tmp_path):
    args = make_arguments(tmp_path)
    assert args.eval_strategy == "steps" and args.save_strategy == "steps"
    assert args.eval_steps == args.save_steps == 697
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "accuracy" and args.greater_is_better is True
    assert args.learning_rate == 2e-5 and args.weight_decay == 0.01
    assert args.warmup_steps == 0.1 and args.lr_scheduler_type == "linear"
    assert args.max_grad_norm == 1.0
    assert args.seed == 42 and args.data_seed == 42
    assert args.save_total_limit == 2
    assert args.report_to in ("none", [], ["none"])


@pytest.mark.parametrize("overrides, error", [
    ({"train_batch_size": 0}, ValueError),
    ({"learning_rate": 0}, ValueError),
    ({"learning_rate": "2e-5"}, TypeError),
    ({"warmup_ratio": 1.0}, ValueError),
    ({"weight_decay": -0.1}, ValueError),
    ({"max_grad_norm": 0}, ValueError),
    ({"eval_steps": 0}, ValueError),
    ({"num_epochs": 0}, ValueError),
    ({"max_steps": 0}, ValueError),
    ({"max_steps": 1.5}, TypeError),
    ({"use_cpu": "no"}, TypeError),
    ({"metric_for_best_model": ""}, ValueError),
    ({"seed": -1}, ValueError),
])
def test_training_arguments_invalid(tmp_path, overrides, error):
    with pytest.raises(error):
        make_arguments(tmp_path, **overrides)


# =================================================== prepare_checkpoint_dir
def test_checkpoint_dir_is_created(tmp_path):
    assert prepare_checkpoint_dir(tmp_path / "new", resume=False, overwrite=False).is_dir()


def test_checkpoint_dir_refuses_to_mix_runs(tmp_path):
    (tmp_path / "checkpoint-10").mkdir()
    with pytest.raises(ValueError, match="--resume"):
        prepare_checkpoint_dir(tmp_path, resume=False, overwrite=False)


def test_checkpoint_dir_overwrite_removes_only_checkpoints(tmp_path):
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "ghi_chu.txt").write_text("giữ lại")
    prepare_checkpoint_dir(tmp_path, resume=False, overwrite=True)
    assert not (tmp_path / "checkpoint-10").exists()
    assert (tmp_path / "ghi_chu.txt").exists()


def test_checkpoint_dir_resume_needs_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="chạy tiếp"):
        prepare_checkpoint_dir(tmp_path, resume=True, overwrite=False)
    (tmp_path / "checkpoint-5").mkdir()
    prepare_checkpoint_dir(tmp_path, resume=True, overwrite=False)
    assert (tmp_path / "checkpoint-5").exists()


def test_checkpoint_dir_invalid_inputs(tmp_path):
    with pytest.raises(ValueError):
        prepare_checkpoint_dir(tmp_path, resume=True, overwrite=True)
    (tmp_path / "file.txt").write_text("x")
    with pytest.raises(NotADirectoryError):
        prepare_checkpoint_dir(tmp_path / "file.txt", resume=False, overwrite=False)
    with pytest.raises(TypeError):
        prepare_checkpoint_dir(None, resume=False, overwrite=False)
    with pytest.raises(TypeError):
        prepare_checkpoint_dir(tmp_path, resume="no", overwrite=False)


# ======================================================== model source, labels
def test_resolve_model_source(tmp_path):
    assert resolve_model_source("distilbert/distilbert-base-uncased") == "distilbert/distilbert-base-uncased"
    assert resolve_model_source(str(tmp_path)) == str(tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_model_source("./khong_co_model")
    with pytest.raises(TypeError):
        resolve_model_source(None)
    with pytest.raises(ValueError):
        resolve_model_source("  ")


@pytest.mark.parametrize("labels, error", [({}, TypeError), (None, TypeError), ({0: "a", 2: "b"}, ValueError), ({1: "a"}, ValueError)])
def test_load_base_model_invalid_labels(labels, error):
    with pytest.raises(error):
        load_base_model("distilbert/distilbert-base-uncased", labels)


# ================================================================ StepTimer
def test_step_timer_measures_after_ignored_steps():
    timer = StepTimer(ignored_steps=2)
    for step in range(1, 7):
        timer.on_step_end(None, SimpleNamespace(global_step=step), None)
    assert timer.steps_timed == 4
    assert timer.steps_per_second() > 0


def test_step_timer_without_enough_steps_raises():
    timer = StepTimer(ignored_steps=5)
    timer.on_step_end(None, SimpleNamespace(global_step=1), None)
    with pytest.raises(RuntimeError):
        timer.steps_per_second()


# ====================================================== cần tải từ Hub (slow)
@pytest.mark.slow
def test_compute_metrics_matches_sklearn_version():
    compute = make_compute_metrics()
    logits = np.array([[2.0, 1.0], [0.0, 3.0], [1.0, 0.0], [0.5, 0.2], [0.1, 0.9]])
    labels = np.array([0, 1, 1, 0, 1])
    result = compute(SimpleNamespace(predictions=logits, label_ids=labels))
    expected = classification_metrics(labels.tolist(), logits.argmax(axis=-1).tolist())
    for name in ("accuracy", "precision", "recall", "f1"):
        assert result[name] == pytest.approx(expected[name])


@pytest.mark.slow
def test_compute_metrics_handles_tuple_and_no_positive_prediction():
    compute = make_compute_metrics()
    logits = np.array([[3.0, 0.0], [2.0, 1.0]])
    result = compute(SimpleNamespace(predictions=(logits,), label_ids=np.array([1, 0])))
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["precision"] == 0.0 and result["f1"] == 0.0


@pytest.mark.slow
@pytest.mark.parametrize("logits, labels", [(np.array([0.1, 0.9]), np.array([1, 0])), (np.zeros((3, 2)), np.array([0, 1]))])
def test_compute_metrics_invalid_shapes(logits, labels):
    with pytest.raises(ValueError):
        make_compute_metrics()(SimpleNamespace(predictions=logits, label_ids=labels))


@pytest.mark.slow
def test_frame_to_dataset(tiny_model_dir):
    tokenizer, _ = load_base_model(str(tiny_model_dir), config.ID2LABEL)
    frame = synthetic_reviews(5, seed=0)
    frame.loc[0, "text"] = "word " * 500
    dataset = frame_to_dataset(frame, tokenizer, 32)
    assert set(dataset.column_names) == {"input_ids", "attention_mask", "labels"}
    assert dataset["labels"] == frame["label"].tolist()
    assert max(len(ids) for ids in dataset["input_ids"]) == 32
    assert len(dataset["input_ids"][1]) < 32  # không pad sẵn


@pytest.mark.slow
def test_frame_to_dataset_invalid(tiny_model_dir):
    tokenizer, _ = load_base_model(str(tiny_model_dir), config.ID2LABEL)
    frame = synthetic_reviews(2, seed=0)
    with pytest.raises(ValueError):
        frame_to_dataset(frame, tokenizer, 1)
    with pytest.raises(ValueError, match="thiếu cột"):
        frame_to_dataset(frame.drop(columns="label"), tokenizer, 32)
    with pytest.raises(ValueError, match="rỗng"):
        frame_to_dataset(frame.iloc[0:0], tokenizer, 32)
    with pytest.raises(TypeError):
        frame_to_dataset([("a", 1)], tokenizer, 32)


def tiny_options(tiny_model_dir, **overrides):
    options = dict(
        model_name=str(tiny_model_dir), id2label=config.ID2LABEL, max_length=32, num_epochs=2,
        learning_rate=5e-4, train_batch_size=8, eval_batch_size=16, gradient_accumulation_steps=1,
        weight_decay=0.01, warmup_ratio=0.1, max_grad_norm=1.0, evals_per_epoch=2,
        early_stopping_patience=2, metric_for_best_model="accuracy", logging_steps=2, save_total_limit=2,
        seed=42, use_cpu=True, gap_threshold=0.05,
    )
    options.update(overrides)
    return options


@pytest.fixture(scope="module")
def full_run(tiny_model_dir, tmp_path_factory):
    folder = tmp_path_factory.mktemp("full_run")
    report = run_training(
        synthetic_reviews(40, seed=1), synthetic_reviews(10, seed=2),
        checkpoint_dir=folder / "ck", final_model_dir=folder / "final", **tiny_options(tiny_model_dir),
    )
    return folder, report


@pytest.mark.slow
def test_full_run_report_and_final_model(full_run):
    folder, report = full_run
    assert report["mode"] == "full"
    assert report["n_train"] == 80 and report["n_validation"] == 20
    assert {"train_metrics", "validation_metrics", "overfitting", "best_checkpoint", "log_history"} <= set(report)
    assert not any("test" in key for key in report)
    assert report["overfitting"]["gap"] == pytest.approx(
        report["train_metrics"]["accuracy"] - report["validation_metrics"]["accuracy"]
    )
    assert report["best_checkpoint"].startswith(str(folder / "ck"))
    assert 0 <= report["validation_metrics"]["accuracy"] <= 1
    assert report["hyperparameters"]["effective_batch_size"] == 8
    assert math.isfinite(report["train_loss"])
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        assert (folder / "final" / name).exists()


@pytest.mark.slow
def test_final_model_reloads_and_predicts_on_cpu(full_run):
    folder, _ = full_run
    tokenizer, model = load_sentiment_model(str(folder / "final"), torch.device("cpu"))
    prediction = predict_sentiment(model, tokenizer, "a good film", 32)[0]
    assert prediction["label"] in config.ID2LABEL.values()


@pytest.mark.slow
def test_resume_continues_from_last_checkpoint_instead_of_restarting(full_run, tiny_model_dir):
    folder, first = full_run
    resumed = run_training(
        synthetic_reviews(40, seed=1), synthetic_reviews(10, seed=2), checkpoint_dir=folder / "ck",
        final_model_dir=folder / "final_resumed", resume=True, **tiny_options(tiny_model_dir),
    )
    logged_steps = [entry["step"] for entry in resumed["log_history"] if "loss" in entry]
    assert logged_steps == sorted(set(logged_steps))  # không có bước nào bị train lại
    assert first["global_steps"] <= resumed["global_steps"] <= resumed["planned_steps"]


@pytest.mark.slow
def test_full_run_logs_no_early_stopping_warning_for_train_evaluation(tiny_model_dir, tmp_path, caplog):
    run_training(
        synthetic_reviews(10, seed=1), synthetic_reviews(4, seed=2), checkpoint_dir=tmp_path / "ck",
        final_model_dir=tmp_path / "final", **tiny_options(tiny_model_dir, num_epochs=1),
    )
    assert "early stopping is disabled" not in caplog.text


@pytest.mark.slow
def test_smoke_run_measures_speed_without_saving(tiny_model_dir, tmp_path):
    report = run_training(
        synthetic_reviews(30, seed=3), synthetic_reviews(8, seed=4), checkpoint_dir=tmp_path / "ck",
        final_model_dir=None, smoke_max_steps=6, smoke_ignored_steps=2, **tiny_options(tiny_model_dir),
    )
    assert report["mode"] == "smoke"
    assert report["global_steps"] == 6
    assert report["train_steps_per_second"] > 0 and report["eval_samples_per_second"] > 0
    assert "train_metrics" not in report


@pytest.mark.slow
def test_smoke_steps_must_exceed_ignored_steps(tiny_model_dir, tmp_path):
    with pytest.raises(ValueError):
        run_training(
            synthetic_reviews(10, seed=3), synthetic_reviews(4, seed=4), checkpoint_dir=tmp_path,
            final_model_dir=None, smoke_max_steps=2, smoke_ignored_steps=2, **tiny_options(tiny_model_dir),
        )


def test_full_run_requires_final_dir(tmp_path):
    with pytest.raises(ValueError, match="final_model_dir"):
        run_training(
            synthetic_reviews(10, seed=3), synthetic_reviews(4, seed=4), checkpoint_dir=tmp_path,
            final_model_dir=None, **tiny_options("unused"),
        )
