"""Bài 2 – fine-tune DistilBERT bằng Trainer của Hugging Face.

Ứng với các bước của đề:
    3. Load a pretrained model and its tokenizer  -> load_base_model, AutoTokenizer
    4. Preprocess the dataset                     -> frame_to_dataset (tokenize, cắt câu; pad theo batch)
    5. Define training arguments                  -> build_training_arguments
    6. Create a Trainer object and finetune       -> run_training
    7. Evaluate the finetuned model               -> run_training đánh giá trên validation (và train);
                                                     tập test để dành cho bước đánh giá cuối.
"""

import math
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from lab3.evaluate import overfitting_gap
from lab3.sentiment import validate_max_length

_MAX_SEED = 2**32 - 1


# --------------------------------------------------------------------------- kiểm tra tham số
def _positive_int(value: Any, name: str) -> int:
    """Giả định: value là số nguyên dương (không nhận bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} phải là số nguyên, nhận được {type(value).__name__}.")
    if value < 1:
        raise ValueError(f"{name} phải >= 1, nhận được {value}.")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    """Giả định: value là số thực hữu hạn > 0 (không nhận bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} phải là số, nhận được {type(value).__name__}.")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} phải là số dương hữu hạn, nhận được {value}.")
    return float(value)


def _ratio(value: Any, name: str) -> float:
    """Giả định: value là số thực trong [0, 1)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} phải là số, nhận được {type(value).__name__}.")
    if not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError(f"{name} phải nằm trong [0, 1), nhận được {value}.")
    return float(value)


def _seed(value: Any) -> int:
    """Giả định: seed là số nguyên 0..2^32-1."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"seed phải là số nguyên, nhận được {type(value).__name__}.")
    if not 0 <= value <= _MAX_SEED:
        raise ValueError(f"seed phải nằm trong khoảng 0..{_MAX_SEED}, nhận được {value}.")
    return int(value)


def _path(value: Any, name: str) -> Path:
    """Giả định: value là str hoặc Path, không rỗng."""
    if value is None or isinstance(value, bool) or not isinstance(value, (str, Path)):
        raise TypeError(f"{name} phải là str hoặc Path, nhận được {type(value).__name__}.")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} không được rỗng.")
    return Path(value)


# --------------------------------------------------------------------------- số bước, thời gian
def steps_per_epoch(n_samples: int, batch_size: int, gradient_accumulation_steps: int) -> int:
    """Số lần cập nhật trọng số trong một epoch.

    Giả định: batch cuối thiếu vẫn tính là 1 batch; cứ gradient_accumulation_steps batch thì cập nhật 1 lần
    (nhóm cuối thiếu vẫn tính là 1 lần).
    """
    samples = _positive_int(n_samples, "n_samples")
    batches = math.ceil(samples / _positive_int(batch_size, "batch_size"))
    return math.ceil(batches / _positive_int(gradient_accumulation_steps, "gradient_accumulation_steps"))


def eval_interval(n_samples: int, batch_size: int, gradient_accumulation_steps: int, evals_per_epoch: int) -> int:
    """Cứ bao nhiêu bước cập nhật thì đánh giá validation (và lưu checkpoint) một lần.

    Giả định: đánh giá khoảng evals_per_epoch lần mỗi epoch; ít nhất 1 bước.
    """
    per_epoch = steps_per_epoch(n_samples, batch_size, gradient_accumulation_steps)
    return max(1, per_epoch // _positive_int(evals_per_epoch, "evals_per_epoch"))


def estimate_full_run(
    train_steps_per_second: float,
    eval_samples_per_second: float,
    n_train: int,
    n_validation: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    evals_per_epoch: int,
    include_train_eval: bool,
) -> dict[str, float]:
    """Ước tính thời gian (giây) của lần train đầy đủ từ tốc độ đo được khi chạy thử.

    Giả định:
    - Tốc độ đo ở lần chạy thử giữ nguyên suốt lần chạy thật (cùng máy, cùng max_length, cùng batch).
    - Số lần đánh giá validation = num_epochs × evals_per_epoch.
    - include_train_eval: có chấm điểm lại toàn bộ tập train sau khi train (để so train với validation).
    - Chưa tính thời gian lưu checkpoint và việc máy nóng lên chạy chậm dần.
    """
    train_speed = _positive_float(train_steps_per_second, "train_steps_per_second")
    eval_speed = _positive_float(eval_samples_per_second, "eval_samples_per_second")
    epochs = _positive_int(num_epochs, "num_epochs")
    validation_samples = _positive_int(n_validation, "n_validation")
    if not isinstance(include_train_eval, bool):
        raise TypeError("include_train_eval phải là bool.")

    total_steps = steps_per_epoch(n_train, batch_size, gradient_accumulation_steps) * epochs
    train_seconds = total_steps / train_speed
    validation_seconds = epochs * _positive_int(evals_per_epoch, "evals_per_epoch") * validation_samples / eval_speed
    train_eval_seconds = (_positive_int(n_train, "n_train") / eval_speed) if include_train_eval else 0.0
    return {
        "total_update_steps": float(total_steps),
        "train_seconds": train_seconds,
        "validation_seconds": validation_seconds,
        "train_eval_seconds": train_eval_seconds,
        "total_seconds": train_seconds + validation_seconds + train_eval_seconds,
    }


# --------------------------------------------------------------------------- dữ liệu
def frame_to_dataset(frame: pd.DataFrame, tokenizer: Any, max_length: int) -> Dataset:
    """Bước 4 của đề: tokenize một tập để đưa vào Trainer.

    Giả định:
    - frame có cột text (đã làm sạch) và label (số nguyên).
    - Câu dài hơn max_length bị cắt (giữ phần đầu). KHÔNG pad ở đây: DataCollatorWithPadding sẽ pad
      theo câu dài nhất của từng batch, đỡ tốn tính toán hơn pad cố định tới max_length.
    - Giữ nguyên thứ tự dòng. Trả về Dataset gồm input_ids, attention_mask, labels.
    - Không tạo token_type_ids: transformers 5 nạp tokenizer của DistilBERT dưới dạng BertTokenizer
      (luôn sinh token_type_ids), nhưng DistilBERT không có tham số này nên cột đó chỉ là dữ liệu thừa.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame phải là pandas.DataFrame, nhận được {type(frame).__name__}.")
    missing = [column for column in ("text", "label") if column not in frame.columns]
    if missing:
        raise ValueError(f"frame thiếu cột: {missing}.")
    if frame.empty:
        raise ValueError("frame rỗng.")
    validate_max_length(max_length, tokenizer)

    dataset = Dataset.from_pandas(
        frame.loc[:, ["text", "label"]].rename(columns={"label": "labels"}), preserve_index=False
    )
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length, return_token_type_ids=False),
        batched=True,
        remove_columns=["text"],
        desc="Tokenize",
    )


# --------------------------------------------------------------------------- model, metric
def resolve_model_source(model_name: str) -> str:
    """Kiểm tra tên model và trả về nguồn để from_pretrained đọc.

    Giả định: model_name là id trên Hub hoặc thư mục local. Chuỗi bắt đầu bằng ".", "/", "~"
    hoặc chứa "\\" được hiểu là thư mục local và phải tồn tại.
    """
    if model_name is None or not isinstance(model_name, str):
        raise TypeError(f"model_name phải là str, nhận được {type(model_name).__name__}.")
    if not model_name.strip():
        raise ValueError("model_name không được rỗng.")
    local = Path(model_name).expanduser()
    if local.exists():
        return str(local)
    if model_name.startswith((".", "/", "~")) or "\\" in model_name:
        raise FileNotFoundError(f"Không tìm thấy thư mục model '{local}'.")
    return model_name


def load_base_model(model_name: str, id2label: dict[int, str]) -> tuple[Any, torch.nn.Module]:
    """Bước 3 của đề: tải tokenizer và model pretrained, gắn lớp phân loại mới.

    Giả định:
    - id2label ánh xạ đủ các nhãn 0..C-1; lớp phân loại cuối được khởi tạo ngẫu nhiên với C đầu ra
      (transformers sẽ in cảnh báo "newly initialized", đó là điều mong đợi).
    - disable_mmap=True: đọc trọng số hẳn vào RAM để tensor được căn hàng bộ nhớ đúng
      (cùng lý do với Bài 1, xem docstring của load_sentiment_model).
    """
    if not isinstance(id2label, dict) or not id2label:
        raise TypeError("id2label phải là dict không rỗng {số: tên nhãn}.")
    if sorted(id2label) != list(range(len(id2label))):
        raise ValueError(f"id2label phải có đủ các khoá 0..{len(id2label) - 1}, nhận được {sorted(id2label)}.")
    source = resolve_model_source(model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForSequenceClassification.from_pretrained(
            source,
            num_labels=len(id2label),
            id2label={int(key): value for key, value in id2label.items()},
            label2id={value: int(key) for key, value in id2label.items()},
            disable_mmap=True,
        )
    except OSError as error:
        raise RuntimeError(f"Không tải được model '{model_name}'. Kiểm tra tên model hoặc internet.") from error
    return tokenizer, model


def make_compute_metrics(positive_label: int = 1) -> Callable:
    """Tạo hàm tính metric cho Trainer bằng thư viện evaluate (đề yêu cầu cài evaluate).

    Giả định: nhị phân; precision/recall/F1 tính cho nhãn positive_label; model không đoán nhãn dương
    lần nào thì precision = 0 thay vì báo lỗi. Lần đầu cần internet để tải script metric.
    """
    if isinstance(positive_label, bool) or not isinstance(positive_label, (int, np.integer)):
        raise TypeError("positive_label phải là số nguyên.")
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        """Giả định: predictions là logits (số mẫu, số lớp), label_ids là nhãn thật cùng số mẫu."""
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = np.asarray(logits)
        labels = np.asarray(eval_prediction.label_ids)
        if logits.ndim != 2:
            raise ValueError(f"logits phải có 2 chiều (số mẫu, số lớp), nhận được {logits.ndim} chiều.")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(f"Số logits ({logits.shape[0]}) khác số nhãn ({labels.shape[0]}).")
        predictions = logits.argmax(axis=-1)
        options = {"pos_label": int(positive_label), "average": "binary", "zero_division": 0}
        return {
            "accuracy": float(accuracy.compute(predictions=predictions, references=labels)["accuracy"]),
            "precision": float(precision.compute(predictions=predictions, references=labels, **options)["precision"]),
            "recall": float(recall.compute(predictions=predictions, references=labels, **options)["recall"]),
            "f1": float(f1.compute(predictions=predictions, references=labels, pos_label=int(positive_label),
                                   average="binary")["f1"]),
        }

    return compute_metrics


# --------------------------------------------------------------------------- tham số train
def build_training_arguments(
    output_dir: str | Path,
    *,
    num_epochs: int,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    weight_decay: float,
    warmup_ratio: float,
    max_grad_norm: float,
    eval_steps: int,
    logging_steps: int,
    save_total_limit: int,
    metric_for_best_model: str,
    seed: int,
    use_cpu: bool,
    max_steps: int = -1,
) -> TrainingArguments:
    """Bước 5 của đề: khai báo tham số train.

    Giả định:
    - Đánh giá validation và lưu checkpoint cùng nhịp (mỗi eval_steps bước), cuối cùng nạp lại
      checkpoint có metric_for_best_model cao nhất (load_best_model_at_end).
    - Optimizer AdamW mặc định của Trainer; learning rate tăng dần trong warmup_ratio đầu rồi giảm tuyến tính.
    - Gradient clipping max_grad_norm; regularization weight_decay; seed cho cả model và thứ tự dữ liệu.
    - Không gửi log đi đâu (report_to="none"); không dùng tiến trình phụ đọc dữ liệu (ổn định trên macOS).
    - max_steps = -1 nghĩa là chạy đủ num_epochs; số dương thì dừng sau đúng số bước đó (dùng khi chạy thử).
    """
    if not isinstance(use_cpu, bool):
        raise TypeError("use_cpu phải là bool.")
    if not isinstance(metric_for_best_model, str) or not metric_for_best_model.strip():
        raise ValueError("metric_for_best_model phải là tên metric (chuỗi không rỗng).")
    if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)):
        raise TypeError(f"max_steps phải là số nguyên, nhận được {type(max_steps).__name__}.")
    if max_steps != -1 and max_steps < 1:
        raise ValueError(f"max_steps phải là -1 hoặc >= 1, nhận được {max_steps}.")
    decay = _ratio(weight_decay, "weight_decay")

    return TrainingArguments(
        output_dir=str(_path(output_dir, "output_dir")),
        num_train_epochs=_positive_int(num_epochs, "num_epochs"),
        max_steps=int(max_steps),
        learning_rate=_positive_float(learning_rate, "learning_rate"),
        per_device_train_batch_size=_positive_int(train_batch_size, "train_batch_size"),
        per_device_eval_batch_size=_positive_int(eval_batch_size, "eval_batch_size"),
        gradient_accumulation_steps=_positive_int(gradient_accumulation_steps, "gradient_accumulation_steps"),
        weight_decay=float(decay),
        warmup_steps=_ratio(warmup_ratio, "warmup_ratio"),
        lr_scheduler_type="linear",
        max_grad_norm=_positive_float(max_grad_norm, "max_grad_norm"),
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=_positive_int(eval_steps, "eval_steps"),
        save_steps=int(eval_steps),
        logging_steps=_positive_int(logging_steps, "logging_steps"),
        save_total_limit=_positive_int(save_total_limit, "save_total_limit"),
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        seed=_seed(seed),
        data_seed=int(seed),
        use_cpu=use_cpu,
        dataloader_num_workers=0,
        report_to="none",
    )


def prepare_checkpoint_dir(path: str | Path, resume: bool, overwrite: bool) -> Path:
    """Kiểm tra thư mục checkpoint trước khi train.

    Giả định:
    - resume=True: phải có sẵn checkpoint để chạy tiếp.
    - resume=False và đã có checkpoint cũ: báo lỗi để không trộn checkpoint của 2 lần chạy khác nhau,
      trừ khi overwrite=True thì xoá các thư mục checkpoint-* cũ (chỉ xoá đúng các thư mục đó).
    - Không được bật resume và overwrite cùng lúc.
    """
    folder = _path(path, "checkpoint_dir")
    if not isinstance(resume, bool) or not isinstance(overwrite, bool):
        raise TypeError("resume và overwrite phải là bool.")
    if resume and overwrite:
        raise ValueError("Không thể vừa --resume vừa --overwrite.")
    if folder.exists() and not folder.is_dir():
        raise NotADirectoryError(f"'{folder}' không phải thư mục.")
    existing = sorted(folder.glob("checkpoint-*")) if folder.exists() else []
    if resume and not existing:
        raise ValueError(f"Không có checkpoint nào trong '{folder}' để chạy tiếp (--resume).")
    if existing and not resume:
        if not overwrite:
            raise ValueError(
                f"'{folder}' đã có {len(existing)} checkpoint từ lần chạy trước. "
                "Dùng --resume để chạy tiếp, hoặc --overwrite để xoá và train lại từ đầu."
            )
        for checkpoint in existing:
            shutil.rmtree(checkpoint)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


class StepTimer(TrainerCallback):
    """Đo thời gian mỗi bước cập nhật khi chạy thử, bỏ qua vài bước khởi động đầu tiên.

    Giả định: đồng bộ GPU (MPS/CUDA) ở cuối mỗi bước để thời gian đo phản ánh tính toán thật.
    """

    def __init__(self, ignored_steps: int):
        self.ignored_steps = int(ignored_steps)
        self.start: float | None = None
        self.steps_timed = 0
        self.elapsed = 0.0

    @staticmethod
    def _synchronize() -> None:
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_step_end(self, args, state, control, **kwargs):
        self._synchronize()
        now = time.perf_counter()
        if state.global_step == self.ignored_steps:
            self.start = now
        elif state.global_step > self.ignored_steps and self.start is not None:
            self.steps_timed = state.global_step - self.ignored_steps
            self.elapsed = now - self.start
        return control

    def steps_per_second(self) -> float:
        if self.start is None or self.steps_timed == 0:
            raise RuntimeError("Chưa đo được tốc độ: số bước chạy thử phải lớn hơn số bước khởi động bỏ qua.")
        return self.steps_timed / self.elapsed


# --------------------------------------------------------------------------- chạy train
def _metrics_without_prefix(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    """Bỏ tiền tố "eval_" / "train_" khỏi tên metric. Giả định: chỉ giữ accuracy, precision, recall, f1, loss."""
    keep = ("accuracy", "precision", "recall", "f1", "loss")
    return {name: float(metrics[f"{prefix}_{name}"]) for name in keep if f"{prefix}_{name}" in metrics}


def run_training(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    model_name: str,
    id2label: dict[int, str],
    checkpoint_dir: str | Path,
    final_model_dir: str | Path | None,
    max_length: int,
    num_epochs: int,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    weight_decay: float,
    warmup_ratio: float,
    max_grad_norm: float,
    evals_per_epoch: int,
    early_stopping_patience: int,
    metric_for_best_model: str,
    logging_steps: int,
    save_total_limit: int,
    seed: int,
    use_cpu: bool,
    gap_threshold: float,
    resume: bool = False,
    smoke_max_steps: int | None = None,
    smoke_ignored_steps: int = 0,
) -> dict[str, Any]:
    """Bước 6 và 7 của đề: fine-tune bằng Trainer, rồi đánh giá model tốt nhất.

    Giả định:
    - train / validation đã chia, làm sạch, loại trùng ở bước dữ liệu. Không có tập test ở đây.
    - Chọn checkpoint tốt nhất và dừng sớm dựa trên validation, không bao giờ dựa trên test.
    - Chạy thật (smoke_max_steps=None): chấm điểm lại cả train để so với validation (cảnh báo overfitting),
      lưu model tốt nhất vào final_model_dir.
    - Chạy thử (smoke_max_steps là số bước): chỉ đo tốc độ, không chấm train, không lưu final model.
    """
    smoke = smoke_max_steps is not None
    if not smoke and final_model_dir is None:
        raise ValueError("Cần final_model_dir khi chạy thật.")
    if smoke and _positive_int(smoke_max_steps, "smoke_max_steps") <= smoke_ignored_steps:
        raise ValueError("smoke_max_steps phải lớn hơn số bước khởi động bỏ qua.")

    set_seed(_seed(seed))
    tokenizer, model = load_base_model(model_name, id2label)
    train_dataset = frame_to_dataset(train, tokenizer, max_length)
    validation_dataset = frame_to_dataset(validation, tokenizer, max_length)

    interval = eval_interval(len(train_dataset), train_batch_size, gradient_accumulation_steps, evals_per_epoch)
    if smoke:
        interval = int(smoke_max_steps)
    arguments = build_training_arguments(
        checkpoint_dir,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        eval_steps=interval,
        logging_steps=min(logging_steps, interval),
        save_total_limit=save_total_limit,
        metric_for_best_model=metric_for_best_model,
        seed=seed,
        use_cpu=use_cpu,
        max_steps=int(smoke_max_steps) if smoke else -1,
    )

    callbacks: list[TrainerCallback] = [
        EarlyStoppingCallback(early_stopping_patience=_positive_int(early_stopping_patience, "early_stopping_patience"))
    ]
    timer = StepTimer(smoke_ignored_steps) if smoke else None
    if timer is not None:
        callbacks.append(timer)

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=make_compute_metrics(),
        callbacks=callbacks,
    )

    started = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=True if resume else None)
    validation_metrics = trainer.evaluate()
    report: dict[str, Any] = {
        "mode": "smoke" if smoke else "full",
        "model_name": model_name,
        "device": str(trainer.args.device),
        "n_train": len(train_dataset),
        "n_validation": len(validation_dataset),
        "hyperparameters": {
            "max_length": int(max_length),
            "num_epochs": int(num_epochs),
            "learning_rate": float(learning_rate),
            "train_batch_size": int(train_batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "effective_batch_size": int(train_batch_size) * int(gradient_accumulation_steps),
            "weight_decay": float(weight_decay),
            "warmup_ratio": float(warmup_ratio),
            "max_grad_norm": float(max_grad_norm),
            "eval_steps": int(interval),
            "early_stopping_patience": int(early_stopping_patience),
            "metric_for_best_model": metric_for_best_model,
            "seed": int(seed),
        },
        "train_runtime_seconds": float(train_result.metrics.get("train_runtime", 0.0)),
        "train_loss": float(train_result.metrics.get("train_loss", float("nan"))),
        "global_steps": int(trainer.state.global_step),
        "validation_metrics": _metrics_without_prefix(validation_metrics, "eval"),
        "eval_samples_per_second": float(validation_metrics["eval_samples_per_second"]),
        "log_history": trainer.state.log_history,
    }

    if smoke:
        report["train_steps_per_second"] = timer.steps_per_second()
        report["wall_seconds"] = time.perf_counter() - started
        return report

    # Early stopping phản ứng với mọi lần evaluate; lần chấm tập train này không phải validation nên gỡ ra trước.
    trainer.remove_callback(EarlyStoppingCallback)
    train_metrics = trainer.evaluate(train_dataset, metric_key_prefix="train")
    report["train_metrics"] = _metrics_without_prefix(train_metrics, "train")
    report["overfitting"] = overfitting_gap(
        report["train_metrics"]["accuracy"], report["validation_metrics"]["accuracy"], gap_threshold
    )
    report["best_checkpoint"] = trainer.state.best_model_checkpoint
    report["best_metric"] = trainer.state.best_metric
    report["planned_steps"] = int(trainer.state.max_steps)
    report["stopped_early"] = trainer.state.global_step < trainer.state.max_steps

    final_dir = _path(final_model_dir, "final_model_dir")
    trainer.save_model(str(final_dir))
    report["final_model_dir"] = str(final_dir)
    report["wall_seconds"] = time.perf_counter() - started
    return report
