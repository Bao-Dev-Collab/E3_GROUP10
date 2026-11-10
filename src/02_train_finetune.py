"""Bài 2: fine-tune DistilBERT cho bài toán phân loại nhị phân trên IMDb.

Script chạy theo từng giai đoạn (--stage). Hiện có:
    data     – đọc IMDb, chia train/validation/test, làm sạch, loại trùng, thống kê độ dài token
               (bước 2 "Load a simple dataset" của đề, phần chuẩn bị dữ liệu).
    baseline – TF-IDF + LogisticRegression, chọn tham số bằng 5-fold CV trên train, báo cáo train
               và validation. Không dùng tập test.
    train    – fine-tune DistilBERT bằng Trainer (bước 3–7 của đề). Không dùng tập test.
               --smoke: chạy thử vài bước để bắt lỗi và ước tính thời gian, không ghi đè model thật.
    evaluate – chấm DistilBERT và baseline trên tập TEST (một lần, sau khi đã chốt model): metrics,
               confusion matrix, đồ thị quá trình train, phân tích theo độ dài, ví dụ đoán sai.

Cách chạy (từ thư mục lab3_DL/):
    python src/02_train_finetune.py --stage data
    python src/02_train_finetune.py --stage baseline
    python src/02_train_finetune.py --stage train --smoke
    python src/02_train_finetune.py --stage train
    python src/02_train_finetune.py --stage train --resume      # chạy tiếp nếu lần trước bị dừng
    python src/02_train_finetune.py --stage evaluate
"""

import argparse
import sys
import tempfile
from pathlib import Path

# Cho phép chạy trực tiếp file này mà vẫn import được package lab3 trong src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab3 import config  # noqa: E402
from lab3.baseline import run_baseline, save_model  # noqa: E402
from lab3.data import load_split, load_tokenizer, prepare_data  # noqa: E402
from lab3.final_evaluation import run_final_evaluation  # noqa: E402
from lab3.train import estimate_full_run, prepare_checkpoint_dir, run_training  # noqa: E402
from lab3.utils import get_device, load_json, save_json  # noqa: E402

DEFAULT_REPORTS = {
    "data": config.DATA_REPORT_PATH,
    "baseline": config.BASELINE_REPORT_PATH,
    "train": config.TRAIN_REPORT_PATH,
    "evaluate": config.EVALUATION_REPORT_PATH,
}

EXIT_OK = 0
EXIT_INPUT_ERROR = 2


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh. Giả định: argv = None nghĩa là đọc từ sys.argv."""
    parser = argparse.ArgumentParser(description="Bài 2: fine-tune DistilBERT trên IMDb.")
    parser.add_argument("--stage", required=True, choices=sorted(DEFAULT_REPORTS), help="Giai đoạn cần chạy.")
    parser.add_argument("--raw-train", type=Path, default=config.RAW_TRAIN_PATH, help="CSV train gốc.")
    parser.add_argument("--raw-test", type=Path, default=config.RAW_TEST_PATH, help="CSV test gốc.")
    parser.add_argument("--splits-dir", type=Path, default=config.SPLITS_DIR, help="Thư mục chứa các tập đã chia.")
    parser.add_argument("--report", type=Path, default=None, help="File JSON báo cáo. Mặc định theo giai đoạn.")
    parser.add_argument("--val-size", type=float, default=config.VAL_SIZE, help="Tỉ lệ validation tách từ train.")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Seed cho chia dữ liệu và baseline.")
    parser.add_argument("--model", default=config.BASE_MODEL_NAME, help="Model lấy tokenizer để đếm token.")
    parser.add_argument("--cv-folds", type=int, default=config.CV_FOLDS, help="Số fold cross-validation (baseline).")
    parser.add_argument("--n-jobs", type=int, default=config.BASELINE_N_JOBS, help="Số tiến trình song song (baseline).")
    parser.add_argument("--baseline-model", type=Path, default=config.BASELINE_MODEL_PATH, help="File lưu model baseline.")
    parser.add_argument("--smoke", action="store_true", help="Chạy thử vài bước để đo tốc độ (train).")
    parser.add_argument("--resume", action="store_true", help="Chạy tiếp từ checkpoint gần nhất (train).")
    parser.add_argument("--overwrite", action="store_true", help="Xoá checkpoint cũ và train lại từ đầu (train).")
    parser.add_argument("--max-length", type=int, default=config.TRAIN_MAX_LENGTH, help="Số token tối đa mỗi review.")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Số epoch.")
    parser.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE, help="Batch size khi train.")
    parser.add_argument("--grad-accum", type=int, default=config.GRADIENT_ACCUMULATION_STEPS, help="Số batch cộng dồn gradient.")
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--device", default=None, help="cuda, mps hoặc cpu. Mặc định: tự chọn.")
    parser.add_argument("--checkpoints-dir", type=Path, default=config.CHECKPOINTS_DIR, help="Thư mục checkpoint.")
    parser.add_argument("--final-model-dir", type=Path, default=config.FINAL_MODEL_DIR, help="Thư mục lưu model tốt nhất.")
    parser.add_argument("--train-report", type=Path, default=config.TRAIN_REPORT_PATH, help="Báo cáo train (evaluate).")
    args = parser.parse_args(argv)
    if args.report is None:
        args.report = config.SMOKE_REPORT_PATH if (args.stage == "train" and args.smoke) else DEFAULT_REPORTS[args.stage]
    return args


def print_data_report(report: dict) -> None:
    """In tóm tắt báo cáo dữ liệu. Giả định: report do prepare_data tạo."""
    print("\nSố dòng gốc:", report["raw_rows"])
    print("Bỏ vì rỗng sau khi làm sạch:", report["empty_after_cleaning_removed"])
    print("Loại trùng:")
    for key, value in report["duplicates"].items():
        print(f"  {key}: {value}")
    print("Số dòng cuối cùng:", report["final_rows"])
    print("Phân bố nhãn:", report["label_counts"])
    stats = report["train_token_length"]
    print(
        f"Độ dài token (chỉ tập train): trung bình {stats['mean']:.1f}, trung vị {stats['median']:.0f}, "
        f"p90 {stats['p90']:.0f}, p95 {stats['p95']:.0f}, dài nhất {stats['max']}"
    )
    for mark, fraction in stats["fraction_longer_than"].items():
        print(f"  Tỉ lệ review dài hơn {mark} token (sẽ bị cắt nếu max_length={mark}): {fraction:.1%}")
    for name, info in report["files"].items():
        print(f"Đã lưu {name}: {info['path']} ({info['rows']} dòng)")


def run_data_stage(args: argparse.Namespace) -> dict:
    """Giai đoạn dữ liệu. Giả định: các tham số đường dẫn và số đã được argparse đọc đúng kiểu."""
    tokenizer = load_tokenizer(args.model)
    report = prepare_data(
        raw_train_path=args.raw_train,
        raw_test_path=args.raw_test,
        output_dir=args.splits_dir,
        val_size=args.val_size,
        seed=args.seed,
        tokenizer=tokenizer,
        thresholds=config.TOKEN_LENGTH_THRESHOLDS,
        allowed_labels=config.ID2LABEL.keys(),
    )
    report["tokenizer"] = args.model
    print_data_report(report)
    saved = save_json(report, args.report)
    print(f"Đã lưu báo cáo: {saved}")
    return report


def print_baseline_report(report: dict) -> None:
    """In kết quả cross-validation và điểm train / validation. Giả định: report do run_baseline tạo."""
    print(f"\nCross-validation {report['cv_folds']} fold trên tập train (xếp theo accuracy):")
    print(f"  {'Hạng':>4}  {'n-gram':<8}{'C':>6}  {'CV accuracy':>16}  {'CV F1':>7}  {'Train acc':>9}")
    for row in report["cv_results"]:
        params = row["params"]
        ngram = "-".join(str(n) for n in params.get("tfidf__ngram_range", []))
        print(
            f"  {row['rank']:>4}  {ngram:<8}{params.get('clf__C', ''):>6}  "
            f"{row['cv_accuracy_mean']:.4f} ± {row['cv_accuracy_std']:.4f}  "
            f"{row['cv_f1_mean']:.4f}  {row['train_accuracy_mean']:>9.4f}"
        )
    print(f"\nTham số tốt nhất: {report['best_params']} (CV accuracy {report['best_cv_accuracy']:.4f})")
    print(f"Số từ / cụm từ TF-IDF: {report['vocabulary_size']:,}")
    for name in ("train_metrics", "validation_metrics"):
        metrics = report[name]
        print(
            f"{name.split('_')[0]:<10}: accuracy {metrics['accuracy']:.4f} | precision {metrics['precision']:.4f} "
            f"| recall {metrics['recall']:.4f} | F1 {metrics['f1']:.4f} ({metrics['count']} review)"
        )
    gap = report["overfitting"]
    print(f"Chênh lệch accuracy train − validation: {gap['gap']:.4f} (ngưỡng {gap['threshold']})")
    if gap["overfitting_warning"]:
        print(
            "CẢNH BÁO: train cao hơn validation quá ngưỡng – mô hình học thuộc một phần tập train (overfitting).",
            file=sys.stderr,
        )


def run_baseline_stage(args: argparse.Namespace) -> dict:
    """Giai đoạn baseline. Giả định: data/splits đã có train.csv, validation.csv; KHÔNG đọc test.csv."""
    labels = config.ID2LABEL.keys()
    train = load_split(args.splits_dir / config.TRAIN_SPLIT_PATH.name, labels)
    validation = load_split(args.splits_dir / config.VALIDATION_SPLIT_PATH.name, labels)
    model, report = run_baseline(
        train,
        validation,
        param_grid=config.BASELINE_PARAM_GRID,
        n_splits=args.cv_folds,
        seed=args.seed,
        gap_threshold=config.OVERFIT_GAP_THRESHOLD,
        n_jobs=args.n_jobs,
        max_features=config.BASELINE_MAX_FEATURES,
    )
    print_baseline_report(report)
    saved_model = save_model(model, args.baseline_model)
    report["model_path"] = str(saved_model)
    saved_report = save_json(report, args.report)
    print(f"Đã lưu model: {saved_model}\nĐã lưu báo cáo: {saved_report}")
    return report


def _format_duration(seconds: float) -> str:
    """Đổi giây sang dạng "1 giờ 23 phút". Giả định: seconds >= 0."""
    minutes = round(seconds / 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours} giờ {minutes} phút" if hours else f"{minutes} phút"


def _training_options(args: argparse.Namespace) -> dict:
    """Gom tham số train từ dòng lệnh và config. Giả định: args đã qua argparse; device đã kiểm tra."""
    device = get_device(args.device)
    return {
        "model_name": args.model,
        "id2label": config.ID2LABEL,
        "max_length": args.max_length,
        "num_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.batch_size,
        "eval_batch_size": config.EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": args.grad_accum,
        "weight_decay": config.WEIGHT_DECAY,
        "warmup_ratio": config.WARMUP_RATIO,
        "max_grad_norm": config.MAX_GRAD_NORM,
        "evals_per_epoch": config.EVALS_PER_EPOCH,
        "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
        "metric_for_best_model": config.METRIC_FOR_BEST_MODEL,
        "logging_steps": config.LOGGING_STEPS,
        "save_total_limit": config.SAVE_TOTAL_LIMIT,
        "seed": args.seed,
        "use_cpu": device.type == "cpu",
        "gap_threshold": config.OVERFIT_GAP_THRESHOLD,
    }


def run_smoke(args: argparse.Namespace, train, validation) -> dict:
    """Chạy thử trên mẫu nhỏ để bắt lỗi và ước tính thời gian lần chạy thật.

    Giả định: checkpoint của lần chạy thử ghi vào thư mục tạm và bị xoá ngay; không đụng outputs/final_model.
    """
    options = _training_options(args)
    sample_train = train.sample(min(len(train), config.SMOKE_TRAIN_SAMPLES), random_state=args.seed)
    sample_validation = validation.sample(min(len(validation), config.SMOKE_VALIDATION_SAMPLES), random_state=args.seed)
    with tempfile.TemporaryDirectory(prefix="lab3_smoke_") as temporary:
        report = run_training(
            sample_train, sample_validation, checkpoint_dir=temporary, final_model_dir=None,
            smoke_max_steps=config.SMOKE_MAX_STEPS, smoke_ignored_steps=config.SMOKE_WARMUP_STEPS_IGNORED, **options,
        )
    estimate = estimate_full_run(
        train_steps_per_second=report["train_steps_per_second"],
        eval_samples_per_second=report["eval_samples_per_second"],
        n_train=len(train),
        n_validation=len(validation),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_epochs=args.epochs,
        evals_per_epoch=config.EVALS_PER_EPOCH,
        include_train_eval=True,
    )
    report["estimate_full_run"] = estimate
    report["within_time_budget"] = estimate["total_seconds"] <= config.TIME_BUDGET_HOURS * 3600

    print(f"\nCHẠY THỬ XONG trên {report['device']} (max_length={args.max_length}, batch={args.batch_size}"
          f"×{args.grad_accum}, {args.epochs} epoch)")
    print(f"  Tốc độ train: {report['train_steps_per_second']:.2f} bước/giây | "
          f"đánh giá: {report['eval_samples_per_second']:.0f} review/giây")
    print(f"  Ước tính lần chạy thật trên {len(train):,} review train, {len(validation):,} review validation:")
    print(f"    Train {int(estimate['total_update_steps']):,} bước: {_format_duration(estimate['train_seconds'])}")
    print(f"    Đánh giá validation {args.epochs * config.EVALS_PER_EPOCH} lần: {_format_duration(estimate['validation_seconds'])}")
    print(f"    Chấm điểm lại tập train: {_format_duration(estimate['train_eval_seconds'])}")
    print(f"    TỔNG: khoảng {_format_duration(estimate['total_seconds'])}")
    if report["within_time_budget"]:
        print(f"  -> Nằm trong giới hạn {config.TIME_BUDGET_HOURS:g} giờ.")
    else:
        print(f"  -> VƯỢT giới hạn {config.TIME_BUDGET_HOURS:g} giờ. Thử chạy lại --smoke với --max-length nhỏ hơn "
              "(vd 192 hoặc 128).", file=sys.stderr)
    return report


def run_train_stage(args: argparse.Namespace) -> dict:
    """Giai đoạn train. Giả định: data/splits có train.csv, validation.csv; KHÔNG đọc test.csv."""
    labels = config.ID2LABEL.keys()
    train = load_split(args.splits_dir / config.TRAIN_SPLIT_PATH.name, labels)
    validation = load_split(args.splits_dir / config.VALIDATION_SPLIT_PATH.name, labels)

    if args.smoke:
        if args.resume or args.overwrite:
            raise ValueError("--smoke không dùng chung với --resume hay --overwrite.")
        report = run_smoke(args, train, validation)
    else:
        checkpoint_dir = prepare_checkpoint_dir(args.checkpoints_dir, args.resume, args.overwrite)
        report = run_training(
            train, validation, checkpoint_dir=checkpoint_dir, final_model_dir=args.final_model_dir,
            resume=args.resume, **_training_options(args),
        )
        train_m, val_m, gap = report["train_metrics"], report["validation_metrics"], report["overfitting"]
        print(f"\nTRAIN XONG sau {_format_duration(report['wall_seconds'])} trên {report['device']}"
              f" ({report['global_steps']}/{report['planned_steps']} bước"
              f"{', dừng sớm' if report['stopped_early'] else ''})")
        print(f"  Checkpoint tốt nhất: {report['best_checkpoint']} (validation {config.METRIC_FOR_BEST_MODEL} "
              f"{report['best_metric']:.4f})")
        for name, metrics in (("train", train_m), ("validation", val_m)):
            print(f"  {name:<10}: accuracy {metrics['accuracy']:.4f} | precision {metrics['precision']:.4f} | "
                  f"recall {metrics['recall']:.4f} | F1 {metrics['f1']:.4f}")
        print(f"  Chênh lệch accuracy train − validation: {gap['gap']:.4f} (ngưỡng {gap['threshold']})")
        if gap["overfitting_warning"]:
            print("CẢNH BÁO: train cao hơn validation quá ngưỡng (overfitting).", file=sys.stderr)
        print(f"  Model tốt nhất đã lưu: {report['final_model_dir']}")

    saved = save_json(report, args.report)
    print(f"Đã lưu báo cáo: {saved}")
    return report


def run_evaluate_stage(args: argparse.Namespace) -> dict:
    """Giai đoạn đánh giá cuối trên tập test.

    Giả định: đã có model DistilBERT tốt nhất, model baseline và báo cáo train; đồ thị và bảng dự đoán
    được lưu cùng thư mục với file báo cáo.
    """
    test = load_split(args.splits_dir / config.TEST_SPLIT_PATH.name, config.ID2LABEL.keys())
    if not args.final_model_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục model '{args.final_model_dir}'. Hãy chạy --stage train trước.")
    report = run_final_evaluation(
        test,
        final_model_dir=args.final_model_dir,
        baseline_model_path=args.baseline_model,
        train_report=load_json(args.train_report),
        id2label=config.ID2LABEL,
        device=get_device(args.device),
        batch_size=config.EVAL_BATCH_SIZE,
        output_dir=args.report.parent,
        figure_names=(config.CONFUSION_MATRIX_FILENAME, config.TRAINING_CURVES_FILENAME),
        predictions_name=config.TEST_PREDICTIONS_FILENAME,
        errors_per_type=config.ERROR_EXAMPLES_PER_TYPE,
        preview_chars=config.ERROR_PREVIEW_CHARS,
    )
    print(f"\nKẾT QUẢ TRÊN TẬP TEST ({report['n_test']:,} review, max_length={report['max_length']}):")
    for name, metrics in report["metrics"].items():
        print(f"  {name:<26}: accuracy {metrics['accuracy']:.4f} | precision {metrics['precision']:.4f} | "
              f"recall {metrics['recall']:.4f} | F1 {metrics['f1']:.4f}")
    print("  Theo độ dài review:")
    for row in report["accuracy_by_length"]:
        values = " | ".join(f"{k} {v:.4f}" for k, v in row.items() if k not in ("group", "count") and v is not None)
        print(f"    {row['group']:<28} ({row['count']:,} review): {values}")
    print(f"  Hai model cùng đúng / chỉ DistilBERT đúng / chỉ baseline đúng / cùng sai: "
          f"{' / '.join(f'{v:,}' for v in report['agreement'].values())}")
    for kind, path in report["files"].items():
        print(f"  Đã lưu {kind}: {path}")
    saved = save_json(report, args.report)
    print(f"Đã lưu báo cáo: {saved}")
    return report


def main(argv: list[str] | None = None) -> int:
    """Chạy giai đoạn được chọn.

    Giả định: lỗi do đầu vào (file thiếu, tham số sai, dữ liệu hỏng...) được in rõ ra stderr, trả mã 2.
    """
    args = parse_args(argv)
    try:
        if args.stage == "data":
            run_data_stage(args)
        elif args.stage == "baseline":
            run_baseline_stage(args)
        elif args.stage == "train":
            run_train_stage(args)
        elif args.stage == "evaluate":
            run_evaluate_stage(args)
    except (TypeError, ValueError, FileNotFoundError, IsADirectoryError, NotADirectoryError, RuntimeError) as error:
        print(f"Lỗi: {error}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
