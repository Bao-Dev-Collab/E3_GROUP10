"""Bài 2 – dữ liệu: đọc IMDb, chia train/validation/test, làm sạch, loại trùng, thống kê độ dài.

Thứ tự xử lý tuân theo RULES.md: chia dữ liệu TRƯỚC mọi bước xử lý khác.
    1. Đọc file CSV gốc và kiểm tra hợp lệ.
    2. Chia train gốc thành train / validation (stratify). Test gốc giữ nguyên làm test.
    3. Làm sạch từng tập (hàm xử lý từng review riêng lẻ, không học gì từ dữ liệu).
    4. Loại review trùng: trong nội bộ train/validation, và giữa các tập (ưu tiên giữ ở test).
    5. Thống kê độ dài token CHỈ trên train để chọn max_length.
    6. Lưu ra data/splits/.
"""

import hashlib
import math
import re
from collections.abc import Callable, Collection, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

REQUIRED_COLUMNS = ("text", "label")
SPLIT_COLUMNS = ("source_split", "source_index", "text", "label")
_LINE_BREAK_TAG = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
_MAX_SEED = 2**32 - 1


# --------------------------------------------------------------------------- kiểm tra đầu vào
def _validate_path(path: Any, name: str) -> Path:
    """Kiểm tra tham số đường dẫn.

    Giả định: path là str hoặc Path, không rỗng. Chỉ kiểm tra kiểu, không kiểm tra tồn tại.
    """
    if path is None or isinstance(path, bool) or not isinstance(path, (str, Path)):
        raise TypeError(f"{name} phải là str hoặc Path, nhận được {type(path).__name__}.")
    if isinstance(path, str) and not path.strip():
        raise ValueError(f"{name} không được rỗng.")
    return Path(path)


def _validate_allowed_labels(allowed_labels: Any) -> set[int]:
    """Kiểm tra tập nhãn hợp lệ.

    Giả định: allowed_labels là tập hợp (list/tuple/set/dict keys...) các số nguyên, có ít nhất 2 nhãn.
    """
    if allowed_labels is None or isinstance(allowed_labels, (str, bytes)) or not isinstance(allowed_labels, Iterable):
        raise TypeError("allowed_labels phải là một tập hợp các số nguyên.")
    labels = list(allowed_labels)
    if any(isinstance(label, bool) or not isinstance(label, (int, np.integer)) for label in labels):
        raise TypeError("Mọi phần tử của allowed_labels phải là số nguyên.")
    if len(set(labels)) < 2:
        raise ValueError("allowed_labels cần ít nhất 2 nhãn khác nhau (bài toán phân loại).")
    return {int(label) for label in labels}


# --------------------------------------------------------------------------- 1. đọc dữ liệu
def _read_csv_strict(path: str | Path, required_columns: Collection[str]) -> tuple[Path, pd.DataFrame]:
    """Đọc một file CSV, giữ nguyên chuỗi gốc.

    Giả định:
    - path trỏ tới một file CSV có dòng tiêu đề chứa đủ required_columns.
    - Ô văn bản được đọc nguyên văn: chuỗi như "NA", "null" không bị pandas đổi thành giá trị thiếu.
    """
    file_path = _validate_path(path, "path")
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file '{file_path}'.")
    if file_path.is_dir():
        raise IsADirectoryError(f"'{file_path}' là thư mục, cần đường dẫn tới file CSV.")
    try:
        frame = pd.read_csv(file_path, dtype={"text": str}, keep_default_na=False, na_values=[])
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"File '{file_path}' rỗng.") from error
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"File '{file_path}' thiếu cột: {missing}.")
    return file_path, frame


def _checked_labels(frame: pd.DataFrame, file_path: Path, labels_allowed: set[int]) -> pd.Series:
    """Kiểm tra cột text, label và trả về label dạng số nguyên.

    Giả định: frame đã có cột text, label; text phải có nội dung; label là số nguyên thuộc labels_allowed.
    """
    if frame.empty:
        raise ValueError(f"File '{file_path}' không có dòng dữ liệu nào.")

    empty_text = frame["text"].str.strip() == ""
    if empty_text.any():
        raise ValueError(f"File '{file_path}' có {int(empty_text.sum())} dòng text rỗng.")

    numeric = pd.to_numeric(frame["label"], errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"File '{file_path}' có {int(numeric.isna().sum())} nhãn không phải số.")
    if (numeric % 1 != 0).any():
        raise ValueError(f"File '{file_path}' có nhãn không phải số nguyên.")
    labels = numeric.astype(int)
    unknown = sorted(set(labels) - labels_allowed)
    if unknown:
        raise ValueError(f"File '{file_path}' có nhãn ngoài tập cho phép {sorted(labels_allowed)}: {unknown}.")
    return labels


def load_raw_csv(path: str | Path, allowed_labels: Collection[int]) -> pd.DataFrame:
    """Đọc file CSV gốc và kiểm tra hợp lệ.

    Giả định:
    - File có dòng tiêu đề và ít nhất 2 cột text, label (cột thừa bị bỏ qua).
    - Mọi ô text đều là chuỗi có nội dung; ô rỗng là lỗi của file gốc nên báo lỗi, không tự bỏ.
    - Ô text được đọc nguyên văn: chuỗi như "NA", "null" là review thật, không bị đổi thành giá trị thiếu.
    - label là số nguyên thuộc allowed_labels.
    Trả về DataFrame gồm source_index (số thứ tự dòng trong file gốc, từ 0), text, label.
    """
    labels_allowed = _validate_allowed_labels(allowed_labels)
    file_path, frame = _read_csv_strict(path, REQUIRED_COLUMNS)
    labels = _checked_labels(frame, file_path, labels_allowed)
    return pd.DataFrame({"source_index": np.arange(len(frame)), "text": frame["text"], "label": labels})


def load_split(path: str | Path, allowed_labels: Collection[int]) -> pd.DataFrame:
    """Đọc một tập đã chia do save_split tạo ra (data/splits/*.csv).

    Giả định:
    - File có đủ các cột SPLIT_COLUMNS.
    - Ô text đọc nguyên văn (review "NA" vẫn là chuỗi "NA"), không rỗng; label là số nguyên thuộc allowed_labels.
    """
    labels_allowed = _validate_allowed_labels(allowed_labels)
    file_path, frame = _read_csv_strict(path, SPLIT_COLUMNS)
    labels = _checked_labels(frame, file_path, labels_allowed)
    return frame.assign(label=labels).loc[:, list(SPLIT_COLUMNS)]


# --------------------------------------------------------------------------- 2. chia dữ liệu
def split_train_validation(frame: pd.DataFrame, val_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chia một tập thành train và validation, giữ nguyên tỉ lệ nhãn (stratify).

    Giả định:
    - frame có cột label, ít nhất 2 lớp, mỗi lớp ít nhất 2 mẫu.
    - 0 < val_size < 1 là tỉ lệ dành cho validation; seed là số nguyên 0..2^32-1.
    - Dữ liệu được xáo trộn trước khi chia, nên thứ tự dòng ban đầu (IMDb xếp theo nhãn) không ảnh hưởng.
    - Sau khi chia, cả train và validation đều phải có đủ mọi lớp.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame phải là pandas.DataFrame, nhận được {type(frame).__name__}.")
    if "label" not in frame.columns:
        raise ValueError("frame thiếu cột label.")
    if isinstance(val_size, bool) or not isinstance(val_size, (int, float)):
        raise TypeError(f"val_size phải là số thực, nhận được {type(val_size).__name__}.")
    if not 0 < val_size < 1:
        raise ValueError(f"val_size phải nằm trong khoảng (0, 1), nhận được {val_size}.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed phải là số nguyên, nhận được {type(seed).__name__}.")
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed phải nằm trong khoảng 0..{_MAX_SEED}, nhận được {seed}.")

    counts = frame["label"].value_counts()
    if len(counts) < 2:
        raise ValueError("Cần ít nhất 2 lớp để chia có stratify.")
    if counts.min() < 2:
        raise ValueError(f"Mỗi lớp cần ít nhất 2 mẫu, lớp ít nhất chỉ có {int(counts.min())}.")
    n_val = math.ceil(val_size * len(frame))
    if n_val < len(counts) or len(frame) - n_val < len(counts):
        raise ValueError(
            f"Không đủ dữ liệu: {len(frame)} dòng với val_size={val_size} không chia được để "
            f"mỗi tập có đủ {len(counts)} lớp."
        )

    train, validation = train_test_split(
        frame, test_size=val_size, stratify=frame["label"], random_state=int(seed), shuffle=True
    )
    # Stratify chia số mẫu theo tỉ lệ từng lớp rồi làm tròn, nên lớp quá ít mẫu có thể được chia 0 mẫu
    # cho một bên dù tổng số mẫu đủ. Kiểm tra kết quả thật thay vì chỉ kiểm tra tổng.
    for name, part in (("train", train), ("validation", validation)):
        missing_classes = sorted(int(label) for label in set(counts.index) - set(part["label"]))
        if missing_classes:
            raise ValueError(
                f"Sau khi chia, tập {name} thiếu lớp {missing_classes}: lớp này quá ít mẫu so với "
                f"val_size={val_size}. Hãy đổi val_size hoặc bổ sung dữ liệu."
            )
    return train.reset_index(drop=True), validation.reset_index(drop=True)


# --------------------------------------------------------------------------- 3. làm sạch
def clean_text(text: str) -> str:
    """Làm sạch một review.

    Giả định:
    - text là str.
    - Chỉ thay thẻ xuống dòng <br>, <br/>, <br /> (không phân biệt hoa thường) bằng khoảng trắng,
      rồi gộp khoảng trắng liên tiếp và bỏ khoảng trắng hai đầu.
    - Các thẻ HTML khác và entity HTML rất hiếm trong IMDb (dưới 20 lần / 50.000 review) nên giữ nguyên.
    - Không đổi chữ hoa/thường: tokenizer uncased tự làm việc đó.
    """
    if not isinstance(text, str):
        raise TypeError(f"text phải là str, nhận được {type(text).__name__}.")
    return " ".join(_LINE_BREAK_TAG.sub(" ", text).split())


def clean_dataframe(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Làm sạch cột text của một tập và bỏ các dòng rỗng sau khi làm sạch.

    Giả định: frame có cột text (str). Trả về (tập đã làm sạch, số dòng bị bỏ vì rỗng).
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame phải là pandas.DataFrame, nhận được {type(frame).__name__}.")
    if "text" not in frame.columns:
        raise ValueError("frame thiếu cột text.")
    cleaned = frame.assign(text=frame["text"].map(clean_text))
    keep = cleaned["text"] != ""
    return cleaned[keep].reset_index(drop=True), int((~keep).sum())


# --------------------------------------------------------------------------- 4. loại trùng
def duplicate_key(text: str) -> str:
    """Khoá dùng để so trùng hai review.

    Giả định: model uncased nên hai review chỉ khác nhau về chữ hoa/thường, khoảng trắng
    hoặc thẻ <br /> được coi là cùng một đầu vào.
    """
    return clean_text(text).lower()


def _drop_conflicting_and_internal_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Trong một tập: bỏ hết các nhóm trùng có nhãn mâu thuẫn, nhóm còn lại giữ bản đầu tiên.

    Giả định: frame có cột text, label. Trả về (tập mới, số dòng mâu thuẫn bị bỏ, số bản trùng bị bỏ).
    """
    keys = frame["text"].map(duplicate_key)
    label_counts = frame.groupby(keys)["label"].transform("nunique")
    conflicting = label_counts > 1
    remaining = frame[~conflicting]
    remaining_keys = keys[~conflicting]
    first = ~remaining_keys.duplicated(keep="first")
    return remaining[first].reset_index(drop=True), int(conflicting.sum()), int((~first).sum())


def remove_duplicates(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Loại review trùng để không rò rỉ dữ liệu giữa các tập.

    Giả định:
    - Ba tập đã được chia và làm sạch, đều có cột text, label.
    - Test là tập chuẩn để so sánh nên KHÔNG bị sửa; trùng trong nội bộ test chỉ được đếm.
    - Ưu tiên giữ: test > validation > train. Review có ở tập ưu tiên cao hơn bị bỏ khỏi tập thấp hơn.
    - Trong nội bộ train / validation: nhóm trùng có nhãn mâu thuẫn bị bỏ hết, nhóm còn lại giữ bản đầu.
    """
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} phải là pandas.DataFrame, nhận được {type(frame).__name__}.")
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} thiếu cột: {missing}.")

    test_keys = set(test["text"].map(duplicate_key))
    test_internal = int(test["text"].map(duplicate_key).duplicated(keep="first").sum())

    validation, val_conflicts, val_internal = _drop_conflicting_and_internal_duplicates(validation)
    in_test = validation["text"].map(duplicate_key).isin(test_keys)
    validation_vs_test = int(in_test.sum())
    validation = validation[~in_test].reset_index(drop=True)
    validation_keys = set(validation["text"].map(duplicate_key))

    train, train_conflicts, train_internal = _drop_conflicting_and_internal_duplicates(train)
    train_key_series = train["text"].map(duplicate_key)
    in_test = train_key_series.isin(test_keys)
    in_validation = train_key_series.isin(validation_keys) & ~in_test
    train = train[~(in_test | in_validation)].reset_index(drop=True)

    report = {
        "train_internal_duplicates_removed": train_internal,
        "train_conflicting_labels_removed": train_conflicts,
        "validation_internal_duplicates_removed": val_internal,
        "validation_conflicting_labels_removed": val_conflicts,
        "train_also_in_test_removed": int(in_test.sum()),
        "train_also_in_validation_removed": int(in_validation.sum()),
        "validation_also_in_test_removed": validation_vs_test,
        "test_internal_duplicates_kept": test_internal,
    }
    return train, validation, report


# --------------------------------------------------------------------------- 5. thống kê độ dài
def token_length_stats(texts: list[str], tokenizer: Callable, thresholds: Collection[int]) -> dict[str, Any]:
    """Thống kê số token mỗi review để chọn max_length.

    Giả định:
    - Chỉ gọi với text của tập TRAIN (không nhìn validation/test khi chọn tham số).
    - Độ dài tính cả token đặc biệt ([CLS], [SEP]) và không cắt câu.
    - thresholds là các số nguyên dương; với mỗi mốc, trả về tỉ lệ review dài hơn mốc đó (sẽ bị cắt).
    """
    if not isinstance(texts, list):
        raise TypeError(f"texts phải là list, nhận được {type(texts).__name__}.")
    if not texts:
        raise ValueError("texts không được rỗng.")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("Mọi phần tử của texts phải là str.")
    if not callable(tokenizer):
        raise TypeError("tokenizer phải gọi được (callable).")
    if thresholds is None or isinstance(thresholds, (str, bytes)) or not isinstance(thresholds, Iterable):
        raise TypeError("thresholds phải là tập hợp các số nguyên dương.")
    marks = list(thresholds)
    if not marks:
        raise ValueError("thresholds không được rỗng.")
    for mark in marks:
        if isinstance(mark, bool) or not isinstance(mark, (int, np.integer)):
            raise TypeError(f"Mốc {mark!r} không phải số nguyên.")
        if mark <= 0:
            raise ValueError(f"Mốc phải dương, nhận được {mark}.")

    encoded = tokenizer(texts, add_special_tokens=True, truncation=False, verbose=False)
    lengths = np.array([len(ids) for ids in encoded["input_ids"]])
    return {
        "count": int(lengths.size),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "max": int(lengths.max()),
        "fraction_longer_than": {str(int(mark)): float((lengths > mark).mean()) for mark in sorted(marks)},
    }


# --------------------------------------------------------------------------- 6. lưu
def save_split(frame: pd.DataFrame, path: str | Path) -> dict[str, Any]:
    """Lưu một tập ra CSV và trả về số dòng, mã SHA-256.

    Giả định: frame có đủ các cột SPLIT_COLUMNS; thư mục cha chưa có sẽ được tạo; file cũ bị ghi đè.
    Xuống dòng luôn là "\\n" (không theo hệ điều hành), để cùng dữ liệu cho ra cùng file và cùng
    SHA-256 trên macOS, Linux lẫn Windows.
    """
    file_path = _validate_path(path, "path")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame phải là pandas.DataFrame, nhận được {type(frame).__name__}.")
    missing = [column for column in SPLIT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"frame thiếu cột: {missing}.")
    if file_path.is_dir():
        raise IsADirectoryError(f"'{file_path}' là thư mục, cần đường dẫn tới file CSV.")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(SPLIT_COLUMNS)].to_csv(file_path, index=False, lineterminator="\n")
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return {"path": str(file_path), "rows": int(len(frame)), "sha256": digest}


def load_tokenizer(model_name: str) -> Any:
    """Tải tokenizer của base model.

    Giả định: model_name là id trên Hugging Face Hub hoặc thư mục local; lần đầu cần internet.
    """
    if model_name is None or not isinstance(model_name, str):
        raise TypeError(f"model_name phải là str, nhận được {type(model_name).__name__}.")
    if not model_name.strip():
        raise ValueError("model_name không được rỗng.")
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except OSError as error:
        raise RuntimeError(f"Không tải được tokenizer '{model_name}'. Kiểm tra tên model hoặc internet.") from error


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Số mẫu mỗi nhãn. Giả định: frame có cột label số nguyên."""
    return {str(label): int(count) for label, count in sorted(frame["label"].value_counts().items())}


# --------------------------------------------------------------------------- toàn bộ quy trình
def prepare_data(
    raw_train_path: str | Path,
    raw_test_path: str | Path,
    output_dir: str | Path,
    val_size: float,
    seed: int,
    tokenizer: Callable,
    thresholds: Collection[int],
    allowed_labels: Collection[int],
) -> dict[str, Any]:
    """Chạy toàn bộ quy trình dữ liệu và lưu train.csv, validation.csv, test.csv vào output_dir.

    Giả định:
    - Hai file gốc đều hợp lệ theo load_raw_csv.
    - Chia train gốc thành train/validation TRƯỚC khi làm sạch, loại trùng hay thống kê.
    - Nhãn của validation/test phải đều đã xuất hiện trong train.
    Trả về báo cáo (dict) gồm kích thước, phân bố nhãn, số dòng bị bỏ, thống kê token, file đã lưu.
    """
    out_dir = _validate_path(output_dir, "output_dir")
    raw_train = load_raw_csv(raw_train_path, allowed_labels)
    raw_test = load_raw_csv(raw_test_path, allowed_labels)

    train, validation = split_train_validation(raw_train, val_size, seed)
    test = raw_test
    unseen = sorted((set(validation["label"]) | set(test["label"])) - set(train["label"]))
    if unseen:
        raise ValueError(f"Validation/test có nhãn chưa xuất hiện trong train: {unseen}.")

    train, train_empty = clean_dataframe(train)
    validation, validation_empty = clean_dataframe(validation)
    test, test_empty = clean_dataframe(test)

    train, validation, duplicates = remove_duplicates(train, validation, test)
    token_stats = token_length_stats(train["text"].tolist(), tokenizer, thresholds)

    files = {}
    for name, frame, source in (("train", train, "train"), ("validation", validation, "train"), ("test", test, "test")):
        files[name] = save_split(frame.assign(source_split=source), out_dir / f"{name}.csv")

    return {
        "seed": int(seed),
        "val_size": float(val_size),
        "raw_rows": {"train": int(len(raw_train)), "test": int(len(raw_test))},
        "empty_after_cleaning_removed": {"train": train_empty, "validation": validation_empty, "test": test_empty},
        "duplicates": duplicates,
        "final_rows": {"train": int(len(train)), "validation": int(len(validation)), "test": int(len(test))},
        "label_counts": {"train": _label_counts(train), "validation": _label_counts(validation), "test": _label_counts(test)},
        "train_token_length": token_stats,
        "files": files,
    }
