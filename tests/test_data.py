"""Test cho lab3.data: không cần tải model (dùng tokenizer giả)."""

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from lab3.data import (
    clean_dataframe,
    clean_text,
    duplicate_key,
    load_raw_csv,
    load_split,
    prepare_data,
    remove_duplicates,
    save_split,
    split_train_validation,
    token_length_stats,
)

LABELS = (0, 1)


def write_csv(path: Path, texts: list, labels: list) -> Path:
    pd.DataFrame({"text": texts, "label": labels}).to_csv(path, index=False)
    return path


def sorted_frame(n_per_class: int = 50) -> pd.DataFrame:
    """Dữ liệu xếp theo nhãn giống IMDb: toàn nhãn 0 trước, toàn nhãn 1 sau."""
    texts = [f"negative review number {i}" for i in range(n_per_class)]
    texts += [f"positive review number {i}" for i in range(n_per_class)]
    labels = [0] * n_per_class + [1] * n_per_class
    return pd.DataFrame({"source_index": range(len(texts)), "text": texts, "label": labels})


def frame(texts: list, labels: list) -> pd.DataFrame:
    return pd.DataFrame({"source_index": range(len(texts)), "text": texts, "label": labels})


# ================================================================ load_raw_csv
def test_load_raw_csv_reads_text_label_and_source_index(tmp_path):
    path = write_csv(tmp_path / "a.csv", ["good", "bad", "ok"], [1, 0, 1])
    result = load_raw_csv(path, LABELS)
    assert list(result.columns) == ["source_index", "text", "label"]
    assert result["source_index"].tolist() == [0, 1, 2]
    assert result["label"].tolist() == [1, 0, 1]


def test_load_raw_csv_keeps_na_like_strings_as_text(tmp_path):
    path = write_csv(tmp_path / "a.csv", ["NA", "null", "None"], [0, 1, 0])
    assert load_raw_csv(path, LABELS)["text"].tolist() == ["NA", "null", "None"]


def test_load_raw_csv_ignores_extra_columns(tmp_path):
    path = tmp_path / "a.csv"
    pd.DataFrame({"id": [7, 8], "text": ["x", "y"], "label": [0, 1]}).to_csv(path, index=False)
    assert list(load_raw_csv(path, LABELS).columns) == ["source_index", "text", "label"]


def test_load_raw_csv_accepts_integral_float_labels(tmp_path):
    path = write_csv(tmp_path / "a.csv", ["x", "y"], [0.0, 1.0])
    assert load_raw_csv(path, LABELS)["label"].tolist() == [0, 1]


def test_load_raw_csv_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_csv(tmp_path / "khong_co.csv", LABELS)


def test_load_raw_csv_directory_raises(tmp_path):
    with pytest.raises(IsADirectoryError):
        load_raw_csv(tmp_path, LABELS)


@pytest.mark.parametrize("bad_path", [None, 123, True, ["a.csv"]])
def test_load_raw_csv_wrong_path_type_raises(bad_path):
    with pytest.raises(TypeError):
        load_raw_csv(bad_path, LABELS)


def test_load_raw_csv_empty_path_string_raises():
    with pytest.raises(ValueError):
        load_raw_csv("  ", LABELS)


def test_load_raw_csv_zero_byte_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(ValueError, match="rỗng"):
        load_raw_csv(path, LABELS)


def test_load_raw_csv_header_only_raises(tmp_path):
    path = tmp_path / "header.csv"
    path.write_text("text,label\n")
    with pytest.raises(ValueError, match="không có dòng"):
        load_raw_csv(path, LABELS)


def test_load_raw_csv_missing_column_raises(tmp_path):
    path = tmp_path / "a.csv"
    pd.DataFrame({"review": ["x"], "label": [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="thiếu cột"):
        load_raw_csv(path, LABELS)


@pytest.mark.parametrize("bad_text", ["", "   "])
def test_load_raw_csv_empty_text_raises(tmp_path, bad_text):
    path = write_csv(tmp_path / "a.csv", ["fine", bad_text], [0, 1])
    with pytest.raises(ValueError, match="text rỗng"):
        load_raw_csv(path, LABELS)


@pytest.mark.parametrize("bad_label, message", [("pos", "không phải số"), (0.5, "số nguyên"), (2, "ngoài"), (-1, "ngoài")])
def test_load_raw_csv_invalid_label_raises(tmp_path, bad_label, message):
    path = write_csv(tmp_path / "a.csv", ["x", "y"], [0, bad_label])
    with pytest.raises(ValueError, match=message):
        load_raw_csv(path, LABELS)


@pytest.mark.parametrize("bad_labels, error", [(None, TypeError), ("01", TypeError), ([0, "1"], TypeError), ([1], ValueError), ([1, 1], ValueError)])
def test_load_raw_csv_invalid_allowed_labels_raises(tmp_path, bad_labels, error):
    path = write_csv(tmp_path / "a.csv", ["x", "y"], [0, 1])
    with pytest.raises(error):
        load_raw_csv(path, bad_labels)


# ====================================================== split_train_validation
def test_split_sizes_follow_val_size():
    train, validation = split_train_validation(sorted_frame(50), 0.1, 42)
    assert (len(train), len(validation)) == (90, 10)


def test_split_on_label_sorted_data_keeps_both_classes_balanced():
    _, validation = split_train_validation(sorted_frame(50), 0.1, 42)
    assert validation["label"].value_counts().to_dict() == {0: 5, 1: 5}


def test_split_keeps_ratio_on_imbalanced_data():
    data = frame([f"t{i}" for i in range(100)], [0] * 80 + [1] * 20)
    _, validation = split_train_validation(data, 0.1, 0)
    assert validation["label"].value_counts().to_dict() == {0: 8, 1: 2}


def test_split_has_no_overlap_and_loses_nothing():
    data = sorted_frame(50)
    train, validation = split_train_validation(data, 0.2, 7)
    train_ids, val_ids = set(train["source_index"]), set(validation["source_index"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(data["source_index"])


def test_split_is_reproducible_with_same_seed():
    first = split_train_validation(sorted_frame(), 0.1, 42)[1]["source_index"].tolist()
    second = split_train_validation(sorted_frame(), 0.1, 42)[1]["source_index"].tolist()
    other = split_train_validation(sorted_frame(), 0.1, 43)[1]["source_index"].tolist()
    assert first == second
    assert first != other


@pytest.mark.parametrize("val_size", [0, 1, -0.1, 1.5])
def test_split_val_size_out_of_range_raises(val_size):
    with pytest.raises(ValueError):
        split_train_validation(sorted_frame(), val_size, 42)


@pytest.mark.parametrize("val_size", ["0.1", None, True])
def test_split_val_size_wrong_type_raises(val_size):
    with pytest.raises(TypeError):
        split_train_validation(sorted_frame(), val_size, 42)


@pytest.mark.parametrize("seed, error", [(-1, ValueError), (2**32, ValueError), (1.5, TypeError), ("42", TypeError), (True, TypeError)])
def test_split_invalid_seed_raises(seed, error):
    with pytest.raises(error):
        split_train_validation(sorted_frame(), 0.1, seed)


def test_split_single_class_raises():
    with pytest.raises(ValueError, match="2 lớp"):
        split_train_validation(frame(["a", "b", "c"], [1, 1, 1]), 0.3, 42)


def test_split_class_with_one_sample_raises():
    with pytest.raises(ValueError, match="ít nhất 2 mẫu"):
        split_train_validation(frame(["a", "b", "c", "d"], [0, 0, 0, 1]), 0.5, 42)


def test_split_too_few_rows_for_both_classes_in_validation_raises():
    with pytest.raises(ValueError, match="Không đủ dữ liệu"):
        split_train_validation(frame(["a", "b", "c", "d"], [0, 0, 1, 1]), 0.1, 42)


@pytest.mark.parametrize(
    "n_negative, n_positive, val_size, missing_in",
    [(98, 2, 0.02, "validation"), (95, 5, 0.1, "validation"), (8, 2, 0.8, "train")],
)
def test_split_raises_when_minority_class_would_vanish(n_negative, n_positive, val_size, missing_in):
    """Tổng số mẫu đủ nhưng stratify làm tròn khiến lớp thiểu số còn 0 mẫu ở một bên."""
    data = frame([f"t{i}" for i in range(n_negative + n_positive)], [0] * n_negative + [1] * n_positive)
    with pytest.raises(ValueError, match=f"tập {missing_in} thiếu lớp"):
        split_train_validation(data, val_size, 42)


def test_split_wrong_input_raises():
    with pytest.raises(TypeError):
        split_train_validation([("a", 0)], 0.1, 42)
    with pytest.raises(ValueError, match="label"):
        split_train_validation(pd.DataFrame({"text": ["a", "b"]}), 0.5, 42)


# ================================================================== clean_text
@pytest.mark.parametrize("raw", ["a<br />b", "a<br/>b", "a<BR>b", "a< br / >b", "a<br /><br />b"])
def test_clean_text_replaces_line_break_tag_variants(raw):
    assert clean_text(raw) == "a b"


def test_clean_text_collapses_whitespace_and_strips():
    assert clean_text("  many\t\tspaces \n and\nlines  ") == "many spaces and lines"


def test_clean_text_keeps_other_tags_unicode_and_case():
    assert clean_text("<i>Great</i> phim hay 😍 &#8230;") == "<i>Great</i> phim hay 😍 &#8230;"


def test_clean_text_can_become_empty():
    assert clean_text("<br /><br />   ") == ""
    assert clean_text("") == ""


def test_clean_text_is_idempotent():
    once = clean_text("A<br />  B <br/>C")
    assert clean_text(once) == once


@pytest.mark.parametrize("value", [None, 123, ["a"], b"a<br />b"])
def test_clean_text_wrong_type_raises(value):
    with pytest.raises(TypeError):
        clean_text(value)


def test_clean_dataframe_drops_rows_empty_after_cleaning():
    cleaned, dropped = clean_dataframe(frame(["ok<br />fine", "<br />", "  ", "good"], [1, 0, 0, 1]))
    assert dropped == 2
    assert cleaned["text"].tolist() == ["ok fine", "good"]
    assert cleaned.index.tolist() == [0, 1]


def test_clean_dataframe_wrong_input_raises():
    with pytest.raises(TypeError):
        clean_dataframe(["a"])
    with pytest.raises(ValueError):
        clean_dataframe(pd.DataFrame({"label": [0]}))


def test_duplicate_key_ignores_case_spaces_and_line_breaks():
    assert duplicate_key("Great  MOVIE<br />Loved it") == duplicate_key("great movie loved it")


# ========================================================== remove_duplicates
def test_remove_duplicates_train_copy_of_test_is_removed_and_test_untouched():
    train = frame(["shared review", "only train"], [1, 0])
    validation = frame(["only val"], [0])
    test = frame(["Shared  Review", "only test"], [1, 0])
    new_train, new_val, report = remove_duplicates(train, validation, test)
    assert new_train["text"].tolist() == ["only train"]
    assert new_val["text"].tolist() == ["only val"]
    assert report["train_also_in_test_removed"] == 1
    assert test["text"].tolist() == ["Shared  Review", "only test"]


def test_remove_duplicates_validation_copy_of_test_is_removed():
    _, new_val, report = remove_duplicates(frame(["t"], [0]), frame(["x", "dup"], [0, 1]), frame(["DUP"], [1]))
    assert new_val["text"].tolist() == ["x"]
    assert report["validation_also_in_test_removed"] == 1


def test_remove_duplicates_train_copy_of_validation_is_removed_from_train():
    new_train, new_val, report = remove_duplicates(frame(["same", "t"], [1, 0]), frame(["same"], [1]), frame(["z"], [0]))
    assert new_train["text"].tolist() == ["t"]
    assert new_val["text"].tolist() == ["same"]
    assert report["train_also_in_validation_removed"] == 1


def test_remove_duplicates_internal_copies_keep_first():
    new_train, _, report = remove_duplicates(frame(["a", "b", "A", "a<br />"], [1, 0, 1, 1]), frame(["v"], [0]), frame(["z"], [0]))
    assert new_train["text"].tolist() == ["a", "b"]
    assert report["train_internal_duplicates_removed"] == 2


def test_remove_duplicates_conflicting_labels_are_all_dropped():
    new_train, _, report = remove_duplicates(frame(["x", "X", "y"], [1, 0, 1]), frame(["v"], [0]), frame(["z"], [0]))
    assert new_train["text"].tolist() == ["y"]
    assert report["train_conflicting_labels_removed"] == 2


def test_remove_duplicates_counts_but_keeps_test_internal_duplicates():
    test = frame(["z", "Z", "w"], [0, 0, 1])
    _, _, report = remove_duplicates(frame(["t"], [0]), frame(["v"], [1]), test)
    assert report["test_internal_duplicates_kept"] == 1
    assert len(test) == 3


def test_remove_duplicates_without_duplicates_changes_nothing():
    train, validation, test = frame(["a", "b"], [0, 1]), frame(["c"], [0]), frame(["d"], [1])
    new_train, new_val, report = remove_duplicates(train, validation, test)
    assert new_train["text"].tolist() == ["a", "b"]
    assert new_val["text"].tolist() == ["c"]
    assert set(report.values()) == {0}


def test_remove_duplicates_wrong_input_raises():
    with pytest.raises(TypeError):
        remove_duplicates(None, frame(["v"], [0]), frame(["z"], [0]))
    with pytest.raises(ValueError, match="thiếu cột"):
        remove_duplicates(frame(["t"], [0]), pd.DataFrame({"text": ["v"]}), frame(["z"], [0]))


# ========================================================== token_length_stats
def test_token_length_stats_values(whitespace_tokenizer):
    # Độ dài = số từ + 2 token đặc biệt: 4 và 8.
    stats = token_length_stats(["a b", "a b c d e f"], whitespace_tokenizer, [10, 5])
    assert stats["count"] == 2
    assert stats["mean"] == 6
    assert stats["median"] == 6
    assert stats["max"] == 8
    assert stats["fraction_longer_than"] == {"5": 0.5, "10": 0.0}


def test_token_length_stats_length_equal_to_threshold_is_not_truncated(whitespace_tokenizer):
    stats = token_length_stats(["one two three"], whitespace_tokenizer, [5])
    assert stats["fraction_longer_than"]["5"] == 0.0


@pytest.mark.parametrize("texts, error", [([], ValueError), (("a",), TypeError), ("a", TypeError), (["a", 1], TypeError)])
def test_token_length_stats_invalid_texts_raises(whitespace_tokenizer, texts, error):
    with pytest.raises(error):
        token_length_stats(texts, whitespace_tokenizer, [5])


@pytest.mark.parametrize("thresholds, error", [([], ValueError), ([0], ValueError), ([-5], ValueError), ([2.5], TypeError), (None, TypeError), ("128", TypeError)])
def test_token_length_stats_invalid_thresholds_raises(whitespace_tokenizer, thresholds, error):
    with pytest.raises(error):
        token_length_stats(["a b"], whitespace_tokenizer, thresholds)


def test_token_length_stats_tokenizer_not_callable_raises():
    with pytest.raises(TypeError):
        token_length_stats(["a"], "tokenizer", [5])


# ================================================================== save_split
def split_frame(texts: list, labels: list) -> pd.DataFrame:
    return frame(texts, labels).assign(source_split="train")


def test_save_split_round_trip_and_checksum(tmp_path):
    data = split_frame(['He said "wow", then left', "Phim hay 😍, tuyệt"], [1, 1])
    info = save_split(data, tmp_path / "sub" / "train.csv")
    loaded = pd.read_csv(info["path"], keep_default_na=False)
    assert loaded["text"].tolist() == data["text"].tolist()
    assert info["rows"] == 2
    assert info["sha256"] == hashlib.sha256(Path(info["path"]).read_bytes()).hexdigest()


def test_save_split_uses_same_line_ending_on_every_os(tmp_path, monkeypatch):
    """Giả lập Windows (os.linesep = \\r\\n): file vẫn phải dùng \\n để SHA-256 giống mọi máy."""
    data = split_frame(["first", "second"], [0, 1])
    unix = save_split(data, tmp_path / "unix.csv")
    monkeypatch.setattr("os.linesep", "\r\n")
    windows = save_split(data, tmp_path / "windows.csv")
    assert b"\r\n" not in Path(windows["path"]).read_bytes()
    assert windows["sha256"] == unix["sha256"]


def test_save_split_invalid_input_raises(tmp_path):
    with pytest.raises(ValueError, match="thiếu cột"):
        save_split(frame(["a"], [0]), tmp_path / "x.csv")
    with pytest.raises(IsADirectoryError):
        save_split(split_frame(["a"], [0]), tmp_path)
    with pytest.raises(TypeError):
        save_split(split_frame(["a"], [0]), None)
    with pytest.raises(TypeError):
        save_split("not a frame", tmp_path / "x.csv")


# ================================================================== load_split
def test_load_split_round_trip_keeps_na_like_text(tmp_path):
    data = split_frame(["NA", "null", "Great, \"quoted\" film"], [0, 1, 1])
    info = save_split(data, tmp_path / "train.csv")
    loaded = load_split(info["path"], LABELS)
    assert loaded["text"].tolist() == ["NA", "null", 'Great, "quoted" film']
    assert loaded["label"].tolist() == [0, 1, 1]
    assert list(loaded.columns) == ["source_split", "source_index", "text", "label"]


def test_load_split_rejects_raw_format_file(tmp_path):
    path = write_csv(tmp_path / "raw.csv", ["x", "y"], [0, 1])
    with pytest.raises(ValueError, match="thiếu cột"):
        load_split(path, LABELS)


def test_load_split_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path / "train.csv", LABELS)


# ================================================================ prepare_data
@pytest.fixture
def raw_files(tmp_path):
    """Train 100 dòng (xếp theo nhãn, có <br />, có 1 cặp trùng nội bộ), test 20 dòng (2 dòng trùng train)."""
    train_texts = [f"bad film number {i}<br />really bad" for i in range(50)]
    train_texts += [f"good film number {i}<br />really good" for i in range(50)]
    train_texts[1] = train_texts[0].upper()  # trùng nội bộ (khác hoa/thường)
    train_labels = [0] * 50 + [1] * 50
    test_texts = [f"test review {i}" for i in range(18)] + [train_texts[10], train_texts[60]]
    test_labels = [i % 2 for i in range(18)] + [0, 1]
    return (
        write_csv(tmp_path / "raw_train.csv", train_texts, train_labels),
        write_csv(tmp_path / "raw_test.csv", test_texts, test_labels),
    )


def run_prepare(raw_files, out_dir, tokenizer, seed=42):
    return prepare_data(raw_files[0], raw_files[1], out_dir, 0.1, seed, tokenizer, [5, 10], LABELS)


def test_prepare_data_end_to_end(tmp_path, raw_files, whitespace_tokenizer):
    report = run_prepare(raw_files, tmp_path / "splits", whitespace_tokenizer)
    splits = {name: pd.read_csv(info["path"], keep_default_na=False) for name, info in report["files"].items()}

    keys = {name: set(data["text"].map(duplicate_key)) for name, data in splits.items()}
    assert keys["train"].isdisjoint(keys["validation"])
    assert keys["train"].isdisjoint(keys["test"])
    assert keys["validation"].isdisjoint(keys["test"])

    assert len(splits["test"]) == 20  # test giữ nguyên
    assert not splits["train"]["text"].str.contains("<br", case=False).any()
    assert set(splits["validation"]["label"]) == {0, 1}
    assert set(splits["validation"]["source_split"]) == {"train"}
    assert report["final_rows"]["train"] + report["final_rows"]["validation"] == 100 - sum(
        report["duplicates"][key] for key in report["duplicates"] if key.endswith("removed")
    )
    assert report["train_token_length"]["count"] == report["final_rows"]["train"]


def test_prepare_data_is_reproducible(tmp_path, raw_files, whitespace_tokenizer):
    first = run_prepare(raw_files, tmp_path / "a", whitespace_tokenizer)["files"]
    second = run_prepare(raw_files, tmp_path / "b", whitespace_tokenizer)["files"]
    assert {k: v["sha256"] for k, v in first.items()} == {k: v["sha256"] for k, v in second.items()}


def test_prepare_data_rejects_label_unseen_in_train(tmp_path, whitespace_tokenizer):
    raw_train = write_csv(tmp_path / "tr.csv", [f"r{i}" for i in range(20)], [0, 1] * 10)
    raw_test = write_csv(tmp_path / "te.csv", ["x", "y"], [0, 2])
    with pytest.raises(ValueError, match="chưa xuất hiện"):
        prepare_data(raw_train, raw_test, tmp_path / "out", 0.2, 42, whitespace_tokenizer, [5], (0, 1, 2))


def test_prepare_data_missing_raw_file_raises(tmp_path, whitespace_tokenizer):
    with pytest.raises(FileNotFoundError):
        prepare_data(tmp_path / "no.csv", tmp_path / "no2.csv", tmp_path / "out", 0.1, 42, whitespace_tokenizer, [5], LABELS)
