"""Test cho lab3.utils: chọn thiết bị và ghi JSON."""

import json

import pytest
import torch

from lab3.utils import get_device, save_json


# ---------------------------------------------------------------- get_device
def test_get_device_auto_returns_supported_device():
    assert get_device().type in {"cuda", "mps", "cpu"}


@pytest.mark.parametrize("name", ["cpu", "CPU", "  cpu  "])
def test_get_device_cpu_accepts_case_and_spaces(name):
    assert get_device(name) == torch.device("cpu")


@pytest.mark.parametrize("name", ["tpu", "", "gpu0"])
def test_get_device_rejects_unknown_name(name):
    with pytest.raises(ValueError):
        get_device(name)


@pytest.mark.parametrize("value", [123, 1.5, ["cpu"], True])
def test_get_device_rejects_wrong_type(value):
    with pytest.raises(TypeError):
        get_device(value)


@pytest.mark.skipif(torch.cuda.is_available(), reason="Máy có CUDA nên không kiểm tra được trường hợp thiếu CUDA.")
def test_get_device_cuda_missing_raises():
    with pytest.raises(RuntimeError):
        get_device("cuda")


# ---------------------------------------------------------------- save_json
def test_save_json_round_trip_keeps_vietnamese(tmp_path):
    data = {"nhãn": "TÍCH CỰC", "điểm": 0.97, "danh_sách": [1, 2, 3], "rỗng": None}
    path = save_json(data, tmp_path / "ket_qua.json")
    assert json.loads(path.read_text(encoding="utf-8")) == data
    assert "TÍCH CỰC" in path.read_text(encoding="utf-8")


def test_save_json_creates_missing_parent_folders(tmp_path):
    path = save_json([1], tmp_path / "a" / "b" / "c.json")
    assert path.exists()


def test_save_json_overwrites_existing_file(tmp_path):
    target = tmp_path / "x.json"
    save_json({"v": 1}, target)
    save_json({"v": 2}, target)
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 2}


@pytest.mark.parametrize("bad_path", [None, 123, True, ["a.json"]])
def test_save_json_rejects_wrong_path_type(bad_path):
    with pytest.raises(TypeError):
        save_json({}, bad_path)


@pytest.mark.parametrize("bad_path", ["", "   "])
def test_save_json_rejects_empty_path(bad_path):
    with pytest.raises(ValueError):
        save_json({}, bad_path)


def test_save_json_rejects_directory_path(tmp_path):
    with pytest.raises(IsADirectoryError):
        save_json({}, tmp_path)


def test_save_json_rejects_unserializable_data(tmp_path):
    target = tmp_path / "x.json"
    with pytest.raises(TypeError):
        save_json({"tập": {1, 2}}, target)
    assert not target.exists()


# ---------------------------------------------------------------- load_json
from lab3.utils import load_json  # noqa: E402


def test_load_json_round_trip(tmp_path):
    data = {"nhãn": ["TÍCH CỰC"], "số": 1.5}
    assert load_json(save_json(data, tmp_path / "a.json")) == data


def test_load_json_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_json(tmp_path / "khong_co.json")
    with pytest.raises(IsADirectoryError):
        load_json(tmp_path)
    with pytest.raises(TypeError):
        load_json(None)
    broken = tmp_path / "hong.json"
    broken.write_text("{không phải json", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        load_json(broken)
