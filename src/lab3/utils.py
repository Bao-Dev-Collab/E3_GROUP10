"""Các hàm tiện ích dùng chung: chọn thiết bị tính toán, ghi file JSON."""

import json
from pathlib import Path
from typing import Any

import torch

_SUPPORTED_DEVICES = ("cuda", "mps", "cpu")


def get_device(prefer: str | None = None) -> torch.device:
    """Chọn thiết bị để chạy model.

    Giả định:
    - prefer là None hoặc một trong "cuda", "mps", "cpu" (không phân biệt hoa thường).
    - prefer = None: tự chọn theo thứ tự ưu tiên cuda -> mps -> cpu.
    - Nếu chỉ định thiết bị mà máy không có thì báo lỗi, không âm thầm đổi sang thiết bị khác.
    """
    if prefer is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if not isinstance(prefer, str):
        raise TypeError(f"prefer phải là str hoặc None, nhận được {type(prefer).__name__}.")

    name = prefer.strip().lower()
    if name not in _SUPPORTED_DEVICES:
        raise ValueError(f"Thiết bị '{prefer}' không hợp lệ. Chỉ nhận: {', '.join(_SUPPORTED_DEVICES)}.")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Máy này không có GPU CUDA.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Máy này không có GPU Apple (MPS).")
    return torch.device(name)


def load_json(path: str | Path) -> Any:
    """Đọc file JSON (UTF-8).

    Giả định: path là str hoặc Path trỏ tới một file JSON hợp lệ do project tạo ra.
    """
    if path is None or isinstance(path, bool) or not isinstance(path, (str, Path)):
        raise TypeError(f"path phải là str hoặc Path, nhận được {type(path).__name__}.")
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Không tìm thấy file '{target}'.")
    if target.is_dir():
        raise IsADirectoryError(f"'{target}' là thư mục, cần đường dẫn tới file JSON.")
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"File '{target}' không phải JSON hợp lệ: {error}") from error


def save_json(data: Any, path: str | Path) -> Path:
    """Ghi dữ liệu ra file JSON (UTF-8, giữ nguyên tiếng Việt).

    Giả định:
    - data chỉ chứa kiểu ghi được ra JSON (dict, list, str, số, bool, None).
    - path là str hoặc Path trỏ tới một file (không phải thư mục). Thư mục cha chưa có sẽ được tạo.
    - File cũ cùng tên bị ghi đè.
    """
    if path is None or isinstance(path, bool) or not isinstance(path, (str, Path)):
        raise TypeError(f"path phải là str hoặc Path, nhận được {type(path).__name__}.")
    if isinstance(path, str) and not path.strip():
        raise ValueError("path không được rỗng.")

    target = Path(path)
    if target.is_dir():
        raise IsADirectoryError(f"'{target}' là thư mục, cần đường dẫn tới một file.")

    # json.dumps tự báo TypeError nếu data có kiểu không ghi được ra JSON.
    text = json.dumps(data, ensure_ascii=False, indent=2)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text + "\n", encoding="utf-8")
    return target
