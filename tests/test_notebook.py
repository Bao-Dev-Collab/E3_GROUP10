"""Test cho notebook tổng hợp: đúng định dạng, code hợp lệ, chạy được từ đầu đến cuối."""

import ast

import nbformat
import pytest
from nbclient import NotebookClient

from lab3 import config

NOTEBOOK = config.PROJECT_ROOT / "notebooks" / "Practice3_HuggingFace_Final.ipynb"
REQUIRED_OUTPUTS = [
    config.FINAL_MODEL_DIR / "model.safetensors",
    config.DATA_REPORT_PATH,
    config.BASELINE_REPORT_PATH,
    config.TRAIN_REPORT_PATH,
    config.EVALUATION_REPORT_PATH,
]


def load() -> nbformat.NotebookNode:
    return nbformat.read(NOTEBOOK, as_version=4)


def test_notebook_is_valid_nbformat():
    nbformat.validate(load())


def test_every_code_cell_is_valid_python():
    for index, cell in enumerate(load().cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"ô {index}")


def test_notebook_does_not_train_or_touch_test_selection():
    source = "\n".join(cell.source for cell in load().cells if cell.cell_type == "code")
    assert "run_training(" not in source  # notebook không train lại
    assert ".train(" not in source


def test_saved_notebook_has_no_error_outputs():
    errors = [o for c in load().cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert errors == []


@pytest.mark.slow
@pytest.mark.skipif(not all(path.exists() for path in REQUIRED_OUTPUTS), reason="Cần chạy đủ các bước train / evaluate trước.")
def test_notebook_runs_end_to_end(tmp_path):
    notebook = load()
    NotebookClient(notebook, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    errors = [o for c in notebook.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert errors == []
