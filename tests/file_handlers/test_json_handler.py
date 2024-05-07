import json
from pathlib import Path

import numpy as np

from src.file_handlers.json_handler import JsonHandler


def test_load(
    json_handler: JsonHandler, array_4_d: np.ndarray, json_file_path: str
) -> None:
    loaded_data = json_handler.load(json_file_path)
    assert np.array_equal(loaded_data, array_4_d)


def test_save(json_handler: JsonHandler, tmpdir: Path) -> None:
    data = np.array([1, 2, 3])
    file_path = str(tmpdir / "test_data.json")
    json_handler.save(data, file_path)
    with open(file_path, "r") as f:
        saved_data = json.load(f)
    assert np.array_equal(saved_data, data)
