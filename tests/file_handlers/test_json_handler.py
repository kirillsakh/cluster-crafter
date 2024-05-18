import json
from pathlib import Path

import numpy as np

from src.file_handlers.json_handler import JsonHandler


def test_load(
    json_handler: JsonHandler, test_array: np.ndarray, json_file_path: str
) -> None:
    loaded_data = json_handler.load(json_file_path)
    np.testing.assert_array_equal(loaded_data, test_array)


def test_save(json_handler: JsonHandler, tmpdir: Path, test_array: np.ndarray) -> None:
    file_path = str(tmpdir / "test_data.json")
    json_handler.save(test_array, file_path)
    with open(file_path, "r") as f:
        saved_data = json.load(f)
    np.testing.assert_array_equal(saved_data, test_array)
