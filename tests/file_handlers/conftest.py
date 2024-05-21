import json
import os
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from src.file_handlers.json_handler import JsonHandler
from src.file_handlers.numpy_handler import NumpyHandler


@pytest.fixture
def json_handler() -> JsonHandler:
    return JsonHandler()


@pytest.fixture
def numpy_handler() -> NumpyHandler:
    return NumpyHandler()


@pytest.fixture
def json_file_path(test_array: np.ndarray, tmpdir: Path) -> Generator[str, None, None]:
    file_path = str(tmpdir / "test_data.json")
    with open(file_path, "w") as f:
        json.dump(test_array.tolist(), f)
    yield file_path
    os.remove(file_path)


@pytest.fixture
def numpy_file_path(test_array: np.ndarray, tmpdir: Path) -> Generator[str, None, None]:
    file_path = str(tmpdir / "test_data.npy")
    np.save(file_path, test_array)
    yield file_path
    os.remove(file_path)
