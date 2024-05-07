import json
import os
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from src.file_handlers.base_handler import FileFormatHandler
from src.file_handlers.json_handler import JsonHandler
from src.file_handlers.numpy_handler import NumpyHandler


@pytest.fixture
def base_handler() -> FileFormatHandler:
    return FileFormatHandler()


@pytest.fixture
def json_handler() -> JsonHandler:
    return JsonHandler()


@pytest.fixture
def numpy_handler() -> NumpyHandler:
    return NumpyHandler()


def ndarray(num_samples: int, num_of_dimensions: int) -> np.ndarray:
    """Fixture to generate N-dimensional arrays."""
    feature_vector_sizes = np.random.randint(1, 6, size=num_of_dimensions - 1)
    shape = (num_samples,) + tuple(feature_vector_sizes)
    return np.random.randn(*shape)


@pytest.fixture
def array_4_d() -> np.ndarray:
    return ndarray(num_samples=300, num_of_dimensions=4)


@pytest.fixture
def json_file_path(array_4_d: np.ndarray, tmpdir: Path) -> Generator[str, None, None]:
    file_path = str(tmpdir / "test_data.json")
    with open(file_path, "w") as f:
        json.dump(array_4_d.tolist(), f)
    yield file_path
    os.remove(file_path)


@pytest.fixture
def numpy_file_path(array_4_d: np.ndarray, tmpdir: Path) -> Generator[str, None, None]:
    file_path = str(tmpdir / "test_data.npy")
    np.save(file_path, array_4_d)
    yield file_path
    os.remove(file_path)
