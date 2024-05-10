from pathlib import Path

import numpy as np

from src.file_handlers.numpy_handler import NumpyHandler


def test_load(
    numpy_handler: NumpyHandler, test_array: np.ndarray, numpy_file_path: str
) -> None:
    loaded_data = numpy_handler.load(numpy_file_path)
    np.testing.assert_array_equal(loaded_data, test_array)


def test_save(
    numpy_handler: NumpyHandler, tmpdir: Path, test_array: np.ndarray
) -> None:
    file_path = str(tmpdir / "test_data.npy")
    numpy_handler.save(test_array, file_path)
    saved_data = np.load(file_path)
    np.testing.assert_array_equal(saved_data, test_array)
