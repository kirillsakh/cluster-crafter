from pathlib import Path

import numpy as np

from src.file_handlers.numpy_handler import NumpyHandler


def test_load(
    numpy_handler: NumpyHandler, array_4_d: np.ndarray, numpy_file_path: str
) -> None:
    loaded_data = numpy_handler.load(numpy_file_path)
    assert np.array_equal(loaded_data, array_4_d)


def test_save(numpy_handler: NumpyHandler, tmpdir: Path) -> None:
    data = np.array([1, 2, 3])
    file_path = str(tmpdir / "test_data.npy")
    numpy_handler.save(data, file_path)
    saved_data = np.load(file_path)
    assert np.array_equal(saved_data, data)
