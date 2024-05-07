import numpy as np

from src.utils import reshape_to_2d


def test_reshape_to_2d_with_2d_array():
    array = np.array([[1, 2, 3], [4, 5, 6]])
    result = reshape_to_2d(array)
    assert np.array_equal(result, array)


def test_reshape_to_2d_with_3d_array():
    array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    expected_result = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    result = reshape_to_2d(array)
    assert np.array_equal(result, expected_result)


def test_reshape_to_2d_with_1d_array():
    array = np.array([1, 2, 3, 4])
    result = reshape_to_2d(array)
    assert np.array_equal(result, array)
