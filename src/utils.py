import numpy as np


def reshape_to_2d(array: np.ndarray) -> np.ndarray:
    """Reshape a numpy array into a 2-dimensional array if its shape length is greater than 2.

    Args:
        array (numpy.ndarray): The input array to reshape.

    Returns:
        numpy.ndarray: The reshaped array. If the input array's shape length is less than or equal to 2,
        it returns the array unchanged.

    Raises:
        ValueError: If the input array is not a numpy array.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError("Input array must be a numpy array")

    if len(array.shape) <= 2:
        return array
    else:
        num_samples = array.shape[0]
        num_feature_vectors = np.prod(array.shape[1:])
        return array.reshape((num_samples, num_feature_vectors))
