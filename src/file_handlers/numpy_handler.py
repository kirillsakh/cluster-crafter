import numpy as np

from ..file_handlers.base_handler import FileFormatHandler


class NumpyHandler(FileFormatHandler):
    """Handles loading and saving data from/to NumPy binary files (.npy)."""

    def load(self, file_path: str) -> np.ndarray:
        """Load data from a NumPy binary file.

        Args:
            file_path (str): The path to the NumPy binary file (.npy).

        Returns:
            numpy.ndarray: The loaded data as a NumPy array.
        """
        return np.load(file_path)

    def save(self, data: np.ndarray, file_path: str):
        """Save data to a NumPy binary file.

        Args:
            data (numpy.ndarray): The data to save.
            file_path (str): The path to save the NumPy binary file (.npy).
        """
        np.save(file_path, data)
