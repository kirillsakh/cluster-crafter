from abc import ABC, abstractmethod

import numpy as np


class FileFormatHandler(ABC):
    """Base class for file format handlers."""

    @abstractmethod
    def load(self, file_path: str) -> np.ndarray:
        """Load data from a file.

        Args:
            file_path (str): The path to the file to read.

        Returns:
            numpy.ndarray: The loaded data as a NumPy array.
        """
        pass

    @abstractmethod
    def save(self, data: np.ndarray, file_path: str) -> None:
        """Save data to a file.

        Args:
            data (np.ndarray): The data to save to the file.
            file_path (str): The path to the file to save the data.
        """
        pass
