import numpy as np


class FileFormatHandler:
    """Base class for file format handlers."""

    def load(self, file_path: str) -> np.ndarray:
        """Load data from a file.

        This method should be implemented by subclasses.

        Args:
            file_path (str): The path to the file to read.

        Returns:
            numpy.ndarray: The loaded data as a NumPy array.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    def save(self, data: np.ndarray, file_path: str) -> None:
        """Save data to a file.

        This method should be implemented by subclasses.

        Args:
            data (np.ndarray): The data to save to the file.
            file_path (str): The path to the file to save the data.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError
