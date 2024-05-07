import json

import numpy as np

from .base_handler import FileFormatHandler


class JsonHandler(FileFormatHandler):
    """Handles loading and saving data from/to JSON files."""

    def load(self, file_path: str) -> np.ndarray:
        """Load data from a JSON file.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            numpy.ndarray: The loaded data as a NumPy array.
        """
        with open(file_path) as f:
            json_data = json.load(f)
        return np.array(json_data)

    def save(self, data: np.ndarray, file_path: str) -> None:
        """Save data to a JSON file.

        Args:
            data (numpy.ndarray): The data to save.
            file_path (str): The path to save the JSON file.
        """
        json_data = json.dumps(data.tolist())
        with open(file_path, "w") as f:
            f.write(json_data)
