from abc import ABC, abstractmethod

import numpy as np


class BaseClustering(ABC):
    """Base class for clustering algorithms."""

    @abstractmethod
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Processes the input data using the clustering algorithm.

        Args:
            data (np.ndarray): Input data to be processed.

        Returns:
            np.ndarray: Processed data.
        """
        pass
