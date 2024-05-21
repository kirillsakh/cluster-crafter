import numpy as np
from sklearn.cluster import DBSCAN as _DBSCAN

from .base_clustering import BaseClustering


class DBSCAN(BaseClustering):
    """DBSCAN clustering algorithm implementation."""

    def __init__(self, eps: float, min_samples: int):
        """
        Initialize the DBSCAN clustering algorithm.

        Args:
            eps (float): The maximum distance between two samples for one to be considered
                as in the neighborhood of the other.
            min_samples (int): The number of samples (or total weight) in a neighborhood for
                a point to be considered as a core point.
        """
        self.algorithm = _DBSCAN(eps=eps, min_samples=min_samples)

    def process(self, data: np.ndarray) -> np.ndarray:
        """Process the input data using DBSCAN clustering algorithm.

        Args:
            data (np.ndarray): Input data to be clustered.

        Returns:
            np.ndarray: Array of cluster labels assigned to each data point.
        """
        return self.algorithm.fit_predict(data)
