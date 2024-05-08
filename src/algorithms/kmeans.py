import numpy as np
from sklearn.cluster import KMeans as _Kmeans  # type: ignore

from src.utils import reshape_to_2d

from .base_clustering import BaseClustering


class KMeans(BaseClustering):
    """KMeans clustering algorithm implementation."""

    def __init__(self, kwargs: dict):
        """
        Initialize KMeans clustering algorithm.

        Args:
            kwargs (dict): Keyword arguments to initialize KMeans.

        """
        self.algorithm: _Kmeans = _Kmeans(**kwargs)

    @classmethod
    def create(cls, kwargs: dict) -> "KMeans":
        """
        Create an instance of KMeans clustering algorithm.

        Args:
            kwargs (dict): Keyword arguments to initialize KMeans.

        Returns:
            KMeans: An instance of KMeans clustering algorithm.

        """
        return cls(kwargs)

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Process data using KMeans clustering algorithm.

        Args:
            data (np.ndarray): Input data to be processed.

        Returns:
            np.ndarray: Cluster labels for each data point.

        """
        data_flat = reshape_to_2d(data)
        return self.algorithm.fit_predict(data_flat)
