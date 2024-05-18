import numpy as np
from sklearn.cluster import KMeans as _Kmeans

from .base_clustering import BaseClustering


class KMeans(BaseClustering):
    """KMeans clustering algorithm implementation."""

    def __init__(self, n_clusters: int):
        """
        Initialize the KMeans clustering algorithm.

        Args:
            n_clusters (int): The number of clusters to form as well as the number of centroids to generate.
        """
        self.algorithm = _Kmeans(n_clusters=n_clusters)

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Process the input data using KMeans clustering algorithm.

        Args:
            data (np.ndarray): Input data to be clustered.

        Returns:
            np.ndarray: Array of cluster labels assigned to each data point.
        """
        return self.algorithm.fit_predict(data)
