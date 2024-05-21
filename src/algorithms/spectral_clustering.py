import numpy as np
from sklearn.cluster import SpectralClustering as _SpectralClustering

from .base_clustering import BaseClustering


class SpectralClustering(BaseClustering):
    """SpectralClustering clustering algorithm implementation."""

    def __init__(self, n_clusters: int, affinity: str):
        """
        Initialize the SpectralClustering clustering algorithm.

        Args:
            n_clusters (int): The number of clusters to form as well as the number of centroids to generate.
            affinity (str): The affinity type to use. This can be 'nearest_neighbors', 'rbf', etc.
        """
        self.algorithm = _SpectralClustering(n_clusters=n_clusters, affinity=affinity)

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Process the input data using the SpectralClustering algorithm.

        Args:
            data (numpy.ndarray): Input data to be processed.

        Returns:
            numpy.ndarray: The processed data.
        """
        return self.algorithm.fit_predict(data)
