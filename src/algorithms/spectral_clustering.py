import numpy as np
from sklearn.cluster import SpectralClustering as _SpectralClustering  # type: ignore

from src.utils import reshape_to_2d

from .base_clustering import BaseClustering


class SpectralClustering(BaseClustering):
    """Spectral clustering algorithm implementation."""

    def __init__(self, kwargs: dict):
        """
        Initialize SpectralClustering instance.

        Args:
            kwargs (dict): Keyword arguments to configure the SpectralClustering algorithm.
        """
        self.algorithm: _SpectralClustering = _SpectralClustering(**kwargs)

    @classmethod
    def create(cls, kwargs: dict) -> "SpectralClustering":
        """
        Create a new instance of SpectralClustering.

        Args:
            kwargs (dict): Keyword arguments to configure the SpectralClustering algorithm.

        Returns:
            SpectralClustering: A new instance of SpectralClustering.
        """
        return cls(kwargs)

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Process the input data using the SpectralClustering algorithm.

        Args:
            data (numpy.ndarray): Input data to be processed.

        Returns:
            numpy.ndarray: The processed data.
        """
        data_flat = reshape_to_2d(data)
        return self.algorithm.fit_predict(data_flat)
