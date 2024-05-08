import numpy as np
from sklearn.cluster import DBSCAN as _DBSCAN  # type: ignore

from src.utils import reshape_to_2d

from .base_clustering import BaseClustering


class DBSCAN(BaseClustering):
    """DBSCAN clustering algorithm implementation.

    Args:
        kwargs (dict): Keyword arguments to initialize the DBSCAN algorithm.

    Attributes:
        algorithm (_DBSCAN): Instance of the DBSCAN algorithm.

    """

    def __init__(self, kwargs: dict):
        """Initialize the DBSCAN clustering algorithm.

        Args:
            kwargs (dict): Keyword arguments to initialize the DBSCAN algorithm.

        """
        self.algorithm: _DBSCAN = _DBSCAN(**kwargs)

    @classmethod
    def create(cls, kwargs: dict) -> "DBSCAN":
        """Create a new instance of the DBSCAN class.

        Args:
            kwargs (dict): Keyword arguments to initialize the DBSCAN algorithm.

        Returns:
            DBSCAN: An instance of the DBSCAN class.

        """
        return cls(kwargs)

    def process(self, data: np.ndarray) -> np.ndarray:
        """Process the input data using the DBSCAN algorithm.

        Args:
            data (np.ndarray): Input data to be clustered.

        Returns:
            np.ndarray: Array of cluster labels assigned to each data point.

        """
        data_flat = reshape_to_2d(data)
        return self.algorithm.fit_predict(data_flat)
