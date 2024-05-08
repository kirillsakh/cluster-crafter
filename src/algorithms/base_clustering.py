import numpy as np


class BaseClustering:
    """Base class for clustering algorithms."""

    def __init__(self, kwargs: dict):
        """
        Initializes the clustering algorithm.

        Args:
            kwargs (dict): Additional keyword arguments for configuring the algorithm.
        """
        self.algorithm = None

    @classmethod
    def create(cls, kwargs: dict) -> "BaseClustering":
        """
        Creates an instance of the clustering algorithm.

        This method should be implemented by subclasses.

        Args:
            kwargs (dict): Additional keyword arguments for configuring the algorithm.

        Returns:
            BaseClustering: An instance of the clustering algorithm.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Processes the input data using the clustering algorithm.

        This method should be implemented by subclasses.

        Args:
            data (np.ndarray): Input data to be processed.

        Returns:
            np.ndarray: Processed data.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError
