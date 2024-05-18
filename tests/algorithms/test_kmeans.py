from unittest.mock import MagicMock

import numpy as np
from sklearn.cluster import KMeans as _Kmeans

from src.algorithms.kmeans import KMeans


class TestKMeans:
    def test_create(self, kmeans_instance: KMeans):
        assert isinstance(kmeans_instance, KMeans)
        assert isinstance(kmeans_instance.algorithm, _Kmeans)

    def test_process(self, kmeans_instance: KMeans, test_array: np.ndarray):
        kmeans_instance.algorithm = MagicMock()
        _ = kmeans_instance.process(test_array)

        kmeans_instance.algorithm.fit_predict.assert_called_once_with(test_array)
