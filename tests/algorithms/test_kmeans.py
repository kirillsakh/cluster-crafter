import numpy as np

from src.algorithms.kmeans import KMeans

from .data import KWARGS_KMEANS


class TestDBSCAN:
    def test_create(self):
        kmeans = KMeans.create(KWARGS_KMEANS)
        assert isinstance(kmeans, KMeans)
        params = kmeans.algorithm.get_params()
        for key, value in KWARGS_KMEANS.items():
            assert params[key] == value

    def test_process(self, mock_kmeans: KMeans):
        data = np.array([[1, 2], [3, 4]])
        mock_kmeans.process(data)
        assert mock_kmeans.algorithm.fit_predict.called_once_with(data)
