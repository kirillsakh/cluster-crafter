from unittest.mock import MagicMock

import numpy as np
from sklearn.cluster import SpectralClustering as _SpectralClustering

from src.algorithms.spectral_clustering import SpectralClustering


class TestSpectralClustering:
    def test_create(self, spectral_clustering_instance: SpectralClustering):
        assert isinstance(spectral_clustering_instance, SpectralClustering)
        assert isinstance(spectral_clustering_instance.algorithm, _SpectralClustering)

    def test_process(
        self, spectral_clustering_instance: SpectralClustering, test_array: np.ndarray
    ):
        spectral_clustering_instance.algorithm = MagicMock()
        _ = spectral_clustering_instance.process(test_array)

        spectral_clustering_instance.algorithm.fit_predict.assert_called_once_with(
            test_array
        )
