import numpy as np

from src.algorithms.spectral_clustering import SpectralClustering

from .data import KWARGS_SPECTRAL


class TestDBSCAN:
    def test_create(self):
        spectral = SpectralClustering.create(KWARGS_SPECTRAL)
        assert isinstance(spectral, SpectralClustering)
        params = spectral.algorithm.get_params()
        for key, value in KWARGS_SPECTRAL.items():
            assert params[key] == value

    def test_process(self, mock_spectral: SpectralClustering):
        data = np.array([[1, 2], [3, 4]])
        mock_spectral.process(data)
        assert mock_spectral.algorithm.fit_predict.called_once_with(data)
