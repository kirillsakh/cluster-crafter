from unittest.mock import MagicMock

import pytest

from src.algorithms.dbscan import DBSCAN
from src.algorithms.kmeans import KMeans
from src.algorithms.spectral_clustering import SpectralClustering

from .data import KWARGS_DBSCAN, KWARGS_KMEANS, KWARGS_SPECTRAL


@pytest.fixture
def mock_dbscan() -> DBSCAN:
    dbscan = DBSCAN.create(KWARGS_DBSCAN)
    dbscan.algorithm = MagicMock()
    dbscan.algorithm.fit_predict = MagicMock()
    return dbscan


@pytest.fixture
def mock_kmeans() -> KMeans:
    kmeans = KMeans.create(KWARGS_KMEANS)
    kmeans.algorithm = MagicMock()
    kmeans.algorithm.fit_predict = MagicMock()
    return kmeans


@pytest.fixture
def mock_spectral() -> SpectralClustering:
    spectral = SpectralClustering.create(KWARGS_SPECTRAL)
    spectral.algorithm = MagicMock()
    spectral.algorithm.fit_predict = MagicMock()
    return spectral
