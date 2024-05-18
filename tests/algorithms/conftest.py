import pytest

from src.algorithms.dbscan import DBSCAN
from src.algorithms.kmeans import KMeans
from src.algorithms.spectral_clustering import SpectralClustering

from .data import KWARGS_DBSCAN, KWARGS_KMEANS, KWARGS_SPECTRAL


@pytest.fixture
def dbscan_instance() -> DBSCAN:
    return DBSCAN(**KWARGS_DBSCAN)  # type: ignore


@pytest.fixture
def kmeans_instance() -> KMeans:
    return KMeans(**KWARGS_KMEANS)  # type: ignore


@pytest.fixture
def spectral_clustering_instance() -> SpectralClustering:
    return SpectralClustering(**KWARGS_SPECTRAL)  # type: ignore
