import pytest

from src.algorithms.dbscan import DBSCAN
from src.algorithms.kmeans import KMeans
from src.algorithms.spectral_clustering import SpectralClustering
from src.config import ConfigContainer
from src.file_handlers.json_handler import JsonHandler
from src.file_handlers.numpy_handler import NumpyHandler

from .data import CONFIG_DATA


class TestConfigContainer:
    def test_config_init(self, config_container: ConfigContainer):
        config = config_container.config()
        assert config == CONFIG_DATA

    def test_get_algorithm_default(self, config_container: ConfigContainer):
        clusterizer = config_container.get_algorithm()
        assert isinstance(clusterizer, KMeans)

    @pytest.mark.parametrize(
        ("clustering_type", "expected_clusterizer"),
        [
            ("dbscan", DBSCAN),
            ("spectral", SpectralClustering),
        ],
    )
    def test_get_algorithm(
        self,
        config_container: ConfigContainer,
        clustering_type: str,
        expected_clusterizer: DBSCAN | SpectralClustering,
    ):
        clustering_config = config_container.config.clustering()
        clustering_config.update(dict(type=clustering_type))
        clusterizer = config_container.get_algorithm()
        assert isinstance(clusterizer, expected_clusterizer)  # type: ignore

    def test_get_file_handler_input(self, config_container: ConfigContainer):
        clustering_config = config_container.config.clustering()
        clustering_config.update(dict(input_format="npy"))
        file_handler = config_container.get_file_handler_input()
        assert isinstance(file_handler, NumpyHandler)

    def test_get_file_handler_output(self, config_container: ConfigContainer):
        clustering_config = config_container.config.clustering()
        clustering_config.update(dict(input_format="json"))
        file_handler = config_container.get_file_handler_input()
        assert isinstance(file_handler, JsonHandler)
