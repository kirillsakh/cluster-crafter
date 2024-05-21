from src.config import ConfigContainer, KMeans
from src.file_handlers.json_handler import JsonHandler
from src.file_handlers.numpy_handler import NumpyHandler

from .data import CONFIG_DATA


class TestConfigContainer:
    def test_config_init(self, config_container: ConfigContainer):
        config = config_container.config()
        assert config == CONFIG_DATA

    def test_get_algorithm(self, config_container: ConfigContainer):
        clusterizer = config_container.get_algorithm()
        assert isinstance(clusterizer, KMeans)

    def test_get_file_handler_input(self, config_container: ConfigContainer):
        clusterin_config = config_container.config.clustering()
        clusterin_config.update(dict(input_format="npy"))
        config_container.config.clustering.override(clusterin_config)
        file_handler = config_container.get_file_handler_input()
        assert isinstance(file_handler, NumpyHandler)

    def test_get_file_handler_output(self, config_container: ConfigContainer):
        clusterin_config = config_container.config.clustering()
        clusterin_config.update(dict(input_format="json"))
        config_container.config.clustering.override(clusterin_config)
        file_handler = config_container.get_file_handler_input()
        assert isinstance(file_handler, JsonHandler)
