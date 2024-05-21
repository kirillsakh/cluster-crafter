from dependency_injector import containers, providers

from src.algorithms.dbscan import DBSCAN
from src.algorithms.kmeans import KMeans
from src.algorithms.spectral_clustering import SpectralClustering
from src.file_handlers.json_handler import JsonHandler
from src.file_handlers.numpy_handler import NumpyHandler


class ConfigContainer(containers.DeclarativeContainer):
    """Container for managing configuration settings."""

    config = providers.Configuration()

    get_algorithm = providers.Selector(
        selector=config.clustering.type,
        kmeans=providers.Factory(KMeans.create, config.clustering.kmeans),
        spectral=providers.Factory(SpectralClustering, config.clustering.spectral),
        dbscan=providers.Factory(DBSCAN, config.clustering.dbscan),
    )

    get_file_handler_input = providers.Selector(
        selector=config.clustering.input_format,
        json=providers.Factory(JsonHandler),
        npy=providers.Factory(NumpyHandler),
    )

    get_file_handler_output = providers.Selector(
        selector=config.clustering.output_format,
        json=providers.Factory(JsonHandler),
        npy=providers.Factory(NumpyHandler),
    )
