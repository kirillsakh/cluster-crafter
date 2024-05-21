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
        dbscan=providers.Factory(
            DBSCAN,
            eps=config.clustering.dbscan.eps,
            min_samples=config.clustering.dbscan.min_samples,
        ),
        kmeans=providers.Factory(
            KMeans, n_clusters=config.clustering.kmeans.n_clusters
        ),
        spectral=providers.Factory(
            SpectralClustering,
            n_clusters=config.clustering.spectral.n_clusters,
            affinity=config.clustering.spectral.affinity,
        ),
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
