from importlib import import_module

from dependency_injector import containers, providers

from .file_handlers.base_handler import FileFormatHandler
from .types import Clusterizer


class ConfigContainer(containers.DeclarativeContainer):
    """Container for handling dependency injection of tool parameters."""

    config = providers.Configuration()

    @providers.Singleton
    def file_handlers(container: containers.DeclarativeContainer) -> dict[str, FileFormatHandler]:
        """Provides file format handlers based on configuration.

        Args:
            container (containers.DeclarativeContainer): The container instance.

        Returns:
            dict[str, FileFormatHandler]: A dictionary mapping format names to file format handlers.
        """
        handlers = {}

        formats = container.config.formats()

        for format_name, handler_path in formats.items():
            module_path, class_name = handler_path.rsplit('.', 1)
            module = import_module(module_path)
            handler = getattr(module, class_name)()
            handlers[format_name] = handler

        return handlers

    @providers.Singleton
    def get_handler(container: containers.DeclarativeContainer, format_name: str) -> FileFormatHandler:
        """Provides a file format handler based on the specified format name.

        Args:
            container (containers.DeclarativeContainer): The container instance.
            format_name (str): The name of the file format.

        Returns:
            FileFormatHandler: The file format handler instance.
        """
        handlers = ConfigContainer.file_handlers(container)
        return handlers[format_name]

    @providers.Singleton
    def clustering_algorithms(container: containers.DeclarativeContainer) -> dict[str, Clusterizer]:
        """Provides clustering algorithms based on configuration.

        Args:
            container (containers.DeclarativeContainer): The container instance.

        Returns:
            dict[str, Clusterizer]: A dictionary mapping algorithm names to clustering algorithms.
        """
        clusterizers = {}

        algorithms = container.config.algorithms()
        hyperparams = container.config.hyperparams()

        for algorithm_name, algorithm_path in algorithms.items():
            module_path, class_name = algorithm_path.rsplit('.', 1)
            module = import_module(module_path)
            params = hyperparams[algorithm_name]
            algorithm = getattr(module, class_name)(**params)
            clusterizers[algorithm_name] = algorithm

        return clusterizers

    @providers.Singleton
    def get_algorithm(container: containers.DeclarativeContainer, algorithm_name: str | None) -> Clusterizer:
        """Provides a clustering algorithm based on the specified algorithm name.

        Args:
            container (containers.DeclarativeContainer): The container instance.
            algorithm_name (str, optional): The name of the clustering algorithm.

        Returns:
            Clusterizer: The clustering algorithm instance.
        """
        clusterizers = ConfigContainer.clustering_algorithms(container)
        if not algorithm_name or algorithm_name not in clusterizers:
            return clusterizers[container.config.default_algorithm()]
        return clusterizers[algorithm_name]

    @classmethod
    def create(cls, config_file: str) -> containers.DeclarativeContainer:
        """Creates a new instance of the ConfigContainer class based on the specified config file.

        Args:
            config_file (str): The path to the configuration file.

        Returns:
            containers.DeclarativeContainer: The ConfigContainer instance.
        """
        container = cls()
        container.config.from_yaml(config_file)
        return container
