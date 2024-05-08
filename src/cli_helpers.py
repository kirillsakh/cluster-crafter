import numpy as np

from src.config import ConfigContainer
from src.constants import DEFAULT_FORMAT_TYPE
from src.utils import get_file_format


def load_data(config_container: ConfigContainer, input_file: str) -> np.ndarray:
    """Load data from the input file.

    Args:
        config_container (ConfigContainer): The container for configuration settings.
        input_file (str): The path to the input file.

    Returns:
        np.ndarray: The loaded data.
    """
    input_file_format = get_file_format(input_file)
    if input_file_format is None:
        input_file_format = DEFAULT_FORMAT_TYPE
    update_config(config_container, {"input_format": input_file_format})
    input_file_handler = config_container.get_file_handler_input()
    return input_file_handler.load(input_file)


def process_data(
    config_container: ConfigContainer, data: np.ndarray, method: str | None
) -> np.ndarray:
    """Process the loaded data using the specified clustering method.

    Args:
        config_container (ConfigContainer): The container for configuration settings.
        data (np.ndarray): The data to be processed.
        method (str): The clustering method to be applied.

    Returns:
        np.ndarray: The processed data.
    """
    if method and method != config_container.config.clustering.type():
        update_config(config_container, {"type": method})
    clusterizer = config_container.get_algorithm()
    labels = clusterizer.process(data)
    return labels


def save_data(
    config_container: ConfigContainer, output_file: str, data: np.ndarray
) -> None:
    """Save the processed data to the output file.

    Args:
        config_container (ConfigContainer): The container for configuration settings.
        output_file (str): The path to the output file.
        data (np.ndarray): The data to be saved.

    Returns:
        None
    """
    output_file_format = get_file_format(output_file)
    if output_file_format is None:
        output_file_format = DEFAULT_FORMAT_TYPE
        output_file += "." + output_file_format
    update_config(config_container, {"output_format": output_file_format})
    output_file_handler = config_container.get_file_handler_output()
    output_file_handler.save(data, output_file)


def update_config(config_container: ConfigContainer, data: dict) -> None:
    """Update the configuration settings.

    Args:
        config_container (ConfigContainer): The container for configuration settings.
        data (dict): The dictionary containing the configuration data to update.

    Returns:
        None
    """
    clustering_config = config_container.config.clustering()
    clustering_config.update(data)
    config_container.config.clustering.override(clustering_config)
