import click
import numpy as np

import options
from src.config import ConfigContainer
from src.utils import get_config_file, get_file_format, reshape_to_2d


@click.command(
    name="cluster-crafter",
    context_settings={"help_option_names": options.help_option_names},
)
@options.input_file
@options.method
@options.output_file
def cli(input_file: str, method: str | None, output_file: str | None):
    """Command-line tool for data clustering.

    This command-line tool is used for clustering data using various algorithms.
    The tool reads configuration settings from a YAML file named `config.yaml` located in the current working directory.
    If the `config.yaml` file is not found in the current working directory, the tool falls back to the default configuration settings.

    If the input file does not have an extension, the default file format specified in the `config.yaml` file will be used.

    If the `method` parameter is not provided, the default clustering algorithm specified in the `config.yaml` file will be used.

    If the `output_file` parameter is not provided, the clustering labels will be saved into a file named `labels` with the default file format specified in the `config.yaml` file.


    Example usage:
        python cli.py --input-file=data.json --method=kmeans --output-file=clustered_data.json
    """

    # Create a container for configuration settings
    config_container = ConfigContainer.create(config_file=get_config_file())

    # Load data from input file
    data = _load_data(config_container, input_file)

    # Perform clustering
    labels = _process_data(config_container, data, method)

    # Save clustered data to output file
    _save_data(config_container, output_file, labels)

def _load_data(config_container: ConfigContainer, input_file: str) -> np.ndarray:
    """Load data from the input file.

    Args:
        config_container (ConfigContainer): The container for configuration settings.
        input_file (str): The path to the input file.

    Returns:
        np.ndarray: The loaded data.
    """
    input_file_format = get_file_format(input_file)
    if not input_file_format:
        input_file_format = config_container.config.default_format()
    input_file_handler = ConfigContainer.get_handler(container=config_container, format_name=input_file_format)
    return input_file_handler.load(input_file)


def _process_data(config_container: ConfigContainer, data: np.ndarray, method: str) -> np.ndarray:
    """Process the loaded data using the specified clustering method.

    Args:
        config_container (ConfigContainer): The container for configuration settings.
        data (np.ndarray): The data to be processed.
        method (str): The clustering method to be applied.

    Returns:
        np.ndarray: The processed data.
    """
    clusterizer = ConfigContainer.get_algorithm(container=config_container, algorithm_name=method)
    data_flat = reshape_to_2d(data)
    return clusterizer.fit_predict(data_flat)


def _save_data(config_container: ConfigContainer, output_file: str, data: np.ndarray) -> None:
    """Save the processed data to the output file.

    Args:
        config_container (ConfigContainer): The container for configuration settings.
        output_file (str): The path to the output file.
        data (np.ndarray): The data to be saved.

    Returns:
        None
    """
    output_file_format = get_file_format(output_file)
    if not output_file_format:
        output_file_format = config_container.config.default_format()
        output_file += '.' + output_file_format
    output_file_handler = ConfigContainer.get_handler(container=config_container, format_name=output_file_format)
    output_file_handler.save(data, output_file)


if __name__ == "__main__":
    cli()
