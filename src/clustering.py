import click

from src import options
from src.cli_helpers import load_data, process_data, save_data
from src.config import ConfigContainer
from src.utils import get_config_file


@click.command(
    name="clustering",
    context_settings={"help_option_names": options.help_option_names},
)
@options.input_file
@options.method
@options.output_file
def cli(input_file: str, method: str | None, output_file: str):
    """Command-line tool for data clustering.

    This command-line tool is used for clustering data using various algorithms.
    The tool reads configuration settings from a YAML file named `config.yaml` located in the current working directory.
    If the `config.yaml` file is not found in the current working directory, the tool falls back to the default configuration settings.

    If the `input_file` does not have an extension, the default file format.

    If the `method` parameter is not provided, the default clustering algorithm specified in the `config.yaml` file will be used.

    If the `output_file` parameter is not provided, the clustering labels will be saved into a file named `labels` with the default file format.


    Example usage:
        python src/clustering.py --help
    """

    config_container = ConfigContainer()
    config_container.config.from_yaml(get_config_file())

    data = load_data(config_container, input_file)

    labels = process_data(config_container, data, method)

    save_data(config_container, output_file, labels)


if __name__ == "__main__":
    cli()
