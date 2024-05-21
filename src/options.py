import click

from .constants import DEFAULT_OUTPUT_FILE_NAME
from .utils import get_config_file, load_from_yaml

help_option_names = ("-h", "--help")

input_file = click.option("-i", "--input-file", required=True, help="Input file path")

config_path = get_config_file()
algorithm_options = [
    k for k in load_from_yaml(config_path)["clustering"].keys() if k != "type"
]

method = click.option(
    "-m", "--method", type=click.Choice(algorithm_options), help="Clustering method"
)

output_file = click.option(
    "-o", "--output-file", default=DEFAULT_OUTPUT_FILE_NAME, help="Output file path"
)
