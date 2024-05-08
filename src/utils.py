import os
from pathlib import Path

import yaml

from .constants import CONFIG_YAML_NAME


def get_config_file() -> Path:
    """Get the path to the configuration file.

    Returns:
        Path: The path to the configuration file.
    """
    if os.path.exists(CONFIG_YAML_NAME):
        config_path = Path(os.getcwd()) / CONFIG_YAML_NAME
    else:
        config_path = Path(__file__).parent.parent / CONFIG_YAML_NAME
    return config_path


def get_file_format(input_file: str) -> str | None:
    """
    Determine the file format based on the provided input file path.

    Args:
        input_file (str): The path of the input file.

    Returns:
        str | None: The file format extracted from the input file path or None if no extension is found.
    """
    if "." in input_file:
        file_extension = input_file.split(".")[-1]
        return file_extension
    else:
        return None


def load_from_yaml(file_path: Path) -> dict:
    """Load a YAML file and return its contents as a dictionary.

    Args:
        file_path (Path): The path to the YAML file.

    Returns:
        dict: A dictionary containing the contents of the YAML file.
    """
    with open(file_path) as f:
        config_dict = yaml.safe_load(f)
    return config_dict
