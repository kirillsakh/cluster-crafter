import os
from pathlib import Path
from typing import Generator

import pytest
import yaml

from src.config import ConfigContainer
from src.constants import CONFIG_YAML_NAME

from .data import CONFIG_DATA


@pytest.fixture
def config_file_path(tmpdir: Path) -> Generator[str, None, None]:
    file_path = str(tmpdir / CONFIG_YAML_NAME)
    with open(file_path, "w") as f:
        yaml.dump(CONFIG_DATA, f)
    yield file_path
    os.remove(file_path)


@pytest.fixture
def config_container(config_file_path: Path) -> ConfigContainer:
    config_container = ConfigContainer()
    config_container.config.from_yaml(config_file_path)
    return config_container
