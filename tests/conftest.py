import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.config import ConfigContainer
from src.constants import CONFIG_YAML_NAME

from .data import CONFIG_DATA


@pytest.fixture
def test_array() -> np.ndarray:
    return np.random.normal(size=(300, 5))


@pytest.fixture
def config_file_path(tmpdir: Path) -> Path:  # type: ignore
    file_path = tmpdir / CONFIG_YAML_NAME
    with open(file_path, "w") as f:
        yaml.dump(CONFIG_DATA, f)
    yield file_path
    os.remove(file_path)


@pytest.fixture
def config_container(config_file_path: Path) -> ConfigContainer:
    config_container = ConfigContainer()
    config_container.config.from_yaml(config_file_path)
    return config_container
