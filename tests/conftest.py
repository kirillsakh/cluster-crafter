import numpy as np
import pytest


@pytest.fixture
def test_array() -> np.ndarray:
    return np.random.normal(size=(300, 5))
