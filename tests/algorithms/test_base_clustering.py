import numpy as np
import pytest

from src.algorithms.base_clustering import BaseClustering


class TestBaseClustering:
    clusterizer = BaseClustering({})

    def test_init(self):
        assert self.clusterizer.algorithm is None

    def test_create(self):
        with pytest.raises(NotImplementedError):
            BaseClustering.create({})

    def test_process_not_implemented(self):
        data = np.array([[1, 2], [3, 4]])
        with pytest.raises(NotImplementedError):
            self.clusterizer.process(data)
