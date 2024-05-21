from unittest.mock import MagicMock

import numpy as np
from sklearn.cluster import DBSCAN as _DBSCAN

from src.algorithms.dbscan import DBSCAN


class TestDBSCAN:
    def test_create(self, dbscan_instance: DBSCAN):
        assert isinstance(dbscan_instance, DBSCAN)
        assert isinstance(dbscan_instance.algorithm, _DBSCAN)

    def test_process(self, dbscan_instance: DBSCAN, test_array: np.ndarray):
        dbscan_instance.algorithm = MagicMock()
        _ = dbscan_instance.process(test_array)

        dbscan_instance.algorithm.fit_predict.assert_called_once_with(test_array)
