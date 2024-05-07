import numpy as np

from src.algorithms.dbscan import DBSCAN

from .data import KWARGS_DBSCAN


class TestDBSCAN:
    def test_create(self):
        dbscan = DBSCAN.create(KWARGS_DBSCAN)
        assert isinstance(dbscan, DBSCAN)
        params = dbscan.algorithm.get_params()
        for key, value in KWARGS_DBSCAN.items():
            assert params[key] == value

    def test_process(self, mock_dbscan: DBSCAN):
        data = np.array([[1, 2], [3, 4]])
        mock_dbscan.process(data)
        assert mock_dbscan.algorithm.fit_predict.called_once_with(data)
