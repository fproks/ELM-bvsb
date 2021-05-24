import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple

stdsc = StandardScaler()
from config import LOGGER

class elmUtils(object):

    @staticmethod
    def _formatData(data: np.ndarray, target: np.ndarray, need: bool) -> Tuple[np.ndarray, np.ndarray]:
        if need is False:
            return data, target
        if data.ndim == 3:
            LOGGER.info("reshape data from 3 dimension to 2 dimension")
            data = data.reshape((data.shape[0], -1))
        assert data.ndim == 2
        assert target.ndim == 1
        assert data.shape[0] == target.shape[0]
        data = stdsc.fit_transform(data / 16.0)
        return data, target

    @staticmethod
    def splitData(data: np.ndarray, target: np.ndarray, test_size=0.5, need=True) -> Tuple[tuple, tuple]:
        (data, target) = elmUtils._formatData(data, target, need)
        dgx_train, dgx_test, dgy_train, dgy_test = train_test_split(data, target, test_size=test_size)
        return (dgx_train, dgy_train), (dgx_test, dgy_test)

    @staticmethod
    def splitDataWithIter(data: np.ndarray, target: np.ndarray, train_size=0.2, iter_size=0.5) -> Tuple[
        tuple, tuple, tuple]:
        assert train_size + iter_size < 1
        (train_data, test_iter_data) = elmUtils.splitData(data, target, 1 - train_size)
        (iter_data, test_data) = elmUtils.splitData(test_iter_data[0], test_iter_data[1],
                                                    (1 - iter_size - train_size) / (1 - train_size), False)
        return train_data, iter_data, test_data
