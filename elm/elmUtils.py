import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
import os

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

    @staticmethod
    def is_simple_numpy_number(data: np.ndarray) -> bool:
        if np.issubdtype(data.dtype, np.integer):
            return True
        if np.issubdtype(data.dtype, np.floating):
            return True
        return False

    @staticmethod
    def processingData(data, target, transformIndex=[], needLabelEncoder=False):
        from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
        if elmUtils.is_simple_numpy_number(data) is False:
            LOGGER.warn(f'precessing data type is {data.dtype},start use OneHotEncoder')
            if len(transformIndex) == 0:
                LOGGER.info("transform Index is empty, data transform to float")
                # data = OrdinalEncoder().fit_transform(data)
            else:
                LOGGER.info(f"transform data with col index {transformIndex}")
                data[:, transformIndex] = OrdinalEncoder().fit_transform(data[:, transformIndex])
            data = data.astype(float)
        if type(data) != np.ndarray:
            data = data.todense()
        # if type(data) != np.ndarray:
        #     LOGGER.warn(f"processing data type is {type(data)},use spare method")
        #     data = StandardScaler(with_mean=False).fit_transform(data)
        # else:
        #     data = StandardScaler().fit_transform(data)
        if needLabelEncoder or elmUtils.is_simple_numpy_number(target) is False:
            LOGGER.warn(f'processing target type is {target.dtype},start ues LabelEncoder')
            target = LabelEncoder().fit_transform(target)
        return data, target


    @staticmethod
    def readDataFileToData(filepath, targetIndex, transformIndex=[], deleteIndex=[], delimiter=",", dtype=np.str_,
                           filing_values=0):
        """
        读取文件,并讲数据转换为数字，获取数据集和标准分类
        targetIndex: 类别所在的列，通常为0 或者-1
        transformIndex: 指定那些列需要被转换为浮点数，通常指那些为文本的列，为空则均不进行转换
        deleteIndex: 指定哪些列需要被删除,为空则只删除target所在的列
        delimiter: 分隔符,指定文件中使用什么分隔符，默认为逗号
        dtype: 指定文件中数据的类型，当文件中数据均为float 或者int 时，可更换为float，默认为字符串
        filing_values: 指定当数据存在缺失时，填充什么数据
        return: 返回匿名内部类, 其中data为数据集，target为对应的类别
        """
        LOGGER.info(f"read file: {os.path.abspath(filepath)}")
        data = np.genfromtxt(filepath, delimiter=delimiter, dtype=dtype, missing_values="",
                             filling_values=filing_values)
        target = data[:, targetIndex].flatten()
        assert type(deleteIndex) is list
        deleteIndex.append(targetIndex)
        data = np.delete(data, deleteIndex, axis=1)
        LOGGER.info(f"read file finish ,data shape is {data.shape}")
        data, target = elmUtils.processingData(data, target, transformIndex=transformIndex)

        class _(object):
            pass

        _.data = data
        _.target = target
        return _
