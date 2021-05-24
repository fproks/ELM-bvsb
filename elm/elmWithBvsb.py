# -*- coding: utf8
import numpy as np
from typing import Tuple

from elm.elm import ELMClassifier
from elm.OSELM import OSELM
import warnings
from sklearn.decomposition import PCA
from config import LOGGER


class BvsbClassifier:
    def __init__(self, X_train: np.ndarray, Y_train: np.ndarray, X_iter: np.ndarray, Y_iter: np.ndarray,
                 x_test: np.ndarray, y_test: np.ndarray, iterNum=0.2,
                 upLimit=0.5):
        assert type(iterNum) is float or type(iterNum) is int
        assert X_train.size != 0 and Y_train.size != 0 and X_iter.size != 0 and Y_iter.size != 0
        if iterNum < 1:
            self.perNum = int(np.ceil(iterNum * Y_train.size))
        else:
            self.perNum = int(np.ceil(iterNum))
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_iter = X_iter
        self.Y_iter = Y_iter
        self.X_test = x_test
        self.Y_test = y_test
        self.elmc = None
        self._score = 0
        self._upperLimit = np.ceil(Y_iter.size * upLimit)
        self._iter_continue = True
        LOGGER.info(f'perNumber is {self.perNum}')

    def createELM(self, n_hidden, activation_func, alpha, random_state):
        assert self.elmc is None
        LOGGER.info("create and init ELM")
        self.elmc = ELMClassifier(n_hidden=n_hidden, activation_func=activation_func, alpha=alpha,
                                  random_state=random_state)

    def createOSELM(self, n_hidden):
        assert self.elmc is None
        LOGGER.info("create and init OSELM")
        self.elmc = OSELM(self.X_train, self.Y_train, n_hidden)
        self.Y_iter = self.elmc.binarizer.fit_transform(self.Y_iter)

    """计算bvsb"""

    """获取分类正确的数据中bvsb靠前的数据索引"""

    def argBvsbWithAccuracy(self, perData: np.ndarray) -> np.ndarray:
        argAcc = BvsbUtils.getAccIndex(self.Y_iter, perData)
        LOGGER.info(f'KNN与ELM匹配个数{argAcc.size}')
        if argAcc.size == 0:
            return np.array([], dtype=int)
        assert argAcc.max() < perData.shape[0]
        bvsbData = BvsbUtils.calculateBvsb(perData)
        arrBvsb = np.c_[bvsbData[argAcc], argAcc]
        argSBvsbAcc = arrBvsb[arrBvsb[:, 0].argsort()][:, 1]
        _iterNum = int(min(self.perNum, self._upperLimit))
        LOGGER.debug(f'欲获取的bvsb-knn数据个数:{_iterNum}')
        LOGGER.debug(f'bvsb-knn 一致后数据个数: {len(argSBvsbAcc)}')
        return argSBvsbAcc[-_iterNum:].astype(int)

    """获取下次需要进行训练的数据，并从迭代集合中删除他们"""

    def getUpdateData(self, preData: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._upperLimit <= 0:
            self._iter_continue = False
            return None
        assert preData.ndim != 1
        argBvab = self.argBvsbWithAccuracy(preData)
        if argBvab.size < (self.perNum * .01):
            self._iter_continue = False
            return None
        self._upperLimit -= argBvab.size
        X_up = self.X_iter[argBvab]
        Y_up = self.Y_iter[argBvab]
        LOGGER.info(f'获取再次迭代数据{len(Y_up)}个')
        self.X_iter = np.delete(self.X_iter, argBvab, axis=0)
        self.Y_iter = np.delete(self.Y_iter, argBvab, axis=0)
        return X_up, Y_up

    def updateTrainData(self, preData: np.ndarray):
        _data = self.getUpdateData(preData)
        if _data is None:
            return None
        LOGGER.info(f'增加数据 {_data[1].size}个')
        self.X_train = np.r_[self.X_train, _data[0]]
        self.Y_train = np.r_[self.Y_train, _data[1]]
        LOGGER.info(f'训练集数据现在有 {self.Y_train.size}个')
        LOGGER.info(f'剩余迭代数据 {self.Y_iter.size}个')
        return _data[1].size

    """数据添加到训练集合中"""

    def getUpdateDateWithoutKNN(self, preData: np.ndarray):
        if self._upperLimit <= 0:
            self._iter_continue = False
            return None
        assert preData.ndim != 1
        if preData.shape[0] <= self._upperLimit:
            self._iter_continue = False
            return None
        _iterNum = int(min(self._upperLimit, self.perNum))
        if _iterNum < (self.perNum * 0.1):
            self._iter_continue = False
            return None
        argbvsbData = BvsbUtils.calculateBvsb(preData).argsort()
        sortArgBvsb = argbvsbData[-_iterNum:].astype(int)
        self._upperLimit -= _iterNum
        Y_iter = self.elmc.binarizer.inverse_transform(preData)
        X_add = self.X_iter[sortArgBvsb]
        Y_add = Y_iter[sortArgBvsb]
        self.X_iter = np.delete(self.X_iter, sortArgBvsb, axis=0)
        return X_add, Y_add

    def updateDataWithoutKNN(self, preData: np.ndarray):
        X_add, Y_add = self.getUpdateDateWithoutKNN(preData)
        LOGGER.info(f'增加数据 {Y_add.size}')
        self.X_train = np.r_[self.X_train, X_add]
        self.Y_train = np.r_[self.Y_train, Y_add]
        LOGGER.info(f'训练集现有数据 {self.Y_train.size}个')
        LOGGER.info(f'剩余迭代数据{self.X_iter.shape[0]}个')
        return Y_add.size

    def score(self, x, y):
        _tmp = self.elmc.score(x, y)
        if _tmp > self._score: self._score = _tmp
        return self._score

    def trainELMWithBvsb(self):
        i = 0
        print("---------------------ELM-BVSB-TRAIN-----------------------------")
        while self._iter_continue:
            i = i + 1
            print(f'--------------------第{i}次训练--------------------')
            self.elmc.fit(self.X_train, self.Y_train)
            preData = self.elmc.predict_with_percentage(self.X_iter)
            score = self.elmc.scoreWithPredict(self.Y_iter, preData)
            LOGGER.info(f'第{i}次迭代后迭代数据集的正确率为{score}')
            LOGGER.debug(f'perData 类型为:{type(preData)}')
            self.updateTrainData(preData)
            LOGGER.debug(f'第{i}次迭代训练后测试集的分类正确率为{self.score(self.X_test, self.Y_test)}')

    def trainOSELMWithBvsb(self):
        i = 0
        print("-------------------------------OSELM-BVSB-TRAIN------------------------------------------")
        LOGGER.info(f'迭代训练前算法对测试集的正确率为{self.elmc.score(self.X_test, self.Y_test)}')
        while self._iter_continue:
            i = i + 1
            print(f'---------------第{i}次训练-------------------')
            predict = self.elmc.predict(self.X_iter)
            score = self.elmc.scoreWithPredict(self.Y_iter, predict)
            LOGGER.info(f'第{i}次迭代后迭代数据集的正确率为{score}')
            _data = self.getUpdateData(predict)
            if _data is None:
                LOGGER.warn("未获取迭代数据，迭代训练结束")
                break
            self.elmc.train(_data[0], _data[1])
            LOGGER.debug(f'第{i}次迭代训练后测试集的分类正确率为{self.elmc.score(self.X_test, self.Y_test)}')

    def trainELMWithoutKNN(self):
        i = 0
        print("-------------------------ELM Without KNN-------------------------")
        while self._iter_continue:
            i = i + 1
            print(f'---------------第{i}次训练-------------------')
            self.elmc.fit(self.X_train, self.Y_train)
            preData = self.elmc.predict_with_percentage(self.X_iter)
            self.updateDataWithoutKNN(preData)
            LOGGER.debug(f'第{i}次迭代训练后测试集的分类正确率为{self.elmc.score(self.X_test, self.Y_test)}')

    def trainOSELMWithoutKNN(self):
        i = 0
        print("----------------------OSELM WITHOUT KNN---------------------------")
        while self._iter_continue:
            i = i + 1
            print(f'---------------第{i}次训练-------------------')
            predict = self.elmc.predict(self.X_iter)
            _data = self.getUpdateDateWithoutKNN(predict)
            if _data is None:
                LOGGER.warn("未获取迭代数据，迭代训练结束")
                break
            LOGGER.info(f'第{i}次训练时进行训练的数据个数:{_data[1].size}')
            self.elmc.train(_data[0], _data[1])
            LOGGER.debug(f'第{i}次迭代训练后测试集的分类正确率为{self.score(self.X_test, self.Y_test)}')


class BvsbUtils(object):

    @staticmethod
    def KNNClassifierResult(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, K=20):
        from sklearn import neighbors
        nbr = neighbors.KNeighborsClassifier(K)
        nbr.fit(x_train, y_train)
        return nbr.predict(x_test)

    @staticmethod
    def classPrediction(perData: np.ndarray, Y: np.ndarray):
        from sklearn.preprocessing import LabelBinarizer
        binarizer = LabelBinarizer(-1, 1)
        binarizer.fit(Y)
        return binarizer.inverse_transform(perData)
        # return perData.argmax(axis=1)

    @staticmethod
    def argCorrectClassify(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
        assert Y_true.shape[0] == Y_pred.shape[0]
        return np.argwhere(Y_true == Y_pred).flatten().astype(int)

    @staticmethod
    def getAccIndex(Y_true, perData):
        return BvsbUtils.argCorrectClassify(Y_true, BvsbUtils.classPrediction(perData, Y_true))

    @staticmethod
    def dimensionReductionWithPCA(data: np.ndarray, n_components=None) -> np.ndarray:
        LOGGER.info("Dimensionality reduction  with PCA")

        def _su(a: list, cp: float):
            p = 0
            for i in range(len(a)):
                p += a[i]
                if p > cp: return i
            return len(a)

        assert data.ndim == 2
        import math
        if n_components is None: n_components = math.ceil(data.shape[1] / 2)
        assert isinstance(n_components, int) or isinstance(n_components, float)
        pca = None
        if isinstance(n_components, int):
            if n_components > min(data.shape):
                n_components = min(data.shape)
                warnings.warn(f'n_components exceed max size,revise to ${n_components}')
                pca = PCA(n_components)
                return pca.fit_transform(data)
        else:
            assert 0 < n_components < 1
            pca = PCA()
            result = pca.fit_transform(data)
            components = _su(pca.explained_variance_ratio_, n_components)
            LOGGER.info(f'Dimensionality reduction components is {components}')
            return result[:, 0:components + 1]

    @staticmethod
    def calculateBvsb(percentageData: np.ndarray) -> np.ndarray:
        p_temp = np.sort(percentageData)[:, -2:]
        return p_temp[:, -1] - p_temp[:, -2]
