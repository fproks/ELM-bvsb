# -*- coding: utf8
import numpy
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

    def createOSELM(self, n_hidden, active_function="sigmoid"):
        assert self.elmc is None
        LOGGER.info("create and init OSELM")
        self.elmc = OSELM(self.X_train, self.Y_train, n_hidden)
        # self.Y_iter = self.elmc.binarizer.transform(self.Y_iter)

    """计算bvsb"""

    """获取分类正确的数据中bvsb靠前的数据索引"""

    def argBvsbWithAccuracy(self, perData: np.ndarray):
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

    def getUpDataIndexWithBvsb(self, predData: np.ndarray,limit=0) -> np.ndarray:
        assert predData.ndim != 1
        preClass = self.elmc.binarizer.inverse_transform(predData)
        argAcc = np.argwhere(self.Y_iter == preClass).flatten().astype(int)  # 相同的索引
        accPreData = predData[argAcc]
        tmp = np.sort(accPreData)
        if tmp.shape[1] >= 2:
            bvsb = tmp[:, -1] - tmp[:, -2]
        else:
            bvsb = tmp.flatten()
        bvsbArg = np.argsort(bvsb)  # bvsb索引
        if limit==0:
            limit=len(bvsbArg)
        real_index = argAcc[bvsbArg[-limit:]].flatten().astype(int)
        return real_index

    def getUpdateDataWithBvsb(self, predData: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._upperLimit <= 0:
            self._iter_continue = False
            return None
        real_index = self.getUpDataIndexWithBvsb(predData,limit= int(min(self.perNum, self._upperLimit)))
        if real_index.size < (self.perNum * 0.1):
            self._iter_continue = False
            return None
        self._upperLimit -= real_index.size
        if len(real_index) == 1:
            print("-------------------------------------------------------")
        X_up = self.X_iter[real_index]
        Y_up = self.Y_iter[real_index]
        self.X_iter = np.delete(self.X_iter, real_index, axis=0)
        self.Y_iter = np.delete(self.Y_iter, real_index, axis=0)
        return X_up, Y_up

    def getUpdataWithoutBVSB(self, predata: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._upperLimit <= 0:
            self._iter_continue = False
            return None
        assert predata.ndim != 1
        preClass = self.elmc.binarizer.inverse_transform(predata)
        argAcc = np.argwhere(self.Y_iter == preClass).flatten().astype(int)  # 相同的索引
        accPreData = predata[argAcc]
        tmp = np.sort(accPreData)[:, -1]
        sortArg = np.argsort(tmp)
        _iterNum = int(min(self.perNum, self._upperLimit))
        real_index = argAcc[sortArg[-_iterNum:]]
        if real_index.size < (self.perNum * 0.1):
            self._iter_continue = False
            return None
        self._upperLimit -= real_index.size
        X_up = self.X_iter[real_index]
        Y_up = self.Y_iter[real_index]
        self.X_iter = np.delete(self.X_iter, real_index, axis=0)
        self.Y_iter = np.delete(self.Y_iter, real_index, axis=0)
        return X_up, Y_up

    def updateTrainDataWithBvsb(self, preData: np.ndarray):
        _data = self.getUpdateDataWithBvsb(preData)
        if _data is None:
            LOGGER.warn("getUpdateTrain is None")
            return None
        return self.mergeTrainData(_data)

    def mergeTrainData(self, _data: Tuple[np.ndarray, np.ndarray]):
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
        _data = self.getUpdateDateWithoutKNN(preData)
        if _data is None:
            return None
        LOGGER.info(f'增加数据 {_data[1].size}')
        self.X_train = np.r_[self.X_train, _data[0]]
        self.Y_train = np.r_[self.Y_train, _data[1]]
        LOGGER.info(f'训练集现有数据 {self.Y_train.size}个')
        LOGGER.info(f'剩余迭代数据{self.X_iter.shape[0]}个')
        return _data[1].size

    def score(self, x, y):
        _tmp = self.elmc.score(x, y)
        if _tmp > self._score: self._score = _tmp
        return self._score

    def fitAndGetUpdateDataIndex(self,limit=0):
        self.elmc.fit(self.X_train, self.Y_train)
        preData = self.elmc.predict_with_percentage(self.X_iter)
        LOGGER.debug(f'perData 类型为:{type(preData)}')
        _data = self.getUpDataIndexWithBvsb(preData,limit=limit)
        return _data

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
            self.updateTrainDataWithBvsb(preData)
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
            _data = self.getUpdateDataWithBvsb(predict)
            if _data is None:
                LOGGER.warn("未获取迭代数据，迭代训练结束")
                break
            self.elmc.fit(_data[0], _data[1])
            LOGGER.debug(f'第{i}次迭代训练后测试集的分类正确率为{self.elmc.score(self.X_test, self.Y_test)}')

    def trainELMWithoutKNN(self):
        i = 0
        print("-------------------------ELM Without KNN-------------------------")
        while self._iter_continue:
            i = i + 1
            print(f'---------------第{i}次训练-------------------')
            self.elmc.fit(self.X_train, self.Y_train)
            preData = self.elmc.predict_with_percentage(self.X_iter)
            if preData is None:
                LOGGER.warn("未获取迭代数据，迭代训练结束")
                break
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
            print(_data[1].shape)
            self.elmc.fit(_data[0], _data[1])
            LOGGER.debug(f'第{i}次迭代训练后测试集的分类正确率为{self.score(self.X_test, self.Y_test)}')

    def trainOSELMWithKNNButBvsb(self):
        i = 0
        print("----------------------OSELM WITH KNN BUT BVSB---------------------------")
        while self._iter_continue:
            i = i + 1
            print(f'---------------第{i}次训练-------------------')
            predict = self.elmc.predict(self.X_iter)
            _data = self.getUpdataWithoutBVSB(predict)
            if _data is None:
                LOGGER.warn("未获取迭代数据，迭代训练结束")
                break
            LOGGER.info(f'第{i}次训练时进行训练的数据个数:{_data[1].size}')
            print(_data[1].shape)
            self.elmc.fit(_data[0], _data[1])
            LOGGER.debug(f'第{i}次迭代训练后测试集的分类正确率为{self.score(self.X_test, self.Y_test)}')

    def trainELMWithKNNButBvsb(self):
        i = 0
        print("------------------------------------ELM WITH KNN BUT BVSB")
        while self._iter_continue:
            i = i + 1
            print(f'-------------------------第{i}次训练----------------------------')
            self.elmc.fit(self.X_train, self.Y_train)
            predict = self.elmc.predict_with_percentage(self.X_iter)
            _data = self.getUpdataWithoutBVSB(predict)
            if _data is None:
                LOGGER.warn("未获取迭代数据，迭代结束")
                break
            LOGGER.info(f'第{i}次训练时添加的数据个数:{_data[1].size}')
            self.mergeTrainData(_data)
            self.elmc.fit(self.X_train, self.Y_train)
            LOGGER.debug(f'第{i}次迭代训练后测试集的分类正确率为{self.score(self.X_test, self.Y_test)}')


class BvsbUtils(object):
    @staticmethod
    def KNNClassifier(x_train: np.ndarray, y_train: np.ndarray, K=20):
        from sklearn import neighbors
        if len(y_train) < K:
            K = len(y_train) - 1
        nbr = neighbors.KNeighborsClassifier(K)
        nbr.fit(x_train, y_train)
        return nbr

    @staticmethod
    def KNNClassifierResult(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, K=20):
        nbr = BvsbUtils.KNNClassifier(x_train, y_train, K)
        return nbr.predict(x_test)

    @staticmethod
    def KNNClassifierProbaRestult(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, K=20):
        return BvsbUtils.KNNClassifier(x_train, y_train, K).predict_proba(x_test)

    @staticmethod
    def SVMClassifierResult(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, K=20):
        from sklearn.svm import SVC
        svm = SVC()
        svm.fit(x_train, y_train)
        return svm.predict(x_test)

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
        if type(data) != numpy.ndarray:
            LOGGER.warn(f'PCA data type is {type(data)}')
            if isinstance(n_components, float):
                LOGGER.warn("data is sparse matrix, use integer n_components")
                raise Exception("data is sparse matrix, please confirm n_components use integer")
            from sklearn.decomposition import TruncatedSVD
            pca = TruncatedSVD(n_components)
            return pca.fit_transform(data)

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
        if isinstance(n_components, int):
            if n_components > min(data.shape):
                n_components = min(data.shape)
                warnings.warn(f"n_components exceed max size,revise to ${n_components}")
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
        if p_temp.shape[1] >= 2:
            return p_temp[:, -1] - p_temp[:, -2]
        else:
            return np.reshape(p_temp, (len(p_temp),))
