# -*- coding: utf8
import numpy as np
from elm.elm import ELMClassifier


class BvsbClassifier:
    def __init__(self, X_train: np.ndarray, Y_train: np.ndarray, X_iter: np.ndarray, Y_iter: np.ndarray, iterNum=0.2,
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
        self.X_test = None
        self.Y_test = None
        self.elmc = None
        self._score = 0
        self._upperLimit = np.ceil(Y_iter.size * upLimit)
        self._iter_continue = True
        print("pernumber is %d" % self.perNum)

    def createELM(self, n_hidden, activation_func, alpha, random_state):
        assert self.elmc is None
        self.elmc = ELMClassifier(n_hidden=n_hidden, activation_func=activation_func, alpha=alpha,
                                  random_state=random_state)

    """计算bvsb"""

    def calculateBvsb(self, percentageData: np.ndarray) -> np.ndarray:
        p_temp = np.sort(percentageData)[:, -2:]
        return p_temp[:, -1] - p_temp[:, -2]

    """获取分类正确的数据中bvsb靠前的数据索引"""

    def argBvsbWithAccuracy(self, perData: np.ndarray) -> np.ndarray:
        argAcc = BvsbUtils.getAccIndex(self.Y_iter, perData)
        print("KNN与ELM匹配个数%d" % argAcc.size)
        if argAcc.size == 0:
            return np.array([], dtype=int)
        assert argAcc.max() < perData.shape[0]
        bvsbData = self.calculateBvsb(perData)
        arrBvsb = np.c_[bvsbData[argAcc], argAcc]
        argSBvsbAcc = arrBvsb[arrBvsb[:, 0].argsort()][:, 1]
        _iterNum = min(self.perNum, self._upperLimit)
        return argSBvsbAcc[-_iterNum:].astype(int)

    """获取下次需要进行训练的数据，并从迭代集合中删除他们"""

    def updateTrainData(self, perData: np.ndarray):
        if self._upperLimit <= 0:
            self._iter_continue = False
            return 0
        assert perData.ndim != 1
        print(perData.shape)
        print(self.X_train.shape)
        # assert perData.shape[1] == self.X_train.shape[1]
        argBvsb = self.argBvsbWithAccuracy(perData)
        if argBvsb.size < (self.perNum * 0.1):
            self._iter_continue = False
            return
        self._upperLimit -= argBvsb.size
        X_add = self.X_iter[argBvsb]
        Y_add = self.Y_iter[argBvsb]
        self.X_iter = np.delete(self.X_iter, argBvsb, axis=0)
        self.Y_iter = np.delete(self.Y_iter, argBvsb, axis=0)
        print("增加数据%d个" % Y_add.size)
        print("")
        self.X_train = np.r_[self.X_train, X_add]
        self.Y_train = np.r_[self.Y_train, Y_add]
        print("训练集数据现在有%d个" % self.Y_train.size)
        print("剩余迭代数据%d个" % self.Y_iter.size)
        return Y_add.size

    """数据添加到训练集合中"""

    def updateDataWithoutKNN(self, preData: np.ndarray):
        if self._upperLimit <= 0:
            self._iter_continue = False
            return 0
        assert preData.ndim != 1
        if preData.shape[0] <= self._upperLimit:
            self._iter_continue = False
            return
        _iterNum = int(min(self._upperLimit, self.perNum))
        if _iterNum < (self.perNum * 0.1):
            self._iter_continue = False
            return
        argbvsbData = self.calculateBvsb(preData).argsort()
        sortArgBvsb = argbvsbData[-_iterNum:].astype(int)
        self._upperLimit -= _iterNum
        Y_iter = self.elmc.binarizer.inverse_transform(preData)
        X_add = self.X_iter[sortArgBvsb]
        Y_add = Y_iter[sortArgBvsb]
        self.X_iter = np.delete(self.X_iter, sortArgBvsb, axis=0)
        print("增加数据%d个" % Y_add.size)
        self.X_train = np.r_[self.X_train, X_add]
        self.Y_train = np.r_[self.Y_train, Y_add]
        print("训练集现有数据%d个" % self.Y_train.size)
        print("剩余迭代数据%d个" % self.X_iter.shape[0])
        return Y_add.size

    def score(self, x, y):
        _tmp = self.elmc.score(x, y)
        if _tmp > self._score: self._score = _tmp
        return self._score

    def TrainELMWithBvsb(self):
        i = 0
        while self._iter_continue:
            i = i + 1
            print("-------------------")
            print("第%d次训练" % i)
            self.elmc.fit(self.X_train, self.Y_train)
            preData = self.elmc.predict_with_percentage(self.X_iter)
            score = self.elmc.scoreWithPredict(self.Y_iter, preData)
            print("根据他们的结果得到的正确率%f" % score)
            addSize = self.updateTrainData(preData)
            print("目前，测试机的分类正确率%f" % (self.score(self.X_test, self.Y_test)))

    def TrainELMWithoutKNN(self):
        i = 0
        while self._iter_continue:
            i = i + 1
            print("-----------ELM Without KNN---------------")
            print("第%d次训练" % i)
            self.elmc.fit(self.X_train, self.Y_train)
            preData = self.elmc.predict_with_percentage(self.X_iter)
            addsize = self.updateDataWithoutKNN(preData)
            print("测试集分类正确率%f" % (self.score(self.X_test, self.Y_test)))


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
