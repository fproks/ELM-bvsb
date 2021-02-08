# -*- coding: utf8
import numpy as np
from elm.elm import ELMClassifier


class BvsbClassifier:
    def __init__(self, X_train: np.ndarray, Y_train: np.ndarray, X_iter: np.ndarray, Y_iter: np.ndarray, iterNum=0.2):
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
        print("根据我的结果得到的正确率%f" % (argAcc.size / self.Y_iter.size))
        if argAcc.size == 0:
            return np.array([], dtype=int)
        assert argAcc.max() < perData.shape[0]
        bvsbData = self.calculateBvsb(perData)
        arrBvsb = np.c_[bvsbData[argAcc], argAcc]
        argSBvsbAcc = arrBvsb[arrBvsb[:, 0].argsort()][:, 1]
        return argSBvsbAcc[-self.perNum:].astype(int)

    """获取下次需要进行训练的数据，并从迭代集合中删除他们"""

    def getBvsbWithAccuracy(self, perData: np.ndarray):
        assert perData.ndim != 1
        print(perData.shape)
        print(self.X_train.shape)
        # assert perData.shape[1] == self.X_train.shape[1]
        argBvsb = self.argBvsbWithAccuracy(perData)
        X_result = self.X_iter[argBvsb]
        Y_result = self.Y_iter[argBvsb]
        self.X_iter = np.delete(self.X_iter, argBvsb, axis=0)
        self.Y_iter = np.delete(self.Y_iter, argBvsb, axis=0)
        return X_result, Y_result

    def updateTrainData(self, preData):
        X_add, Y_add = self.getBvsbWithAccuracy(preData)
        print("增加数据%d个" % Y_add.size)
        if Y_add.size < self.perNum / 2:
            return Y_add.size
        self.X_train = np.r_[self.X_train, X_add]
        self.Y_train = np.r_[self.Y_train, Y_add]
        print("训练集数据现在有%d个" % self.Y_train.size)
        print("剩余迭代数据%d个" % self.Y_iter.size)
        if self.Y_iter.size < self.perNum:
            self.X_train = np.r_[self.X_train, self.X_iter]
            self.Y_train = np.r_[self.Y_train, self.Y_iter]
            self.X_iter = np.array([[]])
            self.Y_iter = np.array([])
        return Y_add.size

    """数据添加到训练集合中"""

    def TrainELMWithBvsb(self):
        i = 0
        while self.Y_iter.size > self.perNum:
            print(i)
            i = i + 1
            self.elmc.fit(self.X_train, self.Y_train)
            preData = self.elmc.predict_with_percentage(self.X_iter)
            score = self.elmc.scoreWithPredict(self.Y_iter, preData)
            print("根据他们的结果得到的正确率%f" % score)
            if score < 0.3:
                break
            addSize = self.updateTrainData(preData)
            print("目前，测试机的分类正确率%f" % (self.elmc.score(self.X_test, self.Y_test)))
            if addSize < self.perNum / 2:
                break
        self.elmc.fit(self.X_train, self.Y_train)


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

    @staticmethod
    def argCorrectClassify(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
        assert Y_true.shape[0] == Y_pred.shape[0]
        return np.argwhere(Y_true == Y_pred).flatten().astype(int)

    @staticmethod
    def getAccIndex(Y_true, perData):
        return BvsbUtils.argCorrectClassify(Y_true, BvsbUtils.classPrediction(perData, Y_true))
