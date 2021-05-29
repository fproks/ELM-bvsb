import numpy as np
from numpy.linalg import pinv
from sklearn.preprocessing import LabelBinarizer
from config import LOGGER


_internal_activation_funcs = {'sine': np.sin,
                              'tanh': np.tanh,
                              'sigmoid': (lambda x: 1.0 / (1.0 + np.exp(-x)))
                              }


def transformYWithOutnumbers(y: np.ndarray, outnumber: int) -> np.ndarray:
    assert y.ndim == 1
    result = np.zeros((y.size, outnumber), int)
    for i in range(y.size):
        result[i, y[i]] = 1
    result[result == 0] = -1
    return result


class OSELM(object):
    def __init__(self, features: np.ndarray, targets: np.ndarray, numHiddenNeurons, active_function="sigmoid"):
        self.binarizer = LabelBinarizer(-1, 1)
        assert features.ndim == 2
        self.inputs = features.shape[1]
        if targets.ndim == 1:
            targets = self.binarizer.fit_transform(targets)
        self.outputs = targets.shape[1]
        self.numHiddenNeurons = numHiddenNeurons
        self.inputWeights = None
        self.bias = None
        self.beta = None
        self.M = None
        if active_function in _internal_activation_funcs:
            self.activate_function = _internal_activation_funcs[active_function]
            LOGGER.info(f"activate function is {active_function}")
        else:
            self.activate_function = _internal_activation_funcs["sigmoid"]
            LOGGER.warn("activate function is not in list, use sigmoid instead")
        self.initializePhase(features, targets)

    def calculateHiddenLayerActivation(self, features):
        V = np.dot(features, np.transpose(self.inputWeights)) + self.bias
        if callable(self.activate_function):
            return self.activate_function(V)
        else:
            LOGGER.warn("activate_func could not callable,use sigmoid instead")
            return sigmoid(V)

    def initializePhase(self, features: np.ndarray, targets: np.ndarray):
        assert features.shape[0] == targets.shape[0]
        if targets.ndim == 1:
            targets = self.binarizer.fit_transform(targets)
        assert targets.shape[1] == self.outputs
        self.inputWeights = np.random.random((self.numHiddenNeurons, self.inputs)) * 2 - 1
        self.bias = np.random.random((1, self.numHiddenNeurons)) * 2 - 1
        H0 = self.calculateHiddenLayerActivation(features)
        self.M = pinv(np.dot(np.transpose(H0), H0))
        self.beta = np.dot(pinv(H0), targets)

    # https://blog.csdn.net/google19890102/article/details/45273309
    def train(self, features: np.ndarray, targets: np.ndarray):
        if targets.ndim == 1:
            targets = self.binarizer.transform(targets)
            if targets.shape[0] != self.outputs:
                targets = transformYWithOutnumbers(targets, self.outputs)
        (numSamples, numOutputs) = targets.shape
        assert features.shape[0] == targets.shape[0]
        assert numOutputs == self.outputs
        H = self.calculateHiddenLayerActivation(features)
        Ht = np.transpose(H)
        try:
            self.M -= np.dot(self.M, np.dot(Ht, np.dot(
                pinv(np.eye(numSamples) + np.dot(H, np.dot(self.M, Ht))),
                np.dot(H, self.M))))
            self.beta += np.dot(np.dot(self.M, Ht), (targets - np.dot(H, self.beta)))
        except np.linalg.LinAlgError:
            LOGGER.error("can not converge, ignore the current training cycle")

    def predict(self, features: np.ndarray):
        H = self.calculateHiddenLayerActivation(features)
        prediction = np.dot(H, self.beta)
        return prediction

    def scoreWithPredict(self, y: np.ndarray, predict):
        from sklearn.metrics import accuracy_score
        _y = self.binarizer.inverse_transform(predict)
        if y.ndim == 2: y = self.binarizer.inverse_transform(y)
        return accuracy_score(y, _y)

    def score(self, X, y):
        return self.scoreWithPredict(y, self.predict(X))
