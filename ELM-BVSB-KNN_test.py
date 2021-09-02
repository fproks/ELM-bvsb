from elm import BvsbClassifier, BvsbUtils, elmUtils
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler

import time

print("------EVM-BVSB-KNN------")
# digits = load_digits()
data = datasets.load_digits()
stdc = StandardScaler()  # 均值归一化
label_size = 0.1

data.data = stdc.fit_transform(data.data / 16.0)
train, iter, test = elmUtils.splitDataWithIter(data.data, data.target, label_size, 0.7)

Y_iter = BvsbUtils.KNNClassifierResult(train[0], train[1], iter[0])
print(Y_iter.size)

tic = time.perf_counter_ns()
bvsbc = BvsbClassifier(train[0], train[1], iter[0], Y_iter, test[0], test[1], iterNum=0.1)
bvsbc.createELM(n_hidden=1000, activation_func="sigmoid", alpha=1.0, random_state=0)
bvsbc.X_test = test[0]
bvsbc.Y_test = test[1]
bvsbc.trainELMWithBvsb()
toc = time.perf_counter_ns()

print(bvsbc.score(test[0], test[1]))
print("ELM-BVSB 项目用时:%d" % ((toc - tic) / 1000 / 1000))
