from elm import BvsbClassifier, BvsbUtils
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler

import time

print("------------")
# digits = load_digits()
digits = datasets.load_digits()
dgx = digits.data
stdc = StandardScaler()  # 均值归一化

dgy = digits.target
print("数据个数:%d" % dgy.size)
dgx, dgy = stdc.fit_transform(digits.data / 16.0), digits.target
dgx_train, dgx_test, dgy_train, dgy_test = train_test_split(dgx, dgy, test_size=0.2)
X_train, X_iter, Y_train, Y_iter = train_test_split(dgx_train, dgy_train, test_size=0.5)
Y_iter = BvsbUtils.KNNClassifierResult(X_train, Y_train, X_iter)
print(Y_iter.size)

tic = time.perf_counter_ns()
bvsbc = BvsbClassifier(X_train, Y_train, X_iter, Y_iter, iterNum=0.1)
bvsbc.createELM(n_hidden=1000, activation_func="tanh", alpha=1.0, random_state=0)
bvsbc.X_test = dgx_test
bvsbc.Y_test = dgy_test
bvsbc.TrainELMWithBvsb()
toc = time.perf_counter_ns()
# bvsbc.elmc.fit(dgx_train,dgy_train)
print("+++++++++++++++++++")
print(bvsbc.score(dgx_test, dgy_test))
print("ELM-BVSB 项目用时:%d" % ((toc - tic) / 1000 / 1000))
