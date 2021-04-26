#!/usr/bin/env python
# encoding: utf-8
from elm import BvsbClassifier, BvsbUtils
from sklearn.model_selection import train_test_split
from sklearn import datasets
from elm import elmUtils
import numpy as np
from sklearn.preprocessing import StandardScaler

import time

print("---------OSELM-BVSB-KNN-----------")
digits = datasets.load_digits()
#
# stdc = StandardScaler()  # 均值归一化
# dgy = digits.target
# print("数据个数:%d" % dgy.size)
# dgx, dgy = stdc.fit_transform(digits.data / 16.0), digits.target
#
# dgx = BvsbUtils.dimensionReductionWithPCA(dgx, 0.95)
#
# dgx_train, dgx_test, dgy_train, dgy_test = train_test_split(dgx, dgy, test_size=0.5)
# X_train, X_iter, Y_train, Y_iter = train_test_split(dgx_train, dgy_train, test_size=0.2)
# Y_iter = BvsbUtils.KNNClassifierResult(X_train, Y_train, X_iter)

(train_data, iter_data, test_data) = elmUtils.splitDataWithIter(BvsbUtils.dimensionReductionWithPCA(digits.data, 0.95),
                                                                digits.target, 0.2, 0.2)
iter_y=BvsbUtils.KNNClassifierResult(train_data[0],train_data[1],iter_data[0])

tic = time.perf_counter_ns()
bvsbc = BvsbClassifier(train_data[0], train_data[1], iter_data[0], iter_y, test_data[0], test_data[1],
                       iterNum=0.1)
bvsbc.createOSELM(n_hidden=1000)
bvsbc.trainOSELMWithBvsb()
toc = time.perf_counter_ns()

print(f'OSELM-BVSB-KNN 正确率为{bvsbc.score(test_data[0], test_data[1])}')
print(f'OSELM-BVSB-KNN 项目用时:{(toc - tic) / 1000 / 1000} ms')
