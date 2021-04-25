#!/usr/bin/env python
# encoding: utf-8
from elm import BvsbClassifier, BvsbUtils
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler

import time

print("---------OSELM-BVSB-----------")
digits = datasets.load_digits()

stdc = StandardScaler()  # 均值归一化
dgy = digits.target
print("数据个数:%d" % dgy.size)
dgx, dgy = stdc.fit_transform(digits.data / 16.0), digits.target

dgx = BvsbUtils.dimensionReductionWithPCA(dgx, 0.95)

dgx_train, dgx_test, dgy_train, dgy_test = train_test_split(dgx, dgy, test_size=0.5)
X_train, X_iter, Y_train, Y_iter = train_test_split(dgx_train, dgy_train, test_size=0.8)
Y_iter = BvsbUtils.KNNClassifierResult(X_train, Y_train, X_iter)

print(f'训练集大小为{Y_train.size}')
print(f'迭代训练集大小为{Y_iter.size}')
print(f'测试集大小为{dgy_test.size}')

tic = time.perf_counter_ns()
bvsbc = BvsbClassifier(X_train, Y_train, X_iter, Y_iter, dgx_test, dgy_test, iterNum=0.1)
bvsbc.createOSELM(n_hidden=1000)
bvsbc.trainOSELMWithoutKNN()
toc = time.perf_counter_ns()

print(f'OSELM-BVSB-KNN 正确率为{bvsbc.score(dgx_test, dgy_test)}')
print(f'OSELM-BVSB-KNN 项目用时:{(toc-tic)/1000/1000} ms')
