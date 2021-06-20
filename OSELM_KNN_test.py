#!/usr/bin/env python
# encoding: utf-8
from elm import BvsbClassifier, BvsbUtils
from sklearn import datasets
from elm import elmUtils
from sklearn.preprocessing import StandardScaler
import time

print("---------OSELM-BVSB-----------")
data=elmUtils.readDataFileToData("data/balance-scale.data", targetIndex=0)
print(f'数据集大小问{data.target.size}')
data.data=StandardScaler().fit_transform(data.data)
(train, iter, test) = elmUtils.splitDataWithIter(data.data, data.target,
                                                 0.3, 0.3)
print(f'训练集大小为{train[1].size}')
print(f'迭代训练集大小为{iter[1].size}')
print(f'测试集大小为{test[1].size}')
iter_y = BvsbUtils.SVMClassifierResult(train[0], train[1], iter[0])
tic = time.perf_counter_ns()
bvsbc = BvsbClassifier(train[0], train[1], iter[0], iter_y, test[0], test[1], iterNum=0.1)
bvsbc.createOSELM(n_hidden=1000)
bvsbc.trainOSELMWithKNNButBvsb()
toc = time.perf_counter_ns()

print(f'OSELM-BVSB-KNN 正确率为{bvsbc.score(test[0], test[1])}')
print(f'OSELM-BVSB-KNN 项目用时:{(toc - tic) / 1000 / 1000} ms')
