#!/usr/bin/env python
# encoding: utf-8
import time

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from elm import BvsbClassifier, BvsbUtils
from elm import elmUtils

print("---------OSELM-BVSB-KNN-----------")
#data = datasets.load_digits()
data=elmUtils.readDataFileToData("data/balance-scale.data", targetIndex=0)
#data=elmUtils.readDataFileToData("./data/abalone.data", targetIndex=-1, transformIndex=[0])
stdc = StandardScaler()

#data.data, data.target = stdc.fit_transform(data.data) / 16.0, data.target
#data.data = BvsbUtils.dimensionReductionWithPCA(data.data, 0.95)
label_size = 0.3

(train_data, iter_data, test_data) = elmUtils.splitDataWithIter(data.data, data.target, label_size, 0.3)
iter_y = BvsbUtils.KNNClassifierResult(train_data[0], train_data[1], iter_data[0])

tic = time.perf_counter_ns()
bvsbc = BvsbClassifier(train_data[0], train_data[1], iter_data[0], iter_y, test_data[0], test_data[1],
                       iterNum=0.2)
bvsbc.createOSELM(n_hidden=1000,active_function="sigmoid")
bvsbc.trainOSELMWithBvsb()
toc = time.perf_counter_ns()

print(f'OSELM-BVSB-KNN 正确率为{bvsbc.score(test_data[0], test_data[1])}')
print(f'OSELM-BVSB-KNN 项目用时:{(toc - tic) / 1000 / 1000} ms')
