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
data = datasets.load_digits()
stdc=StandardScaler()

data.data,data.target=stdc.fit_transform(data.data)/16.0, data.target
data.data=BvsbUtils.dimensionReductionWithPCA(data.data,0.95)
data.data,_,data.target,_=train_test_split(data.data,data.target,test_size=0.7)
label_size=0.3

(train_data, iter_data, test_data) = elmUtils.splitDataWithIter(data.data,data.target, label_size, 0.2)
iter_y=BvsbUtils.KNNClassifierResult(train_data[0],train_data[1],iter_data[0])

tic = time.perf_counter_ns()
bvsbc = BvsbClassifier(train_data[0], train_data[1], iter_data[0], iter_y, test_data[0], test_data[1],
                       iterNum=0.1)
bvsbc.createOSELM(n_hidden=1000)
bvsbc.trainOSELMWithBvsb()
toc = time.perf_counter_ns()

print(f'OSELM-BVSB-KNN 正确率为{bvsbc.score(test_data[0], test_data[1])}')
print(f'OSELM-BVSB-KNN 项目用时:{(toc - tic) / 1000 / 1000} ms')
