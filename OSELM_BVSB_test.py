#!/usr/bin/env python
# encoding: utf-8
from elm import BvsbClassifier, BvsbUtils
from sklearn import datasets
from elm import elmUtils

import time

print("---------OSELM-BVSB-----------")
digits = datasets.load_digits()
print(f'数据集大小问{digits.target.size}')
(train, iters, test) = elmUtils.splitDataWithIter(BvsbUtils.dimensionReductionWithPCA(digits.data, 0.95), digits.target,
                                                  0.2, 0.3)
print(f'训练集大小为{train[1].size}')
print(f'迭代训练集大小为{iters[1].size}')
print(f'测试集大小为{test[1].size}')

tic = time.perf_counter_ns()
bvsbc = BvsbClassifier(train[0], train[1], iters[0], iters[1], test[0], test[1], iterNum=0.1)
bvsbc.createOSELM(n_hidden=1000)
bvsbc.trainOSELMWithoutKNN()
toc = time.perf_counter_ns()

print(f'OSELM-BVSB-KNN 正确率为{bvsbc.score(test[0], test[1])}')
print(f'OSELM-BVSB-KNN 项目用时:{(toc - tic) / 1000 / 1000} ms')
