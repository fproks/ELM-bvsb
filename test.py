#!/usr/bin/env python
# encoding: utf-8
import sklearn.datasets

from elm import BvsbClassifier, BvsbUtils
from sklearn.model_selection import train_test_split
from sklearn import datasets
from elm import elmUtils
import numpy as np
from sklearn.preprocessing import StandardScaler

import time
import torch
import numpy as np

acc_rem = []  #初始化精度列表
time_rem = []   #初始化时间列表

for ii in range(10):

    print("---------OSELM-BVSB-KNN-----------")
    #data = datasets.fetch_olivetti_faces()


    #data = sklearn.datasets.fetch_covtype()
    #data = datasets.load_iris()
    data = datasets.load_digits()
    #ata = datasets.load_wine()#############################################有问题，跑不了
    #data = datasets.load_breast_cancer()
    stdc=StandardScaler()

    #data.data,data.target=stdc.fit_transform(data.data)/16.0, data.target
    #data.data=BvsbUtils.dimensionReductionWithPCA(data.data,0.95)
    label_size=0.3

    (train_data, iter_data, test_data) = elmUtils.splitDataWithIter(data.data,data.target, label_size, 0.2)
    iter_y=BvsbUtils.KNNClassifierResult(train_data[0],train_data[1],iter_data[0])

    tic = time.perf_counter_ns()
    bvsbc = BvsbClassifier(train_data[0], train_data[1], iter_data[0], iter_y, test_data[0], test_data[1],
                           iterNum=0.1)
    bvsbc.createOSELM(n_hidden=1000)
    bvsbc.trainOSELMWithBvsb()

    toc = time.perf_counter_ns()
    acc_temp = bvsbc.score(test_data[0], test_data[1])  #记录每次的精度
    acc_rem.append(acc_temp)            #将每次的精度存入列表
    time_temp = (toc - tic) / 1000 / 1000   #记录每次的时间
    time_rem.append(time_temp)      #将每次的时间存入列表

    print(f'OSELM-BVSB-KNN 正确率为{bvsbc.score(test_data[0], test_data[1])}')
    print(f'OSELM-BVSB-KNN 项目用时:{(toc - tic) / 1000 / 1000} ms')


for i in acc_rem:
    print(f'每一次的精度：{i*100:0.2f}',)   #打印每次精度
acc_mean = np.mean(acc_rem) #求出平均精度
print('平均精度：',acc_mean) #打印平均精度
print('每一次的运行时间：',time_rem) #打印每次时间
time_mean = np.mean(time_rem)   #求出平均时间
print('平均精度：',time_mean)    #打印平均时间

