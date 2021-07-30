from sklearn.svm import SVC
from elm import BvsbClassifier, BvsbUtils
from sklearn.model_selection import train_test_split
from sklearn import datasets
from elm import elmUtils
import numpy as np

from sklearn.preprocessing import StandardScaler

import time
print("-----------------以下为SVM算法---------------")

acc_rem = []  #初始化精度列表
time_rem = []   #初始化时间列表

for ii in range(10):
    # ---------------------------数据集----------------------------------
    #data=elmUtils.readDataFileToData("./data/iris.data",targetIndex=-1)#xxxxxxxxxxxx
    #data=elmUtils.readDataFileToData("./data/bezdekIris.data", targetIndex=-1)
    # data = datasets.load_wine()#OK
    # data = datasets.load_breast_cancer()#OK
    data=elmUtils.readDataFileToData("./data/seeds.data", delimiter = None, targetIndex=-1)#OK
    #data = datasets.load_digits()
    # data=elmUtils.readDataFileToData("./data/glass.data", targetIndex=-1)#xxxxxxxxxxxx
    # data = datasets.fetch_olivetti_faces()#OK
    # ---------------------------数据集---------------------------------

    #data = datasets.fetch_olivetti_faces()#这是第1个
    #data = sklearn.datasets.fetch_covtype()#这是第2个
    #data = datasets.load_iris()#这是第3个
    #data = datasets.load_digits()#这是第4个  再前面加#屏蔽语句，把运行的打开
    #data = datasets.load_wine()#这是第5个
    #data = datasets.load_breast_cancer()#这是第6个

    stdsc = StandardScaler()
    data.data = stdsc.fit_transform(data.data) / 16.0
    label_size = 0.3
    t1 = time.perf_counter_ns()

    train, test = elmUtils.splitData(data.data, data.target, 1 - label_size, True)
    svm = SVC()
    svm.fit(train[0], train[1])
    tmp_acc = svm.score(test[0], test[1])
    print(f'SVM 正确率为: {tmp_acc}')

    t2 = time.perf_counter_ns()

    print(f'SVM平均正确率为{tmp_acc}')
    print(f'SVM用时 {(t2 - t1) / 10000000}秒')

    toc = time.perf_counter_ns()
    acc_temp =tmp_acc #记录每次的精度
    acc_rem.append(acc_temp)            #将每次的精度存入列表
    time_temp = (t2 - t1)/1000000000 #记录每次的时间  / 1000 / 1000
    time_rem.append(time_temp)      #将每次的时间存入列表


print("*****************************************************")
for i in acc_rem:
    print(f'{i*100:0.2f}',)   #打印每次精度
acc_mean = np.mean(acc_rem) #求出平均精度
print("**：{:.2f}".format(acc_mean*100))  #打印平均精度

for i in time_rem:
    print(f'{i*1:0.2f}',)   #打印每次时间
time_mean = np.mean(time_rem)   #求出平均时间
print("**{:.2f}".format(time_mean))    #打印平均时间
print('---------------------以上为SVM算法----------------------') #运行程序


