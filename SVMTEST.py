from sklearn.svm import SVC
from elm import BvsbClassifier, BvsbUtils
from sklearn.model_selection import train_test_split
from sklearn import datasets
from elm import elmUtils

from sklearn.preprocessing import StandardScaler

import time

stdsc = StandardScaler()
#data = datasets.fetch_covtype()


"""
covtype.data 数据规模过大，(58万) 不要使用30%进行分类
"""
data=elmUtils.readDataFileToData("data/abalone.data", targetIndex=-1, transformIndex=[0])
#data=elmUtils.readDataFileToData("./data/balance-scale.data", targetIndex=0)
#data=elmUtils.readDataFileToData("./data/covtype.data",targetIndex=-1,dtype=float)
#data=elmUtils.readDataFileToData("./data/ecoli.data",targetIndex=-1,deleteIndex=[0],delimiter=None)
#data=elmUtils.readDataFileToData("./data/ionosphere.data",targetIndex=-1)
#data=elmUtils.readDataFileToData("./data/iris.data",targetIndex=-1)
#data=elmUtils.readDataFileToData("./data/wine.data",targetIndex=0,dtype=float)
#data=elmUtils.readDataFileToData("./data/yeast.data",targetIndex=-1,deleteIndex=[0],delimiter=None)
#data=elmUtils.readDataFileToData("./data/zoo.data",targetIndex=-1,deleteIndex=[0])
print(data.data.shape)
#data.data = stdsc.fit_transform(data.data) / 16.0

label_size = 0.3

t1 = time.perf_counter_ns()

train, test = elmUtils.splitData(data.data, data.target, 1 - label_size, True)
svm = SVC()
svm.fit(train[0], train[1])
tmp_acc = svm.score(test[0], test[1])
print(f'SVM 正确率为: {tmp_acc}')

t2 = time.perf_counter_ns()

print(f'SVM平均正确率为{tmp_acc}')
print(f'SVM用时 {(t2 - t1) / 1000 / 1000 }毫秒')
