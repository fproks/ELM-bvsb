from elm import BvsbClassifier, BvsbUtils
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from elm import BvsbClassifier, BvsbUtils
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from elm import BvsbClassifier, BvsbUtils
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from elm import ELMClassifier
from sklearn import datasets
from elm import elmUtils,BvsbUtils
import numpy as np

#data = datasets.fetch_olivetti_faces()  # 这是第1个
# data = sklearn.datasets.fetch_covtype()#这是第2个
data = datasets.load_iris()#这是第3个
# data = datasets.load_digits()#这是第4个  再前面加#屏蔽语句，把运行的打开
#data = datasets.load_wine()#这是第5个
# data = datasets.load_breast_cancer()#这是第6个

#data=elmUtils.coverDataFileToData("./data/abalone.data", targetIndex=-1, transformIndex=[0])
#data=elmUtils.readDataFileToData("data/balance-scale.data", targetIndex=0)
#data = datasets.fetch_olivetti_faces()  # 稀疏矩阵,必须转换和降维
# 数据集不全为数字时，needOneHot=True, target 不为数字时，needLabelEncoder=True
#data.data,data.target=elmUtils.processingData(data.data, data.target)
#data.data=BvsbUtils.dimensionReductionWithPCA(data.data,100) #kddcpu99 维度太高，必须进行降维
print(data.data.shape)
label_size=0.3

(train_data, test_data) = elmUtils.splitData(data.data, data.target, 1-label_size)
elmc = ELMClassifier(n_hidden=1000, activation_func='tanh', alpha=1.0, random_state=0)
elmc.fit(train_data[0], train_data[1])
print(elmc.score(test_data[0], test_data[1]))

#data = datasets.fetch_olivetti_faces()  # 这是第1个
# data = sklearn.datasets.fetch_covtype()#这是第2个
# data = datasets.load_iris()#这是第3个
# data = datasets.load_digits()#这是第4个  再前面加#屏蔽语句，把运行的打开
data = datasets.load_wine()#这是第5个
# data = datasets.load_breast_cancer()#这是第6个

