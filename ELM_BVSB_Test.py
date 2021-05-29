from elm import BvsbClassifier, BvsbUtils
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from  elm import elmUtils
import time

print("------------")
# digits = load_digits()
digits = datasets.load_breast_cancer()
dgx = digits.data
stdc = StandardScaler()  # 均值归一化

dgy = digits.target
print("数据个数:%d" % dgy.size)
dgx, dgy = stdc.fit_transform(digits.data / 16.0), digits.target

label_size=0.3
(train_data, iter_data, test_data) = elmUtils.splitDataWithIter(dgx,dgy, label_size, 0.2)

tic = time.perf_counter_ns()
bvsbc = BvsbClassifier(train_data[0], train_data[1], iter_data[0], iter_data[1], test_data[0], test_data[1],
                       iterNum=0.1)
bvsbc.createELM(n_hidden=1000, activation_func="tanh", alpha=1.0, random_state=0)
bvsbc.X_test = test_data[0]
bvsbc.Y_test = test_data[1]
bvsbc.trainELMWithoutKNN()
toc = time.perf_counter_ns()

print("+++++++++++++++++++")
print(bvsbc.score(test_data[0], test_data[1]))
print("ELM-BVSB 项目用时:%d" % ((toc - tic) / 1000 / 1000))
