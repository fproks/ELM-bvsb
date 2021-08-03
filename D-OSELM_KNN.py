from elm import OSELM,elmUtils,BvsbUtils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from elm import BvsbClassifier, BvsbUtils
from elm import elmUtils

#---------------------------数据集----------------------------------
#data=elmUtils.readDataFileToData("./data/iris.data",targetIndex=-1)#xxxxxxxxxxxx
#data=elmUtils.readDataFileToData("./data/bezdekIris.data", targetIndex=-1)
#data = datasets.load_wine()#OK
#data = datasets.load_breast_cancer()#OK
#data=elmUtils.readDataFileToData("./data/seeds.data", delimiter = None, targetIndex=-1)#OK
#data = datasets.load_digits()
#data=elmUtils.readDataFileToData("./data/glass.data", targetIndex=-1)#xxxxxxxxxxxx
#data = datasets.fetch_olivetti_faces()#OK
#---------------------------数据集---------------------------------


data.target = data.target + 1
stdsc = StandardScaler()
label_size = 0.3#已标记样本比例，分别取0.05-0.1-0.2-0.3-0.4

print("-----------------以下为OSELM-KNN(本文)算法（10）---------------")
acc_rem = []  #初始化精度列表
for ii in range(10):
    #data.target = data.target + 1
    #stdsc = StandardScaler()
    #label_size = 0.5
    (train_data, iter_data, test_data) = elmUtils.splitDataWithIter(data.data, data.target, label_size, 0.3)
    iter_y = BvsbUtils.KNNClassifierResult(train_data[0], train_data[1], iter_data[0])
    bvsbc = BvsbClassifier(train_data[0], train_data[1], iter_data[0], iter_y, test_data[0], test_data[1],
                           iterNum=0.1)
    bvsbc.createOSELM(n_hidden=1000, active_function="sigmoid")
    bvsbc.trainOSELMWithKNNButBvsb()
    print(f'OSELM-BVSB-KNN 正确率为{bvsbc.score(test_data[0], test_data[1])}')

    acc_temp = bvsbc.score(test_data[0], test_data[1])  #记录每次的精度
    acc_rem.append(acc_temp)            #将每次的精度存入列表
for i in acc_rem:
    print(f'{i*100:0.2f}',)   #打印每次精度
acc_mean = np.mean(acc_rem) #求出平均精度
print("{:.2f}".format(acc_mean*100))  #打印平均精度

print('---------------------以上为OSELM-BVSB-KNN(10次)算法----------------------') #运行程序

