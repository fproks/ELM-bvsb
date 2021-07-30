from elm import BvsbClassifier, BvsbUtils
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from  elm import elmUtils
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
print("-----------------以下为ELM-BVSB-（10次）--------------")
stdc = StandardScaler()  # 均值归一化
data.data = stdc.fit_transform(data.data / 16.0)
label_size = 0.05                       #已标记样本比例，分别取0.05-0.1-0.2-0.3-0.4
acc_rem = []  #初始化精度列表
for ii in range(10):
    (train_data, iter_data, test_data) = elmUtils.splitDataWithIter(data.data, data.target, label_size, 0.3)
    bvsbc = BvsbClassifier(train_data[0], train_data[1], iter_data[0], iter_data[1], test_data[0], test_data[1],
                           iterNum=0.1)
    bvsbc.createELM(n_hidden=1000, activation_func="tanh", alpha=1.0, random_state=0)
    bvsbc.X_test = test_data[0]
    bvsbc.Y_test = test_data[1]
    bvsbc.trainELMWithoutKNN()
    print(bvsbc.score(test_data[0], test_data[1]))
    acc_temp = bvsbc.score(test_data[0], test_data[1])  #记录每次的精度
    acc_rem.append(acc_temp)            #将每次的精度存入列表

print("*****************************************************")
for i in acc_rem:
    print(f'{i*100:0.2f}',)   #打印每次精度
acc_mean = np.mean(acc_rem) #求出平均精度
print("**{:.2f}".format(acc_mean*100))  #打印平均精度
print('---------------------以上为ELM-BVSB算法（10次）----------------------') #运行程序


