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



data.target=data.target+1
stdc = StandardScaler()
print("-----------------以下为OSELM算法（10）---------------")
label_size = 0.3#已标记样本比例，分别取0.05-0.1-0.2-0.3-0.4
acc_rem = []  # 初始化精度列表
for ii in range(10):#循环10次
    #data.target = data.target + 1
    #stdsc = StandardScaler()
    #label_size = 0.05
    #data.data,data.target=elmUtils.processingData(data.data,data.target)

    print(data.data.shape)
    # data.data=BvsbUtils.dimensionReductionWithPCA(data.data,100)
    dgx_train, dgx_test, dgy_train, dgy_test = train_test_split(data.data, data.target, test_size=1 - label_size)
    oselm = OSELM(dgx_train, dgy_train, 2000, active_function="sigmoid")
    print(oselm.score(dgx_test, dgy_test))

    acc_temp = oselm.score(dgx_test, dgy_test)  # 记录每次的精度
    acc_rem.append(acc_temp)  # 将每次的精度存入列表

print("*****************************************************")
for i in acc_rem:
    print(f'{i*100:0.2f}', )  # 打印每次精度
acc_mean = np.mean(acc_rem)  # 求出平均精度
print("{:.2f}".format(acc_mean*100))  #打印平均精度

#or i in time_rem:
#    print(f'{i * 1:0.2f}', )  # 打印每次时间
#time_mean = np.mean(time_rem)  # 求出平均时间
#print("**{:.2f}".format(time_mean))  # 打印平均时间
print('---------------------以上为OSELM算法（10）----------------------')  # 运行程序
