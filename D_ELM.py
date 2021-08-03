from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from elm import ELMClassifier
from elm import elmUtils
print("-----------------以下为ELM算法（10次）---------------")

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

#data.target = data.target + 1#(这里什么时候要什么时候不要)
stdc = StandardScaler()
label_size = 0.1                    #已标记样本比例，分别取0.05-0.1-0.2-0.3-0.4
acc_rem = []  #初始化精度列表
for ii in range(1):
    # 数据集不全为数字时，needOneHot=True, target 不为数字时，needLabelEncoder=True
    data.data, data.target = elmUtils.processingData(data.data, data.target)
    print(data.data.shape)
    (train_data, test_data) = elmUtils.splitData(data.data, data.target, 1 - label_size)
    elmc = ELMClassifier(n_hidden=1000, activation_func='tanh', alpha=1.0, random_state=0)
    elmc.fit(train_data[0], train_data[1])
    print(elmc.score(test_data[0], test_data[1]))

    acc_temp = elmc.score(test_data[0], test_data[1]) #记录每次的精度
    acc_rem.append(acc_temp)            #将每次的精度存入列表

print("*****************************************************")
for i in acc_rem:
    print(f'{i*100:0.2f}',)   #打印每次精度
acc_mean = np.mean(acc_rem) #求出平均精度
print("{:.2f}".format(acc_mean*100))  #打印平均精度
print("-----------------以下为ELM算法（10次）---------------")



