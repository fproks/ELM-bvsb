from elm import BvsbClassifier, BvsbUtils
# from sklearn.model_selection import train_test_split
# from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from  elm import elmUtils
# from sklearn.neighbors import KNeighborsClassifier
# import operator
from sko.PSO import PSO


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
print("-----------------以下为ELM-BVSB-KNN(本文)算法（10次）---------------")
stdc = StandardScaler()
label_size = 0.10                     #已标记样本比例，分别取0.05-0.1-0.2-0.3-0.4
acc_rem = []  #初始化精度列表
for ii in range(1):
    data.data = stdc.fit_transform(data.data / 16.0)
    (train_data, iter_data, test_data) = elmUtils.splitDataWithIter(data.data, data.target, label_size, 0.3)
    iter_y ,nbr = BvsbUtils.KNNClassifierResult(train_data[0], train_data[1], iter_data[0])  #KNN
    # sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 选择发生频率最高的元素标签
    ######
    # 调用sklearn库函数
    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf.fit(train_data, data.target)

    # 2 15
    tempIndex=20
    tempData=data.data[0:tempIndex]
    tempTarget=data.target[0:tempIndex]

    pred = nbr.predict_proba(iter_data[0])

    classMax=np.max(pred,axis=1)  # 获取未标记样本的最大隶属度

    # classMaxMaxIndex=np.argmax(classMax) # 置信度最大
    # sortedClassCount = sorted(classMaxMaxIndex.items(), key=operator.itemgetter(1), reverse=True)  # 选择发生频率最高的元素标签
    sortIndex= np.argsort(classMax)  # 排序后的原下标
    select_h = 120  # 选出置信度搞的前h个样本
    sort_h_data = sortIndex[sorted(np.argsort(classMax)[-select_h:])]  # 返回原来数据置信度最大的h个数
    # pass
    # print(pred)
    # score = nbr.score(data.data, data.target)
    ################################
    # bvsbc = BvsbClassifier(train_data[0], train_data[1], iter_data[0], iter_y, test_data[0], test_data[1],iterNum=0.1)
    bvsbc = BvsbClassifier(train_data[0], train_data[1], iter_data[0], sort_h_data, test_data[0], test_data[1], iterNum=0.1)
    bvsbc.createELM(n_hidden=1000, activation_func="tanh", alpha=1.0, random_state=0)
    bvsbc.X_test = test_data[0]
    bvsbc.Y_test = test_data[1]
    bvsbc.trainELMWithBvsb()
    print("+++++++++++++++++++")
    print(bvsbc.score(test_data[0], test_data[1]))

    acc_temp = bvsbc.score(test_data[0], test_data[1]) #记录每次的精度
    acc_rem.append(acc_temp)            #将每次的精度存入列表
print("*****************************************************")
for i in acc_rem:
    print(f'{i*100:0.2f}',)   #打印每次精度
acc_mean = np.mean(acc_rem) #求出平均精度
print("**{:.2f}".format(acc_mean*100))  #打印平均精度
print('---------------------以上为ELM-BVSB-KNN(本文)算法（10次）----------------------') #运行程序

