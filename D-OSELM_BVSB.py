from elm import BvsbClassifier, BvsbUtils
from sklearn import datasets
from elm import elmUtils
from sklearn.preprocessing import StandardScaler
import time
import numpy as np

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


print("-----------------以下为OSELM-BVSB算法---------------")
data.target = data.target + 1
stdsc = StandardScaler()
label_size = 0.3#已标记样本比例，分别取0.05-0.1-0.2-0.3-0.4
acc_rem = []  #初始化精度列表
for ii in range(10):#循环次数
    print(f'数据集大小问{data.target.size}')
    data.data = StandardScaler().fit_transform(data.data)
    (train, iter, test) = elmUtils.splitDataWithIter(data.data, data.target,
                                                     0.2, 0.3)
    print(f'训练集大小为{train[1].size}')
    print(f'迭代训练集大小为{iter[1].size}')
    print(f'测试集大小为{test[1].size}')

    #tic = time.perf_counter_ns()
    bvsbc = BvsbClassifier(train[0], train[1], iter[0], iter[1], test[0], test[1], iterNum=0.1)
    bvsbc.createOSELM(n_hidden=1000)
    bvsbc.trainOSELMWithoutKNN()
    #toc = time.perf_counter_ns()

    print(f'OSELM-BVSB 正确率为{bvsbc.score(test[0], test[1])}')
   # print(f'OSELM-BVSB项目用时:{(toc - tic) / 1000 / 1000} ms')

    acc_temp =bvsbc.score(test[0], test[1]) #记录每次的精度
    acc_rem.append(acc_temp)            #将每次的精度存入列表
print("*****************************************************")
for i in acc_rem:
    print(f'{i*100:0.2f}',)   #打印每次精度
acc_mean = np.mean(acc_rem) #求出平均精度
print("{:.2f}".format(acc_mean*100))  #打印平均精度
print('---------------------以上为OSELM-BVSB算法----------------------') #运行程序


