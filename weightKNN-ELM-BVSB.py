'''
@author: linhos
@email: linhos@126.com
@created: 2021.08.12
'''
from elm import BvsbClassifier, BvsbUtils
import numpy as np
from sklearn.preprocessing import StandardScaler
from elm import elmUtils
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets

# ---------------------------数据集----------------------------------
#data = elmUtils.readDataFileToData("./data/bezdekIris.data", targetIndex=-1, deleteIndex=[4])  # 效果不错 83-96-94-93-88
#data = datasets.load_wine()  #
data=elmUtils.readDataFileToData("./data/breast-cancer-wisconsin.data", targetIndex=-1,deleteIndex=[0]) #
# data=elmUtils.readDataFileToData("./uci_data/Seeds.data", delimiter=None,targetIndex=[7]) #
# data = datasets.load_digits()
# data=elmUtils.readDataFileToData("./uci_data/glass.data", targetIndex=-1) # 56-78-75-77-86
# data = datasets.fetch_olivetti_faces()
# ---------------------------数据集---------------------------------
print("-----------------以下为ELM-BVSB-KNN加权算法（10次）---------------")
stdc = StandardScaler()
label_size = 0.08  # 已标记样本比例为10%
acc_rem = []  # 初始化精度列表
#data.data = stdc.fit_transform(data.data / 16.0)
data.target = LabelEncoder().fit_transform(data.target)

bvsbc = None
hidden_nums = 1000  # 隐层结点数量
select_h = 120  # 选出置信度高的前h个样本
for ii in range(10):
    (train_data, iter_data, test_data) = elmUtils.splitDataWithIter(data.data, data.target, label_size, 0.72)
    X_train = train_data[0].copy()
    Y_train = train_data[1].copy()
    X_iter = iter_data[0].copy()
    len_iter = len(X_iter)
    i = 1
    while len(X_iter) > (len_iter / 2):
        nbr = BvsbUtils.KNNClassifier(X_train, Y_train)  # KNN
        # iter_y=nbr.predict(X_iter)
        pred = nbr.predict_proba(X_iter)
        iter_y = np.argmax(pred, axis=1)
        classMax = np.max(pred, axis=1)  # 获取未标记样本的最大隶属度
        sortIndex = np.argsort(classMax)  # 排序后的原下标
        iter_index = np.sort(sortIndex[-select_h:])
        sort_h_y = iter_y[iter_index]  # 返回原来数据置信度最大的h个数
        sort_h_data = X_iter[iter_index]
        len_curr_iter = len(sort_h_y)
        bvsbc = BvsbClassifier(X_train, Y_train, sort_h_data, sort_h_y, test_data[0], test_data[1], iterNum=0.1)
        bvsbc.createELM(n_hidden=hidden_nums, activation_func="tanh", alpha=1.0, random_state=0)
        _data_index = bvsbc.fitAndGetUpdateDataIndex(limit=int(0.2 * len_curr_iter))
        if len(_data_index) != 0:
            X_train = np.r_[bvsbc.X_train, sort_h_data[_data_index]]
            Y_train = np.r_[bvsbc.Y_train, sort_h_y[_data_index]]
            X_iter = np.delete(X_iter, iter_index[_data_index], axis=0)
        else:
            print("没有数据被加入训练集，训练结束")
            break
        print(f"第{ii} 次训练,第{i}次迭代: 正确率为:{bvsbc.score(test_data[0], test_data[1])}")
        i += 1
    acc_temp = bvsbc.score(test_data[0], test_data[1])  # 记录每次的精度
    acc_rem.append(acc_temp)  # 将每次的精度存入列表

print("***************ELM-BVSB-KNN加权算法（10次精度）********************")
for i in acc_rem:
    print(f'{i * 100:0.2f}', )  # 打印每次精度
acc_mean = np.mean(acc_rem)  # 求出平均精度
print("{:.2f}".format(acc_mean * 100))  # 打印平均精度
print("-----------------ELM-BVSB-KNN加权算法（10次）---------------")
