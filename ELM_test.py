from elm import ELMClassifier
from sklearn import datasets
from elm import elmUtils,BvsbUtils


data = datasets.fetch_kddcup99()  # 稀疏矩阵,必须转换和降维
# 数据集不全为数字时，needOneHot=True, target 不为数字时，needLabelEncoder=True
data.data,data.target=elmUtils.processingData(data.data, data.target)
data.data=BvsbUtils.dimensionReductionWithPCA(data.data,100) #kddcpu99 维度太高，必须进行降维
print(data.data.shape)
label_size=0.3

(train_data, test_data) = elmUtils.splitData(data.data, data.target, 1-label_size)
elmc = ELMClassifier(n_hidden=1000, activation_func='tanh', alpha=1.0, random_state=0)
elmc.fit(train_data[0], train_data[1])
print(elmc.score(train_data[0], train_data[1]), elmc.score(test_data[0], test_data[1]))


