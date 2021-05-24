from  sklearn.svm import SVC
from elm import BvsbClassifier, BvsbUtils
from sklearn.model_selection import train_test_split
from sklearn import datasets
from elm import elmUtils

from sklearn.preprocessing import StandardScaler


import time
stdsc = StandardScaler()
data=datasets.fetch_covtype()
data.data=stdsc.fit_transform(data.data)/16.0
data.data,_, data.target,_=train_test_split(data.data,data.target,test_size=0.7)
label_size=0.05

acc=0
print(data.data.shape)
#print(data.target.shape)


t1=time.perf_counter_ns()

for i in range(10):
    train, test = elmUtils.splitData(data.data, data.target, 1 - label_size, True)
    svm=SVC()
    svm.fit(train[0],train[1])
    tmp_acc=svm.score(test[0], test[1])
    print(f'SVM 正确率为: {tmp_acc}')
    acc+=tmp_acc
t2=time.perf_counter_ns()

print(f'SVM平均正确率为{acc/10}')
print(f'SVM用时 {(t2-t1)/1000/1000/10}毫秒')