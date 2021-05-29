import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

_tmp=sio.loadmat("vggdata.mat")
data=_tmp["data"]
stand=_tmp["target"].flatten()
print(stand.shape)

#data=BvsbUtils.dimensionReductionWithPCA(data,0.95)
print(data.shape)
svm=SVC()
train_x,test_x,train_y,test_y=train_test_split(data,stand,test_size=0.95)
print(train_y.shape)
svm.fit(train_x,train_y)
print(svm.score(test_x,test_y))
