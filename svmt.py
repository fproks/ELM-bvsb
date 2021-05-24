import scipy.io as sio
import numpy as  np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = sio.loadmat("./Pavia.mat")["pavia"]
stand = sio.loadmat("./Pavia_gt.mat")["pavia_gt"]
data = sio.loadmat("./SalinasA.mat")["salinasA"]
stand = sio.loadmat("./SalinasA_gt.mat")["salinasA_gt"]

data = np.reshape(data, (-1, data.shape[2]))
stand = np.reshape(stand, (stand.shape[0] * stand.shape[1]))
print(data.shape)
print(stand.shape)
data = data[stand > 0, :]
stand = stand[stand > 0]
print(len(stand))
print(data.shape)

x_train, x_test, y_train, y_test = train_test_split(data, stand, test_size=0.95)
print(len(y_train))
scv = SVC()
scv.fit(x_train, y_train)
print(scv.score(x_test, y_test))
# l=scv.predict(x_test)
# print(len(l))
