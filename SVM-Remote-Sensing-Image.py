import scipy.io as sio
import numpy as  np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import cm

'''
SVM
'''
data = sio.loadmat("./data/PaviaU.mat")["paviaU"]
stand = sio.loadmat("./data/PaviaU_gt.mat")["paviaU_gt"]
[M, N] = stand.shape
data = np.reshape(data, (-1, data.shape[2]))
stand = np.reshape(stand, (stand.shape[0] * stand.shape[1]))
label_index = np.nonzero(stand)
data = data[stand > 0, :]
stand = stand[stand > 0]

print(f'样本个数:{len(label_index)}')
label_size = 0.05



x_train, x_test, y_train, y_test = train_test_split(data, stand, test_size=1 - label_size)
scv = SVC()
scv.fit(x_train, y_train)
print(f' overall accuracy :{scv.score(data, stand) * 100:.2f}')
y_pred = scv.predict(data)
confusion = metrics.confusion_matrix(stand, y_pred)
np.savetxt(f"SVM_matrix_{label_size}.csv", confusion, fmt="%d", delimiter=",")
average_accuracy = np.mean(metrics.precision_score(y_pred, stand, average=None))
print(f'average accuracy: {(average_accuracy * 100):.2f}')
print(f'kappa is :{metrics.cohen_kappa_score(stand, y_pred)}')

predict = np.zeros(M * N)
predict[label_index] = y_pred
predict = np.reshape(predict, (M, N))
img = plt.imshow(predict.astype(int), cm.jet)
plt.show()
sio.savemat('res.mat', {'y_pred': predict})



