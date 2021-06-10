from sklearn.calibration import CalibratedClassifierCV
from elm import ELMClassifier,OSELM
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

data = load_digits()

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target,test_size=0.7)
elmc = ELMClassifier(n_hidden=1000, activation_func='tanh', alpha=1.0, random_state=0)
oselm = OSELM(x_train, y_train, 1000,active_function="sigmoid")

#cccv=CalibratedClassifierCV(oselm,cv=2,method="isotonic")
cccv=CalibratedClassifierCV(elmc,cv=2,method="isotonic")
cccv.fit(x_train,y_train)
r=cccv.predict_proba(x_test)


#https://blog.csdn.net/yolohohohoho/article/details/99679680

print(r)
