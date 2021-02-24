from elm import ELMClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

stdsc = StandardScaler()

# digits = load_digits()
digits = datasets.fetch_olivetti_faces()
dgx = digits.data
dgy = digits.target

dgx, dgy = stdsc.fit_transform(digits.data / 16.0), digits.target

print(dgy.shape)

dgx_train, dgx_test, dgy_train, dgy_test = train_test_split(dgx, dgy, test_size=0.2)
X_train, X_iter, Y_train, Y_iter = train_test_split(dgx_train, dgy_train, test_size=0.5)

elmc = ELMClassifier(n_hidden=1000, activation_func='tanh', alpha=1.0, random_state=0)
elmc.fit(X_train, Y_train)
# elmc.score(dgx_test,dgy_test)
print(Y_train.shape)
print(dgy_test.shape)
print(elmc.score(X_train, Y_train), elmc.score(dgx_test, dgy_test))
