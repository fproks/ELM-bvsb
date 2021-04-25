from elm import OSELM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import numpy as np

stdsc = StandardScaler()
digits = datasets.load_digits()
dgx = digits.data
dgy = digits.target

dgx = stdsc.fit_transform(dgx / 16.0)
dgx_train, dgx_test, dgy_train, dgy_test = train_test_split(dgx, dgy, test_size=0.5)

inputs = dgx_train.shape[1]
outputs = dgy.max() + 1
oselm = OSELM(dgx_train, dgy_train, 1000)

print(oselm.score(dgx_test, dgy_test))


