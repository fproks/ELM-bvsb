from elm import OSELM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import numpy as np

stdsc = StandardScaler()
data = datasets.load_digits()

label_size=0.3

data.data = stdsc.fit_transform(data.data / 16.0)
dgx_train, dgx_test, dgy_train, dgy_test = train_test_split(data.data, data.target, test_size=1-label_size)

oselm = OSELM(dgx_train, dgy_train, 1000,active_function="sigmoid")

print(oselm.score(dgx_test, dgy_test))


