from elm import OSELM,elmUtils,BvsbUtils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import numpy as np

stdsc = StandardScaler()
data = datasets.fetch_kddcup99()

label_size=0.3


data.data,data.target=elmUtils.processingData(data.data,data.target)

print(data.data.shape)
#data.data=BvsbUtils.dimensionReductionWithPCA(data.data,100)
dgx_train, dgx_test, dgy_train, dgy_test = train_test_split(data.data, data.target, test_size=1-label_size)

oselm = OSELM(dgx_train, dgy_train, 1000,active_function="sigmoid")

print(oselm.score(dgx_test, dgy_test))


