from elm import ELMClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from elm import elmUtils

# stdsc = StandardScaler()
# digits = load_digits()
digits = datasets.load_digits()
# dgx = digits.data
# dgy = digits.target
# dgx, dgy = stdsc.fit_transform(digits.data / 16.0), digits.target
# print(dgy.shape)
# dgx_train, dgx_test, dgy_train, dgy_test = train_test_split(dgx, dgy, test_size=0.5)


(train_data, test_data) = elmUtils.splitData(digits.data, digits.target, 0.5)
elmc = ELMClassifier(n_hidden=1000, activation_func='tanh', alpha=1.0, random_state=0)
elmc.fit(train_data[0], train_data[1])
# elmc.score(dgx_test,dgy_test)
print(elmc.score(train_data[0], train_data[1]), elmc.score(test_data[0], test_data[1]))
