from elm import ELMClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

stdsc = StandardScaler()

# digits = load_digits()
digits = datasets.load_digits()
dgx = digits.data
dgy = digits.target

dgx, dgy = stdsc.fit_transform(digits.data / 16.0), digits.target

print(dgy.shape)

dgx_train, dgx_test, dgy_train, dgy_test = train_test_split(dgx, dgy, test_size=0.5)

elmc = ELMClassifier(n_hidden=1000, activation_func='tanh', alpha=1.0, random_state=0)
elmc.fit(dgx_train, dgy_train)
# elmc.score(dgx_test,dgy_test)
print(elmc.score(dgx_train, dgy_train), elmc.score(dgx_test, dgy_test))
