from elm import ELMClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

stdsc = StandardScaler()

digits = load_digits()


np.savetxt("y,csv", digits.target, delimiter=",", fmt="%d")
np.savetxt("111.csv", digits.data, delimiter=",")

dgx, dgy = stdsc.fit_transform(digits.data / 16.0), digits.target

print(dgy)

dgx_train, dgx_test, dgy_train, dgy_test = train_test_split(dgx, dgy, test_size=0.2)

elmc = ELMClassifier(n_hidden=500, activation_func='hardlim', alpha=1.0, random_state=0)
elmc.fit(dgx_train, dgy_train)

percent=elmc.predict_with_percentage(dgx_test)
np.savetxt("111.csv",percent)
ps=np.sort(percent)[:,-2:]
ps_t=ps[:,-1]-ps[:,-2]
num=int(np.ceil(np.shape(ps_t)[0]*0.2))
arg_ps=np.argsort(ps_t)
arg_ps=arg_ps[:num]
nd=percent[arg_ps]
print(nd)
#print(elmc.score(dgx_train, dgy_train), elmc.score(dgx_test, dgy_test))
