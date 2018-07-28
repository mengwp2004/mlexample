from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np

#Load boston housing dataset as an example
boston = load_boston()

print(type(boston))

X = boston["data"]
print(type(X))
Y = boston["target"]
print(X.shape)
print(X)
print(Y.shape)
print(Y)
names = boston["feature_names"]
print(names)


rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
for i in range(X.shape[1]):
     score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                              cv=ShuffleSplit(len(X), 3, .3))
     print(score)
     scores.append((round(np.mean(score), 3), names[i]))
print sorted(scores, reverse=True)

print(ShuffleSplit(len(X), 3, .3))

