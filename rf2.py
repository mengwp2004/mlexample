from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np

size = 10000
np.random.seed(seed=10)
X_seed = np.random.normal(0, 1, size)
X0 = X_seed + np.random.normal(0, .1, size)
X1 = X_seed + np.random.normal(0, .1, size)
X2 = X_seed + np.random.normal(0, .1, size)
X = np.array([X0, X1, X2]).T
Y = X0 + X1 + X2

rf = RandomForestRegressor(n_estimators=20, max_features=2)
rf.fit(X, Y);
print ("Scores for X0, X1, X2:", map(lambda x:round (x,3),
                                    rf.feature_importances_))
