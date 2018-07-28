from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import numpy as np

size = 100

#We run the method 10 times with different random seeds
for i in range(10):
    print "Random seed %s" % i
    np.random.seed(seed=i)
    X_seed = np.random.normal(0, 1, size)
    X1 = X_seed + np.random.normal(0, .1, size)
    X2 = X_seed + np.random.normal(0, .1, size)
    X3 = X_seed + np.random.normal(0, .1, size)
    Y = X1 + X2 + X3 + np.random.normal(0, 1, size)
    X = np.array([X1, X2, X3]).T


    lr = LinearRegression()
    lr.fit(X,Y)
    #print "Linear model:", pretty_print_linear(lr.coef_)
    print "Linear model:",lr.coef_


    ridge = Ridge(alpha=10)
    ridge.fit(X,Y)
    print ("Ridge model:", ridge.coef_)
    #print ("Ridge model:", pretty_print_linear(ridge.coef_))
