from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston



def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)

boston = load_boston()
scaler = StandardScaler()
X = scaler.fit_transform(boston["data"])
print(X.shape)
Y = boston["target"]
names = boston["feature_names"]
print(names)
lasso = Lasso(alpha=.001)
lasso.fit(X, Y)
print(lasso.coef_)
#print ("Lasso model: ", pretty_print_linear(lasso.coef_, names, sort = True))
