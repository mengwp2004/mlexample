#encoding=utf-8

import matplotlib.pyplot as plt
import csv
import numpy as np
#with open('sh300.csv') as csvfile:
#with open('600783.csv') as csvfile:
from  array import *

#from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.metrics import confusion_matrix
from sklearn.qda import QDA
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV

datalen = 5000


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def getdata():
  X= np.zeros((datalen,334))
  y = array('f')
  i =0

  with open('HFT_XY_unselected.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row  in reader:
           
           #l= row.keys()
           #print(len(l))
           #print(l)
           #print(row['realY'])
           #print(row['predictY'])
           y.append(float(row['realY']))
           del row['realY']
           row1 = array('f')
           for unit in row.values():
               #print(type(unit))
               row1.append((float(unit)))
           #print(type(row1))
           #m = np.asarray(row1)
           #if (i == 1) :
           #   X= row1
           #   i =i+1
           #   continue
           #print("====")
           #print(type(row1))
           #print(type(X))
           #print(len(X))
           X[i] = np.asarray(row1)
           #np.vstack((y,float(row['realY'])))
           #print(type(X))
           #X.append(row1)
           #y.append(float(row['realY']))
           #print(row.values())
           #print(type(row))
           #print len(row)
           #print row
           i = i+1
           #print '-------------------------'
           if i>= datalen :
                break

  return X,np.asarray(y)


if __name__ == "__main__":
    # Create a lagged series of the S&P500 US stock market index
    X,y=getdata()
    print(X)
    print(X.shape)
    print(type(X))


    print(y)
    print(y.shape)
    print(type(y))
    print(y.shape)

    #
    #n_components = 30
    #pca = PCA(n_components=n_components)
    #X = pca.fit_transform(X)

    print(X.shape)
    print(type(X))
    print(y.shape)
    print(type(y))
    print(y.shape)


    estimator = Ridge()
    #selector = RFECV(estimator, step=1, cv=5)
    
    selector = ExtraTreesRegressor(n_estimators=50)
    selector = selector.fit(X, y)
    print("Optimal number of features : %d" % selector.n_features_)
    X= selector.transform(X)
    print(X.shape)
  
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

    # build a classifier
    clf = RandomForestRegressor(n_estimators=20)
    # use a full grid over all parameters
    param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False]}

    # run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    start = time()
    grid_search.fit(X, y)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)



    #check feature important
    #use for classfication

    #forest = ExtraTreesClassifier(n_estimators=250,
    #                          random_state=0)
    #forest.fit(X, y)
    #importances = forest.feature_importances_
    #print("important:")
    #print(importances)


    # Create the (parametrised) models
    print("Hit Rates/Confusion Matrices:\n")
    #models = [("LR", LogisticRegression()), 
    #          ("LDA", LDA()), 
    #          ("QDA", QDA()),
    #          ("LSVC", LinearSVC()),
    #          ("RSVM", SVC(
    #          	C=1000000.0, cache_size=200, class_weight=None,
    #            coef0=0.0, degree=3, gamma=0.0001, kernel='rbf',
    #            max_iter=-1, probability=False, random_state=None,
    #            shrinking=True, tol=0.001, verbose=False)
    #          ),
    #          ("RF", RandomForestClassifier(
    #          	n_estimators=1000, criterion='gini', 
    #            max_depth=None, min_samples_split=2, 
    #            min_samples_leaf=1, max_features='auto', 
    #            bootstrap=True, oob_score=False, n_jobs=1, 
    #            random_state=None, verbose=0)
    #          )]
    
    models =[ ("LR", LinearRegression()),
              ("ElasticNet",ElasticNet()),
              ("Ridge",Ridge()),
              ("RandomForestRegressor",RandomForestRegressor(n_estimators=50)),
              ("SGDRegressor",SGDRegressor()),
              ("DecisionTreeRegressor",DecisionTreeRegressor()),
              ("ExtraTreesRegressor",ExtraTreesRegressor(n_estimators=50))] 

    # Iterate through the models
    for m in models:
        
        # Train each of the models on the training set
        m[1].fit(X_train, y_train)

        # Make an array of predictions on the test set
        pred = m[1].predict(X_test)
        # The coefficients
        print(m[1])
        #print('Coefficients: \n', m[1].coef_)
        # The mean squared error
        print("Mean squared error: %.2f"
             % mean_squared_error(y_test, pred))
        # Explained variance score: 1 is perfect prediction
        print('R^2 score: %.2f' % r2_score(y_test, pred))
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    plt.plot(y_test)
    ax2 = fig.add_subplot(2,2,2)
    #print(pred)
    plt.plot(pred)
    plt.show()
